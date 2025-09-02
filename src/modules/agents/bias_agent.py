from typing import Tuple
import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SN

from modules.layer.attn_utils import MultiHeadGroupAttn, GroupTemp

class MHA_QAgent(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        H = self.hidden_size = int(getattr(args, "hidden_size", 64))
        self.n_heads = int(getattr(args, "n_head", 4))
        self.A_move = int(args.output_normal_actions)

        # Flags
        self.use_group_card_temp = bool(getattr(args, "use_group_card_temp", False))
        self.spectral_norm = bool(getattr(args, "spectral_norm", False))
        self.q_strat = str(getattr(args, "q_strat", None))
        self.use_film = bool(getattr(args, "use_film", False))
        self.mask_invalid_targets = bool(getattr(args, "mask_invalid_targets", False))

        # Parse input dims (SPECTRA convention)
        self.own_feats_dim, self.enemy_feats_dim, self.ally_feats_dim = input_shape
        self.enemy_feats_dim = self.enemy_feats_dim[-1]
        self.ally_feats_dim = self.ally_feats_dim[-1]

        if getattr(args, "obs_agent_id", False):
            self.own_feats_dim += 1
        if getattr(args, "obs_last_action", False):
            self.own_feats_dim += 1

        # Embeddings
        self.embed_own   = nn.Linear(self.own_feats_dim, H)
        self.embed_ally  = nn.Linear(self.ally_feats_dim, H)
        self.embed_enemy = nn.Linear(self.enemy_feats_dim, H)

        # Spectral norm
        if self.spectral_norm:
            self.embed_own = SN(self.embed_own)
            self.embed_ally = SN(self.embed_ally)
            self.embed_enemy = SN(self.embed_enemy)

        # Group cardinality temperature
        if self.use_group_card_temp:
            self.groupTemp_A = GroupTemp()
            self.groupTemp_E = GroupTemp()

        # Multi‑head group cross‑attention (own query → allies / enemies)
        self.attn_allies  = MultiHeadGroupAttn(H, self.n_heads)
        self.attn_enemies = MultiHeadGroupAttn(H, self.n_heads)

        # Reccurent core
        self.gru = nn.GRUCell(3*H, H)

        # Movements heads
        self.q_move = nn.Linear(H, self.A_move)
        if self.spectral_norm:
            self.q_move = SN(self.q_move)

        # Shoot heads
        self.z_to_K = nn.Linear(H, H, bias=False)
        self.E_to_K = nn.Linear(H, H, bias=False)
        if self.spectral_norm:
            self.z_to_K = SN(self.z_to_K)
            self.E_to_K = SN(self.E_to_K)

        # (Optional) FiLM from aggregated enemy context
        if self.use_film:
            self.film = nn.Sequential(
                nn.Linear(H, H), nn.ReLU(inplace=True),
                nn.Linear(H, 2*H)  # splits to gamma, beta
            )

        # Bias
        if self.q_strat == "opp_bias":
            self.family_bias = nn.Linear(H, 1)
        elif self.q_strat == "affine_bias":
            self.calib = nn.Sequential(
                nn.Linear(H, H), nn.ReLU(inplace=True),
                nn.Linear(H, 4)  # -> [gamma_m, beta_m, gamma_s, beta_s]
                )


    def init_hidden(self):
        # controller expands this to [bs, n_agents, H]
        return self.embed_own.weight.new_zeros(1, self.hidden_size)

    def forward(self, inputs: Tuple, prev_h):
        """
        Inputs: (bs, own[Bn,1,Do], allies[Bn,Na,Da], enemies[Bn,Ne,De], embedding_indices)
        Output: Q [Bn,1, A_move + Ne], hidden' [bs,n_agents,H]
        """
        bs, own_raw, ally_raw, enemy_raw, embedding_indices = inputs
        Bn = own_raw.shape[0]
        nA = ally_raw.shape[1]
        nE = enemy_raw.shape[1]
        H = self.hidden_size

        # masks (all‑zeros rows are padding)
        ally_mask  = ~th.all(ally_raw  == 0, dim=-1)      # [Bn,nA]
        enemy_mask = ~th.all(enemy_raw == 0, dim=-1)      # [Bn,nE]

        # Optionally append id / last action to own obs (algo‑side flags)
        if getattr(self.args, "obs_agent_id", False):
            agent_idx = embedding_indices[0].reshape(-1, 1, 1)
            own_raw = th.cat([own_raw, agent_idx], dim=-1)
        if getattr(self.args, "obs_last_action", False):
            last_act = embedding_indices[-1].reshape(-1, 1, 1)
            own_raw = th.cat([own_raw, last_act], dim=-1)

        # Embeddings
        e_own = self.embed_own(own_raw).squeeze(1)        # [Bn,H]
        A = self.embed_ally(ally_raw)                     # [Bn,nA,H]
        E = self.embed_enemy(enemy_raw)                   # [Bn,nE,H]

        # (OPTIONAL) Group temps for cardinality neutralization
        temp_A = temp_E = None
        if self.use_group_card_temp:
            # all [Bn]
            temp_A = self.groupTemp_A(ally_mask.float().sum(-1))
            temp_E = self.groupTemp_E(enemy_mask.float().sum(-1))

        # Cross‑attn summaries (query = own, groups = allies/enemies)
        zA, _ = self.attn_allies(e_own, A, ally_mask, group_temp=temp_A)   # [Bn,H]
        zE, _ = self.attn_enemies(e_own, E, enemy_mask, group_temp=temp_E) # [Bn,H]

        # Fuse
        u_cat = th.cat([e_own, zA, zE], dim=-1)  # [Bn,3H]

        prev_h = prev_h.view(Bn, H)  # [Bn, H]
        z = self.gru(u_cat, prev_h)                # [Bn, H]
        hidden_out = z.view(bs, -1, H)             # [B, n_agents, H]

        # ------------- Move actions -------------
        logits_move = self.q_move(z)                      # [Bn, A_move]

        # FiLM on per-enemy features using aggregated enemy context
        if self.use_film:
            gamma, beta = self.film(zE).chunk(2, dim=-1)       # each [Bn, H]
            # small bounded modulation for stability
            gamma, beta = th.tanh(gamma)*0.1, th.tanh(beta)*0.1
            E = E * (1+gamma.unsqueeze(1)) + beta.unsqueeze(1)

        # ------------- Shoot actions -------------
        zk = self.z_to_K(z)                               # [Bn, H]
        Ek = self.E_to_K(E)                               # [Bn, Ne, H]
        logits_shoot = th.einsum("bd,bnd->bn", zk, Ek)    # [Bn, Ne]

        # Mask invalid enemies
        if self.mask_invalid_targets:
            logits_shoot = logits_shoot.masked_fill(~enemy_mask, th.finfo(logits_shoot.dtype).min)
       
        # ---- ------------- ----
        if self.q_strat == "opp_bias":
            b = self.family_bias(z)                       # [Bn, 1]
            logits_move = logits_move + b
            logits_shoot = logits_shoot - b
        elif self.q_strat == "affine_bias":
            gamma_m, beta_m, gamma_s, beta_s = self.calib(z).chunk(4, dim=-1)
            gamma_m = F.softplus(gamma_m) + 1e-3
            gamma_s = F.softplus(gamma_s) + 1e-3

            logits_move = logits_move * gamma_m + beta_m    # [Bn, A_move]
            logits_shoot = logits_shoot * gamma_s + beta_s  # [Bn, nE]
        
        Q = th.cat([logits_move, logits_shoot], dim=-1).unsqueeze(1)  # [Bn,1,A_move+nE]

        return Q, hidden_out