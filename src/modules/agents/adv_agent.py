from typing import Tuple
import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modules.layer.attn_utils import MultiHeadGroupAttn  # reuse your existing light utility


class MHA_QAgent(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        self.hidden_size = int(getattr(args, "hidden_size", 64))
        self.n_heads = int(getattr(args, "n_head", 4))
        self.A_move = int(args.output_normal_actions)

        # Parse input dims (SPECTRA convention)
        self.own_feats_dim, self.enemy_feats_dim, self.ally_feats_dim = input_shape
        self.enemy_feats_dim = self.enemy_feats_dim[-1]
        self.ally_feats_dim = self.ally_feats_dim[-1]
        if getattr(args, "obs_agent_id", False):
            self.own_feats_dim += 1
        if getattr(args, "obs_last_action", False):
            self.own_feats_dim += 1

        H = self.hidden_size

        # Embeddings
        self.embed_own   = nn.Linear(self.own_feats_dim, H)
        self.embed_ally  = nn.Linear(self.ally_feats_dim, H)
        self.embed_enemy = nn.Linear(self.enemy_feats_dim, H)

        # Multi‑head group cross‑attention (own query → allies / enemies)
        self.cross_att  = MultiHeadGroupAttn(3*H, self.n_heads)

        # Minimal fusion → agent feature
        self.gru = nn.GRU(3*H,H)

    def init_hidden(self):
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
        e_own = self.embed_own(own_raw).squeeze(1)        # [Bn,1,H] -> squeeze -> [Bn,H]
        A = self.embed_ally(ally_raw)                     # [Bn,nA,H]
        E = self.embed_enemy(enemy_raw)                   # [Bn,nE,H]

        u_cat = th.cat([e_own, A, E], dim=-1)             # [Bn,3H]
        z, _ = self.cross_att(u_cat, ally_mask, group_temp=None)   # [Bn,H]

        # Cross‑attn summaries (query = own, groups = allies/enemies)
        zA, _ = self.attn_allies(e_own, A, ally_mask, group_temp=None)   # [Bn,H]
        zE, _ = self.attn_enemies(e_own, E, enemy_mask, group_temp=None) # [Bn,H]

        # Post‑attn conditional modulation of summaries
        if self.use_cln:
            zA = self.cln_A(zA, e_own)
            zE = self.cln_E(zE, e_own)

        # Fuse
        h = self.gru(th.cat([e_own, zA, zE], dim=-1), prev_h)    # [Bn,H]

