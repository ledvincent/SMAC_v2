#!/usr/bin/env python3
import math
from typing import Optional

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modules.layer.attn_utils import ConditionalLayerNorm, MultiHeadGroupAttn
from modules.layer.attn_utils import GeomBias, GroupTemp
from modules.layer.attn_utils import _masked_mean

"""
MoGA‑GRU (final, SS-compatible, no extended-action path)
-------------------------------------------------------
- Same IO as SPECTRA's SS_RNNAgent:
    forward(inputs, hidden_state) -> (Q, hidden')
  inputs is a tuple: (bs, own[Bn,1,Do], allies[Bn,Na,Da], enemies[Bn,Ne,De], embedding_indices)
  hidden_state: [bs, n_agents, H]
  outputs: Q [Bn,1, A_move + Ne], hidden' [bs,n_agents,H]
- No self-attention; O(n_allies + n_enemies)
- Grouped single-query **multi-head** cross-attn (ally/enemy), with **per-head enemy logit bias**
- Optional group-cardinality temperature per group
- Mixture over {ally, enemy, self} → **GRU** (like SPECTRA)
- Movement head: prototype attention (PI)
- Shooting head: pointer/QK (PE) **over enemies only** (no extended-action variant)
- Calibration: shared V, type biases, temperatures, within-family mean-centering
"""

# ------------------------------- agent -------------------------------

class SS_MoGA_RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.n_heads = args.n_head
        self.output_normal_actions = args.output_normal_actions

        # Optional geometry bias & relative XY config
        self.use_geom_bias = getattr(args, "use_geom_bias_enemies", True)
        self.geom_bias_mode = getattr(args, "geom_bias_mode", "shared")  # bias shared or per-head
        self.geom_tanh_c = getattr(args, "geom_tanh_c", 2.0)
        self.use_group_card_temp = getattr(args, "use_group_card_temp", True)
        self.use_refine_shoot = getattr(args, "use_refine_shoot", False)
        self.prototype_movement = getattr(args, "prototype_movement", True)
        self.hs_query = getattr(args, "hs_query", False)

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

        # Optional: Group cardinality temperature T_G = softplus(alpha*log(n)+beta)
        if self.use_group_card_temp:
            self.groupTemp_A = GroupTemp()
            self.groupTemp_E = GroupTemp()

        # Optional: Conditions on previous hidden state
        if self.hs_query:
            # residual correction from hidden state (starts as no-op)
            self.q_resid = nn.Linear(H, H, bias=False)
            nn.init.zeros_(self.q_resid.weight)
            # tiny gate driven by h_prev (starts ~0, so own-only at init)
            self.q_gate = nn.Linear(H, 1)
            nn.init.zeros_(self.q_gate.weight)
            nn.init.constant_(self.q_gate.bias, -4.0)  # sigmoid(-4) ~ 0.018

        # Grouped single‑query multi‑head cross‑attention (ally/enemy)
        self.grp_attn_allies  = MultiHeadGroupAttn(H, self.n_heads)
        self.grp_attn_enemies = MultiHeadGroupAttn(H, self.n_heads)

        # Post‑attn conditional norm on summaries
        self.cln_A = ConditionalLayerNorm(H)
        self.cln_E = ConditionalLayerNorm(H)

        # Layer norm
        self.q_ln = nn.LayerNorm(H)
        self.inp_ln = nn.LayerNorm(3*H)

        # Recurrent core
        self.gru = nn.GRUCell(3*H, H)
        self.pre_ln = nn.LayerNorm(H)

        # Family calibration
        self.V_head    = nn.Linear(H, 1)
        self.type_bias = nn.Linear(H, 2)   # [b_move, b_shoot]
        self.log_tau_move  = nn.Parameter(th.zeros(1))
        self.log_tau_shoot = nn.Parameter(th.zeros(1))

        # Movement head: prototype attention (PI) or MLP
        if self.prototype_movement:
            self.W_move_q = nn.Linear(H, H)
            self.prototypes = nn.Parameter(th.randn(self.output_normal_actions, H) * (1.0 / math.sqrt(H)))
        else:
            self.move_head = nn.Sequential(nn.Linear(H, H), nn.ReLU(inplace=True), nn.Linear(H, self.output_normal_actions))

        # Shooting head: pointer (Query‑Key) **enemies only**
        self.Wq_shoot = nn.Linear(H, H)
        self.Wk_enemy = nn.Linear(H, H)

    def init_hidden(self):
        return self.embed_own.weight.new_zeros(1, self.hidden_size)


    # ------------------------------ forward ----------------------------

    def forward(self, inputs, hidden_state):
        bs, own_raw, ally_raw, enemy_raw, embedding_indices = inputs
        Bn = own_raw.shape[0]
        nA = ally_raw.shape[1]
        nE = enemy_raw.shape[1]
        assert nA > 0 and nE > 0

        H = self.hidden_size

        # Masks (pad rows are all‑zeros)
        ally_mask  = ~th.all(ally_raw  == 0, dim=-1)      # [Bn,nA]
        enemy_mask = ~th.all(enemy_raw == 0, dim=-1)      # [Bn,nE]

        # Append indices if requested (algo args not env.args) 
        # (if args.env_args.obs_agent_id=True then setting args.obs_agent_id=True is redundant)
        # args.obs_agent_id and args.obs_last_action are only used here while args.env_args are used in the environment
        if getattr(self.args, "obs_agent_id", False):
            agent_idx = embedding_indices[0].reshape(-1, 1, 1)
            own_raw = th.cat([own_raw, agent_idx], dim=-1)
        if getattr(self.args, "obs_last_action", False):
            last_act = embedding_indices[-1].reshape(-1, 1, 1)
            own_raw = th.cat([own_raw, last_act], dim=-1)

        # Embeddings
        e_own = self.embed_own(own_raw)                   # [Bn,1,H]
        A = self.embed_ally(ally_raw)                     # [Bn,nA,H]
        E = self.embed_enemy(enemy_raw)                   # [Bn,nE,H]

        # Hidden-state–conditioned query (state‑aware attention)
        q_own = e_own.squeeze(1)                   # [Bn, H]
        h_prev = hidden_state.reshape(-1, H)       # [Bn, H]
        if self.hs_query:
            has_state = (h_prev.abs().sum(dim=-1, keepdim=True) > 0).float() # 0 at t=0
            g = th.sigmoid(self.q_gate(h_prev)) * has_state  # [Bn,1]
            q_src = q_own + g * self.q_resid(h_prev)   # residual correction only
        else:
            q_src = q_own

        # ----- Optional additions -----
        # Group temps for cardinality neutralization
        temp_A = temp_E = None
        if self.use_group_card_temp:
            # all [Bn]
            cntA = ally_mask.float().sum(-1)
            cntE = enemy_mask.float().sum(-1)
            temp_A = self.groupTemp_A(cntA)
            temp_E = self.groupTemp_E(cntE)
        # ----------------------------------

        # Grouped cross‑attention (ally/enemy)
        # Summary of cross-attention between query (own_context) and group (allies/enemies)
        # both of [Bn,H] shape
        zA, _ = self.grp_attn_allies(q_src, A, ally_mask, group_temp=temp_A)
        zE, _ = self.grp_attn_enemies(q_src, E, enemy_mask, group_temp=temp_E)

        # Post‑attn conditional modulation of summaries
        zA = self.cln_A(zA, e_own.squeeze(1))
        zE = self.cln_E(zE, e_own.squeeze(1))

        # Normalize q_src (keep scales comparable)
        qn = self.q_ln(q_src)

        u_cat = th.cat([q_src, zA, zE], dim=-1)     # [Bn,3H]
        u_cat = self.inp_ln(u_cat)  # Normalize before GRU

        # GRU update (attention -> GRU, like SPECTRA)
        h = self.gru(u_cat, h_prev)                           # [Bn,H]

        # Family calibration parts 
        V = self.V_head(h)                                # [Bn,1]
        b_move, b_shoot = self.type_bias(h).chunk(2, dim=-1) # [Bn, 1] x2 family bias
        tau_move  = th.exp(self.log_tau_move ) + 1e-6
        tau_shoot = th.exp(self.log_tau_shoot) + 1e-6

        # (optional) let the enemy context tweak shooting temp (zero-init; safe)
        if self.use_tau_shoot_mod and nE > 0:
            tau_shoot = tau_shoot * (1.0 + 0.1 * th.tanh(self.tau_shoot_mod(zE)))  # [Bn,1]

        # Movement head (PI)
        if self.prototype_movement:
            q_mv = self.W_move_q(h)                       # [Bn,H]
            A_move = F.linear(q_mv, self.prototypes) / math.sqrt(H)  # [Bn, A_move]
        else:
            A_move = self.move_head(h)
        A_move = A_move - A_move.mean(dim=-1, keepdim=True)
        Q_move = (V + b_move + (A_move / tau_move))       # [Bn, A_move]

        # Shooting head (PE over enemies only)
        q_sh = self.Wq_shoot(h)                           # [Bn,H]
        K_enemy = self.Wk_enemy(E)                        # [Bn,nE,H]
        a = th.einsum("bh,bnh->bn", q_sh, K_enemy) / math.sqrt(H)  # [Bn,nE]

        if self.use_refine_shoot and nE > 0:
            h_rep = h.unsqueeze(1).expand(-1, nE, -1)
            feats = th.cat([h_rep, E, h_rep*E, (h_rep - E).abs(), a.unsqueeze(-1)], dim=-1)
            a = self.ptr_mlp(feats).squeeze(-1)          # [Bn,nE]
        # Center within valid enemies
        den_e = enemy_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_e = (a * enemy_mask.float()).sum(dim=1, keepdim=True) / den_e
        a = a - mean_e
        very_neg = th.finfo(a.dtype).min
        a = th.where(enemy_mask, a, th.full_like(a, very_neg))
        Q_shoot = (V + b_shoot + (a / tau_shoot))         # [Bn,nE]

        # Final concat with movement
        Q = th.cat([Q_move, Q_shoot], dim=-1).unsqueeze(1)  # [Bn,1, A_move + nE]
        return Q, h.view(bs, -1, H) 


if __name__ == "__main__":
    # Minimal self-test matching SPECTRA IO
    class Args: pass
    args = Args()
    args.hidden_size = 64
    args.n_head = 4
    args.output_normal_actions = 8
    args.obs_agent_id = False
    args.obs_last_action = False
    args.use_geom_bias_enemies = True
    args.own_xy_slice = slice(0,2)
    args.enemy_xy_slice = slice(0,2)
    args.use_group_card_temp = True
    args.use_refine_shoot = True
    args.prototype_movement = True

    agent = SS_MoGA_RNNAgent(
        input_shape=(16, (None, 10), (None, 12)),
        args=args,
    )

    B, nA, nE = 2, 3, 5
    Bn = B * 4
    own = th.randn(Bn, 1, 16)
    allies = th.randn(Bn, nA, 12)
    enemies = th.randn(Bn, nE, 10)
    inputs = (B, own, allies, enemies, [th.zeros(Bn,1,1), th.zeros(Bn,1,1)])
    hid = th.zeros(B, nA+1, 64)

    Q, hid2 = agent(inputs, hid)
    print("Q:", Q.shape, "hidden:", hid2.shape)
