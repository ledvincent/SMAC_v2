#!/usr/bin/env python3
import math
from typing import Optional

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modules.layer.attn_utils import MultiHeadGroupAttn, GroupTemp

# ------------------------------- agent -------------------------------

class CustomAtt_RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_allies = args.n_allies
        self.n_enemies = args.n_enemies
        self.n_actions = args.n_actions
        self.n_heads = args.hpn_head_num
        self.hidden_dim = H = args.hidden_dim

        self.own_feats_dim, self.enemy_feats_dim, self.ally_feats_dim = input_shape
        self.enemy_feats_dim = self.enemy_feats_dim[-1]
        self.ally_feats_dim = self.ally_feats_dim[-1]
        self.output_normal_actions = self.args.output_normal_actions
        
        # Optional flags
        self.use_group_card_temp = bool(getattr(args, "use_group_card_temp", False))
        self.use_bias = bool(getattr(args, "use_bias", False))

        if self.args.obs_agent_id:
            self.own_feats_dim += 1
        if self.args.obs_last_action:
            self.own_feats_dim += 1

        # Embeddings
        hyper_out_dim = (self.own_feats_dim * H) + H + 4
        if self.use_bias: hyper_out_dim += 1
        self.hyper_own = nn.Sequential(
            nn.Linear(self.own_feats_dim, H),
            nn.ReLU(inplace=True),
            nn.Linear(H, hyper_out_dim))

        self.enemy_embedding = nn.Linear(self.enemy_feats_dim, self.hidden_dim)
        self.ally_embedding = nn.Linear(self.ally_feats_dim, self.hidden_dim)

        # Group cardinality temperature
        if self.use_group_card_temp:
            self.groupTemp_A = GroupTemp()
            self.groupTemp_E = GroupTemp()

        # Separate or combined ally/enemy cross-attn (no geometry priors)
        self.use_combined_attn = bool(getattr(args, "use_combined_attn", False))
        if self.use_combined_attn:
            self.cross_attn = MultiHeadGroupAttn(H, self.n_heads)
        else:
            self.cross_attn_ally  = MultiHeadGroupAttn(H, self.n_heads)
            self.cross_attn_enemy = MultiHeadGroupAttn(H, self.n_heads)

        # GRU
        self.gru = nn.GRUCell(3*H, H)

        # Movements heads
        self.q_move = nn.Linear(H, self.A_move)

        # Shoot heads
        self.z_to_K = nn.Linear(H, H, bias=False)
        self.E_to_K = nn.Linear(H, H, bias=False)

        # (Optional) FiLM from aggregated enemy context
        if self.use_film:
            self.film = nn.Sequential(
                nn.Linear(H, H), nn.ReLU(inplace=True),
                nn.Linear(H, 2*H)  # splits to gamma, beta
            )
        
        # Nice-to-have init: keep hyper calibration near neutral
        with th.no_grad():
            # last linear initially zeros → calib/prior start neutral (we still get nonzero W,b from first layer)
            self.hyper_own[-1].weight.mul_(0.01)
            self.hyper_own[-1].bias.zero_()

    def init_hidden(self):
        return None

    # ------------------------------ forward ----------------------------

    def forward(self, inputs, hidden_state):
        '''
        Inputs: (batch_size, own features, ally features, enemy features, embedding_indices)
        hidden_state : [bs, n_agents, H]
        Output: Q [Bn,1, A_move + Ne], hidden' [bs,n_agents,H]
        '''
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

        # ----- Hypernet slices -----
        own_vec = own_raw.squeeze(1)                      # [Bn, D_own]
        h_out   = self.hyper_own(own_vec)                 # [Bn, hyper_out_dim]

        # W_own, b_own
        w_len = self.own_feats_dim * H
        W_flat = h_out[:, :w_len]
        b_own  = h_out[:, w_len:w_len+H]
        off    = w_len + H
        
        # Ally/Enemy Embeddings
        ally_e = self.ally_embedding(ally_raw)                  # [Bn, Na, H]
        enemy_e = self.enemy_embedding(enemy_raw)               # [Bn, Ne, H]

        # Group temps for cardinality neutralization
        temp_A = temp_E = None
        if self.use_group_card_temp:
            # all [Bn]
            temp_A = self.groupTemp_A(ally_mask.float().sum(-1))
            temp_E = self.groupTemp_E(enemy_mask.float().sum(-1))

        # Single-query (own) cross-attention
        zA, _ = self.cross_attn_ally(own_e, ally_e, ally_mask, group_temp=temp_A)
        zE, _ = self.cross_attn_enemy(own_e, enemy_e, enemy_mask, group_temp=temp_E)

        # FiLM on per-enemy features using aggregated enemy context
        if self.use_film:
            gamma, beta = self.film(zE).chunk(2, dim=-1)       # each [Bn, H]
            # small bounded modulation for stability
            gamma, beta = th.tanh(gamma)*0.1, th.tanh(beta)*0.1
            enemy_e = enemy_e * (1+gamma.unsqueeze(1)) + beta.unsqueeze(1)

        # Recurrent core
        u_cat = th.cat([own_e, zA, zE], dim=-1)                 # [Bn, 3H]
        z = self.gru(u_cat, hidden_state)                         # [Bn, H]

        # Heads
        # Movements
        logits_move = self.q_move(z)                                  # [Bn, A_move]

        # Shooting (per-enemy bilinear)
        zk = self.z_to_K(z)                                      # [Bn, H]
        Ek = self.E_to_K(enemy_e)                                # [Bn, Ne, H]
        logits_shoot = th.einsum("bd,bmd->bm", zk, Ek)                 # [Bn, Ne]

        # Mask invalid enemies
        # logits_shoot = logits_shoot.masked_fill(~enemy_mask, float('-inf'))

        # Final Q and new hidden
        Q = th.cat([logits_move, logits_shoot], dim=-1).unsqueeze(1)  # [Bn,1,A_move+Ne]
        return Q, z.view(bs, -1, H)