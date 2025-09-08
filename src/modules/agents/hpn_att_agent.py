import torch as th
import torch.nn as nn

from modules.layer.attn_utils import MultiHeadGroupAttn, GroupTemp

class HPNAttAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(HPNAttAgent, self).__init__()
        self.args = args
        self.n_heads = self.args.n_head
        self.n_actions = self.args.n_actions


class CustomAtt_RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        self.n_heads = args.n_head
        self.hidden_dim = H = self.args.hidden_size
        self.n_actions = self.args.n_actions
        self.A_move = self.args.output_normal_actions

        self.own_feats_dim, self.enemy_feats_dim, self.ally_feats_dim = input_shape
        self.enemy_feats_dim = self.enemy_feats_dim[-1]
        self.ally_feats_dim = self.ally_feats_dim[-1]

        if self.args.obs_agent_id:
            self.own_feats_dim += 1
        if self.args.obs_last_action:
            self.own_feats_dim += 1 

        # Flags for optional hyper outputs
        self.combined_att       = bool(getattr(args, "combined_att", False))
        self.hpn_bias           = bool(getattr(args, "hpn_bias", False))
        self.use_film_from_own  = bool(getattr(args, "use_film_from_own", True))
        self.use_gate           = bool(getattr(args, "use_gate", False))
        self.use_group_card_temp = bool(getattr(args, "use_group_card_temp", False))

        # -------- Hypernet over own features --------
        # Output layout: [ W( H×D ), b_own(H), calib(4), move_prior(A_move)?, FiLM(2H)?, gate(1)? ]
        hyper_out_dim = (self.own_feats_dim * H) + H
        if self.hpn_bias:
            hyper_out_dim += 4
        if self.use_film_from_own: 
            hyper_out_dim += 2 * H
        if self.use_gate:          
            hyper_out_dim += 1

        self.hyper_own = nn.Sequential(
            nn.Linear(self.own_feats_dim, H),
            nn.ReLU(inplace=True),
            nn.Linear(H, hyper_out_dim)
        )

        # -------- Entity embeddings --------
        self.enemy_embedding = nn.Linear(self.enemy_feats_dim, H)
        self.ally_embedding  = nn.Linear(self.ally_feats_dim, H)

        # -------- Cross-attention --------

        if self.combined_att:
            self.cross_attn = MultiHeadGroupAttn(2*H, self.n_heads)
            self.split_comb = nn.Linear(H, 2*H) 
            if self.use_group_card_temp:
                self.groupTemp = GroupTemp()
        else:
            self.cross_attn_ally  = MultiHeadGroupAttn(H, self.n_heads)
            self.cross_attn_enemy = MultiHeadGroupAttn(H, self.n_heads)
            if self.use_group_card_temp:
                self.groupTemp_A = GroupTemp()
                self.groupTemp_E = GroupTemp()
            

        # -------- Recurrent fuse --------
        self.gru = nn.GRUCell(3 * H, H)

        # -------- Heads --------
        self.q_move = nn.Linear(H, self.A_move)
        self.z_to_K = nn.Linear(H, H, bias=False)
        self.E_to_K = nn.Linear(H, H, bias=False)

        # Nice-to-have init: keep hyper calibration near neutral
        with th.no_grad():
            # last linear initially zeros → calib/prior start neutral (we still get nonzero W,b from first layer)
            self.hyper_own[-1].weight.mul_(0.01)
            self.hyper_own[-1].bias.zero_()

    def init_hidden(self, bs: int, n_agents: int):
        return self.enemy_embedding.weight.new_zeros(bs * n_agents, self.hidden_dim)

    def forward(self, inputs, hidden_state):
        bs, own_raw, ally_raw, enemy_raw, embedding_indices = inputs
        Bn, Na, Ne = own_raw.size(0), ally_raw.size(1), enemy_raw.size(1)
        H = self.hidden_dim

        ally_mask  = ~th.all(ally_raw  == 0, dim=-1)
        enemy_mask = ~th.all(enemy_raw == 0, dim=-1)

        if getattr(self.args, "obs_agent_id", False):
            agent_idx = embedding_indices[0].reshape(Bn, 1, -1).to(own_raw.dtype)
            own_raw = th.cat([own_raw, agent_idx], dim=-1)
        if getattr(self.args, "obs_last_action", False):
            last_act = embedding_indices[-1].reshape(Bn, 1, -1).to(own_raw.dtype)
            own_raw = th.cat([own_raw, last_act], dim=-1)

        # ----- Hypernet slices -----
        own_vec = own_raw.squeeze(1)                      # [Bn, D_own]
        h_out   = self.hyper_own(own_vec)                 # [Bn, hyper_out_dim]

        # W_own, b_own
        w_len = self.own_feats_dim * H
        W_flat = h_out[:, :w_len]
        b_own  = h_out[:, w_len:w_len+H]
        off    = w_len + H

        # (Optional) Per-family calibration deltas (-> gammas ~1, betas ~0)
        # Use small, bounded transforms for stability
        beta_m = beta_s = 0.0
        gamma_m = gamma_s = 1.0
        if self.hpn_bias:
            delta = h_out[:, off:off+4]; off += 4
            d_gm, b_m, d_gs, b_s = delta.chunk(4, dim=-1)
            gamma_m = 1.0 + 0.1 * th.tanh(d_gm)              # [Bn,1] or [Bn,H]? here scalars per batch
            gamma_s = 1.0 + 0.1 * th.tanh(d_gs)
            beta_m  = 0.1 * th.tanh(b_m)
            beta_s  = 0.1 * th.tanh(b_s)

        # (Optional) FiLM for enemies (role-only)
        gamma_E = beta_E = None
        if self.use_film_from_own:
            film = h_out[:, off:off+2*H]; off += 2*H
            gE, bE = film.chunk(2, dim=-1)
            gamma_E = 1.0 + 0.1 * th.tanh(gE)            # [Bn, H]
            beta_E  = 0.1 * th.tanh(bE)                  # [Bn, H]

        # (Optional) ally/enemy gate (role-only)
        gate = None
        if self.use_gate:
            gate = th.sigmoid(h_out[:, off:off+1])       # [Bn,1]
            off += 1

        # Project own with per-agent W (Perceptron)
        W = W_flat.view(Bn, H, self.own_feats_dim)       # [Bn, H, D_own]
        e_own = th.einsum('bhd,bd->bh', W, own_vec) + b_own  # [Bn, H]

        # Entities
        A = self.ally_embedding(ally_raw)                # [Bn, Na, H]
        E = self.enemy_embedding(enemy_raw)              # [Bn, Ne, H]
        if self.use_film_from_own:
            E = E * gamma_E.unsqueeze(1) + beta_E.unsqueeze(1)

        # -------- Cross-attn summaries --------
        # (Optional) Group cardinality temperature
        temp_A = temp_E = temp = None
        if self.use_group_card_temp:
            if self.combined_att:
                temp_A = self.groupTemp_A(ally_mask.float().sum(-1))
                temp_E = self.groupTemp_E(enemy_mask.float().sum(-1))
            else:
                comb_mask = ally_mask.float().sum(-1) + enemy_mask.float().sum(-1)
                temp = self.groupTemp()

        # Combined or separate cross-attn
        if self.combined_att:
            z, _ = self.cross_attn(e_own, th.cat([A, E], comb_mask, dim=-1), None, group_temp=temp_A)  # [Bn,H]
            zA, zE = self.split_comb(z).chunk(2, dim=-1)
        else:
            zA, _ = self.cross_attn_ally(e_own,  A, ally_mask,  group_temp=temp_A)  # [Bn,H]
            zE, _ = self.cross_attn_enemy(e_own, E, enemy_mask, group_temp=temp_E)  # [Bn,H]

            
        # Recurrent fuse (optionally gate ally vs enemy)
        if gate is None:
            u_cat = th.cat([e_own, zA, zE], dim=-1)
        else:
            u_cat = th.cat([e_own, gate*zA, (1.0-gate)*zE], dim=-1)

        h_in = e_own.new_zeros(Bn, H) if hidden_state is None else hidden_state.view(Bn, H)
        z = self.gru(u_cat, h_in)                        # [Bn, H]

        # Heads
        logits_move = self.q_move(z)                     # [Bn, A_move]

        zk = self.z_to_K(z)                              # [Bn, H]
        Ek = self.E_to_K(E)                              # [Bn, Ne, H]
        logits_shoot = th.einsum('bd,bnd->bn', zk, Ek)   # [Bn, Ne]

        # Masks
        neg_inf = -1e9 if logits_shoot.dtype == th.float32 else th.finfo(logits_shoot.dtype).min
        logits_shoot = logits_shoot.masked_fill(~enemy_mask, neg_inf)

        # Role-conditioned calibration (state-only)
        q_move  = gamma_m * logits_move + beta_m         # [Bn, A_move]
        q_shoot = gamma_s * logits_shoot + beta_s        # [Bn, Ne]

        Q = th.cat([q_move, q_shoot], dim=-1).unsqueeze(1)  # [Bn,1,A_move+Ne]
        hidden_out = z.view(bs, -1, H)
        return Q, hidden_out
