import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.layer.attn_utils import MultiHeadGroupAttn

# ---------- small utilities ----------

def masked_mean(x, mask, dim):
    # x: [..., M, D], mask: [..., M] in {0,1}
    eps = 1e-8
    w = mask.unsqueeze(-1)  # [..., M, 1]
    s = (x * w).sum(dim=dim)
    d = w.sum(dim=dim).clamp_min(eps)
    return s / d

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, n_layers=2, dropout=0.0):
        super().__init__()
        layers = []
        d = in_dim
        for i in range(n_layers - 1):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.LayerNorm(hidden)]
            if dropout > 0: layers += [nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class RoleHead(nn.Module):
    """
    Simple unsupervised role head.
    Inputs: per-agent fused h0, pooled ally & enemy summaries
    Output: role logits, role probs, role vector used for conditioning
    """
    def __init__(self, d_in, n_roles, use_gumbel=True, tau=1.0):
        super().__init__()
        self.use_g = use_gumbel
        self.tau = tau
        self.n_roles = n_roles
        self.mlp = MLP(d_in, hidden=128, out_dim=n_roles, n_layers=2)

    def forward(self, x):
        logits = self.mlp(x)                              # [B*N, K]
        probs = F.softmax(logits, dim=-1)
        if self.training and self.use_g:
            r = F.gumbel_softmax(logits, tau=self.tau, hard=True, dim=-1)  # straight-through
        else:
            # eval: soft or hard; use soft by default to keep gradients smooth if eval runs under no_grad anyway
            r = probs
        return logits, probs, r

# ---------- the agent ----------

class CAPERNNAgent(nn.Module):
    """
    Capability-Aware, PE-Equivariant RNN agent
    Interface compatible with SPECTRA/HPN:
      __init__(input_shape, args)
      forward(inputs, hidden_state) -> (Q, hidden')
    Expectation:
      - 'inputs' is either a tuple (own, allies, enemies, *extras) or a dict with keys.
      - If flat, we rely on args to slice: n_allies, n_enemies, d_own, d_all, d_en.
    """
    def __init__(self, input_shape, args):
        super().__init__()
        # ---- args / shapes ----
        self.d_own   = getattr(args, "d_own", None)
        self.d_all   = getattr(args, "d_all", None)
        self.d_en    = getattr(args, "d_en", None)
        self.n_all   = getattr(args, "n_allies", None)
        self.n_en    = getattr(args, "n_enemies", None)
        self.n_move  = getattr(args, "n_move_actions", None)
        self.n_types = getattr(args, "n_unit_types", 0)    # 0 if unknown/not present
        self.types_are_last = getattr(args, "unit_type_bits_last", True)  # common in SMAC
        self.combine_entities = getattr(args, "combine_entities", False)

        # core dims
        self.d_model = getattr(args, "embed_dim", 192)
        self.hid_rnn = getattr(args, "rnn_hidden_dim", 256)
        self.n_heads = getattr(args, "attn_heads", 4)
        self.n_roles = getattr(args, "n_roles", 4)
        self.role_tau = getattr(args, "role_tau", 1.0)
        self.use_gumbel = getattr(args, "role_gumbel", True)

        # ---- type embedding ----
        t_dim = getattr(args, "type_emb_dim", 8)
        if self.n_types > 0:
            self.type_emb = nn.Embedding(self.n_types, t_dim)
        else:
            self.type_emb = None
            t_dim = 0

        # ---- encoders ----
        self.enc_self  = MLP((self.d_own or input_shape) + t_dim, self.d_model, self.d_model, n_layers=2)
        self.enc_ally  = MLP((self.d_all or input_shape) + t_dim, self.d_model, self.d_model, n_layers=2)
        self.enc_enemy = MLP((self.d_en  or input_shape) + t_dim, self.d_model, self.d_model, n_layers=2)

        # ---- attention ----
        self.attn_ally  = MultiHeadGroupAttn(self.d_model, self.n_heads)
        self.attn_enemy = MultiHeadGroupAttn(self.d_model, self.n_heads)
        self.fuse = MLP(self.d_model * 3, self.d_model, self.d_model, n_layers=2)  # [self, cA, cE] -> h0

        # ---- role head ----
        # input: h0 || pooled_ally || pooled_enemy
        self.pool_proj = MLP(self.d_model * 3, 128, 128, n_layers=2)
        self.role_head = RoleHead(d_in=128, n_roles=self.n_roles, use_gumbel=self.use_gumbel, tau=self.role_tau)

        # ---- RNN ----
        rnn_in = self.d_model + self.n_roles
        rnn_type = getattr(args, "rnn_type", "gru").lower()
        self.rnn_type = rnn_type
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(rnn_in, self.hid_rnn, batch_first=True)
        else:
            self.rnn = nn.GRU(rnn_in, self.hid_rnn, batch_first=True)
        self.pre_rnn_ln = nn.LayerNorm(rnn_in)
        self.post_rnn_ln = nn.LayerNorm(self.hid_rnn)

        # ---- shared trunk for heads ----
        self.trunk = MLP(self.hid_rnn, self.d_model, self.d_model, n_layers=2)

        # ---- movement head ----
        self.move_head = nn.Linear(self.d_model + self.n_roles, self.n_move)

        # ---- target (shoot) head: pointer-style ----
        attn_qdim = getattr(args, "target_query_dim", self.d_model)
        self.q_proj = nn.Linear(self.d_model + self.n_roles, attn_qdim, bias=False)
        self.k_proj = nn.Linear(self.d_model, attn_qdim, bias=False)

        # buffers for debug (optional)
        self.register_buffer("_dummy", torch.zeros(1), persistent=False)
        self.last_role_probs = None  # for external logging if you want

    # --------- helpers to parse and embed types ---------

    def _get_type_emb(self, x, n_types):
        if n_types <= 0 or self.type_emb is None:
            return torch.zeros(*x.shape[:-1], 0, device=x.device, dtype=x.dtype)
        # assume type one-hot is at the end of the feature vector for that entity
        onehot = x[..., -n_types:] if self.types_are_last else x[..., :n_types]
        idx = onehot.argmax(dim=-1).clamp(min=0)  # [..]
        return self.type_emb(idx)                 # [.., t_dim]

    def _split_inputs(self, inputs):
        """
        Returns own [B*N, 1, d_own], allies [B*N, Na, d_all], enemies [B*N, Ne, d_en]
        This function supports tuple or dict. If flat, you'll need to adapt to your layout.
        """
        if isinstance(inputs, (list, tuple)):
            # Common SPECTRA/HPN: (own, allies, enemies, *extras)
            own   = inputs[0]  # [B*N, 1, d_own] or [B, N, 1, d_own]
            allies= inputs[1]  # [B*N, Na, d_all] or [B, N, Na, d_all]
            enemies=inputs[2]  # [B*N, Ne, d_en]  or [B, N, Ne, d_en]
        elif isinstance(inputs, dict):
            own, allies, enemies = inputs["own"], inputs["allies"], inputs["enemies"]
        else:
            raise ValueError("Unsupported input structure; pass (own, allies, enemies, ...) or dict.")

        # collapse [B, N, ...] to [B*N, ...] if needed
        def collapse(x):
            if x.dim() == 4:  # [B, N, M, D]
                B, N, M, D = x.shape
                return x.view(B * N, M, D)
            elif x.dim() == 3:  # [B*N, M, D]
                return x
            else:
                raise ValueError("Unexpected input dims.")
        own    = collapse(own)
        allies = collapse(allies)
        enemies= collapse(enemies)
        return own, allies, enemies

    # --------- forward ---------

    def forward(self, inputs, hidden_state):
        """
        inputs: (own, allies, enemies, *extras) with shapes collapsed to [B*N, M, D] or [B, N, M, D]
        hidden_state: [B*N, H] for GRU, or tuple for LSTM
        returns: (Q, hidden')
          Q shape: [B*N, A_move + Ne]
        """
        device = self._dummy.device
        own, allies, enemies = self._split_inputs(inputs)

        BN = own.size(0)
        Na = allies.size(1)
        Ne = enemies.size(1)

        # masks from zero-padding (sum(abs) > 0)
        ally_mask = (allies.abs().sum(dim=-1) > 0).to(own.dtype)  # [BN, Na]
        enemy_mask= (enemies.abs().sum(dim=-1) > 0).to(own.dtype) # [BN, Ne]

        # --- type embeddings and encoders ---
        # own is [BN, 1, d_own] -> squeeze to [BN, d_own] for per-agent enc
        own_flat = own.squeeze(1)                                  # [BN, d_own]
        t_self = self._get_type_emb(own_flat, self.n_types)        # [BN, t_dim]
        self_emb = self.enc_self(torch.cat([own_flat, t_self], dim=-1))  # [BN, D]

        # allies/enemies
        t_ally = self._get_type_emb(allies, self.n_types)          # [BN, Na, t_dim]
        t_enemy= self._get_type_emb(enemies, self.n_types)         # [BN, Ne, t_dim]
        ally_emb = self.enc_ally(torch.cat([allies, t_ally], dim=-1))    # [BN, Na, D]
        enemy_emb= self.enc_enemy(torch.cat([enemies, t_enemy], dim=-1)) # [BN, Ne, D]

        # --- cross-attention ---
        if self.combine_entities:
            # Concatenate sets; share the enemy key/value projection (ally & enemy encoders already separate)
            set_emb  = torch.cat([ally_emb, enemy_emb], dim=1)               # [BN, Na+Ne, D]
            set_mask = torch.cat([ally_mask, enemy_mask], dim=1)             # [BN, Na+Ne]
            c_set = self.attn_enemy(self_emb.unsqueeze(1), set_emb, set_mask)  # reuse module, OK
            cA, cE = c_set, c_set
        else:
            cA = self.attn_ally(self_emb.unsqueeze(1), ally_emb, ally_mask)    # [BN, D]
            cE = self.attn_enemy(self_emb.unsqueeze(1), enemy_emb, enemy_mask) # [BN, D]

        # --- fuse ---
        h0 = self.fuse(torch.cat([self_emb, cA, cE], dim=-1))                 # [BN, D]

        # --- pooled summaries for role head ---
        pA = masked_mean(ally_emb, ally_mask, dim=1) if Na > 0 else torch.zeros(BN, self.d_model, device=device)
        pE = masked_mean(enemy_emb, enemy_mask, dim=1) if Ne > 0 else torch.zeros(BN, self.d_model, device=device)
        pooled = self.pool_proj(torch.cat([self_emb.unsqueeze(1), pA.unsqueeze(1), pE.unsqueeze(1)], dim=1).reshape(BN, -1))

        # --- roles ---
        role_logits, role_probs, r = self.role_head(pooled)                   # [BN, K] each
        self.last_role_probs = role_probs.detach()

        # --- recurrent core ---
        rnn_in = torch.cat([h0, r], dim=-1).unsqueeze(1)                      # [BN, 1, D+K]
        rnn_in = self.pre_rnn_ln(rnn_in)
        if self.rnn_type == "lstm":
            if hidden_state is None:
                h0_rnn = torch.zeros(1, BN, self.hid_rnn, device=device)
                c0_rnn = torch.zeros(1, BN, self.hid_rnn, device=device)
                hidden_state = (h0_rnn, c0_rnn)
            out, new_hidden = self.rnn(rnn_in, hidden_state)
        else:
            if hidden_state is None:
                h0_rnn = torch.zeros(1, BN, self.hid_rnn, device=device)
            else:
                h0_rnn = hidden_state
            out, new_hidden = self.rnn(rnn_in, h0_rnn)                        # out: [BN, 1, H]
        h1 = self.post_rnn_ln(out.squeeze(1))                                  # [BN, H]

        # --- shared trunk ---
        z = self.trunk(h1)                                                     # [BN, D]

        # --- movement head (condition on role) ---
        z_move = torch.cat([z, r], dim=-1)                                     # [BN, D+K]
        Q_move = self.move_head(z_move)                                         # [BN, A_move]
        # center (dueling-style)
        Q_move = Q_move - Q_move.mean(dim=-1, keepdim=True)

        # --- target head (pointer to enemies, no geometry bias) ---
        # build queries/keys and masked dot-product
        z_shoot = torch.cat([z, r], dim=-1)                                     # [BN, D+K]
        q = self.q_proj(z_shoot)                                                # [BN, Dq]
        if Ne > 0:
            k = self.k_proj(enemy_emb)                                          # [BN, Ne, Dq]
            logits = torch.einsum('bd,bmd->bm', q, k) / (q.size(-1) ** 0.5)     # [BN, Ne]
            # mask out invalid enemies
            logits = logits.masked_fill(enemy_mask == 0, float('-inf'))
            Q_shoot = logits
            # center over valid targets
            # (replace -inf with very small before mean)
            masked = torch.where(enemy_mask.bool(), Q_shoot, torch.zeros_like(Q_shoot))
            denom = enemy_mask.sum(dim=-1, keepdim=True).clamp_min(1)
            mean_valid = masked.sum(dim=-1, keepdim=True) / denom
            Q_shoot = Q_shoot - mean_valid
        else:
            Q_shoot = torch.zeros(BN, 0, device=device)

        # --- final Q: concat movement then shoot actions ---
        Q = torch.cat([Q_move, Q_shoot], dim=-1)                                # [BN, A_move + Ne]
        return Q, new_hidden
