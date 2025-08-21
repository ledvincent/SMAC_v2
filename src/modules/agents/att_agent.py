# hf_dpa_agent.py
import torch as th
import torch.nn as nn

# Adjust import path if your repo differs
from modules.layer.ss_attention import CrossAttentionBlock, QueryKeyBlock


# ---------- FiLM (γ-only, stable) ----------
class FiLM(nn.Module):
    """y = (1 + 0.5 * tanh(W ctx)) ⊙ x  (no β; anchored around identity)"""
    def __init__(self, hidden: int):
        super().__init__()
        self.to_gamma = nn.Linear(hidden, hidden)

    def forward(self, x: th.Tensor, ctx: th.Tensor) -> th.Tensor:
        # x: [B*N, L, H], ctx: [B*N, H]
        gamma = 1.0 + 0.5 * th.tanh(self.to_gamma(ctx)).unsqueeze(1)  # [B*N,1,H]
        return gamma * x


# ---------- Low-rank adapter (LoRA-style) ----------
class LowRankAdapter(nn.Module):
    """
    x' = x + (x @ V) * s(ctx) @ U^T
    - rank r << H, same adapter applied to all tokens in a set (PE-safe)
    """
    def __init__(self, hidden: int, rank: int = 4):
        super().__init__()
        self.U = nn.Parameter(th.randn(hidden, rank) / (hidden ** 0.5))
        self.V = nn.Parameter(th.randn(hidden, rank) / (hidden ** 0.5))
        self.to_s = nn.Linear(hidden, rank)

    def forward(self, x: th.Tensor, ctx: th.Tensor) -> th.Tensor:
        # x: [B*N, L, H], ctx: [B*N, H]
        s  = self.to_s(ctx)                  # [B*N, r]
        XV = x @ self.V                      # [B*N, L, r]
        delta = (XV * s.unsqueeze(1)) @ self.U.t()   # [B*N, L, H]
        return x + delta


class SS_HF_RNNAgent(nn.Module):
    """
    Lite dual-path SAQA agent with FiLM + LoRA adapters + optional heuristic top-k.

    input_shape: (own_dim, enemy_shape, ally_shape) like SPECTra/HPN
    args must contain:
        .hidden_size, .n_head, .n_actions, .output_normal_actions
        .env_args (dict), .obs_agent_id (bool), .obs_last_action (bool)
    optional in args:
        .topk_enemies (int, default 0=disable)
        .use_heuristic_topk (bool, default False)
        .adapter_rank (int, default 4)
        .heuristic_feat_idx (dict or None)  # indices in RAW enemy_feats:
            # defaults assume SMACv2-style:
            # {'avail':0, 'dist':1, 'relx':2, 'rely':3, 'hp':4, 'shield':5}
    Controller input order supported here: (bs, own, ally, enemy, embedding).
    """

    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args

        # -- LORA / FILM options
        self.use_lora = bool(getattr(args, "use_lora", True))
        self.use_film = bool(getattr(args, "use_film", False))

        # ---- sizes / options
        self.H = args.hidden_size
        self.n_head = args.n_head
        self.n_actions = args.n_actions
        self.output_normal_actions = args.output_normal_actions

        self.topk_enemies = int(getattr(args, "topk_enemies", 0))
        self.use_heuristic_topk = bool(getattr(args, "use_heuristic_topk", False))
        self.adapter_rank = int(getattr(args, "adapter_rank", 4))
        # default indices per your starcraft2.py snippet; adjust if needed
        self.heuristic_idx = getattr(args, "heuristic_feat_idx",
                                     {'avail': 0, 'dist': 1, 'relx': 2, 'rely': 3, 'hp': 4, 'shield': 5})

        # ---- input dims (exactly like SPECTra/HPN)
        self.own_feats_dim, self.enemy_feats_shape, self.ally_feats_shape = input_shape
        self.enemy_feats_dim = self.enemy_feats_shape[-1]
        self.ally_feats_dim  = self.ally_feats_shape[-1]

        if self.args.obs_agent_id:
            self.own_feats_dim += 1
        if self.args.obs_last_action:
            self.own_feats_dim += 1

        # ---- token embedders (Linear → H)
        self.embed_own   = nn.Linear(self.own_feats_dim, self.H)
        self.embed_enemy = nn.Linear(self.enemy_feats_dim, self.H)
        self.embed_ally  = nn.Linear(self.ally_feats_dim,  self.H)

        # ---- context from raw own feats (Linear → H)
        self.context_proj = nn.Linear(self.own_feats_dim, self.H)

        # ---- FiLM modulators (γ-only)
        if self.use_film:
            self.film_enems  = FiLM(self.H)
            self.film_allies = FiLM(self.H)
        else:
            self.film_enems  = self.film_allies = nn.Identity()

        if self.use_lora:
            self.adapt_q  = LowRankAdapter(self.H, self.adapter_rank)
            self.adapt_kE = LowRankAdapter(self.H, self.adapter_rank)
            self.adapt_kA = LowRankAdapter(self.H, self.adapter_rank)
        else:
            self.adapt_q = self.adapt_kE = self.adapt_kA = nn.Identity()

        # ---- dual SAQA (masked)
        self.attn_enems  = CrossAttentionBlock(d=self.H, h=self.n_head)
        self.attn_allies = CrossAttentionBlock(d=self.H, h=self.n_head)

        # ---- tiny scalar gate (3H → 1)
        self.gate = nn.Linear(3 * self.H, 1)

        # ---- temporal memory
        self.rnn = nn.GRUCell(self.H, self.H)

        # ---- action heads
        self.normal_actions_net = nn.Linear(self.H, self.output_normal_actions)  # noop/stop/move...
        self.attack_bias_head   = nn.Linear(self.H, 1)                           # scalar "attack" bias
        nn.init.zeros_(self.attack_bias_head.weight)
        nn.init.zeros_(self.attack_bias_head.bias)

        # ---- pointer scorer over enemies
        self.pointer_qk = QueryKeyBlock(d=self.H, h=self.n_head)  # [B*N, 1, nE]

    # device-friendly hidden init
    def init_hidden(self):
        return self.normal_actions_net.weight.new_zeros(1, self.H)

    @staticmethod
    def _padding_masks(own_feats, ally_feats, enemy_feats):
        # True where token row is valid (not all-zero padding)
        own_mask   = ~(th.all(own_feats   == 0, dim=-1))  # [B*N, 1]
        ally_mask  = ~(th.all(ally_feats  == 0, dim=-1))  # [B*N, nA]
        enemy_mask = ~(th.all(enemy_feats == 0, dim=-1))  # [B*N, nE]
        return own_mask, ally_mask, enemy_mask

    # ---------- Heuristic top-k (optional) ----------
    def _heuristic_topk(self, raw_enemy_feats, enemy_mask, k, explore_p=0.05):
        """
        raw_enemy_feats: [B*N, nE, D_enemy]  (pre-embedding)
        enemy_mask     : [B*N, nE] (bool)
        returns keep_mask: [B*N, nE] (bool)
        """
        if k <= 0:
            return enemy_mask

        D = raw_enemy_feats.size(-1)
        idx = self.heuristic_idx

        def safe_get(t, i, default_val=0.0):
            if i is None or i >= D:  # out-of-range -> 0
                return th.zeros_like(t[..., 0]) + default_val
            return t[..., i]

        avail  = safe_get(raw_enemy_feats, idx.get('avail', 0), 0.0)   # 1 if in attack range
        dist   = safe_get(raw_enemy_feats, idx.get('dist', 1), 1.0)    # normalized by sight
        hp     = safe_get(raw_enemy_feats, idx.get('hp',   None), 0.5) # if absent, neutral 0.5
        shield = safe_get(raw_enemy_feats, idx.get('shield', None), 0.5)

        # Score: prefer in-range, nearer, low HP/shield
        score = 2.0 * avail - dist + 0.5 * (1.0 - hp) + 0.25 * (1.0 - shield)
        score = score.masked_fill(~enemy_mask, float('-inf'))

        if self.training and explore_p > 0.0:
            noise = th.empty_like(score).exponential_().neg()
            score = score + explore_p * noise

        k = min(k, score.size(1))
        _, idxs = th.topk(score, k=k, dim=1)
        keep = th.zeros_like(enemy_mask)
        keep.scatter_(1, idxs, True)
        return enemy_mask & keep

    def forward(self, inputs, hidden_state):
        """
        inputs (controller order confirmed by you):
            (bs, own_feats, ally_feats, enemy_feats, embedding_indices)
        hidden_state: [B, N, H]
        returns: Q: [B*N, 1, A_normal + nE], next_hidden: [B, N, H]
        """
        # -------- unpack (ally before enemy) --------
        bs, own_feats, ally_feats, enemy_feats, embedding_indices = inputs
        Bn = own_feats.size(0)
        nA = ally_feats.shape[1]
        nE = enemy_feats.shape[1]
        self.n_agents = nA + 1

        # ---- append indices to raw own feats (as in SPECTra)
        if self.args.obs_agent_id:
            agent_idx = embedding_indices[0].reshape(-1, 1, 1).to(own_feats.device, own_feats.dtype)
            own_feats = th.cat((own_feats, agent_idx), dim=-1)
        if self.args.obs_last_action:
            last_act = embedding_indices[-1]
            if last_act is None or not isinstance(last_act, th.Tensor):
                last_act = th.zeros(Bn, 1, 1, device=own_feats.device, dtype=own_feats.dtype)
            else:
                last_act = last_act.reshape(-1, 1, 1).to(own_feats.device, own_feats.dtype)
            own_feats = th.cat((own_feats, last_act), dim=-1)

        # ---- masks
        own_mask, ally_mask, enemy_mask = self._padding_masks(own_feats, ally_feats, enemy_feats)
        ally_attn_mask  = ally_mask.unsqueeze(1).to(own_feats.dtype)    # [B*N,1,nA]
        enemy_attn_mask = enemy_mask.unsqueeze(1).to(own_feats.dtype)   # [B*N,1,nE]

        # ---- context & token embeddings
        ctx      = self.context_proj(own_feats.squeeze(1))              # [B*N, H]
        e_own    = self.embed_own(own_feats)                            # [B*N, 1, H]
        e_allies = self.embed_ally(ally_feats)                          # [B*N, nA, H]
        e_enems  = self.embed_enemy(enemy_feats)                        # [B*N, nE, H]

        # ---- FiLM (γ-only)
        e_allies = self.film_allies(e_allies, ctx) if self.use_film else e_allies
        e_enems  = self.film_enems(e_enems,  ctx) if self.use_film else e_enems

        # ---- LoRA adapters before attention
        e_own_q   = self.adapt_q(e_own,   ctx) if self.use_lora else e_own
        e_alliesK = self.adapt_kA(e_allies, ctx) if self.use_lora else e_allies
        e_enemsK  = self.adapt_kE(e_enems,  ctx) if self.use_lora else e_enems

        # ---- dual SAQA with masks
        zA = self.attn_allies(e_own_q, e_alliesK, masks=ally_attn_mask)   # [B*N,1,H]
        zE = self.attn_enems( e_own_q, e_enemsK,  masks=enemy_attn_mask)  # [B*N,1,H]

        # ---- scalar gate (broadcast over channels)
        fuse_in = th.cat([zA, zE, e_own], dim=-1).squeeze(1)              # [B*N,3H]
        g = th.sigmoid(self.gate(fuse_in)).unsqueeze(1)                   # [B*N,1,1]
        z = g * zE + (1.0 - g) * zA                                       # [B*N,1,H]

        # ---- GRU memory (shared params, per-agent states)
        z_flat = z.squeeze(1)                                             # [B*N,H]
        h_prev = hidden_state.reshape(-1, self.H)                         # [B*N,H]
        h_next = self.rnn(z_flat, h_prev)                                 # [B*N,H]
        hq = h_next.unsqueeze(1)                                          # [B*N,1,H]

        # ---- fixed actions head
        q_normal = self.normal_actions_net(hq)                            # [B*N,1,A_normal]

        # ---- optional heuristic top-k pruning (pre-pointer)
        enemy_keep_mask = enemy_mask                                      # [B*N,nE] bool
        if self.use_heuristic_topk and self.topk_enemies > 0 and nE > self.topk_enemies:
            enemy_keep_mask = self._heuristic_topk(
                raw_enemy_feats=enemy_feats,  # RAW (pre-embedding)
                enemy_mask=enemy_mask,
                k=int(self.topk_enemies),
                explore_p=0.05 if self.training else 0.0
            )

        # ---- pointer scores (QK) with LoRA-adapted query/keys
        hq_ptr   = self.adapt_q(hq,       ctx)                            # [B*N,1,H]
        eE_ptr_k = self.adapt_kE(e_enems, ctx)                            # [B*N,nE,H]
        q_ptr = self.pointer_qk(hq_ptr, eE_ptr_k)                         # [B*N,1,nE]

        # ---- attack bias (type-like flavor)
        attack_bias = self.attack_bias_head(hq.squeeze(1)).unsqueeze(1)   # [B*N,1,1]
        q_attack = q_ptr + attack_bias                                    # [B*N,1,nE]

        # ---- mask invalid / pruned enemy targets
        invalid = (~enemy_keep_mask).unsqueeze(1)                         # [B*N,1,nE]
        q_attack = q_attack.masked_fill(invalid, -1e9)

        # ---- concat to per-agent Q (SPECTra-compatible)
        Q = th.cat([q_normal, q_attack], dim=-1)                          # [B*N,1,A_normal+nE]

        # ---- reshape hidden back to [B,N,H]
        bs_int = int(bs)
        next_hidden = h_next.reshape(bs_int, self.n_agents, self.H)
        return Q, next_hidden
