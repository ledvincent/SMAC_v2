from __future__ import annotations
from typing import Any, Tuple

import torch as th
import torch.nn as nn

# Adjust import path if your repo differs
from modules.layer.ss_attention import CrossAttentionBlock, QueryKeyBlock
from modules.layer.adjust import FiLM, LowRankAdapter

# Attention, FILM, LORA, pointer, heuristic
# -> ALF_RNNAgent

class ALF_RNNAgent(nn.Module):
    """
    Dual‑path attention agent with optional FiLM/LoRA/Top‑K/Pointer
    Input / Controller output: (bs, own, ally, enemy, embedding_indices)
        - own_feats : [B*N, 1, D_own]
        - ally_feats : [B*N, nA, D_ally]
        - enemy_feats : [B*N, nE, D_enemy]
        - embedding_indices: tuple containing agent_id scalar and optional last_action

    Output:
        - Q: [B*N, 1, A_normal + nE]  (concat of fixed action logits and per-enemy logits)
        - next_hidden: [B, N, H]
        
    Flags in args (all optional):
        - use_film
        - use_lora
        - use_heuristic_topk
        - use_pointer
    """

    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args

        # ---- toggles
        self.use_film = bool(getattr(args, "use_film", False))
        self.use_lora = bool(getattr(args, "use_lora", True))
        self.use_pointer = bool(getattr(args, "use_pointer", True))
        self.use_topk = bool(getattr(args, "use_heuristic_topk", False))
        self.topk = int(getattr(args, "topk_enemies", 0))
        self.rank = int(getattr(args, "adapter_rank", 4)) 

        # ---- sizes
        self.H = int(args.hidden_size)
        self.n_head = int(args.n_head)
        self.A_norm = int(args.output_normal_actions)
        self.n_actions = int(args.n_actions)

        # ---- input dims
        own_dim, enemy_shape, ally_shape = input_shape
        self.D_own = int(own_dim)
        self.D_e = int(enemy_shape[-1])
        self.D_a = int(ally_shape[-1])
        if getattr(args, "obs_agent_id", False): self.D_own += 1
        if getattr(args, "obs_last_action", False): self.D_own += 1

        # ---- embeddings + context
        self.embed_own = nn.Linear(self.D_own, self.H)
        self.embed_ally = nn.Linear(self.D_a, self.H)
        self.embed_enemy = nn.Linear(self.D_e, self.H)
        self.ctx_proj = nn.Linear(self.D_own, self.H)

        # ---- FiLM (or identity)
        if self.use_film:
            self.film_a = FiLM(self.H)
            self.film_e = FiLM(self.H)
        else:
            self.film_a = self.film_e = nn.Identity()

        # ---- LoRA (or identity)
        if self.use_lora:
            self.adapt_q = LowRankAdapter(self.H, self.rank)
            self.adapt_kA = LowRankAdapter(self.H, self.rank)
            self.adapt_kE = LowRankAdapter(self.H, self.rank)
        else:
            self.adapt_q = self.adapt_kA = self.adapt_kE = nn.Identity()

        # ---- attention + fusion + memory
        self.attn_A = CrossAttentionBlock(d=self.H, h=self.n_head)
        self.attn_E = CrossAttentionBlock(d=self.H, h=self.n_head)
        self.gate = nn.Linear(3*self.H, 1) # scalar gate over [zA,zE,e_own]
        self.rnn = nn.GRUCell(self.H, self.H)

        # ---- action heads
        self.head_norm = nn.Linear(self.H, self.A_norm)
        if self.use_pointer:
            self.ptr = QueryKeyBlock(d=self.H, h=self.n_head) # [B*N,1,nE]
            self.abias = nn.Linear(self.H, 1)
            nn.init.zeros_(self.abias.weight); nn.init.zeros_(self.abias.bias)
        else:
            self.bilin = nn.Bilinear(self.H, self.H, 1) # (hq, e_i) -> scalar

    # device-friendly hidden init
    def init_hidden(self):
        return self.normal_actions_net.weight.new_zeros(1, self.H)

    # ---------- Heuristic top-k (optional) ----------
    @staticmethod
    def _enemy_col_idx(D: int):
        idx = {"shoot": 0, "dist": 1, "hp": None, "shield": None}
        if D >= 5: idx["hp"] = 4
        if D >= 6: idx["shield"] = 5
        return idx

    '''Computes a score per enemy from raw enemy_feats
    and keeps only the top-k enemies
    Arbitrary scoring function'''
    def _heuristic_topk(self, raw_enemy_feats, enemy_mask, k, explore_p=0.05):
        """
        raw_enemy_feats: [B*N, nE, D_enemy]  (pre-embedding)
        enemy_mask     : [B*N, nE] (bool)
        returns keep_mask: [B*N, nE] (bool)
        """
        if k <= 0:
            return enemy_mask

        D = raw_enemy_feats.size(-1)

        # Columns per SMAC build_obs_agent() order for enemies
        shoot = raw_enemy_feats[..., 0]                          # 1 if targetable now
        dist  = raw_enemy_feats[..., 1].clamp_min(1e-6)          # normalized by sight
        hp    = raw_enemy_feats[..., 4] if D >= 5 else None      # normalized health (if provided)
        sh    = raw_enemy_feats[..., 5] if D >= 6 else None      # normalized shield (if Protoss & provided)
        # All scalars

        # Effective HP (use shield if present; otherwise just HP)
        if hp is None:
            ehp = None
        else:
            ehp = hp if sh is None else (0.7 * hp + 0.3 * sh)

        # Scoring: prefer shootable, nearer, and low effective HP
        w_shoot, w_range, w_ehp = 3.0, 1.25, 1.5
        score = w_shoot * shoot + w_range * (1.0 / dist) # -> [B*N, nE]
        if ehp is not None:
            score = score + w_ehp * (1.0 - ehp)

        # Mask out invalid enemies (out of fov, dead)
        score = score.masked_fill(~enemy_mask, float("-inf"))

        # Mild exploration during training
        # Add randmo noise to score during training to avoid deterministic selection
        if self.training and explore_p > 0.0:
            noise = th.empty_like(score).exponential_().neg()
            score = score + explore_p * noise

        # Top-k selection
        # take k largest scores per row (for each agent) and build a boolean mask
        k = min(k, score.size(1))
        _, idxs = th.topk(score, k=k, dim=1)
        keep = th.zeros_like(enemy_mask)
        keep.scatter_(1, idxs, True) # [B*N, nE]
        # Return mask where True for enemies that are valid and in top-k
        return enemy_mask & keep

    def forward(self, inputs, hidden_state):
        """
        inputs (controller order confirmed by you):
            (bs, own_feats, ally_feats, enemy_feats, embedding_indices)
        hidden_state: [B, N, H]
        returns: Q: [B*N, 1, A_normal + nE], next_hidden: [B, N, H]
        """
        # Get features from controller
        bs, own_feats, ally_feats, enemy_feats, embedding_indices = inputs
        Bn = own_feats.size(0)
        nA = ally_feats.shape[1]
        nE = enemy_feats.shape[1]
        n_agents = nA + 1

        # Own features
        # Add agent id
        if self.args.obs_agent_id:
            agent_idx = embedding_indices[0].reshape(-1, 1, 1).to(own_feats.device, own_feats.dtype)
            own_feats = th.cat((own_feats, agent_idx), dim=-1)
        # Add agent's last action
        if self.args.obs_last_action:
            last_act = embedding_indices[-1]
            if last_act is None or not isinstance(last_act, th.Tensor):
                last_act = th.zeros(Bn, 1, 1, device=own_feats.device, dtype=own_feats.dtype)
            else:
                last_act = last_act.reshape(-1, 1, 1).to(own_feats.device, own_feats.dtype)
            own_feats = th.cat((own_feats, last_act), dim=-1)

        # Create masks for valid entities
        ally_mask  = ~(th.all(ally_feats  == 0, dim=-1))  # [B*N, nA]
        enemy_mask = ~(th.all(enemy_feats == 0, dim=-1))
        ally_attn_mask  = ally_mask.unsqueeze(1).to(own_feats.dtype)    # [B*N,1,nA]
        enemy_attn_mask = enemy_mask.unsqueeze(1).to(own_feats.dtype)   # [B*N,1,nE]

        # ---- Project features ----
        # Context used for LoRA adapters and FiLM
        ctx      = self.context_proj(own_feats.squeeze(1))              # [B*N, H]
        # feature embeddings
        e_own    = self.embed_own(own_feats)                            # [B*N, 1, H]
        e_allies = self.embed_ally(ally_feats)                          # [B*N, nA, H]
        e_enems  = self.embed_enemy(enemy_feats)                        # [B*N, nE, H]

        # (optional) Use FiLM (or identity)
        # Adjust the embeddings using the context
        e_allies = self.film_a(e_allies, ctx)
        e_enems  = self.film_e(e_enems,  ctx)

        # (optional) LoRA adapters before attention (or identity)
        # Returns unchanged embeddings if args.use_lora is False (see init)
        own_q   = self.adapt_q(e_own,   ctx)
        Ka = self.adapt_kA(e_allies, ctx)
        Ke  = self.adapt_kE(e_enems,  ctx)

        # ----- Dual cross‑attention (masked) -----
        zA = self.attn_A(own_q, Ka, masks=ally_attn_mask)    # [B*N,1,H]
        zE = self.attn_E( own_q, Ke, masks=enemy_attn_mask)  # [B*N,1,H]

        # ----- Gate‑fusion of ally/enemy paths with the raw own token -----
        fuse_in = th.cat([zA, zE, e_own], dim=-1).squeeze(1)              # [B*N,3H]
        g = th.sigmoid(self.gate(fuse_in)).unsqueeze(1)                   # [B*N,1,1]
        z = g * zE + (1.0 - g) * zA                                       # [B*N,1,H]

        # GRU memory (shared params, per-agent states)
        h_prev = hidden_state.reshape(-1, self.H)                         # [B*N,H]
        h_next = self.rnn(z.squeeze(1), h_prev)                           # [B*N,H]
        hq = h_next.unsqueeze(1)                                          # [B*N,1,H]

        # Fixed action logits
        q_norm = self.head_norm(hq)                            # [B*N,1,A_normal]

        # (optional) heuristic top-k pruning (pre-pointer)
        enemy_keep_mask = enemy_mask                                      # [B*N,nE] bool
        if self.use_topk and self.topk>0 and nE>self.topk:
            enemy_keep_mask = self._heuristic_topk(
                raw_enemy_feats=enemy_feats,  # RAW (pre-embedding)
                enemy_mask=enemy_mask,
                k=self.topk,
                explore_p=0.05 if self.training else 0.0
            )

        # Per‑enemy logits: pointer or bilinear fallback
        if self.use_pointer:
            q_ptr = self.ptr(self.adapt_q(hq, ctx), self.adapt_kE(e_enems, ctx)) # [B*N,1,nE]
            bias = self.abias(hq.squeeze(1)).unsqueeze(1) # [B*N,1,1]
            q_enemy = q_ptr + bias
        else:
            q_enemy = self.bilin(hq.expand(-1, nE, -1), e_enems).transpose(1, 2) # [B*N,1,nE]


        # 12) Mask pruned/invalid enemies and concatenate
        q_enemy = q_enemy.masked_fill((~enemy_keep_mask).unsqueeze(1), -1e9)
        Q = th.cat([q_norm, q_enemy], dim=-1) # [B*N,1,A_norm+nE]


        # 13) Reshape hidden back to [B,N,H] and return
        bs_int = int(bs.item()) if isinstance(bs, th.Tensor) else int(bs)
        return Q, h_next.reshape(bs_int, n_agents, self.H)
