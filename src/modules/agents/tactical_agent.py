# modules/agents/tactical_agent.py
import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modules.layer.attn_utils import MultiHeadGroupAttn, GroupTemp

# ---------- utils ----------
def masked_mean(x, mask, dim):
    eps = 1e-8
    w = mask.unsqueeze(-1).to(x.dtype)         # [..., M, 1]
    s = (x * w).sum(dim=dim)
    d = w.sum(dim=dim).clamp_min(eps)
    return s / d

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, n_layers=2, dropout=0.0):
        super().__init__()
        layers=[]; d=in_dim
        for _ in range(n_layers-1):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.LayerNorm(hidden)]
            if dropout>0: layers += [nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# ---------- Agent (HPN-compatible) ----------
class CAPERNN_ContRoles_HPN(nn.Module):
    """
    Forward signature matches HPN controller:
      inputs = (bs, own_feats, ally_feats, enemy_feats, embedding_indices)
        own_feats   : [B*N, 1, d_own]
        ally_feats  : [B*N, Na, d_all]
        enemy_feats : [B*N, Ne, d_en]
      hidden_state : [B, N, hidden]  (GRUCell-style like SS_RNNAgent)

    Returns:
      Q: [B*N, output_normal_actions + Ne]   (interact head ALWAYS points to enemies)
      hidden': [B, N, hidden]
    """
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args

        # Per-entity dims provided by controller
        own_ctx_dim, enemy_feats_dim, ally_feats_dim = input_shape
        d_en  = enemy_feats_dim[-1] if isinstance(enemy_feats_dim, (list, tuple)) else enemy_feats_dim
        d_all = ally_feats_dim[-1]  if isinstance(ally_feats_dim, (list, tuple))  else ally_feats_dim

        # HPN/SS_RNNAgent-style flags
        self.obs_agent_id    = getattr(self.args, "obs_agent_id", False)
        self.obs_last_action = getattr(self.args, "obs_last_action", False)

        # Core sizes from HPN config (aligns with ss_rnn_agent.py)
        self.hidden_size = self.args.hidden_size
        self.n_head      = self.args.n_head
        self.n_actions   = self.args.n_actions
        self.n_move      = self.args.output_normal_actions         # normal (non-interact) actions
        self.role_dim    = getattr(self.args, "role_dim", 8)

        # Own feature dimension after optional extras
        self.own_feats_dim = own_ctx_dim
        if self.obs_agent_id:    self.own_feats_dim += 1
        if self.obs_last_action: self.own_feats_dim += 1
        self.ally_feats_dim  = d_all
        self.enemy_feats_dim = d_en

        # Embeddings
        self.own_embedding     = nn.Linear(self.own_feats_dim,  self.hidden_size)
        self.allies_embedding  = nn.Linear(self.ally_feats_dim, self.hidden_size)
        self.enemies_embedding = nn.Linear(self.enemy_feats_dim,self.hidden_size)

        # Group cardinality temperature
        self.use_group_card_temp = bool(getattr(self.args, "use_group_card_temp", False))
        if self.use_group_card_temp:
            self.groupTemp_A = GroupTemp()
            self.groupTemp_E = GroupTemp()

        # Separate ally/enemy cross-attn (no geometry priors)
        self.attn_ally  = MultiHeadGroupAttn(self.hidden_size, self.n_head)
        self.attn_enemy = MultiHeadGroupAttn(self.hidden_size, self.n_head)

        # Fuse own + contexts
        self.fuse = MLP(self.hidden_size * 3, self.hidden_size, self.hidden_size, n_layers=2)

        # Continuous role embedding (no fixed K)
        self.pool_proj = MLP(self.hidden_size * 3, 128, 128, n_layers=2)
        self.role_head = MLP(128, 128, self.role_dim, n_layers=2)
        self.role_ln   = nn.LayerNorm(self.role_dim)
        self.gating_alpha = getattr(self.args, "gating_alpha", 1.0)

        # GRUCell core (HPN baseline style)
        self.rnn = nn.GRUCell(self.hidden_size + self.role_dim, self.hidden_size)

        # Heads
        self.normal_actions_net = nn.Linear(self.hidden_size, self.n_move)  # q_normal
        # Pointer interact head: queries from (hidden+role), keys from ENEMIES ONLY
        self.q_proj = nn.Linear(self.hidden_size + self.role_dim, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size,               self.hidden_size, bias=False)

        # Side channel for learner regularizers
        self.register_buffer("_dummy", th.zeros(1), persistent=False)
        self.last_role_embed = None  # [B*N, role_dim]

    def init_hidden(self):
        # Same convention as ss_rnn_agent: [1, hidden] zeros
        return self.own_embedding.weight.new(1, self.hidden_size).zero_()

    def forward(self, inputs, hidden_state):
        """
        inputs: (bs, own_feats, ally_feats, enemy_feats, embedding_indices)
        hidden_state: [B, N, H]
        """
        bs, own_feats, ally_feats, enemy_feats, embedding_indices = inputs
        device = own_feats.device

        BN = own_feats.shape[0]       # B*N
        Na = ally_feats.shape[1]
        Ne = enemy_feats.shape[1]
        n_agents = Na + 1

        # Masks from zero padding
        own_mask   = ~th.all(own_feats == 0, dim=-1)       # [BN, 1]
        ally_mask  = ~th.all(ally_feats == 0, dim=-1)      # [BN, Na]
        enemy_mask = ~th.all(enemy_feats == 0, dim=-1)     # [BN, Ne]

        # Append agent_id / last_action if configured (same as ss_rnn_agent)
        if self.obs_agent_id and embedding_indices and embedding_indices[0] is not None:
            agent_indices = embedding_indices[0].reshape(-1, 1, 1)  # [B*N,1,1]
            own_feats = th.cat((own_feats, agent_indices), dim=-1)
        if self.obs_last_action and embedding_indices and embedding_indices[-1] is not None:
            last_action_indices = embedding_indices[-1].reshape(-1, 1, 1)  # [B*N,1,1]
            own_feats = th.cat((own_feats, last_action_indices), dim=-1)

        # Embed entities
        own_e   = self.own_embedding(own_feats)       # [BN, 1, H]
        ally_e  = self.allies_embedding(ally_feats)   # [BN, Na, H]
        enemy_e = self.enemies_embedding(enemy_feats) # [BN, Ne, H]

        # Group temps for cardinality neutralization
        temp_A = temp_E = None
        if self.use_group_card_temp:
            # all [BN]
            temp_A = self.groupTemp_A(ally_mask.float().sum(-1))
            temp_E = self.groupTemp_E(enemy_mask.float().sum(-1))

        # Single-query (own) cross-attention
        self_emb = own_e[:, 0]  # [BN, H]
        cA, _ = self.attn_ally (self_emb, ally_e,  ally_mask,  group_temp=temp_A)
        cE, _ = self.attn_enemy(self_emb, enemy_e, enemy_mask, group_temp=temp_E)

        # Fuse & pooled summaries (for role)
        h0 = self.fuse(th.cat([self_emb, cA, cE], dim=-1))                 # [BN, H]
        pA = masked_mean(ally_e,  ally_mask,  dim=1) if Na > 0 else th.zeros(BN, self.hidden_size, device=device)
        pE = masked_mean(enemy_e, enemy_mask, dim=1) if Ne > 0 else th.zeros(BN, self.hidden_size, device=device)
        pooled = self.pool_proj(th.cat([self_emb, pA, pE], dim=-1))        # [BN, 128]

        # Continuous role embedding r
        r = th.tanh(self.role_head(pooled))                                # [BN, R]
        r = self.role_ln(r)
        self.last_role_embed = r.detach()

        # GRUCell (per agent)
        h_prev = hidden_state.reshape(-1, self.hidden_size)                # [B*N, H]
        rnn_in = th.cat([h0, self.gating_alpha * r], dim=-1)               # [BN, H+R]
        h_cur  = self.rnn(rnn_in, h_prev)                                  # [BN, H]

        # Heads
        # (1) normal actions
        q_normal = self.normal_actions_net(h_cur).unsqueeze(1)             # [BN,1,A_norm]

        # (2) interact head -> enemies ONLY
        if Ne > 0:
            q_vec = self.q_proj(th.cat([h_cur, self.gating_alpha * r], dim=-1)).unsqueeze(1)  # [BN,1,H]
            k_vec = self.k_proj(enemy_e)                                                      # [BN,Ne,H]
            logits = th.einsum("bij,bmj->bim", q_vec, k_vec) / math.sqrt(q_vec.size(-1))      # [BN,1,Ne]
            logits = logits.masked_fill(enemy_mask.unsqueeze(1) == 0, float("-inf"))
            # center over valid enemies to stabilize scales
            masked = th.where(enemy_mask.unsqueeze(1), logits, th.zeros_like(logits))
            denom  = enemy_mask.sum(dim=-1, keepdim=True).clamp_min(1).unsqueeze(1)
            mean_valid = masked.sum(dim=-1, keepdim=True) / denom
            q_interact = logits - mean_valid                                                  # [BN,1,Ne]
        else:
            q_interact = th.zeros(BN, 1, 0, device=device)

        # Final Q and new hidden
        Q = th.cat([q_normal, q_interact], dim=-1).squeeze(1)              # [BN, A_norm + Ne]
        h_out = h_cur.view(bs, n_agents, self.hidden_size)                 # [B, N, H]
        return Q, h_out
