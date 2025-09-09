import torch as th
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional

# ------------------------------- utils -------------------------------
class GroupTemp(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(th.tensor(0.0))
        self.beta  = nn.Parameter(th.tensor(0.0))

    def forward(self, counts):
        # T = softplus(alpha * log(n) + beta)
        n = counts.clamp(min=1.0)
        return F.softplus(self.alpha * th.log(n) + self.beta) + 1e-6

# ------------------------------- layers ------------------------------

class MultiHeadGroupAttn(nn.Module):
    """Single-query multi-head cross-attention over a group (allies OR enemies).
    Inputs:
    N: number of entities in the group (enemy:nE, ally:nA)
      q_src: [B, H]    (own context query)
      group: [B, N, H] ally/enemy features. keys/values come from this set
      mask: [B, N]     (bool) mask out invalid entities
      group_temp: [B] (optional group temperature)
    """
    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.H = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads # head dimension
        self.q = nn.Linear(self.H, self.H, bias=False)
        self.k = nn.Linear(self.H, self.H, bias=False)
        self.v = nn.Linear(self.H, self.H, bias=False)
        self.out = nn.Linear(self.H, self.H, bias=False)

    def forward(self, q_src: th.Tensor, group: th.Tensor, mask: th.Tensor,
                group_temp: Optional[th.Tensor] = None):
        # q_src: [B,H], group: [B,N,H], mask: [B,N]
        B, N, H = group.shape
        # Safety checks
        assert H == self.H
        assert q_src.shape == (B, H)

        n_heads, head_dim = self.n_heads, self.head_dim

        Q = self.q(q_src).view(B, n_heads, head_dim).unsqueeze(2)                     # [B, H] ->  view -> [B,n_heads,1,head_dim]
        K = self.k(group).view(B, N, n_heads, head_dim).permute(0, 2, 1, 3)     # [B, N, H] ->  view+permute ->[B,n_heads,N,head_dim]
        V = self.v(group).view(B, N, n_heads, head_dim).permute(0, 2, 1, 3)     # [B, N, H] ->  view+permute ->[B,n_heads,N,head_dim]

        # Scaled dot-product attention (QK^T/sqrt(head_dim))
        # divide by sqrt(head_dim) (head dimension) for variance stabilization
        logits = (Q @ K.transpose(-2, -1)).squeeze(-2) / math.sqrt(head_dim)  # [B,n_heads,N]

        # Group temperature
        # Divide group logit vector by a scalar temperature
        # Counteracts the effect of the group size on attention
        if group_temp is not None:
            logits = logits / group_temp.unsqueeze(1).unsqueeze(-1)
        # -------------------------

        # Mask out invalid entities
        logits = logits.masked_fill(~mask.unsqueeze(1), -1e9)      # [B,n_heads,N]
        # Softmax (masked out entities have prob=0 after softmax)
        attn = th.softmax(logits, dim=-1)                              # [B,n_heads,N]
        # softmax(QK^T/sqrt(head_dim)) @ V
        ctx = (attn.unsqueeze(-2) @ V).squeeze(-2)                     # [B,n_heads,head_dim]
        # Merge heads
        ctx = ctx.reshape(B, H)                  # [B,H=n_heads*head_dim]
        return self.out(ctx), attn      # ([B,H], [B,n_heads,N])

class SetAttentionBlock(nn.Module):
    def __init__(self, q_hidden_dim: int, k_hidden_dim: int, hidden_dim: int, n_heads: int):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.H = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # Project to shared space H
            
        self.project_keys    = nn.Linear(k_hidden_dim, hidden_dim)

    def forward(self, queries: th.Tensor, keys: th.Tensor) -> th.Tensor:
        # queries: [B, N, q_hidden_dim], keys: [B, M, k_hidden_dim]
        B, N, _ = queries.shape
        M = keys.shape[1]
        h, p = self.n_heads, self.head_dim

        # Linear projections -> [B, N, H], [B, M, H]
        Q = self.project_queries(queries)
        K = self.project_keys(keys)

        # Split into heads -> [B, h, N, p], [B, h, M, p]
        Q = Q.view(B, N, h, p).permute(0, 2, 1, 3).contiguous()
        K = K.view(B, M, h, p).permute(0, 2, 1, 3).contiguous()

        # Scaled dot-product per head: [B, h, N, p] x [B, h, p, M] -> [B, h, N, M]
        scores_h = th.matmul(Q, K.transpose(-1, -2)) / (p ** 0.5)

        # Average over heads -> [B, N, M]
        scores = scores_h.mean(dim=1)
        return scores

class ConditionalLayerNorm(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.ln = nn.LayerNorm(hidden, elementwise_affine=False)
        self.gamma = nn.Linear(hidden, hidden, bias=False)
        self.beta  = nn.Linear(hidden, hidden, bias=False)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, x: th.Tensor, ctx: th.Tensor) -> th.Tensor:
        return (1 + self.gamma(ctx)) * self.ln(x) + self.beta(ctx)