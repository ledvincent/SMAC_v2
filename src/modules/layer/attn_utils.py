import torch as th
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional

# ------------------------------- utils -------------------------------

def _masked_mean(x, mask, dim):
    """Masked mean over dim; mask True=valid. Works for 2D/3D+ by auto-expanding mask."""
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    m = mask.to(dtype=x.dtype)
    num = (x * m).sum(dim=dim)
    den = m.sum(dim=dim).clamp(min=1.0)
    return num / den

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
      logit_bias: [B, heads, N] (optional per-head bias)
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

class ConditionalLayerNorm(nn.Module):
    """LayerNorm(x) then affine from context (starts as identity)."""
    def __init__(self, hidden):
        super().__init__()
        self.ln = nn.LayerNorm(hidden, elementwise_affine=False)
        self.gamma = nn.Linear(hidden, hidden, bias=False)
        self.beta  = nn.Linear(hidden, hidden, bias=False)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, x: th.Tensor, ctx: th.Tensor) -> th.Tensor:
        return (1 + self.gamma(ctx)) * self.ln(x) + self.beta(ctx)