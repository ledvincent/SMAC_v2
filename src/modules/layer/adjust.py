import torch as th
import torch.nn as nn

'''Learns per-feature scale/shift from a context vector to adjust the feature embeddings'''
class FiLM(nn.Module):
    # Use the same network across all contexts
    """y = (1 + 0.5 * tanh(W ctx)) ⊙ x  """
    def __init__(self, hidden: int, alpha = 0.5):
        super().__init__()
        self.to_gamma = nn.Linear(hidden, hidden)
        self.alph = alpha

    def forward(self, x: th.Tensor, ctx: th.Tensor):
        # x: [B*N, L, H], ctx: [B*N, H]
        gamma = 1.0 + self.alpha * th.tanh(self.to_gamma(ctx)).unsqueeze(1)  # [B*N,1,H]
        return gamma * x

'''Inserts small low-rank matrices into weight updates so the model can adapt attention/MLP weights cheaply'''
class LowRankAdapter(nn.Module):
    """
    x' = x + (x @ V) * s(ctx) @ U^T
    - hidden: channel size H
    - rank r << H
    """
    # Same adapter is applied to all tokens (PE safe)
    def __init__(self, hidden: int, rank: int = 4):
        super().__init__()
        self.U = nn.Parameter(th.randn(hidden, rank) / (hidden ** 0.5))
        self.V = nn.Parameter(th.randn(hidden, rank) / (hidden ** 0.5))
        self.to_s = nn.Linear(hidden, rank)

    def forward(self, x: th.Tensor, ctx: th.Tensor):
        # x: [B*N, L, H], ctx: [B*N, H]
        s  = self.to_s(ctx)                  # [B*N, r]
        XV = x @ self.V                      # [B*N, L, r]
        delta = (XV * s.unsqueeze(1)) @ self.U.t()   # [B*N, L, H]
        return x + delta