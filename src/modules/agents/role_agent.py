import torch
import torch.nn as nn

from modules.layer.attn_utils import MultiHeadGroupAttn

class TypeAwareAgent(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_actions = args.n_actions
        self.hidden_dim = hidden_dim= args.hidden_dim
        
        # 1. Entity encoders (PE within entity types)
        self.own_encoder = nn.Sequential(
            nn.Linear(args.own_feats_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Shared encoder for allies/enemies with type conditioning
        self.entity_encoder = nn.Sequential(
            nn.Linear(args.entity_feats_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 2. Type-aware attention mechanism
        self.type_compatibility_matrix = nn.Parameter(
            torch.randn(3, 3) * 0.1  # 3x3 for unit type interactions
        )
        
        # 3. Cross-attention layers
        self.ally_attention = TypedMultiHeadAttention(hidden_dim, n_heads=4)
        self.enemy_attention = TypedMultiHeadAttention(hidden_dim, n_heads=4)
        
        # 4. Hypernet for action head (conditioned on own type)
        self.hypernet = nn.Sequential(
            nn.Linear(3, hidden_dim),  # Input: unit type one-hot
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * args.n_actions)
        )
        
    def forward(self, own_feats, ally_feats, enemy_feats, avail_actions):
        bs = own_feats.shape[0]
        
        # Extract unit types (last 3 dims are one-hot)
        own_type = own_feats[..., -3:]  # [bs, 1, 3]
        ally_types = ally_feats[..., -3:] if ally_feats.shape[1] > 0 else None
        enemy_types = enemy_feats[..., -3:] if enemy_feats.shape[1] > 0 else None
        
        # Encode own features
        own_hidden = self.own_encoder(own_feats)  # [bs, 1, hidden_dim]
        
        # Process allies with type-aware attention
        if ally_feats.shape[1] > 0:
            ally_hidden = self.entity_encoder(ally_feats)
            # Type compatibility weights
            ally_compat = self.compute_type_compatibility(own_type, ally_types)
            ally_context = self.ally_attention(own_hidden, ally_hidden, ally_hidden, 
                                              type_weights=ally_compat)
        else:
            ally_context = torch.zeros_like(own_hidden)
        
        # Process enemies with type-aware attention  
        if enemy_feats.shape[1] > 0:
            enemy_hidden = self.entity_encoder(enemy_feats)
            enemy_compat = self.compute_type_compatibility(own_type, enemy_types)
            enemy_context = self.enemy_attention(own_hidden, enemy_hidden, enemy_hidden,
                                                type_weights=enemy_compat)
        else:
            enemy_context = torch.zeros_like(own_hidden)
        
        # Combine contexts
        combined = own_hidden + ally_context + enemy_context  # [bs, 1, hidden_dim]
        combined = combined.squeeze(1)  # [bs, hidden_dim]
        
        # Generate action weights using hypernet conditioned on unit type
        action_weights = self.hypernet(own_type.squeeze(1))  # [bs, hidden_dim * n_actions]
        action_weights = action_weights.view(bs, self.hidden_dim, self.n_actions)
        
        # Compute Q-values
        q_values = torch.bmm(combined.unsqueeze(1), action_weights).squeeze(1)  # [bs, n_actions]
        
        # Mask unavailable actions
        q_values[avail_actions == 0] = -float('inf')
        
        return q_values
    
    def compute_type_compatibility(self, own_type, other_types):
        """Compute compatibility scores between unit types"""
        own_idx = own_type.argmax(-1)  # [bs, 1]
        other_idx = other_types.argmax(-1)  # [bs, n_entities]
        
        # Look up compatibility scores
        compat_scores = self.type_compatibility_matrix[own_idx, other_idx]
        
        # Mask out non-observable entities (all zeros)
        entity_mask = (other_types.sum(-1) > 0).float()
        
        return compat_scores * entity_mask