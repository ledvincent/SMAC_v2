import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.layer.attn_utils import MultiHeadGroupAttn

class RoleEmergentAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RoleEmergentAgent, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.hidden_dim = args.rnn_hidden_dim
        
        # Role embedding dimension (continuous latent space)
        self.role_dim = getattr(args, 'role_dim', 32)
        
        # Type-aware encoding: Extract unit capability features
        self.capability_encoder = nn.Sequential(
            nn.Linear(4, 64),  # sight_range, shoot_range, cooldown, max_cooldown
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32)
        )
        
        # Context encoder for environmental observations
        self.own_encoder = nn.Linear(input_shape, 64)
        self.ally_encoder = nn.Sequential(
            nn.Linear(args.ally_feats_size, 32),
            nn.ReLU()
        )
        self.enemy_encoder = nn.Sequential(
            nn.Linear(args.enemy_feats_size, 32),
            nn.ReLU()
        )
        
        # Attention mechanism for processing multiple allies/enemies
        self.ally_attention = nn.MultiheadAttention(
            embed_dim=32, 
            num_heads=4,
            batch_first=True
        )
        self.enemy_attention = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=4, 
            batch_first=True
        )
        
        # Role discovery network - generates continuous role embedding
        self.role_generator = nn.Sequential(
            nn.Linear(64 + 32 + 64, 128),  # own + capability + context
            nn.ReLU(),
            nn.Linear(128, self.role_dim * 2)  # mean and log_var for VAE-style
        )
        
        # Dynamics predictor (inspired by R3DM) - predicts future based on role
        self.dynamics_head = nn.Sequential(
            nn.Linear(self.role_dim + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)  # predicted next state features
        )
        
        # GRU for temporal modeling
        self.gru = nn.GRU(
            64 + self.role_dim,  # context + role
            self.hidden_dim,
            batch_first=True
        )
        
        # Q-value heads conditioned on role
        self.q_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim + self.role_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.n_actions)
            ) for _ in range(3)  # multiple heads for robustness
        ])
        
        # Final Q aggregator
        self.q_aggregator = nn.Linear(3, 1)
        
    def forward(self, inputs, hidden_state):
        bs, own_feats, ally_feats, enemy_feats, embedding_indices = inputs
        
        # Extract capability features from own_feats
        # Assuming last 4 features are: sight_range, shoot_range, cooldown, max_cooldown
        capability_features = own_feats[:, -4:]
        capability_embed = self.capability_encoder(capability_features)
        
        # Encode own features
        own_embed = F.relu(self.own_encoder(own_feats))
        
        # Process allies with attention (handle variable number)
        if ally_feats.shape[1] > 0:
            ally_embeds = self.ally_encoder(ally_feats)
            # Self-attention among allies
            ally_context, _ = self.ally_attention(
                ally_embeds, ally_embeds, ally_embeds
            )
            ally_summary = ally_context.mean(dim=1)  # Aggregate
        else:
            ally_summary = torch.zeros(bs, 32).to(own_feats.device)
        
        # Process enemies with attention (handle variable number)
        if enemy_feats.shape[1] > 0:
            enemy_embeds = self.enemy_encoder(enemy_feats)
            # Self-attention among enemies
            enemy_context, _ = self.enemy_attention(
                enemy_embeds, enemy_embeds, enemy_embeds
            )
            enemy_summary = enemy_context.mean(dim=1)  # Aggregate
        else:
            enemy_summary = torch.zeros(bs, 32).to(own_feats.device)
        
        # Combine context
        context = torch.cat([
            own_embed,
            capability_embed,
            ally_summary,
            enemy_summary
        ], dim=-1)
        
        # Generate role embedding (VAE-style for continuous latent space)
        role_params = self.role_generator(context)
        role_mean = role_params[:, :self.role_dim]
        role_log_var = role_params[:, self.role_dim:]
        
        # Reparameterization trick for differentiable sampling
        if self.training:
            std = torch.exp(0.5 * role_log_var)
            eps = torch.randn_like(std)
            role_embedding = role_mean + eps * std
        else:
            role_embedding = role_mean  # Use mean during evaluation
        
        # Predict future dynamics based on role (for R3DM-style MI objective)
        predicted_next = self.dynamics_head(
            torch.cat([role_embedding, own_embed], dim=-1)
        )
        
        # Temporal processing with GRU
        gru_input = torch.cat([context[:, :64], role_embedding], dim=-1)
        gru_input = gru_input.unsqueeze(1)  # Add time dimension
        gru_out, hidden_state = self.gru(gru_input, hidden_state)
        gru_out = gru_out.squeeze(1)
        
        # Generate Q-values with multiple heads conditioned on role
        q_inputs = torch.cat([gru_out, role_embedding], dim=-1)
        q_values_list = []
        for q_head in self.q_heads:
            q_values_list.append(q_head(q_inputs).unsqueeze(-1))
        
        # Aggregate Q-values from different heads
        q_stack = torch.cat(q_values_list, dim=-1)
        head_weights = F.softmax(self.q_aggregator.weight, dim=0)
        q_values = (q_stack * head_weights.view(1, 1, -1)).sum(dim=-1)
        
        # Store role info for training objectives (can be accessed externally)
        self.last_role_embedding = role_embedding
        self.last_role_mean = role_mean
        self.last_role_log_var = role_log_var
        self.last_predicted_next = predicted_next
        
        return q_values, hidden_state