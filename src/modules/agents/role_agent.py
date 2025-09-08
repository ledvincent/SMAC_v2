import torch as th
import torch.nn as nn
import torch.nn.functional as F

from modules.layer.attn_utils import MultiHeadGroupAttn, GroupTemp

class RoleEmergentAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RoleEmergentAgent, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_heads = args.n_head
        self.hidden_dim = H = args.hidden_size
        self.output_normal_actions = self.args.output_normal_actions
        self.rnn_hidden_dim = args.rnn_hidden_size

        self.own_feats_dim, self.enemy_feats_dim, self.ally_feats_dim = input_shape
        self.enemy_feats_dim = self.enemy_feats_dim[-1]
        self.ally_feats_dim = self.ally_feats_dim[-1]
        
        if self.args.obs_agent_id:
            self.own_feats_dim += 1
        if self.args.obs_last_action:
            self.own_feats_dim += 1
        
        # Role embedding dimension (continuous latent space)
        self.role_dim = getattr(args, 'role_dim', 32)
        self.use_ensemble = getattr(args, 'use_ensemble', False)
        
        # Type-aware encoding: Extract unit capability features
        self.capability_encoder = nn.Sequential(
            nn.Linear(4, H),  # sight_range, shoot_range, cooldown, max_cooldown
            nn.ReLU(), 
            nn.LayerNorm(H)
        )
        
        # Context encoder for environmental observations
        self.own_encoder = nn.Linear(self.own_feats_dim, H)
        self.ally_encoder = nn.Sequential(
            nn.Linear(self.ally_feats_dim, H),
            # nn.ReLU(),
            nn.LayerNorm(H)
        )
        self.enemy_encoder = nn.Sequential(
            nn.Linear(self.enemy_feats_dim, H),
            # nn.ReLU(),
            nn.LayerNorm(H)
        )
        
        # Attention mechanism for processing multiple allies/enemies
        self.ally_Attention  = MultiHeadGroupAttn(H, self.n_heads)
        self.enemy_Attention = MultiHeadGroupAttn(H, self.n_heads)

        # Feature projection
        self.context_proj = nn.Sequential(
            nn.Linear(H*4, H),
            nn.ReLU(),
            nn.LayerNorm(H)
        )
        
        # Role discovery network - generates continuous role embedding
        self.role_generator = nn.Sequential(
            nn.Linear(H, H),  # own + capability + context
            nn.ReLU(),
            nn.Linear(H, self.role_dim * 2)  # mean and log_var for VAE-style
        )
        
        # Dynamics predictor (inspired by R3DM) - predicts future based on role
        self.dynamics_head = nn.Sequential(
            nn.Linear(self.role_dim + H, H),
            nn.ReLU(),
            nn.Linear(H, H)  # predicted next state features
        )
        
        # GRU for temporal modeling
        self.gru = nn.GRUCell(
            H + self.role_dim,  # context + role
            self.rnn_hidden_dim
        )
        
        # Final q_values computation
        if self.use_ensemble:
            self.n_q_heads = 3
            self.q_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.rnn_hidden_dim + self.role_dim, H),
                    nn.ReLU(),
                    nn.Linear(H, self.n_actions)
                ) for _ in range(self.n_q_heads)
            ])
            self.q_aggregator = nn.Sequential(
                nn.Linear(self.role_dim, self.n_q_heads), nn.Softmax(dim=-1)
            )
        else:
            self.q_network = nn.Sequential(
                nn.Linear(self.rnn_hidden_dim + self.role_dim, H*2),
                nn.ReLU(),
                nn.Linear(H*2, self.n_actions)
            )

    def init_hidden(self):
        # Same convention as ss_rnn_agent: [1, hidden] zeros
        return self.own_encoder.weight.new_zeros(1, self.rnn_hidden_dim)
        
    def forward(self, inputs, hidden_state):
        bs, own_feats, ally_feats, enemy_feats, embedding_indices = inputs
        self.n_agents = ally_feats.shape[1] + 1
        own_feats = own_feats.squeeze(1)

        # Optionally append id / last action to own obs (algo‑side flags)
        if getattr(self.args, "obs_agent_id", False):
            agent_idx = embedding_indices[0].reshape(-1, 1, 1)
            own_feats = th.cat([own_feats, agent_idx], dim=-1)
        if getattr(self.args, "obs_last_action", False):
            last_act = embedding_indices[-1].reshape(-1, 1, 1)
            own_feats = th.cat([own_feats, last_act], dim=-1)
        
        # Extract capability features from own_feats (use_full_feat=True)
        # Assuming the indexed features are: sight_range, shoot_range, cooldown, max_cooldown
        capability_features = own_feats[..., -7:-3]
        capability_embed = self.capability_encoder(capability_features)
        
        # Encode own features
        own_embed = self.own_encoder(own_feats)
        ally_embed = self.ally_encoder(ally_feats)
        enemy_embed = self.enemy_encoder(enemy_feats)

        # Masks
        ally_mask = (ally_feats.sum(dim=-1) != 0)
        enemy_mask = (enemy_feats.sum(dim=-1) != 0)

        # Cross-Attention
        ally_context, ally_attn_w   = self.ally_Attention(own_embed, ally_embed, ally_mask)
        enemy_context, enemy_attn_w = self.enemy_Attention(own_embed, enemy_embed, enemy_mask)

        # ally_context, ally_attn_w   = self.ally_Attention(own_embed, th.cat((own_feats, ally_feats), dim=1), ally_mask)
        # enemy_context, enemy_attn_w = self.enemy_Attention(own_embed, th.cat((own_feats, enemy_feats), dim=1), enemy_mask)


        context = th.cat([own_embed, capability_embed, ally_context, enemy_context], dim=-1)
        context = self.context_proj(context)

        # Generate role embedding (VAE-style for continuous latent space)
        role_params = self.role_generator(context)
        role_mean = role_params[:, :self.role_dim]
        role_log_var = role_params[:, self.role_dim:]
        
        # Reparameterization trick
        if self.training:
            std = th.exp(0.5 * role_log_var)
            eps = th.randn_like(std)
            role_embedding = role_mean + eps * std
        else:
            role_embedding = role_mean  # Use mean during evaluation
        
        # Predict future dynamics based on role (for R3DM-style MI objective)
        predicted_next_context = self.dynamics_head(
            th.cat([role_embedding, context], dim=-1)
        )
        
        # Temporal processing with GRU
        gru_input = th.cat([context, role_embedding], dim=-1) # [B,Hr+R]
        hidden_state = hidden_state.reshape(-1, self.rnn_hidden_dim)
        rnn_output = self.gru(gru_input, hidden_state) # [bs*n_agents,rnn_hidden_dim]
        # rnn_output = rnn_output.unsqueeze(1)  # [bs*n_agents,1,rnn_hidden_dim]

        # Generate Q-values with multiple heads conditioned on role
        q_in = th.cat([rnn_output, role_embedding], dim=-1)                                 # [B,Hr+R]
        if self.use_ensemble:
            qs = th.stack([head(q_in) for head in self.q_heads], dim=-1) # [B,A,K]
            w  = self.q_aggregator(role_embedding).unsqueeze(1)                    # [B,1,K]
            q_values = (qs * w).sum(dim=-1)                              # [B,A]
        else:
            q_values = self.q_network(q_in)                              # [B,A]
        
        return q_values, rnn_output.view(bs, self.n_agents, self.rnn_hidden_dim)