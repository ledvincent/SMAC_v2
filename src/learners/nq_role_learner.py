import copy
import time
import os 
import torch as th
from torch.nn import functional as F
from torch.optim import RMSprop, Adam

from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.qatten import QattenMixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.ss_mixer import SSMixer
from modules.mixers.flex_qmix import FlexQMixer
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
from utils.th_utils import get_parameters_num

def calculate_target_q(target_mac, batch, n_agents, enable_parallel_computing=False, thread_num=4):
    if enable_parallel_computing:
        th.set_num_threads(thread_num)
    with th.no_grad():
        # Set target mac to testing mode
        target_mac.set_evaluation_mode()
        target_mac_out = []
        
        target_mac.init_hidden(batch.batch_size, n_agents)
        for t in range(batch.max_seq_length):
            target_agent_outs = target_mac.forward(batch, t=t, test_mode = False)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time
        return target_mac_out


def calculate_n_step_td_target(args, target_mixer, target_max_qvals, batch, rewards, terminated, mask, gamma, td_lambda,
                               enable_parallel_computing=False, thread_num=4, q_lambda=False, target_mac_out=None):
    if enable_parallel_computing:
        th.set_num_threads(thread_num)

    with th.no_grad():
        # Set target mixing net to testing mode
        target_mixer.eval()
        # Calculate n-step Q-Learning targets
        
        target_max_qvals = target_mixer(target_max_qvals, batch["state"])

        if q_lambda:
            raise NotImplementedError
            qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
            qvals = target_mixer(qvals, batch["state"])
            targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals, gamma, td_lambda)
        else:
            targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, gamma, td_lambda)
        return targets.detach()

def _role_kl(mu: th.Tensor, logvar: th.Tensor, reduce="none"):
    # mu, logvar: [..., R]
    kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)   # [..., R]
    kl = kl.sum(dim=-1)                                    # sum over R -> [...]
    if reduce == "mean":
        return kl.mean()
    return kl

def _beta_schedule(step: int, beta_max=1e-3, warmup=50_000):
    s = min(1.0, float(step) / float(warmup))
    return beta_max * s


class NQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda else 'cpu')
        self.params = list(mac.parameters())

        self.role_kl_beta_max = getattr(args, "role_kl_beta", 1e-3)
        self.role_kl_warmup  = getattr(args, "role_kl_warmup_steps", 50_000)


        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":  # 31.521K
            self.mixer = Mixer(args)
        elif args.mixer == "ss_mixer":
            self.mixer = SSMixer(args)
        elif args.mixer == "flex_mixer":
            self.mixer = FlexQMixer(args)
        else:
            raise "mixer error"

        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        print('Mixer Size: ', get_parameters_num(self.mixer.parameters()))
        
        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0
        self.avg_time = 0

        self.enable_parallel_computing = (not self.args.use_cuda) and getattr(self.args, 'enable_parallel_computing',
                                                                              False)
        # self.enable_parallel_computing = False
        if self.enable_parallel_computing:
            from multiprocessing import Pool
            # Multiprocessing pool for parallel computing.
            self.pool = Pool(1)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):

        start_time = time.time()
        if self.args.use_cuda and str(self.mac.get_device()) == "cpu":
            self.mac.cuda()

        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        if self.enable_parallel_computing:
            target_mac_out = self.pool.apply_async(
                calculate_target_q,
                (self.target_mac, batch, self.args.n_agents, True, self.args.thread_num)
            )

        # Calculate estimated Q-Values
        self.mac.set_train_mode()
        mac_out = []

        # Store role information for auxiliary losses
        role_embeddings = []  # [time, n_agents, batch, role_dim]
        role_means = []
        role_log_vars = []
        predicted_next_contexts = []
        
        self.mac.init_hidden(batch.batch_size, n_agents = self.args.n_agents)

        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t, test_mode = False)
            mac_out.append(agent_outs)

            # ------------------------------------------------------------------------------
            # Collect role information from each agent at this timestep
            if hasattr(self.mac.agents[0], 'last_role_embedding'):
                t_role_embeddings = []
                t_role_means = []
                t_role_log_vars = []
                t_predicted_next = []
                
                for agent_id in range(self.args.n_agents):
                    agent = self.mac.agents[agent_id]
                    if hasattr(agent, 'last_role_embedding'):
                        t_role_embeddings.append(agent.last_role_embedding)
                        t_role_means.append(agent.last_role_mean)
                        t_role_log_vars.append(agent.last_role_log_var)
                        if hasattr(agent, 'predicted_next_context'):
                            t_predicted_next.append(agent.predicted_next_context)
                
                if len(t_role_embeddings) > 0:
                    role_embeddings.append(th.stack(t_role_embeddings, dim=0))  # [n_agents, batch, role_dim]
                    role_means.append(th.stack(t_role_means, dim=0))
                    role_log_vars.append(th.stack(t_role_log_vars, dim=0))
                    if len(t_predicted_next) > 0:
                        predicted_next_contexts.append(th.stack(t_predicted_next, dim=0))
            # ------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------

        # --- Role KL regularizer (optional; present if the agent provided stats) ---
        aux = self.mac.pop_aux() if hasattr(self.mac, "pop_aux") else {}
        role_kl_loss = None
        if "role_mean" in aux and "role_log_var" in aux:
            # aux: [B, T, n_agents, R], match TD time indexing (use [:, :-1])
            mu = aux["role_mean"][:, :-1]      # [B, T-1, n_agents, R]
            lv = aux["role_log_var"][:, :-1]   # [B, T-1, n_agents, R]

            # Per-(B,t,agent) KL
            kl_btna = _role_kl(mu, lv, reduce="none")  # [B, T-1, n_agents]

            # Reuse your TD mask & termination handling
            kl_mask = batch["filled"][:, :-1].float()  # [B, T-1, 1] or [B, T-1]
            kl_mask[:, 1:] = kl_mask[:, 1:] * (1 - batch["terminated"][:, :-1].float())
            if kl_mask.dim() == 2:
                kl_mask = kl_mask.unsqueeze(-1)        # [B, T-1, 1]
            kl_mask = kl_mask.expand_as(kl_btna)       # [B, T-1, n_agents]

            denom = kl_mask.sum().clamp_min(1.0)
            role_kl_loss = (kl_btna * kl_mask).sum() / denom
            # ------------------------------------------------------------------------------

        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        # TODO: double DQN action, COMMENT: do not need copy
        mac_out[avail_actions == 0] = -9999999
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            if self.enable_parallel_computing:
                target_mac_out = target_mac_out.get()
            else:
                target_mac_out = calculate_target_q(self.target_mac, batch, self.args.n_agents)

            # Max over target Q-Values/ Double q learning
            # mac_out_detach = mac_out.clone().detach()
            # TODO: COMMENT: do not need copy
            mac_out_detach = mac_out
            # mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]

            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)

            assert getattr(self.args, 'q_lambda', False) == False
            if self.args.mixer.find("qmix") != -1 and self.enable_parallel_computing:
                targets = self.pool.apply_async(
                    calculate_n_step_td_target(
                    self.args, self.target_mixer, target_max_qvals, batch, rewards, terminated, mask, self.args.gamma,
                    self.args.td_lambda, True, self.args.thread_num, False, None)
                )
            else:
                targets = calculate_n_step_td_target(
                    self.args, self.target_mixer, target_max_qvals, batch, rewards, terminated, mask, self.args.gamma,
                    self.args.td_lambda
                )

        # Set mixing net to training mode
        self.mixer.train()
        # Mixer
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        if self.args.mixer.find("qmix") != -1 and self.enable_parallel_computing:
            targets = targets.get()

        td_error = (chosen_action_qvals - targets)
        td_error2 = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask

        mask_elems = mask.sum()
        loss = masked_td_error.sum() / mask_elems

        if role_kl_loss is not None:
            beta = _beta_schedule(self.train_t, self.role_kl_beta_max, self.role_kl_warmup)
            loss = loss + beta * role_kl_loss

        # ------------------------------------------------------------------------------
        # =================
        # Role-specific losses
        # =================
        auxiliary_losses = {}
        total_aux_loss = 0
        # Only compute if we have role information
        if len(role_embeddings) > 0:
            # Stack across time
            role_embeddings = th.stack(role_embeddings, dim=0)  # [time, n_agents, batch, role_dim]
            role_means = th.stack(role_means, dim=0)
            role_log_vars = th.stack(role_log_vars, dim=0)
            
            # 1. KL Regularization Loss (prevents role collapse)
            kl_loss = self.compute_kl_loss(role_means, role_log_vars, mask[:, :, 0])  # mask is [batch, time, 1]
            self.auxiliary_losses['role_kl'] = kl_loss.item()
            total_aux_loss += self.args.role_kl_weight * kl_loss  # Default: 0.001
            
            # 2. Mutual Information Loss (R3DM - makes roles predictive)
            if len(predicted_next_contexts) > 0 and len(predicted_next_contexts) == batch.max_seq_length - 1:
                predicted_next = th.stack(predicted_next_contexts, dim=0)  # [time-1, n_agents, batch, context_dim]
                mi_loss = self.compute_mi_loss(
                    predicted_next, 
                    batch["obs"][:, 1:],  # Next observations
                    self.mac.agents[0].own_encoder,  # Encoder to get actual next context
                    mask[:, :-1, 0]
                )
                self.auxiliary_losses['role_mi'] = mi_loss.item()
                total_aux_loss += self.args.role_mi_weight * mi_loss  # Default: 0.01
            
            # 3. Diversity Loss (encourages different roles for different unit types)
            if self.args.n_agents > 1:
                diversity_loss = self.compute_diversity_loss(role_embeddings, mask[:, :, 0])
                self.auxiliary_losses['role_diversity'] = diversity_loss.item()
                total_aux_loss += self.args.role_diversity_weight * diversity_loss  # Default: 0.005
        
        # Total loss
        loss += total_aux_loss

        # ------------------------------------------------------------------------------

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.train_t += 1
        self.avg_time += (time.time() - start_time - self.avg_time) / self.train_t
        print("Avg cost {} seconds".format(self.avg_time))

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            # For log
            with th.no_grad():
                mask_elems = mask_elems.item()
                td_error_abs = masked_td_error.abs().sum().item() / mask_elems
                q_taken_mean = (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents)
                target_mean = (targets * mask).sum().item() / (mask_elems * self.args.n_agents)
            self.logger.log_stat("loss_td", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("td_error_abs", td_error_abs, t_env)
            self.logger.log_stat("q_taken_mean", q_taken_mean, t_env)
            self.logger.log_stat("target_mean", target_mean, t_env)
            self.logger.log_stat("Training_avg_time", self.avg_time, t_env)
            self.log_stats_t = t_env
        th.cuda.empty_cache()

    # ------------------------------------------------------------------------------
    def compute_mi_loss(self, predicted_next, next_obs, encoder, mask):
        """
        Compute mutual information loss between role and future dynamics
        Args:
            predicted_next: [time-1, n_agents, batch, context_dim]
            next_obs: [batch, time, n_agents, obs_dim]
            encoder: The own_encoder network to process observations
            mask: [batch, time-1]
        """
        batch_size, time_minus_1, n_agents, obs_dim = next_obs[:, 1:].shape
        
        # Reshape and encode actual next observations
        next_obs_flat = next_obs[:, 1:].reshape(-1, obs_dim)  # [batch*time*n_agents, obs_dim]
        with th.no_grad():
            actual_next_context = encoder(next_obs_flat)  # [batch*time*n_agents, context_dim]
        actual_next_context = actual_next_context.reshape(batch_size, time_minus_1, n_agents, -1)
        actual_next_context = actual_next_context.permute(1, 2, 0, 3)  # [time-1, n_agents, batch, context_dim]
        
        # MSE loss between predicted and actual
        mi_loss = F.mse_loss(predicted_next, actual_next_context, reduction='none')
        mi_loss = mi_loss.mean(dim=-1)  # Average over context_dim
        mi_loss = mi_loss.mean(dim=1)  # Average over agents
        mi_loss = mi_loss.permute(1, 0)  # [batch, time-1]
        
        # Apply mask
        mi_loss = (mi_loss * mask).sum() / mask.sum()
        
        return mi_loss


    def compute_diversity_loss(self, role_embeddings, mask):
        """
        Encourage diverse roles across different agents
        Args:
            role_embeddings: [time, n_agents, batch, role_dim]
            mask: [batch, time]
        """
        time_steps, n_agents, batch_size, role_dim = role_embeddings.shape
        
        if n_agents < 2:
            return th.tensor(0.0, device=role_embeddings.device)
        
        diversity_loss = 0
        valid_pairs = 0
        
        # Compute pairwise similarities
        for t in range(time_steps):
            for i in range(n_agents):
                for j in range(i + 1, n_agents):
                    # Compute distance between agent i and j at time t
                    dist = th.norm(role_embeddings[t, i] - role_embeddings[t, j], dim=-1)  # [batch]
                    
                    # Penalize if too similar (small distance)
                    similarity_penalty = th.exp(-2 * dist)  # Exponential penalty for similarity
                    
                    # Apply mask
                    similarity_penalty = (similarity_penalty * mask[:, t]).sum() / (mask[:, t].sum() + 1e-8)
                    diversity_loss += similarity_penalty
                    valid_pairs += 1
        
        if valid_pairs > 0:
            diversity_loss = diversity_loss / valid_pairs
        
        return diversity_loss
        # ------------------------------------------------------------------------------

        
    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        path = os.path.expanduser(path)
        
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.args.mixer != "vdn":
            state_dict = th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage)
            model_dict = self.mixer.state_dict()
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
            self.mixer.load_state_dict(filtered_state_dict, strict=False)
            
    def __del__(self):
        if self.enable_parallel_computing:
            self.pool.close()
