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

        self.role_kl_beta_max = float(getattr(args, "role_kl_beta", 1e-3))
        self.role_kl_warmup  = int(getattr(args, "role_kl_warmup_steps", 50000))
        self.role_diversity = getattr(args, "role_diversity", False)
        self.role_div_weight = float(getattr(args, "role_diversity_weight", 0.0))
        self.role_div_across_types_only = getattr(args, "role_div_across_types_only", True)
        self.role_div_kernel_gamma = float(getattr(args, "role_div_kernel_gamma", 2.0))

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
        
        self.mac.init_hidden(batch.batch_size, n_agents = self.args.n_agents)

        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t, test_mode = False)
            mac_out.append(agent_outs)


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

        # ------------------------------------------------------------------------------
        # =================
        # Role-specific losses
        # =================
        # Pull auxiliary role stats from MAC (requires CustomMAC.forward caching & pop_aux())
        aux = self.mac.pop_aux() if hasattr(self.mac, "pop_aux") else {}

        # Rebuild the 2D timestep mask (B, T-1) to align all aux losses with TD loss
        mask_t = batch["filled"][:, :-1].float()
        term2 = batch["terminated"][:, :-1].float()
        if mask_t.dim() == 3:   # sometimes [B,T-1,1]
            mask_t = mask_t.squeeze(-1)
        if term2.dim() == 3:
            term2 = term2.squeeze(-1)
        mask_t[:, 1:] = mask_t[:, 1:] * (1 - term2[:, :-1])

        total_aux = 0.0

        # -------- KL divergence --------
        if "role_mean" in aux and "role_log_var" in aux:
            # aux tensors are [B, T, n_agents, R] — use T-1 like TD
            mu = aux["role_mean"][:, :-1]         # [B,T-1,Na,R]
            lv = aux["role_log_var"][:, :-1]      # [B,T-1,Na,R]

            # KL per (B,t,agent)
            kl_btna = _role_kl(mu, lv)            # [B,T-1,Na,R]
            kl_btna = kl_btna.sum(dim=-1)         # sum over R -> [B,T-1,Na]

            # Use the same expanded mask already used for TD
            # (your 'mask' variable below is already expanded to [B,T-1,Na])
            mask_btna = self._expand_timestep_mask_like(kl_btna, mask_t)

            denom = mask_btna.sum().clamp_min(1.0)
            kl_loss = (kl_btna * mask_btna).sum() / denom

            beta = _beta_schedule(self.train_t, self.role_kl_beta_max, self.role_kl_warmup)
            total_aux = total_aux + beta * kl_loss
            self.logger.log_stat("role_kl", kl_loss.item(), t_env)
            self.logger.log_stat("role_kl_beta", beta, t_env)

        # -------- 2) Diversity across agents (RBF on L2 distance), across different unit types only --------
        if self.role_diversity:
            if self.role_div_weight > 0.0 and "role_mean" in aux and aux["role_mean"].size(2) > 1:
                Z = aux["role_mean"][:, :-1]                  # [B,T-1,Na,R] — role means for stability
                B, Tm1, Na, R = Z.shape
                Zbt = Z.reshape(B*Tm1, Na, R)                    # [BT,Na,R]

                # Pairwise L2 distances and exponential kernel
                D = th.cdist(Zbt, Zbt, p=2)                   # [BT,Na,Na]
                K = th.exp(-self.role_div_kernel_gamma * D)   # [BT,Na,Na]

                # Off-diagonal mask
                offdiag = 1 - th.eye(Na, device=K.device, dtype=K.dtype)  # [Na,Na]
                offdiag = offdiag.unsqueeze(0).expand(B*Tm1, -1, -1)      # [BT,Na,Na]

                # Optionally exclude same-type pairs (keeps “last 3” features = unit_type one-hot)
                if self.role_div_across_types_only:
                    types = batch["obs"][:, :-1, :, -3:]      # [B,T-1,Na,3]
                    t_idx = types.argmax(dim=-1)              # [B,T-1,Na]
                    ti = t_idx.reshape(B*Tm1, Na, 1)
                    tj = t_idx.reshape(B*Tm1, 1, Na)
                    diff_types = (ti != tj).to(K.dtype)       # [BT,Na,Na]
                    pair_mask = offdiag * diff_types
                else:
                    pair_mask = offdiag

                # Average kernel value over valid (i,j) pairs per (B,t)
                pair_count = pair_mask.sum(dim=(1,2)).clamp_min(1.0)      # [BT]
                K_mean_bt = (K * pair_mask).sum(dim=(1,2)) / pair_count   # [BT]
                K_mean = K_mean_bt.view(B, Tm1)                            # [B,T-1]

                # Weight by timestep mask
                div_loss = (K_mean * mask_t).sum() / mask_t.sum().clamp_min(1.0)

                total_aux = total_aux + self.role_div_weight * div_loss
                self.logger.log_stat("role_div", div_loss.item(), t_env)

        # Add aux to base TD loss
        loss = loss + total_aux
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

    # -------------------------------------------------------------------------------
    def _role_kl(self, mu, logvar):  # returns [..., R]
        return 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)

    def _beta_schedule(self, step: int, beta_max=1e-3, warmup=50_000):
        s = min(1.0, float(step) / float(warmup))
        return beta_max * s
    
    def _expand_timestep_mask_like(self, x: th.Tensor, mask_t: th.Tensor) -> th.Tensor:
        m = mask_t
        while m.dim() < x.dim():
            m = m.unsqueeze(-1)
        return m.expand_as(x)

    # -------------------------------------------------------------------------------
        
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
