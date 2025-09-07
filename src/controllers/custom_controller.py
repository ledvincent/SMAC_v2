#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
import torch as th

from .basic_controller import BasicMAC


class DataParallelAgent(th.nn.DataParallel):
    def init_hidden(self):
        # make hidden states on same device as model
        return self.module.init_hidden()


# This multi-agent controller shares parameters between agents
class CustomMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(CustomMAC, self).__init__(scheme, groups, args)
        self.n_enemies = args.n_enemies
        self.n_allies = self.n_agents - 1

    # Add new func
    def _get_obs_component_dim(self):
        move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim = self.args.obs_component  # [4, (6, 5), (4, 5), 1]
        enemy_feats_dim_flatten = np.prod(enemy_feats_dim)
        ally_feats_dim_flatten = np.prod(ally_feats_dim)
        return (move_feats_dim, enemy_feats_dim_flatten, ally_feats_dim_flatten, own_feats_dim), (
            enemy_feats_dim, ally_feats_dim)

    def _build_inputs(self, batch, t, test_mode):
        bs = batch.batch_size
        obs_component_dim, _ = self._get_obs_component_dim()
        raw_obs_t = batch["obs"][:, t]  # [batch, agent_num, obs_dim]
        move_feats_t, enemy_feats_t, ally_feats_t, own_feats_t = th.split(raw_obs_t, obs_component_dim, dim=-1)
        enemy_feats_t = enemy_feats_t.reshape(bs * self.n_agents * self.n_enemies,
                                              -1)  # [bs * n_agents * n_enemies, fea_dim]
        ally_feats_t = ally_feats_t.reshape(bs * self.n_agents * self.n_allies,
                                            -1)  # [bs * n_agents * n_allies, a_fea_dim]
        # merge move features and own features to simplify computation.
        context_feats = [move_feats_t, own_feats_t]  # [batch, agent_num, own_dim]
        own_context = th.cat(context_feats, dim=2).reshape(bs * self.n_agents, -1)  # [bs * n_agents, own_dim]

        embedding_indices = []
        if self.args.obs_agent_id:
            # agent-id indices, [bs, n_agents]
            embedding_indices.append(th.arange(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1))
        if self.args.obs_last_action:
            # action-id indices, [bs, n_agents]
            if t == 0:
                embedding_indices.append(None)
            else:
                embedding_indices.append(batch["actions"][:, t - 1].squeeze(-1))
        return bs, own_context, enemy_feats_t, ally_feats_t, embedding_indices

    def _get_input_shape(self, scheme):
        move_feats_dim, enemy_feats_dim, ally_feats_dim, own_feats_dim = self.args.obs_component
        own_context_dim = move_feats_dim + own_feats_dim
        return own_context_dim, enemy_feats_dim, ally_feats_dim

    def _build_inputs(self, batch, t, test_mode):
        bs = batch.batch_size
        obs_component_dim, (enemy_shape, ally_shape) = self._get_obs_component_dim()
        move_dim, _, _, own_dim = obs_component_dim
        Ne, d_en  = enemy_shape
        Na, d_all = ally_shape

        raw_obs_t = batch["obs"][:, t]                    # [bs, n_agents, obs_dim]

        # Split flat obs into components (HPN already arranged obs in this order)
        move_feats_t, enemy_feats_t, ally_feats_t, own_feats_t = th.split(raw_obs_t, obs_component_dim, dim=-1)
        # move_feats_t: [bs, n_agents, move_dim]
        # enemy_feats_t: [bs, n_agents, Ne*d_en]
        # ally_feats_t : [bs, n_agents, Na*d_all]
        # own_feats_t  : [bs, n_agents, own_dim]

        # -------- reshape per-entity groups to 3D --------
        # [bs, n_agents, Ne, d_en] -> [bs*n_agents, Ne, d_en]
        if Ne > 0:
            enemy_feats_t = enemy_feats_t.view(bs, self.n_agents, Ne, d_en).reshape(bs * self.n_agents, Ne, d_en)
        else:
            enemy_feats_t = th.zeros(bs * self.n_agents, 0, d_en, device=batch.device)

        # [bs, n_agents, Na, d_all] -> [bs*n_agents, Na, d_all]
        if Na > 0:
            ally_feats_t = ally_feats_t.view(bs, self.n_agents, Na, d_all).reshape(bs * self.n_agents, Na, d_all)
        else:
            ally_feats_t = th.zeros(bs * self.n_agents, 0, d_all, device=batch.device)

        # Merge move + own into "own_context": [bs, n_agents, move_dim+own_dim] -> [bs*n_agents, 1, d_own]
        own_context = th.cat([move_feats_t, own_feats_t], dim=-1)  # [bs, n_agents, move_dim+own_dim]
        own_context = own_context.reshape(bs * self.n_agents, 1, move_dim + own_dim)

        # -------- embedding indices (unchanged) --------
        embedding_indices = []
        if self.args.obs_agent_id:
            # agent-id indices, [bs, n_agents]
            embedding_indices.append(th.arange(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1))
        if self.args.obs_last_action:
            # last actions at t-1, [bs, n_agents] or None at t==0
            if t == 0:
                embedding_indices.append(None)
            else:
                embedding_indices.append(batch["actions"][:, t - 1].squeeze(-1))

        return bs, own_context, enemy_feats_t, ally_feats_t, embedding_indices
