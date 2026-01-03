# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
C-ASE HRL Agent: High-Level Controller for C-ASE

This implements the hybrid discrete-continuous action space as described in the paper:
- Discrete: skill label c (via Gumbel-Softmax)
- Continuous: style latent z (from spherical Gaussian)

The HLC outputs (skill_label, latent_z) which are passed to the pre-trained C-ASE LLC.
"""

import copy
from datetime import datetime
from gym import spaces
import numpy as np
import os
import time
import yaml

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import central_value
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common import a2c_common
from rl_games.common import datasets
from rl_games.common import schedulers
from rl_games.common import vecenv

import torch
import torch.nn.functional as F
from torch import optim

import learning.common_agent as common_agent
import learning.ase_agent as ase_agent
import learning.ase_models as ase_models
import learning.ase_network_builder as ase_network_builder

from tensorboardX import SummaryWriter


class CASEHRLAgent(common_agent.CommonAgent):
    """
    C-ASE High-Level Controller Agent

    Outputs hybrid action space:
    - skill_logits: probability distribution over skill labels (nlabels)
    - latent_z: continuous latent code (style_latent_size)

    Total action dimension = nlabels + style_latent_size
    """

    def __init__(self, base_name, config):
        # Load LLC config to get dimensions
        with open(os.path.join(os.getcwd(), config['llc_config']), 'r') as f:
            llc_config = yaml.load(f, Loader=yaml.SafeLoader)
            llc_config_params = llc_config['params']

        # Override LLC config with llc_* parameters from command line
        llc_nlabels = config.get('llc_nlabels', -1)
        llc_skill_latent_size = config.get('llc_skill_latent_size', -1)
        llc_style_latent_size = config.get('llc_style_latent_size', -1)
        llc_numAMPObsSteps = config.get('llc_numAMPObsSteps', -1)
        llc_if_dropout = config.get('llc_if_dropout', False)
        llc_dropout_rate = config.get('llc_dropout_rate', 0.1)
        llc_enable_srf = config.get('llc_enable_srf', False)
        llc_srf_scale = config.get('llc_srf_scale', 50.0)
        llc_labellength = config.get('llc_labellength', 1)

        # Update LLC network params
        if llc_nlabels > 0:
            llc_config_params['network']['nlabels'] = llc_nlabels
            llc_config_params['config']['nlabels'] = llc_nlabels
        if llc_skill_latent_size > 0:
            llc_config_params['network']['skill_latent_size'] = llc_skill_latent_size
        if llc_style_latent_size > 0:
            llc_config_params['config']['style_latent_size'] = llc_style_latent_size
        if llc_numAMPObsSteps > 0:
            llc_config_params['network']['obs_steps'] = llc_numAMPObsSteps
            llc_config_params['config']['obs_steps'] = llc_numAMPObsSteps
        if llc_labellength > 0:
            llc_config_params['network']['label_length'] = llc_labellength
            llc_config_params['config']['label_length'] = llc_labellength

        # Update LLC dropout params
        llc_config_params['network']['if_dropout'] = llc_if_dropout
        llc_config_params['network']['dropout_rate'] = llc_dropout_rate

        # Update LLC SRF params
        llc_config_params['config']['enable_srf'] = llc_enable_srf
        llc_config_params['config']['srf_scale'] = llc_srf_scale

        # Save LLC feature flags for logging
        self._llc_if_dropout = llc_if_dropout
        self._llc_dropout_rate = llc_dropout_rate
        self._llc_enable_srf = llc_enable_srf
        self._llc_srf_scale = llc_srf_scale

        self._style_latent_size = llc_config_params['config']['style_latent_size']

        # Get nlabels from config
        self._nlabels = config.get('nlabels', 87)
        self._gumbel_temperature = config.get('gumbel_temperature', 1.0)

        # Set controller name for logging
        self._controller_name = 'HLC'

        super().__init__(base_name, config)

        self._task_size = self.vec_env.env.task.get_task_obs_size()

        self._llc_steps = config['llc_steps']
        llc_checkpoint = config['llc_checkpoint']
        assert(llc_checkpoint != ""), "LLC checkpoint must be provided"
        self._build_llc(llc_config_params, llc_checkpoint)

        return

    def env_step(self, actions):
        """
        Execute environment step with hybrid actions.

        actions: [batch, nlabels + style_latent_size]
            - actions[:, :nlabels]: skill logits
            - actions[:, nlabels:]: latent z
        """
        actions = self.preprocess_actions(actions)
        obs = self.obs['obs']

        # Split hybrid action into skill label and latent z
        skill_logits = actions[:, :self._nlabels]
        latent_z = actions[:, self._nlabels:]

        # Convert skill logits to one-hot via Gumbel-Softmax (hard)
        skill_label = F.gumbel_softmax(skill_logits, tau=self._gumbel_temperature, hard=True)
        # Get skill index for LLC
        skill_idx = torch.argmax(skill_label, dim=-1).float()

        rewards = 0.0
        disc_rewards = 0.0
        done_count = 0.0
        terminate_count = 0.0

        for t in range(self._llc_steps):
            llc_actions = self._compute_llc_action(obs, skill_idx, latent_z)
            obs, curr_rewards, curr_dones, infos = self.vec_env.step(llc_actions)

            rewards += curr_rewards
            done_count += curr_dones
            terminate_count += infos['terminate']

            amp_obs = infos['amp_obs']
            curr_disc_reward = self._calc_disc_reward(amp_obs)
            disc_rewards += curr_disc_reward

        rewards /= self._llc_steps
        disc_rewards /= self._llc_steps

        dones = torch.zeros_like(done_count)
        dones[done_count > 0] = 1.0
        terminate = torch.zeros_like(terminate_count)
        terminate[terminate_count > 0] = 1.0

        infos['terminate'] = terminate
        infos['disc_rewards'] = disc_rewards

        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return self.obs_to_tensors(obs), torch.from_numpy(rewards).to(self.ppo_device).float(), torch.from_numpy(dones).to(self.ppo_device), infos

    def cast_obs(self, obs):
        obs = super().cast_obs(obs)
        self._llc_agent.is_tensor_obses = self.is_tensor_obses
        return obs

    def preprocess_actions(self, actions):
        # Clamp latent z part, keep skill logits unclamped for Gumbel-Softmax
        skill_logits = actions[:, :self._nlabels]
        latent_z = torch.clamp(actions[:, self._nlabels:], -1.0, 1.0)
        clamped_actions = torch.cat([skill_logits, latent_z], dim=-1)
        if not self.is_tensor_obses:
            clamped_actions = clamped_actions.cpu().numpy()
        return clamped_actions

    def play_steps(self):
        self.set_eval()

        epinfos = []
        done_indices = []
        update_list = self.update_list

        for n in range(self.horizon_length):
            self.obs = self.env_reset(done_indices)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])

            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data('rewards', n, shaped_rewards)
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)

            self.experience_buffer.update_data('disc_rewards', n, infos['disc_rewards'])

            terminated = infos['terminate'].float()
            terminated = terminated.unsqueeze(-1)
            next_vals = self._eval_critic(self.obs)
            next_vals *= (1.0 - terminated)
            self.experience_buffer.update_data('next_values', n, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            done_indices = done_indices[:, 0]

        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_next_values = self.experience_buffer.tensor_dict['next_values']

        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_disc_rewards = self.experience_buffer.tensor_dict['disc_rewards']
        mb_rewards = self._combine_rewards(mb_rewards, mb_disc_rewards)

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size

        return batch_dict

    def _load_config_params(self, config):
        super()._load_config_params(config)

        self._task_reward_w = config['task_reward_w']
        self._disc_reward_w = config['disc_reward_w']
        self._nlabels = config.get('nlabels', 87)
        return

    def _get_mean_rewards(self):
        rewards = super()._get_mean_rewards()
        rewards *= self._llc_steps
        return rewards

    def _setup_action_space(self):
        super()._setup_action_space()
        # Hybrid action space: skill_logits (nlabels) + latent_z (style_latent_size)
        self.actions_num = self._nlabels + self._style_latent_size
        return

    def init_tensors(self):
        super().init_tensors()

        del self.experience_buffer.tensor_dict['actions']
        del self.experience_buffer.tensor_dict['mus']
        del self.experience_buffer.tensor_dict['sigmas']

        batch_shape = self.experience_buffer.obs_base_shape
        action_dim = self._nlabels + self._style_latent_size

        self.experience_buffer.tensor_dict['actions'] = torch.zeros(batch_shape + (action_dim,),
                                                                dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['mus'] = torch.zeros(batch_shape + (action_dim,),
                                                                dtype=torch.float32, device=self.ppo_device)
        self.experience_buffer.tensor_dict['sigmas'] = torch.zeros(batch_shape + (action_dim,),
                                                                dtype=torch.float32, device=self.ppo_device)

        self.experience_buffer.tensor_dict['disc_rewards'] = torch.zeros_like(self.experience_buffer.tensor_dict['rewards'])
        self.tensor_list += ['disc_rewards']

        return

    def _build_llc(self, config_params, checkpoint_file):
        network_params = config_params['network']
        network_builder = ase_network_builder.ASEBuilder()
        network_builder.load(network_params)

        network = ase_models.ModelASEContinuous(network_builder)
        llc_agent_config = self._build_llc_agent_config(config_params, network)

        self._llc_agent = ase_agent.ASEAgent('llc', llc_agent_config)
        self._llc_agent.restore(checkpoint_file)
        print("Loaded C-ASE LLC checkpoint from {:s}".format(checkpoint_file))
        self._llc_agent.set_eval()
        return

    def _build_llc_agent_config(self, config_params, network):
        llc_env_info = copy.deepcopy(self.env_info)
        obs_space = llc_env_info['observation_space']
        obs_size = obs_space.shape[0]
        obs_size -= self._task_size
        llc_env_info['observation_space'] = spaces.Box(obs_space.low[:obs_size], obs_space.high[:obs_size])

        # Note: LLC action_space is already correct from environment
        # When llc_enable_srf is True, run.py sets env SRF config, so action_space = 76

        config = config_params['config']
        # LLC is only used for inference, use separate directory
        config['train_dir'] = 'runs/llc_inference'
        config['network'] = network
        config['num_actors'] = self.num_actors
        config['features'] = {'observer': self.algo_observer}
        config['env_info'] = llc_env_info

        return config

    def _compute_llc_action(self, obs, skill_idx, latent_z):
        """
        Compute LLC action with skill label and latent z.

        Args:
            obs: observations [batch, obs_dim]
            skill_idx: skill label index [batch]
            latent_z: style latent [batch, style_latent_size]
        """
        llc_obs = self._extract_llc_obs(obs)
        # Strip label before preprocessing (like ase_agent.py does)
        llc_obs_no_label = llc_obs[:, :-self._llc_agent.label_length]
        processed_obs = self._llc_agent._preproc_obs(llc_obs_no_label)

        # Normalize latent z to unit sphere
        z = F.normalize(latent_z, dim=-1)

        # Call LLC's eval_actor with skill label
        mu, _ = self._llc_agent.model.a2c_network.eval_actor(
            obs=processed_obs,
            label_con=skill_idx,
            ase_latents=z
        )
        llc_action = mu
        llc_action = self._llc_agent.preprocess_actions(llc_action)

        return llc_action

    def _extract_llc_obs(self, obs):
        obs_size = obs.shape[-1]
        llc_obs = obs[..., :obs_size - self._task_size]
        return llc_obs

    def _calc_disc_reward(self, amp_obs):
        """Calculate discriminator reward for HRL.
        Uses simpler calculation without 3D reshape since HRL amp_obs is 2D.
        """
        with torch.no_grad():
            disc_logits = self._llc_agent._eval_disc(amp_obs)
            prob = 1 / (1 + torch.exp(-disc_logits))
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.ppo_device)))
            disc_r *= self._llc_agent._disc_reward_scale
        # Ensure shape is [num_envs, 1] for experience buffer
        if disc_r.dim() == 1:
            disc_r = disc_r.unsqueeze(-1)
        return disc_r

    def _combine_rewards(self, task_rewards, disc_rewards):
        combined_rewards = self._task_reward_w * task_rewards + self._disc_reward_w * disc_rewards
        return combined_rewards

    def _record_train_batch_info(self, batch_dict, train_info):
        super()._record_train_batch_info(batch_dict, train_info)
        train_info['disc_rewards'] = batch_dict['disc_rewards']
        return

    def _log_train_info(self, train_info, frame):
        super()._log_train_info(train_info, frame)

        disc_reward_std, disc_reward_mean = torch.std_mean(train_info['disc_rewards'])
        self.writer.add_scalar('info/disc_reward_mean', disc_reward_mean.item(), frame)
        self.writer.add_scalar('info/disc_reward_std', disc_reward_std.item(), frame)
        return

    def _log_case_features(self):
        """Log LLC feature configuration for HRL training."""
        print("\n" + "=" * 60)
        print("The LLC you are controlling has:")
        print("=" * 60)

        # EFM (Element-wise Feature Masking / Dropout) - LLC feature
        llc_if_dropout = self._llc_if_dropout
        llc_dropout_rate = self._llc_dropout_rate
        efm_status = "ENABLED" if llc_if_dropout else "DISABLED"
        print(f"[{'✓' if llc_if_dropout else '✗'}] Element-wise Feature Masking (EFM): {efm_status}")
        if llc_if_dropout:
            print(f"    - Dropout rate: {llc_dropout_rate}")

        # SRF (Skeletal Residual Forces) - LLC feature
        llc_enable_srf = self._llc_enable_srf
        llc_srf_scale = self._llc_srf_scale
        srf_status = "ENABLED" if llc_enable_srf else "DISABLED"
        print(f"[{'✓' if llc_enable_srf else '✗'}] Skeletal Residual Forces (SRF): {srf_status}")
        if llc_enable_srf:
            print(f"    - SRF scale: {llc_srf_scale}")

        print("=" * 60 + "\n")

        # Log to TensorBoard
        feature_text = f"LLC - EFM: {efm_status}, SRF: {srf_status}"
        self.writer.add_text('config/llc_features', feature_text, 0)
