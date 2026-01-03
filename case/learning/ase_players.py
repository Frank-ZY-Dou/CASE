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

import torch
import yaml
import os
import threading
import sys
import select

from isaacgym.torch_utils import *
from rl_games.algos_torch import players

from learning import amp_players
from learning import ase_network_builder

class ASEPlayer(amp_players.AMPPlayerContinuous):
    def __init__(self, config):
        self._style_latent_size = config['style_latent_size']
        self._latent_steps_min = config.get('latent_steps_min', np.inf)
        self._latent_steps_max = config.get('latent_steps_max', np.inf)
        self.label_length =  config['label_length']
        self._enc_reward_scale = config['enc_reward_scale']
        self.obs_steps = config['obs_steps']

        # Skill label update settings
        self._if_update_skill_label = config.get('if_update_skill_label', False)
        self._skill_label_step = config.get('skill_label_step', 100)
        self._skill_label_step_count = 0
        self._clip2group_mapping_path = config.get('clip2group_mapping', None)
        self._skill_names = []
        self._num_skills = config.get('nlabels', 87)

        # User control settings
        self._user_control = config.get('user_control', False)
        self._user_skill_input = None
        self._input_thread = None
        self._stop_input_thread = False

        # Load skill names from mapping file
        if self._clip2group_mapping_path is not None and os.path.exists(self._clip2group_mapping_path):
            self._load_skill_names()

        super().__init__(config)

        if (hasattr(self, 'env')):
            batch_size = self.env.task.num_envs
        else:
            batch_size = self.env_info['num_envs']
        self._ase_latents = torch.zeros((batch_size, self._style_latent_size), dtype=torch.float32,
                                         device=self.device)

        return

    def _load_skill_names(self):
        """Load skill names from the clip2group mapping file."""
        try:
            with open(self._clip2group_mapping_path, 'r') as f:
                mapping = yaml.safe_load(f)
            if 'groups' in mapping:
                # Sort by skill name to maintain order (skill_00, skill_01, ...)
                sorted_skills = sorted(mapping['groups'].keys())
                self._skill_names = sorted_skills
                print(f"[Skill Label Update] Loaded {len(self._skill_names)} skill names")
        except Exception as e:
            print(f"[Skill Label Update] Failed to load skill names: {e}")
            self._skill_names = [f"skill_{i:02d}" for i in range(self._num_skills)]

    def run(self):
        self._reset_latent_step_count()

        # Start keyboard input thread if user control is enabled
        if self._user_control:
            self._start_input_thread()

        try:
            super().run()
        finally:
            # Stop input thread when done
            if self._user_control:
                self._stop_input_thread = True
                if self._input_thread is not None:
                    self._input_thread.join(timeout=1.0)
        return

    def get_action(self, obs_dict, is_determenistic=False):
        self._update_latents()
        self._update_skill_labels()

        obs = obs_dict['obs']
        if len(obs.size()) == len(self.obs_shape):
            obs = obs.unsqueeze(0)

        obs_processed = obs[:, :-self.label_length]
        obs_processed = self._preproc_obs(obs_processed)
        full_obs_with_label = torch.cat((obs_processed,  obs[:, -self.label_length:]), dim=1)
        ase_latents = self._ase_latents

        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : full_obs_with_label,
            'rnn_states' : self.states,
            'ase_latents': ase_latents
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if is_determenistic:
            current_action = mu
        else:
            current_action = action
        current_action = torch.squeeze(current_action.detach())
        return  players.rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))

    def env_reset(self, env_ids=None):
        obs = super().env_reset(env_ids)
        self._reset_latents(env_ids)
        return obs
    
    def _build_net_config(self):
        config = super()._build_net_config()
        config['ase_latent_shape'] = (self._style_latent_size,)
        return config
    
    def _reset_latents(self, done_env_ids=None):
        if (done_env_ids is None):
            num_envs = self.env.task.num_envs
            done_env_ids = to_torch(np.arange(num_envs), dtype=torch.long, device=self.device)

        rand_vals = self.model.a2c_network.sample_latents(len(done_env_ids))
        self._ase_latents[done_env_ids] = rand_vals
        self._change_char_color(done_env_ids)

        return

    def _update_latents(self):
        if (self._latent_step_count <= 0):
            self._reset_latents()
            self._reset_latent_step_count()

            if (self.env.task.viewer):
                # print("Sampling new amp latents------------------------------")
                num_envs = self.env.task.num_envs
                env_ids = to_torch(np.arange(num_envs), dtype=torch.long, device=self.device)
                self._change_char_color(env_ids)
        else:
            self._latent_step_count -= 1
        return
    
    def _reset_latent_step_count(self):
        self._latent_step_count = np.random.randint(self._latent_steps_min, self._latent_steps_max)
        return

    def _calc_amp_rewards(self, amp_obs, ase_latents):
        disc_r = self._calc_disc_rewards(amp_obs)
        enc_r = self._calc_enc_rewards(amp_obs, ase_latents)
        output = {
            'disc_rewards': disc_r,
            'enc_rewards': enc_r
        }
        return output
    
    def _calc_enc_rewards(self, amp_obs, ase_latents):
        with torch.no_grad():
            enc_pred = self._eval_enc(amp_obs)
            err = self._calc_enc_error(enc_pred, ase_latents)
            enc_r = torch.clamp_min(-err, 0.0)
            enc_r *= self._enc_reward_scale

        return enc_r
    
    def _calc_enc_error(self, enc_pred, ase_latent):
        err = enc_pred * ase_latent
        err = -torch.sum(err, dim=-1, keepdim=True)
        return err
    
    def _eval_enc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_enc(proc_amp_obs, self.obs_steps, self.label_length)

    def _amp_debug(self, info):
        with torch.no_grad():
            amp_obs = info['amp_obs']
            amp_obs = amp_obs
            ase_latents = self._ase_latents
            disc_pred = self._eval_disc(amp_obs, self.obs_steps, self.label_length)
            amp_rewards = self._calc_amp_rewards(amp_obs, ase_latents)
            disc_reward = amp_rewards['disc_rewards']
            # enc_reward = amp_rewards['enc_rewards']
            # disc_pred = disc_pred.detach().cpu().numpy()
            # disc_reward = disc_reward.cpu().numpy()
            # enc_reward = enc_reward.cpu().numpy()[:,0]
            # print(">>> disc_pred: ", disc_pred)
            # print("     >>> disc_reward:", disc_reward)
            # print("     >>> enc_reward:", enc_reward)
            # print()
        return

    def _change_char_color(self, env_ids):
        base_col = np.array([0.4, 0.4, 0.4])
        range_col = np.array([0.0706, 0.149, 0.2863])
        range_sum = np.linalg.norm(range_col)

        rand_col = np.random.uniform(0.0, 1.0, size=3)
        rand_col = range_sum * rand_col / np.linalg.norm(rand_col)
        rand_col += base_col
        self.env.task.set_char_color(rand_col, env_ids)
        return

    def _update_skill_labels(self):
        """Update skill labels based on user input or automatic stepping."""
        # Handle user control mode
        if self._user_control:
            if self._user_skill_input is not None:
                new_skill_idx = self._user_skill_input
                self._user_skill_input = None  # Clear the input

                # Get number of environments
                num_envs = self.env.task.num_envs

                # Update the label buffer
                new_labels = torch.full((num_envs,), new_skill_idx, dtype=torch.long, device=self.device)
                self.env.task.label_buf[:] = new_labels

                # Get skill name and print
                if new_skill_idx < len(self._skill_names):
                    skill_name = self._skill_names[new_skill_idx]
                else:
                    skill_name = f"skill_{new_skill_idx:02d}"

                print(f"\n[User Control] Switching to skill {new_skill_idx} ({skill_name})")
            return

        # Handle automatic update mode
        if not self._if_update_skill_label:
            return

        self._skill_label_step_count += 1

        if self._skill_label_step_count >= self._skill_label_step:
            self._skill_label_step_count = 0

            # Get number of environments
            num_envs = self.env.task.num_envs

            # Sample new random skill labels
            new_skill_idx = np.random.randint(0, self._num_skills)
            new_labels = torch.full((num_envs,), new_skill_idx, dtype=torch.long, device=self.device)

            # Update the label buffer in the environment
            self.env.task.label_buf[:] = new_labels

            # Get skill name and print
            if new_skill_idx < len(self._skill_names):
                skill_name = self._skill_names[new_skill_idx]
            else:
                skill_name = f"skill_{new_skill_idx:02d}"

            print(f"[Skill Update] Step {self._skill_label_step}: Switching to skill {new_skill_idx} ({skill_name})")

        return

    def _start_input_thread(self):
        """Start the keyboard input listener thread."""
        print("\n" + "="*60)
        print("[User Control] Keyboard input enabled!")
        print(f"[User Control] Enter skill number (0-{self._num_skills-1}) to switch:")
        print("[User Control] Available skills:")
        for i, name in enumerate(self._skill_names[:min(10, len(self._skill_names))]):
            print(f"  {i}: {name}")
        if len(self._skill_names) > 10:
            print(f"  ... and {len(self._skill_names) - 10} more skills")
        print("="*60 + "\n")

        self._input_thread = threading.Thread(target=self._input_listener, daemon=True)
        self._input_thread.start()

    def _input_listener(self):
        """Listen for keyboard input in a separate thread."""
        while not self._stop_input_thread:
            try:
                # Use select for non-blocking input on Linux
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    user_input = sys.stdin.readline().strip()
                    if user_input:
                        try:
                            skill_idx = int(user_input)
                            if 0 <= skill_idx < self._num_skills:
                                self._user_skill_input = skill_idx
                            else:
                                print(f"[User Control] Invalid skill number. Please enter 0-{self._num_skills-1}")
                        except ValueError:
                            print(f"[User Control] Please enter a valid number (0-{self._num_skills-1})")
            except Exception as e:
                pass