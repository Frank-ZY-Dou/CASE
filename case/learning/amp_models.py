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

import torch.nn as nn
from rl_games.algos_torch.models import ModelA2CContinuousLogStd

class ModelAMPContinuous(ModelA2CContinuousLogStd):
    def __init__(self, network):
        super().__init__(network)
        return

    def build(self, config):
        net = self.network_builder.build('amp', **config)
        for name, _ in net.named_parameters():
            print(name)
        return ModelAMPContinuous.Network(net)

    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network):
            super().__init__(a2c_network)
            return

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            result = super().forward(input_dict)
            if (is_train):
                amp_obs_con = input_dict['amp_obs']
                amp_obs = amp_obs_con[0]
                obs_steps = amp_obs_con[1]
                label_length = amp_obs_con[2]

                # # Frank assert
                # vis_  = amp_obs[0].reshape(10,141)[:,-1]
                # first_ele = vis_[0]
                # for ele in vis_:
                #     assert  first_ele == ele
                #     first_ele = ele
                # ######## end


                disc_agent_logit = self.a2c_network.eval_disc(amp_obs=amp_obs, obs_steps=obs_steps, label_length=label_length)
                result["disc_agent_logit"] = disc_agent_logit

                amp_obs_replay_con = input_dict['amp_obs_replay']
                amp_obs_replay = amp_obs_replay_con[0]
                assert obs_steps == amp_obs_replay_con[1]
                assert label_length == amp_obs_replay_con[2]
                disc_agent_replay_logit = self.a2c_network.eval_disc(amp_obs=amp_obs_replay, obs_steps=obs_steps, label_length=label_length)
                result["disc_agent_replay_logit"] = disc_agent_replay_logit

                amp_demo_obs = input_dict['amp_obs_demo']
                disc_demo_logit = self.a2c_network.eval_disc(amp_obs=amp_demo_obs, obs_steps=obs_steps, label_length=label_length)
                result["disc_demo_logit"] = disc_demo_logit

            return result