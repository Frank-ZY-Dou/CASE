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

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from rl_games.algos_torch import network_builder
from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np

DISC_LOGIT_INIT_SCALE = 1.0

class LinearConditionalMaskLogits(nn.Module):
    ''' runs activated logits through fc and masks out the appropriate discriminator score according to class number'''

    def __init__(self, nc, nlabels):
        super().__init__()
        self.nlabels = nlabels
        self.fc = nn.Linear(nc, nlabels)


    def forward(self, inp, y=None, take_best=False, get_features=False):
        out = self.fc(inp)
        if get_features: return out
        y = y.long()
        original_shape = y.shape
        #torch.Size([16, 2048, 210])
        # disc_mlp_out.shape
        # torch.Size([16, 2048, 512])
        # disc_logits.shape
        # torch.Size([16, 2048, 1])
        # stop inside

        # torch.Size([16, 2048, 2])
        # torch.Size([16, 2048, 1])

        if not take_best:
            out = out.view(-1,self.nlabels)
            # [16, 2048, 2] -> [32768, 2]
            index = Variable(torch.LongTensor(range(out.size(0))))
            # [32768]
            if y.is_cuda:
                index = index.cuda()
            y = y.view(-1)
            # [32768,1]
            return out[index,y].reshape(original_shape)
        else:
            # high activation means real, so take the highest activations
            best_logits, _ = out.max(dim=1)
            return best_logits

class AMPBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            if self.is_continuous:
                if (not self.space_config['learn_sigma']):
                    actions_num = kwargs.get('actions_num')
                    sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
                    sigma_init(self.sigma)
            # print("-->")
            amp_input_shape = kwargs.get('amp_input_shape')
            self._build_disc(amp_input_shape, kwargs, label_length=params['label_length'], nlabels=params['nlabels'])

            return

        def load(self, params):
            super().load(params)

            self._disc_units = params['disc']['units']
            self._disc_activation = params['disc']['activation']
            self._disc_initializer = params['disc']['initializer']
            return

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)

            actor_outputs = self.eval_actor(obs)
            value = self.eval_critic(obs)

            output = actor_outputs + (value, states)

            return output

        def eval_actor(self, obs):
            a_out = self.actor_cnn(obs)
            a_out = a_out.contiguous().view(a_out.size(0), -1)
            a_out = self.actor_mlp(a_out)
                     
            if self.is_discrete:
                logits = self.logits(a_out)
                return logits

            if self.is_multi_discrete:
                logits = [logit(a_out) for logit in self.logits]
                return logits

            if self.is_continuous:
                mu = self.mu_act(self.mu(a_out))
                if self.space_config['fixed_sigma']:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))

                return mu, sigma
            return

        def eval_critic(self, obs):
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)              
            value = self.value_act(self.value(c_out))
            return value

        def eval_disc(self, amp_obs, obs_steps, label_length):
            amp_obs_shape = amp_obs.shape
            if len(amp_obs_shape)==3:
                # this is for reward computation.
                amp_obs = amp_obs.reshape(amp_obs_shape[0],amp_obs_shape[1], obs_steps, int(amp_obs_shape[2]/obs_steps))
                amp_obs_ = amp_obs[:,:,:,:-label_length]
                label_con_ = amp_obs[:,:,0,-label_length] # we have 10 , but we only need one.
                amp_obs_ = amp_obs_.reshape(amp_obs_shape[0],amp_obs_shape[1], obs_steps * int(amp_obs_shape[2]/obs_steps - label_length))
                disc_mlp_out = self._disc_mlp(amp_obs_)
                disc_logits = self._disc_logits(disc_mlp_out, label_con_)
                assert self._disc_mlp[0].training == False
                assert self.training == False



                
            if len(amp_obs_shape) == 2:
                amp_obs = amp_obs.reshape(amp_obs_shape[0], obs_steps,
                                          int(amp_obs_shape[1] / obs_steps))
                amp_obs_ = amp_obs[:, :, :-label_length]
                label_con_ = amp_obs[:, -1, -label_length]  # we have 10 , but we only need one.

                amp_obs_ = amp_obs_.reshape(amp_obs_shape[0],
                                            obs_steps * int(amp_obs_shape[1] / obs_steps - label_length))

                disc_mlp_out = self._disc_mlp(amp_obs_)
                disc_logits = self._disc_logits(disc_mlp_out, label_con_)
                # assert self._disc_mlp[0].training == True
                # assert self.training == True
            return disc_logits

        def get_disc_logit_weights(self):
            return torch.flatten(self._disc_logits.fc.weight)

        def get_disc_weights(self):
            weights = []
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    weights.append(torch.flatten(m.weight))

            weights.append(torch.flatten(self._disc_logits.fc.weight))
            return weights

        def _build_disc(self, input_shape,  numAMPObsSteps, label_length, nlabels ):
            self._disc_mlp = nn.Sequential()

            true_input_shape = (int(input_shape[0] / numAMPObsSteps) - label_length) * numAMPObsSteps
            mlp_args = {
                'input_size' : true_input_shape,
                'units' : self._disc_units, 
                'activation' : self._disc_activation, 
                'dense_func' : torch.nn.Linear,
                "if_dropout": self.if_dropout,
                "dropout_rate": self.dropout_rate
            }

            self._disc_mlp = self._build_mlp(**mlp_args)

            mlp_out_size = self._disc_units[-1]


            self._disc_logits = LinearConditionalMaskLogits(
                mlp_out_size, nlabels)

            mlp_init = self.init_factory.create(**self._disc_initializer)
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias) 

            torch.nn.init.uniform_(self._disc_logits.fc.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
            torch.nn.init.zeros_(self._disc_logits.fc.bias)
            return

    def build(self, name, **kwargs):
        net = AMPBuilder.Network(self.params, **kwargs)
        return net