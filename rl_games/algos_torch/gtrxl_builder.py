"""GTrXL Network Builder for rl_games.

Integrates the GTrXL architecture as a network option for PPO/A2C training,
following the same patterns as A2CBuilder for LSTM networks.
"""

import torch
import torch.nn as nn

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.network_builder import NetworkBuilder
from rl_games.algos_torch.gtrxl import GTrXL


class GTrXLBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)
            self.num_seqs = kwargs.pop('num_seqs', 1)

            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

            # Determine observation input size
            if isinstance(input_shape, dict):
                obs_dim = input_shape['observation'][0]
            else:
                obs_dim = input_shape[0]

            # Build GTrXL backbone
            self.gtrxl = GTrXL(
                input_dim=obs_dim,
                d_model=self.embedding_dim,
                num_layers=self.gtrxl_num_layers,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                memory_length=self.memory_length,
                dropout=self.gtrxl_dropout,
                gate_bias=self.gru_gate_bias,
            )

            out_size = self.embedding_dim

            # Value head
            self.value = self._build_value_layer(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            # Action heads
            if self.is_discrete:
                self.logits = nn.Linear(out_size, actions_num)
            if self.is_multi_discrete:
                self.logits = nn.ModuleList([
                    nn.Linear(out_size, num) for num in actions_num
                ])
            if self.is_continuous:
                self.mu = nn.Linear(out_size, actions_num)
                self.mu_act = self.activations_factory.create(self.space_config['mu_activation'])
                mu_init = self.init_factory.create(**self.space_config['mu_init'])
                self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation'])
                sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

                if self.fixed_sigma:
                    self.sigma = nn.Parameter(
                        torch.zeros(actions_num, requires_grad=True, dtype=torch.float32),
                        requires_grad=True,
                    )
                else:
                    self.sigma = nn.Linear(out_size, actions_num)

            # Initialize action/value heads
            mlp_init = self.init_factory.create(**self.initializer)
            for m in [self.value]:
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

            if self.is_discrete:
                mlp_init(self.logits.weight)
                if self.logits.bias is not None:
                    nn.init.zeros_(self.logits.bias)

            if self.is_continuous:
                mu_init(self.mu.weight)
                if self.fixed_sigma:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)
            dones = obs_dict.get('dones', None)
            seq_length = obs_dict.get('seq_length', 1)

            batch_size = obs.size(0)
            num_seqs = batch_size // seq_length

            # Reshape to (seq_length, num_seqs, obs_dim) for transformer
            out = obs.reshape(num_seqs, seq_length, -1)
            out = out.transpose(0, 1)  # (seq_length, num_seqs, obs_dim)

            # Unpack memory state from rl_games format
            memory = self._unpack_memory(states, num_seqs)

            # Handle done masks
            done_masks = None
            if dones is not None:
                done_masks = dones.reshape(num_seqs, seq_length, -1)
                done_masks = done_masks.transpose(0, 1).squeeze(-1)  # (seq_length, num_seqs)

            # Forward through GTrXL
            out, new_memory = self.gtrxl(out, memory=memory, done_masks=done_masks)

            # Reshape back to (batch_size, d_model)
            out = out.transpose(0, 1)  # (num_seqs, seq_length, d_model)
            out = out.contiguous().reshape(batch_size, -1)

            # Pack memory back to rl_games format
            new_states = self._pack_memory(new_memory)

            # Compute outputs
            value = self.value_act(self.value(out))

            if self.is_discrete:
                logits = self.logits(out)
                return logits, value, new_states

            if self.is_multi_discrete:
                logits = [logit(out) for logit in self.logits]
                return logits, value, new_states

            if self.is_continuous:
                mu = self.mu_act(self.mu(out))
                if self.fixed_sigma:
                    sigma = self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(out))
                return mu, mu * 0 + sigma, value, new_states

        def _pack_memory(self, memory_list):
            """Pack GTrXL memory into rl_games state format.

            GTrXL memory: list of num_layers tensors, each (mem_len, batch, d_model)
            rl_games state: tuple of tensors with shape (dim0, batch, dim2)

            We stack all layers: (num_layers * mem_len, batch, d_model)
            """
            if memory_list is None:
                return (torch.zeros(
                    self.gtrxl.num_layers * self.gtrxl.memory_length,
                    1, self.gtrxl.d_model
                ),)
            stacked = torch.cat(memory_list, dim=0)  # (num_layers * mem_len, batch, d_model)
            return (stacked,)

        def _unpack_memory(self, states, num_seqs):
            """Unpack rl_games state format into GTrXL memory.

            rl_games state: tuple containing tensor of shape (num_layers * mem_len, batch, d_model)
            GTrXL memory: list of num_layers tensors, each (mem_len, batch, d_model)
            """
            if states is None:
                return None

            # Handle tuple format
            if isinstance(states, (tuple, list)):
                if len(states) == 1:
                    packed = states[0]
                else:
                    packed = states[0]
            else:
                packed = states

            mem_len = self.gtrxl.memory_length
            num_layers = self.gtrxl.num_layers

            # Split packed tensor back into per-layer memories
            memory = []
            for i in range(num_layers):
                start = i * mem_len
                end = (i + 1) * mem_len
                memory.append(packed[start:end])

            return memory

        def is_separate_critic(self):
            return False

        def is_rnn(self):
            return True

        def get_default_rnn_state(self):
            """Return initial memory state.

            Shape: tuple of 1 tensor with shape
            (num_layers * memory_length, num_seqs, d_model)

            This matches rl_games convention where each tensor has
            shape (dim0, num_seqs, dim2).
            """
            total_mem = self.gtrxl.num_layers * self.gtrxl.memory_length
            return (torch.zeros(total_mem, self.num_seqs, self.gtrxl.d_model),)

        def load(self, params):
            self.separate = False  # GTrXL uses shared trunk

            # GTrXL-specific params
            gtrxl_params = params.get('gtrxl', {})
            self.embedding_dim = gtrxl_params.get('embedding_dim', 256)
            self.gtrxl_num_layers = gtrxl_params.get('num_layers', 3)
            self.num_heads = gtrxl_params.get('num_heads', 8)
            self.d_ff = gtrxl_params.get('d_ff', None)  # Default: 4 * embedding_dim
            self.memory_length = gtrxl_params.get('memory_length', 64)
            self.gtrxl_dropout = gtrxl_params.get('dropout', 0.0)
            self.gru_gate_bias = gtrxl_params.get('gru_gate_bias', 2.0)

            # Standard params
            self.value_activation = params.get('value_activation', 'None')
            self.initializer = params.get('initializer', {'name': 'default'})
            self.has_space = 'space' in params

            if self.has_space:
                self.is_multi_discrete = 'multi_discrete' in params['space']
                self.is_discrete = 'discrete' in params['space']
                self.is_continuous = 'continuous' in params['space']
                if self.is_continuous:
                    self.space_config = params['space']['continuous']
                    self.fixed_sigma = self.space_config['fixed_sigma']
                elif self.is_discrete:
                    self.space_config = params['space']['discrete']
                elif self.is_multi_discrete:
                    self.space_config = params['space']['multi_discrete']
            else:
                self.is_discrete = False
                self.is_continuous = False
                self.is_multi_discrete = False

    def build(self, name, **kwargs):
        net = GTrXLBuilder.Network(self.params, **kwargs)
        return net
