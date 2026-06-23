# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn


class ShadowInjectionModel(nn.Module):
    """
    Injection adapter applied before each base decoder layer (except layer 0).

    For every adapted layer the correction is computed as a low-rank transformation of the difference between the base
    hidden states and the parallel shadow hidden states::

        delta = hidden - shadow_hidden
        delta_t = dropout(delta @ W_down) @ W_up
        hidden' = hidden + alpha * delta_t

    The ``W_up`` projection is zero-initialized so that the adapter is a no-op at the start of training.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        injection_hidden_size: int,
        dropout: float,
        alpha: float,
        initialization_std: float = 0.02,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if injection_hidden_size < 1:
            raise ValueError(f"injection_hidden_size must be >= 1, got {injection_hidden_size}")
        self.injection_downs = nn.Parameter(torch.randn((num_layers, hidden_size, injection_hidden_size)))
        nn.init.normal_(self.injection_downs, mean=0.0, std=initialization_std)

        self.injection_ups = nn.Parameter(torch.zeros((num_layers, injection_hidden_size, hidden_size)))
        self.dropout = nn.Dropout(dropout)
        self.alpha = float(alpha)

    def forward(
        self,
        hidden_states: torch.Tensor,
        shadow_hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        delta = hidden_states - shadow_hidden_states
        delta_t = torch.einsum("btd,dk->btk", delta, self.injection_downs[layer_idx])
        delta_t = self.dropout(delta_t)
        delta_t = torch.einsum("btk,kd->btd", delta_t, self.injection_ups[layer_idx])
        return hidden_states + self.alpha * delta_t


class ShadowUpdateModel(nn.Module):
    """
    Updates the per-step ``shadow_hidden_states`` after each base decoder layer.

    A gated residual update evolves the shadow hidden states as the base model processes each layer::

        h_in = LN(hidden)
        ht = T(h_in)
        g = sigmoid(G(h_in))
        shadow' = shadow + g * (ht - shadow)
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        gate_hidden_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if gate_hidden_size < 2:
            raise ValueError(f"gate_hidden_size must be >= 2, got {gate_hidden_size}")

        self.update_gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, gate_hidden_size, bias=False),
                    nn.SiLU(),
                    nn.Linear(gate_hidden_size, hidden_size, bias=False),
                    nn.Sigmoid(),
                )
                for _ in range(num_layers)
            ]
        )
        self.update_transforms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, gate_hidden_size, bias=False),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(gate_hidden_size, hidden_size, bias=False),
                )
                for _ in range(num_layers)
            ]
        )
        self.hidden_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        shadow_hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        h_in = self.hidden_norm(hidden_states)
        ht = self.update_transforms[layer_idx](h_in)
        g = self.update_gates[layer_idx](h_in)
        return shadow_hidden_states + g * (ht - shadow_hidden_states)
