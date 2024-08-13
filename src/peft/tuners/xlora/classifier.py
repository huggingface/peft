# Copyright 2023-present the HuggingFace Inc. team.
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
from __future__ import annotations

import builtins
from typing import Optional, Union

import torch
import torch.nn as nn

from .config import XLoraConfig


Number = Union[builtins.int, builtins.float, builtins.bool]


class TemperatureScaledSoftmax(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, logits):
        # Scale logits by the temperature
        scaled_logits = logits / self.temperature
        # Apply softmax to the scaled logits
        return self.softmax(scaled_logits)


class XLoraClassifier(nn.Module):
    """
    A classifier to select LoRA layers for XLora.
    """

    def __init__(
        self,
        model: nn.Module,  # PeftModel
        config: XLoraConfig,
        n_classes: int,
        n_layers: int,
        device: torch.device,
    ):
        """
        Construct an X-LoRA classifier from a model, config and some metadata. Note that n_layers is the number of LoRA
        adapter layers, not the number of model layers.
        """
        super().__init__()

        self.n_classes = n_classes
        self.n_layers = n_layers
        self.config = config
        self.log_scalings = []
        self.softmax = TemperatureScaledSoftmax(temperature=self.config.softmax_temperature)
        self.override_scaling_pass_value: Number = config.scaling_pass_value

        self.scalings_logging = False

        self.dtype = next(model.parameters()).dtype
        add_dropout = config.xlora_dropout_p > 0.0

        layers = []
        if self.config.xlora_depth == 1:
            if config.layerwise_scalings:  # bias=False if we have just one layer
                last = nn.Linear(config.hidden_size, n_classes * n_layers, bias=True).to(device).to(self.dtype)
            else:
                last = nn.Linear(config.hidden_size, n_classes, bias=True).to(device).to(self.dtype)
        else:
            if self.config.xlora_depth <= 0:
                raise ValueError("X-LoRA depth must be strictly positive.")

            layers.append(nn.Linear(config.hidden_size, config.xlora_size, bias=True).to(device).to(self.dtype))

            layers.append(nn.ReLU())
            if add_dropout:
                layers.append(nn.Dropout(p=config.xlora_dropout_p))

            for _ in range(config.xlora_depth - 2):
                layers.append(nn.Linear(config.xlora_size, config.xlora_size, bias=True).to(device).to(self.dtype))

                layers.append(nn.ReLU())
                if add_dropout:
                    layers.append(nn.Dropout(p=config.xlora_dropout_p))

            if config.layerwise_scalings:
                last = nn.Linear(config.xlora_size, n_classes * n_layers, bias=True).to(device).to(self.dtype)
            else:
                last = nn.Linear(config.xlora_size, n_classes, bias=True).to(device).to(self.dtype)
        self.layers = nn.Sequential(*layers, last)

    def make_dummy_scalings(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Make some dummy scalings for the scalings pass (the one to get the logits for the X-LoRA classifier). These are
        of shape (batch_size, seq_len, n_layers, n_classes) and filled with the override scalings pass value. Note that
        n_layers is the number of LoRA adapter layers, not the number of model layers.
        """
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            device = input_ids.device
            seq_len = input_ids.shape[1]
        else:
            batch_size = inputs_embeds.shape[0]
            device = inputs_embeds.device
            seq_len = inputs_embeds.shape[1]

        return torch.full(  # type: ignore
            (batch_size, seq_len, self.n_layers, self.n_classes),
            self.override_scaling_pass_value,
        ).to(device=device, dtype=self.dtype)

    def forward(
        self,
        result,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Using the hidden states of the model, predict `n_classes` LoRA alpha values. Returns the scalings.
        """
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
        else:
            batch_size = inputs_embeds.shape[0]
            seq_len = inputs_embeds.shape[1]

        hidden_states = result.hidden_states  # type: ignore

        hidden_state = hidden_states[-1]  # Get the last hidden state

        ### Classifier run
        # hidden_state=[batch_size, seq_len, hidden_size]
        logits = self.layers.forward(hidden_state)

        ### Repeat to make layerwise scalings
        ### If layerwise_scalings=False, then the classifier only outputs logits which are not layer-wise.
        ### So, we expand them to the correct shape.
        if not self.config.layerwise_scalings:
            logits = logits.unsqueeze(2)
            logits = logits.expand(-1, -1, self.n_layers, -1)

        ### Classifier run

        scalings = logits.reshape(batch_size, seq_len, self.n_layers, self.n_classes)
        # scalings = [batch_size, seq_len, n_layers, n_classes]

        if self.config.enable_softmax:
            scalings = self.softmax(scalings)

        if self.scalings_logging:
            self.log_scalings.append(scalings)

        return scalings

    def _get_bucketed_scalings(self) -> dict[int, tuple[list[int], list[torch.Tensor]]]:
        """
        Returns bucketed scalings, bucketed by seq_len. Each value consists of the positions (the first) and the
        associated tensors. The positions are paired with the associated tensors and give the position in the scaling
        log. Each scaling is a tensor of shape (batch_size, seq_len, n_layers, n_classes)).
        """
        seqlens_map: dict[int, tuple[list[int], list[torch.Tensor]]] = {}
        for i, scaling in enumerate(self.log_scalings):
            seq_len = scaling.shape[1]
            if seq_len not in seqlens_map:
                seqlens_map[seq_len] = ([i], [scaling])
            else:
                seqlens_map[seq_len][0].append(i)
                seqlens_map[seq_len][1].append(scaling)

        return seqlens_map

    def _set_override_scaling_pass_value(self, value: Union[Number, None]):
        if value is None:
            self.override_scaling_pass_value = 1 / self.n_classes
        else:
            self.override_scaling_pass_value = value
        self.config.scaling_pass_value = self.override_scaling_pass_value
