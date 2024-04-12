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

import builtins
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy
import torch
import torch.nn as nn
from transformers.modeling_outputs import (  # type: ignore
    ModelOutput,
)

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


@dataclass
class InhibitorFlagPayload:
    batch_size: int
    seq_len: int
    override_scaling_pass_value: Number


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
        super().__init__()

        self.__dict__["model"] = model  # We want to hide this from Pytorch...
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.config = config
        self.log_scalings = []
        self.softmax = TemperatureScaledSoftmax(temperature=self.config.softmax_temperature)
        self.override_scaling_pass_value: Number = config.scaling_pass_value

        self.scalings_logging = False

        dtype = next(model.parameters()).dtype
        bias_flag = config.use_bias

        layers = []
        if self.config.xlora_depth == 1:
            if config.layerwise_scalings:  # bias=False if we have just one layer
                last = nn.Linear(config.hidden_size, n_classes * n_layers, bias=bias_flag).to(device).to(dtype)
            else:
                last = nn.Linear(config.hidden_size, n_classes, bias=bias_flag).to(device).to(dtype)
        elif self.config.xlora_depth == 2:
            layers.append(nn.Linear(config.hidden_size, config.xlora_size, bias=bias_flag).to(device).to(dtype))

            if config.enable_relu_and_dropout:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=config.xlora_dropout_p))

            if config.layerwise_scalings:
                last = nn.Linear(config.xlora_size, n_classes * n_layers, bias=bias_flag).to(device).to(dtype)
            else:
                last = nn.Linear(config.xlora_size, n_classes, bias=bias_flag).to(device).to(dtype)
        else:
            if self.config.xlora_depth <= 0:
                raise ValueError("X-LoRA depth must be strictly positive.")

            layers.append(nn.Linear(config.hidden_size, config.xlora_size, bias=bias_flag).to(device).to(dtype))

            if config.enable_relu_and_dropout:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=config.xlora_dropout_p))

            for _ in range(config.xlora_depth - 2):
                layers.append(nn.Linear(config.xlora_size, config.xlora_size, bias=bias_flag).to(device).to(dtype))

                if config.enable_relu_and_dropout:
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(p=config.xlora_dropout_p))

            if config.layerwise_scalings:
                last = nn.Linear(config.xlora_size, n_classes * n_layers, bias=bias_flag).to(device).to(dtype)
            else:
                last = nn.Linear(config.xlora_size, n_classes, bias=bias_flag).to(device).to(dtype)
        self.layers = nn.Sequential(*layers, last)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Using the hidden states of the model, predict `n_classes` LoRA alpha values. Sets the scalings.
        """
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if input_ids is not None:
            seq_len = input_ids.shape[1]
        else:
            seq_len = inputs_embeds.shape[1]

        # For type checking
        model = self.model
        with torch.no_grad():
            with model.disable_adapter():
                kwargs["output_hidden_states"] = True
                kwargs["return_dict"] = True

                result: ModelOutput = model.forward(
                    *args,
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    _xlora_classifier_inhibitor_flag=InhibitorFlagPayload(
                        batch_size=batch_size,
                        seq_len=seq_len,
                        override_scaling_pass_value=self.override_scaling_pass_value,
                    ),
                    **kwargs,
                )

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

    def _get_bucketed_scalings(self) -> Dict[int, Tuple[List[int], List[torch.Tensor]]]:
        """
        Returns bucketed scalings, bucketed by seq_len. Each value consists of the positions (the first)
        and the associated tensors. The positions are paired with the associated tensors and give the position
        in the scaling log. Each scaling is a tensor of shape (batch_size, seq_len, n_layers, n_classes)).
        """
        seqlens_map: Dict[int, Tuple[List[int], List[torch.Tensor]]] = {}
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
