# coding=utf-8
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

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import transpose


class IA3Layer(BaseTunerLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        is_feedforward: bool,
    ):
        self.scaling = {}
        self.ia3_l = nn.ParameterDict({})
        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features
        self.is_feedforward = is_feedforward

    def update_layer(self, adapter_name, init_ia3_weights):
        # Actual trainable parameters
        if self.is_feedforward:
            weight = torch.randn((1, self.in_features))
        else:
            weight = torch.randn((self.out_features, 1))
        self.ia3_l[adapter_name] = nn.Parameter(weight)
        if init_ia3_weights:
            self.reset_ia3_parameters(adapter_name)
        self.to(self.weight.device)

    def reset_ia3_parameters(self, adapter_name):
        if adapter_name in self.ia3_l.keys():
            # initialize learned vector with torch.ones
            nn.init.constant_(self.ia3_l[adapter_name], 1.0)


class Linear(nn.Linear, IA3Layer):
    # (IA)^3 implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_feedforward: bool = False,  # Set to True if the layer is treated as a feedforward layer
        **kwargs,
    ) -> None:
        init_ia3_weights = kwargs.pop("init_ia3_weights", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        IA3Layer.__init__(self, in_features=in_features, out_features=out_features, is_feedforward=is_feedforward)
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, init_ia3_weights)
        self.active_adapter = adapter_name

        self.is_feedforward = is_feedforward

    def merge(self) -> None:
        if self.active_adapter not in self.ia3_l.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return

        self.weight = transpose(self.weight, self.fan_in_fan_out)
        self.weight.data = torch.mul(self.weight.data, self.ia3_l[self.active_adapter].data)
        self.weight = transpose(self.weight, self.fan_in_fan_out)

        self.merged = True

    def unmerge(self) -> None:
        if self.active_adapter not in self.ia3_l.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        warnings.warn("Unmerge result can be inaccurate for (IA)^3.")
        self.weight = transpose(self.weight, self.fan_in_fan_out)
        # divide by (IA)^3 vector. Add tolerace to avoid division by zero
        self.weight.data = torch.div(self.weight.data, self.ia3_l[self.active_adapter].data + 1e-8)
        self.weight = transpose(self.weight, self.fan_in_fan_out)

        self.merged = False

    def _linear(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.active_adapter not in self.ia3_l.keys():
            return self._linear(x)

        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self._linear(x)
        elif self.merged:
            result = self._linear(x)
        else:
            dtype = self.ia3_l[self.active_adapter].dtype
            ia3_scaling = self.ia3_l[self.active_adapter].flatten()
            if self.is_feedforward:
                x = x.to(dtype)
                # TODO: self.weight.dtype can be != self.ia3_l[self.active_adapter].dtype
                # e.g. bf16 vs fp32. Is that okay?
                interm = (x * ia3_scaling).to(self.weight.dtype)
                result = self._linear(interm)
            else:
                result = self._linear(x)
                result = result.to(dtype) * ia3_scaling

        result = result.to(previous_dtype)
        return result
