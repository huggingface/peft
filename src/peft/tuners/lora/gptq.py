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

import torch

from peft.tuners.lora.layer import LoraLayer


class QuantLinear(torch.nn.Module, LoraLayer):
    def __init__(
        self,
        adapter_name,
        quant_linear_module,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ):
        torch.nn.Module.__init__(self)
        LoraLayer.__init__(
            self, in_features=quant_linear_module.infeatures, out_features=quant_linear_module.outfeatures
        )
        self.quant_linear_module = quant_linear_module
        self.weight = quant_linear_module.qweight
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def forward(self, x: torch.Tensor):
        # note: logic differs from default Linear because merging is not supported
        result = self.quant_linear_module(x)

        if (
            self.disable_adapters
            or (self.active_adapter not in self.lora_A.keys())
            or (self.r[self.active_adapter] == 0)
        ):
            return result

        lora_A = self.lora_A[self.active_adapter]
        lora_B = self.lora_B[self.active_adapter]
        dropout = self.lora_dropout[self.active_adapter]
        scaling = self.scaling[self.active_adapter]

        requires_conversion = not torch.is_autocast_enabled()
        if requires_conversion:
            expected_dtype = result.dtype
            x = x.to(lora_A.weight.dtype)

        output = lora_B(lora_A(dropout(x)))
        if requires_conversion:
            output = output.to(expected_dtype)
        output = output * scaling
        result += output
        return result

    # TODO: Check if it is better as suggested by users https://github.com/PanQiWei/AutoGPTQ/pull/102
    # def reset_lora_parameters(self, adapter_name):
    #     if adapter_name in self.lora_A.keys():
    #         torch.nn.init.xavier_uniform_(self.lora_A[adapter_name].weight)
    #         torch.nn.init.zeros_(self.lora_B[adapter_name].weight)
