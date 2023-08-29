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

from .layer import AdaLoraLayer


class SVDQuantLinear(torch.nn.Module, AdaLoraLayer):
    def __init__(
        self,
        adapter_name,
        quant_linear_module,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        torch.nn.Module.__init__(self)
        AdaLoraLayer.__init__(
            self, in_features=quant_linear_module.infeatures, out_features=quant_linear_module.outfeatures
        )
        self.quant_linear_module = quant_linear_module
        self.weight = quant_linear_module.qweight
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.quant_linear_module(x)

        if (
            self.disable_adapters
            or (self.active_adapter not in self.lora_A.keys())
            or (self.r[self.active_adapter] == 0)
        ):
            return result

        requires_conversion = not torch.is_autocast_enabled()
        if requires_conversion:
            expected_dtype = result.dtype
            if x.dtype != torch.float32:
                x = x.float()

        lora_A = self.lora_A[self.active_adapter]
        lora_B = self.lora_B[self.active_adapter]
        lora_E = self.lora_E[self.active_adapter]
        dropout = self.lora_dropout[self.active_adapter]
        scaling = self.scaling[self.active_adapter]
        ranknum = self.ranknum[self.active_adapter] + 1e-5

        output = (dropout(x) @ (lora_A * lora_E).T @ lora_B.T) * scaling / ranknum
        # TODO: here, the dtype conversion is applied on the *whole expression*,
        # not the intermediate result, unlike for SVDLinear8bitLT and
        # SVDLinear4bit, is that correct?
        if requires_conversion:
            output = output.to(expected_dtype)
        result = result + output
        return result
