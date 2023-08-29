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

import bitsandbytes as bnb
import torch

from .layer import IA3Layer


class Linear8bitLt(bnb.nn.Linear8bitLt, IA3Layer):
    # (IA)^3 implemented in a dense layer
    def __init__(
        self,
        adapter_name,
        in_features,
        out_features,
        is_feedforward,
        **kwargs,
    ) -> None:
        bnb.nn.Linear8bitLt.__init__(
            self,
            in_features,
            out_features,
            bias=kwargs.get("bias", True),
            has_fp16_weights=kwargs.get("has_fp16_weights", True),
            memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
            threshold=kwargs.get("threshold", 0.0),
            index=kwargs.get("index", None),
        )
        IA3Layer.__init__(self, in_features=in_features, out_features=out_features, is_feedforward=is_feedforward)

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        init_ia3_weights = kwargs.pop("init_ia3_weights", True)
        self.update_layer(adapter_name, init_ia3_weights)
        self.active_adapter = adapter_name
        self.is_feedforward = is_feedforward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.disable_adapters or (self.active_adapter not in self.ia3_l.keys()):
            return super().forward(x)

        requires_conversion = (not torch.is_autocast_enabled()) and (x.dtype != torch.float32)
        if requires_conversion:
            x = x.float()

        ia3_scaling = self.ia3_l[self.active_adapter].flatten()
        if self.is_feedforward:
            result = super().forward(x * ia3_scaling)
            expected_dtype = result.dtype
        else:
            result = super().forward(x)
            expected_dtype = result.dtype
            result = result * ia3_scaling

        if requires_conversion:
            result = result.to(expected_dtype)

        return result
