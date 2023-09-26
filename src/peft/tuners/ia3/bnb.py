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

from peft.import_utils import is_bnb_4bit_available, is_bnb_available

from .layer import IA3Layer


if is_bnb_available():

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
            self.is_feedforward = is_feedforward

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            init_ia3_weights = kwargs.pop("init_ia3_weights", True)
            self.update_layer(adapter_name, init_ia3_weights)
            self.set_adapter(adapter_name)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.disable_adapters:
                return super().forward(x)

            ia3_scaling = 1
            for active_adapter in self.active_adapters:
                if active_adapter not in self.ia3_l.keys():
                    continue
                ia3_scaling *= self.ia3_l[active_adapter].flatten()

            requires_conversion = (not torch.is_autocast_enabled()) and (x.dtype != torch.float32)
            if requires_conversion:
                x = x.float()
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


if is_bnb_4bit_available():

    class Linear4bit(bnb.nn.Linear4bit, IA3Layer):
        # IA3 implemented in a dense layer
        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            is_feedforward,
            **kwargs,
        ) -> None:
            bnb.nn.Linear4bit.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                compute_dtype=kwargs.get("compute_dtype", torch.float32),
                compress_statistics=kwargs.get("compress_statistics", True),
                quant_type=kwargs.get("quant_type", "nf4"),
            )
            IA3Layer.__init__(self, in_features=in_features, out_features=out_features, is_feedforward=is_feedforward)
            self.is_feedforward = is_feedforward

            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            init_ia3_weights = kwargs.pop("init_ia3_weights", True)
            self.update_layer(adapter_name, init_ia3_weights)
            self.set_adapter(adapter_name)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.disable_adapters:
                return super().forward(x)

            ia3_scaling = 1
            for active_adapter in self.active_adapters:
                if active_adapter not in self.ia3_l.keys():
                    continue
                ia3_scaling *= self.ia3_l[active_adapter].flatten()

            requires_conversion = (not torch.is_autocast_enabled()) and (x.dtype != torch.float32)
            if requires_conversion:
                x = x.float()
            if self.is_feedforward:
                result = super().forward(x * ia3_scaling)
                expected_dtype = result.dtype
            else:
                result = super().forward(x)
                expected_dtype = result.dtype
                result = result * ia3_scaling

            result = result.clone()
            # adalora.py and lora.py both suggest that this is necessary for 4-bit training on older versions of Pytorch.
            # This has been duplicated here.

            if requires_conversion:
                result = result.to(expected_dtype)

            return result
