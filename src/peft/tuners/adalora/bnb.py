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

from .layer import AdaLoraLayer


if is_bnb_available():

    class SVDLinear8bitLt(bnb.nn.Linear8bitLt, AdaLoraLayer):
        # Low-rank matrix for SVD-based adaptation
        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
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
            AdaLoraLayer.__init__(self, in_features=in_features, out_features=out_features)
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            init_lora_weights = kwargs.pop("init_lora_weights", True)
            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
            self.active_adapter = adapter_name

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            result = super().forward(x)

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

            output = dropout(x) @ (lora_A * lora_E).T @ lora_B.T
            if requires_conversion:
                output = output.to(expected_dtype)
            output = output * scaling / ranknum
            result = result + output
            return result


if is_bnb_4bit_available():

    class SVDLinear4bit(bnb.nn.Linear4bit, AdaLoraLayer):
        # Low-rank matrix for SVD-based adaptation
        def __init__(
            self,
            adapter_name,
            in_features,
            out_features,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
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
            AdaLoraLayer.__init__(self, in_features=in_features, out_features=out_features)
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            init_lora_weights = kwargs.pop("init_lora_weights", True)
            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
            self.active_adapter = adapter_name

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            result = super().forward(x)

            if (
                self.disable_adapters
                or (self.active_adapter not in self.lora_A.keys())
                or (self.r[self.active_adapter] == 0)
            ):
                return result

            # As per Tim Dettmers, for 4bit, we need to defensively clone here.
            # The reason is that in some cases, an error can occur that backprop
            # does not work on a manipulated view. This issue may be solved with
            # newer PyTorch versions but this would need extensive testing to be
            # sure.
            result = result.clone()

            lora_A = self.lora_A[self.active_adapter]
            lora_B = self.lora_B[self.active_adapter]
            lora_E = self.lora_E[self.active_adapter]
            dropout = self.lora_dropout[self.active_adapter]
            scaling = self.scaling[self.active_adapter]
            ranknum = self.ranknum[self.active_adapter] + 1e-5

            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                compute_dtype = lora_A.weight.dtype
                if x.dtype != compute_dtype:
                    x = x.to(compute_dtype)

            output = dropout(x) @ (lora_A * lora_E).T @ lora_B.T
            if requires_conversion:
                output = output.to(expected_dtype)
            output = output * scaling / ranknum
            result = result + output
            return result
