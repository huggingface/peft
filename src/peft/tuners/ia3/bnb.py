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

from typing import Any

import torch

from peft.import_utils import is_bnb_4bit_available, is_bnb_available

from .layer import IA3Layer


if is_bnb_available():

    class Linear8bitLt(torch.nn.Module, IA3Layer):
        # (IA)^3 implemented in a dense layer
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            is_feedforward: bool,
            init_ia3_weights: bool = True,
            **kwargs,
        ) -> None:
            super().__init__()
            IA3Layer.__init__(self, base_layer, is_feedforward=is_feedforward)

            # Freezing the pre-trained weight matrix
            self.get_base_layer().weight.requires_grad = False
            self._active_adapter = adapter_name
            self.update_layer(adapter_name, init_ia3_weights)

        def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            # note: no check for self.merged because merging is not supported (yet)
            if self.disable_adapters:
                return self.base_layer(x)

            ia3_scaling = 1
            for active_adapter in self.active_adapters:
                if active_adapter not in self.ia3_l.keys():
                    continue
                ia3_scaling *= self.ia3_l[active_adapter].flatten()

            requires_conversion = (not torch.is_autocast_enabled()) and (x.dtype != torch.float32)
            if requires_conversion:
                x = x.float()
            if self.is_feedforward:
                result = self.base_layer(x * ia3_scaling)
                expected_dtype = result.dtype
            else:
                result = self.base_layer(x)
                expected_dtype = result.dtype
                result = result * ia3_scaling

            if requires_conversion:
                result = result.to(expected_dtype)

            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "ia3." + rep


if is_bnb_4bit_available():

    class Linear4bit(torch.nn.Module, IA3Layer):
        # IA3 implemented in a dense layer
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            is_feedforward: bool,
            init_ia3_weights: bool = True,
            **kwargs,
        ) -> None:
            super().__init__()
            IA3Layer.__init__(self, base_layer, is_feedforward=is_feedforward)

            # Freezing the pre-trained weight matrix
            self.get_base_layer().weight.requires_grad = False
            self._active_adapter = adapter_name
            self.update_layer(adapter_name, init_ia3_weights)

        def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            # note: no check for self.merged because merging is not supported (yet)
            if self.disable_adapters:
                return self.base_layer(x)

            ia3_scaling = 1
            for active_adapter in self.active_adapters:
                if active_adapter not in self.ia3_l.keys():
                    continue
                ia3_scaling *= self.ia3_l[active_adapter].flatten()

            requires_conversion = (not torch.is_autocast_enabled()) and (x.dtype != torch.float32)
            if requires_conversion:
                x = x.float()
            if self.is_feedforward:
                result = self.base_layer(x * ia3_scaling)
                expected_dtype = result.dtype
            else:
                result = self.base_layer(x)
                expected_dtype = result.dtype
                result = result * ia3_scaling

            result = result.clone()
            # adalora.py and lora.py both suggest that this is necessary for 4-bit training on older versions of Pytorch.
            # This has been duplicated here.

            if requires_conversion:
                result = result.to(expected_dtype)

            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "ia3." + rep
