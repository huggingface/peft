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
from typing import Any, List, Optional

import torch

from peft.import_utils import is_eetq_available
from peft.tuners.lora.layer import LoraLayer
from peft.tuners.tuners_utils import BaseTunerLayer


if is_eetq_available():
    from eetq import EetqLinear

    class EetqLoraLinear(torch.nn.Module, LoraLayer):
        def __init__(
            self,
            base_layer,
            adapter_name,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            init_lora_weights: bool = True,
            use_rslora: bool = False,
            **kwargs,
        ):
            super().__init__()
            LoraLayer.__init__(self, base_layer)

            # self.base_layer and self.quant_linear_module are the same; we need the former for consistency and the latter
            # for backwards compatibility
            self.quant_linear_module = base_layer

            self._active_adapter = adapter_name
            self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora)

        def forward(self, x: torch.Tensor):
            result = self.quant_linear_module(x)

            if self.disable_adapters:
                return result

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = result.dtype
                    x = x.to(lora_A.weight.dtype)

                output = lora_B(lora_A(dropout(x)))
                if requires_conversion:
                    output = output.to(expected_dtype)
                output = output * scaling
                result = result + output
            return result

        def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
            raise AttributeError("Merging LoRA layers is not supported for Eetq layers.")

        def unmerge(self) -> None:
            raise AttributeError("Unmerging LoRA layers is not supported for Eetq layers.")

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "lora." + rep


def dispatch_eetq(
    target: torch.nn.Module,
    adapter_name: str,
    **kwargs: Any,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if is_eetq_available() and isinstance(target_base_layer, EetqLinear):
        new_module = EetqLoraLinear(target, adapter_name, **kwargs)
        target.weight = target_base_layer.weight

        if hasattr(target, "bias"):
            target.bias = target_base_layer.bias

    return new_module
