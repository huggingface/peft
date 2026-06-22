# Copyright 2025-present the HuggingFace Inc. team.
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

import warnings
from typing import Any, Optional

import torch

from peft.import_utils import is_quanto_available
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

from .config import LoraConfig
from .layer import Linear


class QuantoLoraLinear(Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter not in self.lora_A.keys():
                continue

            base_layer = self.get_base_layer()
            weight = base_layer.weight

            if hasattr(weight, "dequantize"):
                weight = weight.dequantize()

            if safe_merge:
                orig_dtype = weight.dtype
                if active_adapter not in self.lora_variant:
                    delta_weight = self.get_delta_weight(active_adapter)
                    new_weight = weight.clone() + delta_weight.to(orig_dtype)
                else:
                    new_weight = self.lora_variant[active_adapter].merge_safe(self, active_adapter, weight.clone())

                if not torch.isfinite(new_weight).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )
                weight = new_weight
            else:
                if active_adapter not in self.lora_variant:
                    delta_weight = self.get_delta_weight(active_adapter)
                    weight = weight + delta_weight
                else:
                    self.lora_variant[active_adapter].merge_unsafe(self, active_adapter, weight)

            base_layer.weight = torch.nn.Parameter(weight)

            if self.lora_bias[active_adapter]:
                if getattr(base_layer, "bias", None) is None:
                    raise RuntimeError(
                        "Impossible to merge LoRA with `lora_bias=True` because the base layer has no bias."
                    )
                bias = self.lora_B[active_adapter].bias
                if bias is not None:
                    base_layer.bias.data = base_layer.bias.data + bias * self.scaling[active_adapter]

            self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter not in self.lora_A.keys():
                continue

            base_layer = self.get_base_layer()
            weight = base_layer.weight

            if hasattr(weight, "dequantize"):
                weight = weight.dequantize()

            if active_adapter not in self.lora_variant:
                delta_weight = self.get_delta_weight(active_adapter)
                weight = weight - delta_weight.to(weight.dtype)
            else:
                weight = self.lora_variant[active_adapter].unmerge(self, active_adapter, weight)

            base_layer.weight = torch.nn.Parameter(weight)

            if self.lora_bias[active_adapter]:
                bias = self.lora_B[active_adapter].bias
                if bias is not None:
                    base_layer.bias.data = base_layer.bias.data - bias * self.scaling[active_adapter]

    def __repr__(self) -> str:
        rep = super().__repr__()
        return rep.replace("lora.Linear", "lora.QuantoLoraLinear")


def dispatch_quanto(
    target: torch.nn.Module,
    adapter_name: str,
    config: LoraConfig,
    **kwargs: Any,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if not is_quanto_available():
        return new_module

    from optimum.quanto import QLinear

    if isinstance(target_base_layer, QLinear):
        new_module = QuantoLoraLinear(target, adapter_name, config=config, **kwargs)

    return new_module
