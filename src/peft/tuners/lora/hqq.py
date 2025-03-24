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
from __future__ import annotations

import copy
import warnings
from typing import Any, Optional

import torch

from peft.import_utils import is_hqq_available
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .layer import LoraLayer, LoraVariant


if is_hqq_available():
    from hqq.core.quantize import HQQLinear

    class HqqLoraLinear(torch.nn.Module, LoraLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            init_lora_weights: bool = True,
            use_rslora: bool = False,
            use_dora: bool = False,
            lora_bias: bool = False,
            **kwargs,
        ) -> None:
            if lora_bias:
                raise ValueError(f"{self.__class__.__name__} does not support lora_bias yet, set it to False")

            super().__init__()
            LoraLayer.__init__(self, base_layer)
            self.fan_in_fan_out = False

            self._active_adapter = adapter_name
            self.update_layer(
                adapter_name,
                r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_lora_weights=init_lora_weights,
                use_rslora=use_rslora,
                use_dora=use_dora,
                lora_bias=lora_bias,
            )

        def resolve_lora_variant(self, *, use_dora: bool, **kwargs) -> Optional[LoraVariant]:
            if not use_dora:
                return None

            from .variants import DoraLinearVariant

            return DoraLinearVariant()

        def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
            """
            Merge the active adapter weights into the base weights

            Args:
                safe_merge (`bool`, *optional*):
                    If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                    before merging the weights. This is useful if you want to check if the merge operation will produce
                    NaNs. Defaults to `False`.
                adapter_names (`list[str]`, *optional*):
                    The list of adapter names that should be merged. If None, all active adapters will be merged.
                    Defaults to `None`.
            """
            adapter_names = check_adapters_to_merge(self, adapter_names)
            if not adapter_names:
                # no adapter to merge
                return

            for active_adapter in adapter_names:
                if active_adapter not in self.lora_A.keys():
                    continue

                layer = self.get_base_layer()
                quant_config = {**copy.deepcopy(layer.quant_config), "offload_meta": layer.offload_meta}

                output = layer.dequantize()
                if active_adapter not in self.lora_variant:  # vanilla LoRA
                    lora_data = self.get_delta_weight(active_adapter)
                    w_data = output + lora_data
                else:
                    w_data = self.lora_variant[active_adapter].merge_safe(self, active_adapter, output)

                if safe_merge and not torch.isfinite(w_data).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )

                new_hqq_layer = HQQLinear(None, quant_config, compute_dtype=layer.compute_dtype, device=layer.device)
                quant_config.pop("offload_meta", None)
                new_hqq_layer.quantize(w_data, **quant_config)
                self.base_layer = new_hqq_layer
                self.merged_adapters.append(active_adapter)

        def unmerge(self) -> None:
            """
            This method unmerges all merged adapter layers from the base weights.
            """
            if not self.merged:
                warnings.warn("Already unmerged. Nothing to do.")
                return

            while len(self.merged_adapters) > 0:
                active_adapter = self.merged_adapters.pop()
                if active_adapter not in self.lora_A.keys():
                    continue

                layer = self.get_base_layer()
                quant_config = {**copy.deepcopy(layer.quant_config), "offload_meta": layer.offload_meta}
                output = layer.dequantize()

                if active_adapter not in self.lora_variant:  # vanilla LoRA
                    lora_data = self.get_delta_weight(active_adapter)
                    w_data = output.to(lora_data.dtype).to(lora_data.device) - lora_data
                else:
                    w_data = self.lora_variant[active_adapter].unmerge(self, active_adapter, output)

                new_hqq_layer = HQQLinear(None, quant_config, compute_dtype=layer.compute_dtype, device=layer.device)
                quant_config.pop("offload_meta", None)
                new_hqq_layer.quantize(w_data, **quant_config)
                self.base_layer = new_hqq_layer

        def get_delta_weight(self, adapter):
            return (
                transpose(
                    self.lora_B[adapter].weight @ self.lora_A[adapter].weight,
                    False,
                )
                * self.scaling[adapter]
            )

        def _mixed_batch_forward(
            self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
        ) -> torch.Tensor:
            # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
            # extra argument that allows mixing different adapters in the same batch at inference time.
            result = self.base_layer(x, *args, **kwargs)

            unique_adapters = set(adapter_names)
            sub_batch_indices_list = []
            for adapter in unique_adapters:
                sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

            for i, active_adapter in enumerate(unique_adapters):
                if active_adapter == "__base__":
                    continue
                if active_adapter not in self.lora_A.keys():
                    continue

                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = result.dtype
                    x = self._cast_input_dtype(x, lora_A.weight.dtype)

                # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
                # layer output
                sub_batch = x[sub_batch_indices_list[i]]
                output = lora_B(lora_A(dropout(sub_batch))) * scaling
                if requires_conversion:
                    output = output.to(expected_dtype)
                result[sub_batch_indices_list[i]] += output

            return result

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            self._check_forward_args(x, *args, **kwargs)
            adapter_names = kwargs.pop("adapter_names", None)

            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                result = self.base_layer(x, *args, **kwargs)
            elif adapter_names is not None:
                result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
            elif self.merged:
                result = self.base_layer(x, *args, **kwargs)
            else:
                result = self.base_layer(x, *args, **kwargs)

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
                        x = self._cast_input_dtype(x, lora_A.weight.dtype)

                    if active_adapter not in self.lora_variant:  # vanilla LoRA
                        result = result + lora_B(lora_A(dropout(x))) * scaling
                    else:
                        result = self.lora_variant[active_adapter].forward(
                            self,
                            active_adapter=active_adapter,
                            x=x,
                            result=result,
                        )

                    if requires_conversion:
                        result = result.to(expected_dtype)

            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "lora." + rep


def dispatch_hqq(target: torch.nn.Module, adapter_name: str, **kwargs):
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if is_hqq_available() and isinstance(target_base_layer, HQQLinear):
        new_module = HqqLoraLinear(target_base_layer, adapter_name, **kwargs)

    return new_module
