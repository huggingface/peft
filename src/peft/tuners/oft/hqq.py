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
from typing import Optional

import torch

from peft.import_utils import is_hqq_available
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

from .layer import OFTLayer


if is_hqq_available():
    from hqq.core.quantize import HQQLinear

    class HqqOFTLinear(torch.nn.Module, OFTLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            r: int = 8,
            oft_block_size: int = 0,
            module_dropout: float = 0.0,
            init_weights: bool = True,
            coft: bool = False,
            eps: float = 6e-5,
            block_share: bool = False,
            use_cayley_neumann: bool = False,
            num_cayley_neumann_terms: int = 5,
            **kwargs,
        ) -> None:
            super().__init__()
            OFTLayer.__init__(self, base_layer)
            self.fan_in_fan_out = False

            self._active_adapter = adapter_name
            self.update_layer(
                adapter_name,
                r,
                oft_block_size=oft_block_size,
                module_dropout=module_dropout,
                init_weights=init_weights,
                coft=coft,
                eps=eps,
                block_share=block_share,
                use_cayley_neumann=use_cayley_neumann,
                num_cayley_neumann_terms=num_cayley_neumann_terms,
            )

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
                oft_data = self.get_delta_weight(active_adapter)

                output = torch.transpose(output, 0, 1)
                w_data = torch.mm(oft_data, output.to(oft_data.dtype))
                w_data = torch.transpose(w_data, 0, 1)
                w_data = output.to(oft_data.dtype).to(oft_data.device)

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
                if active_adapter not in self.oft_R.keys():
                    continue

                layer = self.get_base_layer()
                quant_config = {**copy.deepcopy(layer.quant_config), "offload_meta": layer.offload_meta}
                output = layer.dequantize()

                oft_data = self.get_delta_weight(active_adapter)

                output = torch.transpose(output, 0, 1)
                w_data = torch.mm(oft_data.t(), output.to(oft_data.dtype))
                w_data = torch.transpose(w_data, 0, 1)
                w_data = w_data.to(oft_data.dtype).to(oft_data.device)

                new_hqq_layer = HQQLinear(None, quant_config, compute_dtype=layer.compute_dtype, device=layer.device)
                quant_config.pop("offload_meta", None)
                new_hqq_layer.quantize(w_data, **quant_config)
                self.base_layer = new_hqq_layer

        def get_delta_weight(self, adapter):
            return self.oft_R[adapter].get_weight()

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            self._check_forward_args(x, *args, **kwargs)
            adapter_names = kwargs.pop("adapter_names", None)

            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                result = self.base_layer(x, *args, **kwargs)
            elif self.merged:
                result = self.base_layer(x, *args, **kwargs)
            else:
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.oft_R.keys():
                        continue
                    oft_R = self.oft_R[active_adapter]

                    requires_conversion = not torch.is_autocast_enabled()
                    if requires_conversion:
                        expected_dtype = x.dtype
                        x = self._cast_input_dtype(x, oft_R.weight.dtype)

                    x = oft_R(x)

            result = self.base_layer(x, *args, **kwargs)
            if requires_conversion:
                result = result.to(expected_dtype)
            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "oft." + rep


def dispatch_hqq(target: torch.nn.Module, adapter_name: str, **kwargs):
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if is_hqq_available() and isinstance(target_base_layer, HQQLinear):
        new_module = HqqOFTLinear(target_base_layer, adapter_name, **kwargs)

    return new_module
