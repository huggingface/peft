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

import math
import warnings
from typing import Any, Literal, Optional

import torch
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

from .utils import BlockCircularConvolution, get_circulant_fast


class C3ALayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("c3a_kernel",)
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("block_size",)

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.block_size = {}
        self.c3a_kernel = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

    def get_delta_weight(self, adapter) -> torch.Tensor:
        if adapter not in self.c3a_kernel.keys():
            raise ValueError(f"Adapter {adapter} not found.")
        base_layer_weight = self.get_base_layer().weight
        base_layer_weight_dtype = base_layer_weight.dtype
        c3a_kernel = self.c3a_kernel[adapter]

        delta_weight = get_circulant_fast(c3a_kernel.to(torch.float32)).to(base_layer_weight_dtype)
        return delta_weight / base_layer_weight.size(-1)

    def update_layer(self, adapter_name, block_size, init_weights):
        if block_size <= 0:
            raise ValueError(f"`block_size` should be a positive integer value but the value passed is {block_size}")
        if self.in_features % block_size != 0:
            raise ValueError(
                f"The block size should be a factor of the input size. However, the input size is {self.in_features} and the block size is {block_size}"
            )
        if self.out_features % block_size != 0:
            raise ValueError(
                f"The block size should be a factor of the output size. However, the output size is {self.out_features} and the block size is {block_size}"
            )

        self.block_size[adapter_name] = block_size

        weight = self.get_base_layer().weight
        self.c3a_kernel[adapter_name] = nn.Parameter(
            torch.zeros(
                self.out_features // block_size,
                self.in_features // block_size,
                block_size,
                dtype=torch.float32,  # Currently, only fp32 is widely supported for FFT (fp16 is only supported on GPU with shapes of powers of 2, bf16 lacks FFT support)
                device=weight.device,
            )
        )

        self.reset_c3a_parameters(adapter_name, init_weights)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    @torch.no_grad()
    def reset_c3a_parameters(self, adapter_name, init_weights):
        if init_weights is True:
            return

        if adapter_name in self.c3a_kernel.keys():
            if init_weights == "gaussian":
                nn.init.normal_(self.c3a_kernel[adapter_name])
            elif init_weights in ["xavier_uniform", False]:  # Support test cases where False presents
                fan_in, fan_out = self.in_features, self.out_features
                std = 1.0 * math.sqrt(2.0 / float(fan_in + fan_out))
                a = math.sqrt(3.0) * std
                nn.init.uniform_(self.c3a_kernel[adapter_name], -a, a)
            elif init_weights == "kaiming_uniform":
                fan_in = self.in_features
                a = 1.0 * math.sqrt(1.0 / float(fan_in))
                nn.init.uniform_(self.c3a_kernel[adapter_name], -a, a)
            else:
                raise ValueError(f"Unknown init_weights: {init_weights}")


class C3ALinear(nn.Module, C3ALayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        block_size: int,
        init_weights: bool | Literal["gaussian", "kaiming_uniform", "xavier_uniform"],
        **kwargs,
    ) -> None:
        super().__init__()
        C3ALayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, block_size, init_weights)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.c3a_kernel.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    orig_weights = orig_weights + delta_weight

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    base_layer.weight.data = base_layer.weight.data + delta_weight

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
            if active_adapter in self.c3a_kernel.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            x = x.to(torch.float32)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.c3a_kernel.keys():
                    continue
                c3a_kernel = self.c3a_kernel[active_adapter].to(torch.float32)
                x = BlockCircularConvolution.apply(x, c3a_kernel) / x.size(-1)
                result += x.to(result.dtype)

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "c3a." + rep
