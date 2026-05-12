# Copyright 2026-present the HuggingFace Inc. team.
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

import warnings
from typing import Any, Optional

import torch
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

from .config import BeftConfig


class BeftLayer(BaseTunerLayer):
    # All names of layers that may contain adapter weights
    adapter_layer_names = ("beft_bias",)

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.beft_bias = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")
        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, adapter_name: str, config: BeftConfig, **kwargs):
        base_layer = self.get_base_layer()
        if base_layer.bias is None:
            warnings.warn(
                "Detected that the base layer has no bias term. "
                "Note you cannot merge the BEFT adapter into the base layer."
            )
        init_weights = config.init_weights
        inference_mode = config.inference_mode
        weight = torch.randn((1, self.out_features))
        self.beft_bias[adapter_name] = nn.Parameter(weight)
        if init_weights:
            self.reset_beft_parameters(adapter_name)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=inference_mode)

    def reset_beft_parameters(self, adapter_name):
        if adapter_name in self.beft_bias.keys():
            nn.init.constant_(self.beft_bias[adapter_name], 0.0)


class Linear(nn.Module, BeftLayer):
    # BEFT implemented in a dense layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        config: BeftConfig,
        **kwargs,
    ) -> None:
        super().__init__()
        BeftLayer.__init__(self, base_layer)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, config=config)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter bias into the base bias

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original bias and check for NaNs before
                merging the bias. This is useful if you want to check if the merge operation will produce NaNs.
                Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.beft_bias.keys():
                base_layer = self.get_base_layer()

                if base_layer.bias is None:
                    raise ValueError(f"Base layer has no bias, cannot merge bias adapter {active_adapter}")

                orig_dtype = base_layer.bias.data.dtype
                beft_bias = self.beft_bias[active_adapter].data

                if safe_merge:
                    output_bias = (base_layer.bias.data + beft_bias.squeeze()).clone()

                    if not torch.isfinite(output_bias).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.bias.data = output_bias.to(orig_dtype)
                else:
                    base_layer.bias.data.add_(beft_bias.squeeze()).to(orig_dtype)

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base bias.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.beft_bias.keys():
                base_layer = self.get_base_layer()

                if base_layer.bias is None:
                    raise ValueError(f"Base layer has no bias, cannot unmerge bias adapter {active_adapter}")

                orig_dtype = base_layer.bias.data.dtype
                # minus BEFT bias.
                beft_bias = self.beft_bias[active_adapter].data
                base_layer.bias.data = (base_layer.bias.data - beft_bias.squeeze()).to(orig_dtype)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            beft_bias = None
            for active_adapter in self.active_adapters:
                if active_adapter not in self.beft_bias.keys():
                    continue
                current_bias = self.beft_bias[active_adapter].flatten()
                if beft_bias is None:
                    beft_bias = current_bias
                else:
                    beft_bias = beft_bias + current_bias

            result = self.base_layer(x, *args, **kwargs)
            if beft_bias is not None:
                orig_dtype = result.dtype
                result = (result + beft_bias).to(orig_dtype)

        return result
