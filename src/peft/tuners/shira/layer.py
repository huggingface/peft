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

import copy
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge


class ShiraLayer(BaseTunerLayer):
    # List all names of layers that may contain trainable adapter weights
    adapter_layer_names = ("shira_weight",)
    # All names of other adapter-related parameters
    other_param_names = ("r", "scaling", "shira_indices")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.scaling = {}
        self.shira_weight = nn.ParameterDict({})
        self.shira_indices = {}
        self.weight_shape = base_layer.weight.shape  # Assumes SHiRA is on some layer with "weight" parameter

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            raise NotImplementedError("Only nn.Linear layers supported currently")

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    def update_layer(
        self,
        adapter_name,
        mask,
        r,
        init_weights: bool = True,
        inference_mode: bool = False,
        **kwargs,
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        self.scaling[adapter_name] = (
            1.0  # Default scale during training. Can be set to any (non-negative) value during inference.
        )
        # The number of shira weights in this layer is determined by r such that the total number of weights is the same as a LoRA Layer (for direct comparisons)
        num_shira_weight = r * (self.in_features + self.out_features)
        if num_shira_weight > self.in_features * self.out_features:
            raise ValueError(
                f"The set rank {r} results in more shira params than the total number of params in the base layer {self.in_features * self.out_features} and this is not allowed."
            )

        # Actual trainable parameters
        # We have used a vector parameter with fixed indices that we use inside a torch.sparse_coo_tensor in get_delta_weight function.
        # Directly using a torch.sparse_coo_tensor as a parameter could have been possible but we ran into some issues similar to:
        # https://github.com/pytorch/pytorch/issues/79542.
        shira_init_weight = torch.zeros(num_shira_weight) if init_weights else torch.randn(num_shira_weight)
        self.shira_weight[adapter_name] = nn.Parameter(
            shira_init_weight.to(self.base_layer.weight.dtype).to(self.base_layer.weight.device),
            requires_grad=True,
        )

        if mask is not None:
            # Compute the shira_indices from the mask. Make sure the mask is formed using r*(self.in_features + self.out_features) and not some other K.
            mask_indices = torch.where(mask == 1.0)
            self.shira_indices[adapter_name] = torch.cat(
                [mask_indices[0].unsqueeze(0), mask_indices[1].unsqueeze(0)], 0
            ).to(torch.int)
            self.shira_indices[adapter_name] = self.shira_indices[adapter_name].to(self.base_layer.weight.device)

            if self.shira_indices[adapter_name].shape[1] != self.shira_weight[adapter_name].shape[0]:
                raise ValueError(
                    f"The SHiRA indices and weights are not the same dimensions for adapter {adapter_name} in layer {self.base_layer}"
                )

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=inference_mode)

    def reset_shira_parameters(self, adapter_name):
        nn.init.zeros_(self.shira_weight[adapter_name])

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale


class Linear(nn.Module, ShiraLayer):
    # SHiRA implemented in a dense layer
    def __init__(
        self,
        base_layer,
        mask,
        adapter_name: str,
        r: int = 0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stored weight like (fan_in, fan_out)
        init_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        ShiraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        if self.base_layer is not self.get_base_layer():
            raise ValueError("SHiRA does not support nested base layers")

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, mask, r, init_weights=init_weights)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """

        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.shira_weight.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()

                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.shira_weight.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """

        # In multi-gpu environment, the indices are at the wrong gpu.  This is needed to correct this.
        self.shira_indices[adapter] = self.shira_indices[adapter].to(self.shira_weight[adapter].device)
        return torch.sparse_coo_tensor(
            self.shira_indices[adapter], self.shira_weight[adapter] * self.scaling[adapter], self.weight_shape
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            new_weight = copy.deepcopy(self.base_layer.weight.data)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.shira_weight.keys():
                    continue
                new_weight += self.get_delta_weight(active_adapter)

            result = F.linear(x, new_weight, bias=self.base_layer.bias)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "shira." + rep
