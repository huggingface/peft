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

import warnings
from functools import lru_cache
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from peft.tuners.tuners_utils import BaseTunerLayer, _get_in_out_features, check_adapters_to_merge
from peft.utils import quantization_extra_repr, resolve_quantization_backend

from .config import ShiraConfig


# use a LRU_cache so that the warning is only ever called once and not repeated for every layer/step/epoch
@lru_cache(None)
def _warn_once_about_module_hooks(shira_layer):
    # ShiRA has a forward method that uses the base weights instead of the .forward call of the base layer.
    # This ignores any hook set on the base layer. Inform the user about this so that they can register the
    # hooks on the PEFT module instead.
    base_layer = shira_layer.get_base_layer()
    if any(
        [
            base_layer._forward_hooks,
            base_layer._forward_pre_hooks,
            base_layer._backward_hooks,
            base_layer._backward_pre_hooks,
        ]
    ):
        warnings.warn(
            "One of the base layers adapted with ShiRA has backward/forward (pre) hooks set which will be ignored "
            "by the adapter's forward implementation. Please set the hooks on the adapted layer instead (i.e., "
            "apply the hooks on the same path but after applying the PEFT config)."
        )


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
        self.quantization_backend = resolve_quantization_backend(
            self.get_base_layer(), get_apply_tensor_subclass=kwargs.get("get_apply_tensor_subclass")
        )

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        self.in_features, self.out_features = _get_in_out_features(base_layer)
        if None in (self.in_features, self.out_features):
            raise TypeError("Only nn.Linear layers supported currently")

        self.kwargs = kwargs

    def update_layer(
        self,
        adapter_name,
        mask,
        r,
        config: ShiraConfig,
        inference_mode: bool = False,
        **kwargs,
    ):
        init_weights = config.init_weights

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
            shira_init_weight,
            requires_grad=True,
        )

        if mask is not None:
            # Compute the shira_indices from the mask. Make sure the mask is formed using r*(self.in_features + self.out_features) and not some other K.
            mask_indices = torch.where(mask == 1.0)
            self.shira_indices[adapter_name] = torch.cat(
                [mask_indices[0].unsqueeze(0), mask_indices[1].unsqueeze(0)], 0
            ).to(torch.int)
            self.shira_indices[adapter_name] = self.shira_indices[adapter_name]

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
        config: ShiraConfig,
        r: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        ShiraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = config.fan_in_fan_out
        if self.base_layer is not self.get_base_layer():
            raise ValueError("SHiRA does not support nested base layers")

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, mask, r, config=config)

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
                    orig_weight = self.get_base_weight().clone()
                    orig_weight += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weight).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    self.set_base_weight(orig_weight)
                else:
                    orig_weight = self.get_base_weight()
                    delta_weight = self.get_delta_weight(active_adapter)
                    orig_weight += delta_weight
                    self.set_base_weight(orig_weight)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.shira_weight.keys():
                orig_weight = self.get_base_weight()
                orig_weight -= self.get_delta_weight(active_adapter)
                self.set_base_weight(orig_weight)

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
            self.shira_indices[adapter],
            self.shira_weight[adapter] * self.scaling[adapter],
            (self.out_features, self.in_features),
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        elif self.quantization_backend and not self.quantization_backend.supports_merge:
            # For the normal forward path, self.get_base_weight() needs to be called, but if the quantization backend
            # doesn't support it, we use a pattern that avoid dequantizing the base weight. The disadvantage is that
            # this is slower.
            base_result = self.base_layer(x, *args, **kwargs)
            new_weight = torch.zeros(self.out_features, self.in_features).to(base_result)

            for active_adapter in self.active_adapters:
                if active_adapter not in self.shira_weight.keys():
                    continue
                new_weight += self.get_delta_weight(active_adapter)

            result = base_result + F.linear(x, new_weight)
        else:
            _warn_once_about_module_hooks(self)
            new_weight = self.get_base_weight().clone()

            for active_adapter in self.active_adapters:
                if active_adapter not in self.shira_weight.keys():
                    continue
                new_weight += self.get_delta_weight(active_adapter)

            result = F.linear(x, new_weight, bias=self.base_layer.bias)

        return result

    def supports_lora_conversion(self, adapter_name: str = "default") -> bool:
        # delta weight is sparse, which does not work with SVD
        return False

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "shira." + rep

    def extra_repr(self) -> str:
        return quantization_extra_repr(self)
