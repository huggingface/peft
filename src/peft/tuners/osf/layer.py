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
from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners._buffer_dict import BufferDict
from peft.tuners.tuners_utils import BaseTunerLayer

from .utils import (
    decompose_weight_matrix,
    reconstruct_weight_matrix,
)


class OSFLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names: tuple[str, ...] = ("osf_svd_params",)
    # All names of other parameters that may contain adapter-related parameters
    other_param_names: tuple[str, ...] = ("_osf_U_high", "_osf_S_high", "_osf_V_high")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.effective_rank = {}
        # Map adapter_name -> ParameterDict{"U_low", "S_low", "V_low"}
        self.osf_svd_params = nn.ModuleDict({})
        # Store high-rank (frozen) components as buffers that track device moves
        self._osf_U_high = BufferDict({})
        self._osf_S_high = BufferDict({})
        self._osf_V_high = BufferDict({})
        # Track hook handles for cleanup
        self.hook_handles = []
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        # Get layer dimensions
        base_layer = self.get_base_layer()
        # Prefer the universally available weight shape when possible.
        if (
            hasattr(base_layer, "weight")
            and isinstance(base_layer.weight, torch.Tensor)
            and base_layer.weight.ndim == 2
        ):
            # For Linear-like modules, weight is [out_features, in_features]
            out_features, in_features = base_layer.weight.shape
        elif isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear, RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        elif hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            in_features, out_features = None, None
            warnings.warn(
                f"Unsupported layer type '{type(base_layer)}' encountered; could not infer in/out features.",
                UserWarning,
            )

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, adapter_name: str, effective_rank: int, **kwargs):
        """Update layer to add a new OSF adapter."""
        if effective_rank <= 0:
            raise ValueError(
                f"`effective_rank` should be a positive integer value but the value passed is {effective_rank}"
            )

        # Store the rank for this adapter
        self.effective_rank[adapter_name] = effective_rank

        # Perform SVD decomposition on the base layer weight
        base_layer = self.get_base_layer()
        weight = base_layer.weight.data
        svd_dict = decompose_weight_matrix(weight, top_k=effective_rank)

        # Store high-rank (frozen) components as buffers
        self._osf_U_high[adapter_name] = svd_dict["U_high"]
        self._osf_S_high[adapter_name] = svd_dict["S_high"]
        self._osf_V_high[adapter_name] = svd_dict["V_high"]

        # Create ParameterDict for trainable low-rank components
        svd_params = nn.ParameterDict(
            {
                "U_low": svd_dict["U_low"],
                "S_low": svd_dict["S_low"],
                "V_low": svd_dict["V_low"],
            }
        )
        self.osf_svd_params[adapter_name] = svd_params

        # Attach gradient hooks for orthogonal projection
        self._attach_hooks(adapter_name)

        # Set the adapter as active
        self.set_adapter(self.active_adapters)

    def _attach_hooks(self, adapter_name: str):
        """Attach gradient hooks for the given adapter."""
        if adapter_name not in self.osf_svd_params:
            return

        svd_module = self.osf_svd_params[adapter_name]

        def hook(grad, name: str, adapter: str, layer: OSFLayer):
            # Project gradient to be orthogonal to high-rank subspace for U_low/V_low
            # Access buffers dynamically to ensure they're on the correct device
            if name == "U_low":
                U_high = layer._osf_U_high[adapter]
                proj = U_high @ (U_high.transpose(0, 1) @ grad)
                return grad - proj
            elif name == "V_low":
                V_high = layer._osf_V_high[adapter]
                proj = (grad @ V_high.transpose(0, 1)) @ V_high
                return grad - proj
            return grad

        # Store hook handles for later cleanup
        handle_u = svd_module["U_low"].register_hook(partial(hook, name="U_low", adapter=adapter_name, layer=self))
        handle_v = svd_module["V_low"].register_hook(partial(hook, name="V_low", adapter=adapter_name, layer=self))

        self.hook_handles.extend([handle_u, handle_v])

    def _detach_hooks(self):
        """Remove all gradient hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def _reconstruct_weight(self, adapter_name: str) -> torch.Tensor:
        """Reconstruct weight matrix from SVD components for given adapter."""
        if adapter_name not in self.osf_svd_params:
            return self.get_base_layer().weight

        svd_module = self.osf_svd_params[adapter_name]
        svd_dict = {
            "U_high": self._osf_U_high[adapter_name],
            "S_high": self._osf_S_high[adapter_name],
            "V_high": self._osf_V_high[adapter_name],
            "U_low": svd_module["U_low"],
            "S_low": svd_module["S_low"],
            "V_low": svd_module["V_low"],
        }
        return reconstruct_weight_matrix(svd_dict)

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
        if adapter_names is None:
            adapter_names = self.active_adapters

        for active_adapter in adapter_names:
            if active_adapter in self.osf_svd_params.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weight = base_layer.weight.data.clone()
                    new_weight = self._reconstruct_weight(active_adapter)

                    if not torch.isfinite(new_weight).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = new_weight.to(orig_weight.dtype)
                else:
                    new_weight = self._reconstruct_weight(active_adapter)
                    base_layer.weight.data = new_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        # For OSF, unmerging means restoring the original weight
        # Since we modify the weight in-place, we need to store the original weight
        # This is a limitation of the current OSF implementation
        warnings.warn("OSF does not support unmerging. Original weights are permanently modified.")

    def __del__(self):
        """Cleanup hooks on deletion."""
        self._detach_hooks()


class Linear(nn.Module, OSFLayer):
    # OSF implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        effective_rank: int = None,
        **kwargs,
    ) -> None:
        super().__init__()
        OSFLayer.__init__(self, base_layer, **kwargs)

        # Set default effective_rank if not provided
        if effective_rank is None:
            # Default to 50% of min dimension
            effective_rank = min(self.in_features, self.out_features) // 2

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, effective_rank, **kwargs)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters:
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            # Use reconstructed weight for forward pass
            base_layer = self.get_base_layer()
            bias = base_layer.bias

            # Use the active adapter's reconstructed weight
            active_adapter = self.active_adapters[0] if self.active_adapters else None
            if active_adapter and active_adapter in self.osf_svd_params:
                weight = self._reconstruct_weight(active_adapter)
                result = F.linear(x, weight, bias)
            else:
                result = self.base_layer(x, *args, **kwargs)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "osf." + rep


def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    osf_config,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Linear):
        new_module = Linear(target, adapter_name, **kwargs)

    return new_module
