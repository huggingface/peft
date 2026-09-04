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
import torch.nn.functional as F
from torch import nn

from peft.tuners._buffer_dict import BufferDict
from peft.tuners.tuners_utils import BaseTunerLayer

from .config import OSFConfig
from .utils import (
    decompose_weight_matrix,
)


class OSFLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names: tuple[str, ...] = ("osf_svd_params",)
    # All names of other parameters that may contain adapter-related parameters
    other_param_names: tuple[str, ...] = (
        "_osf_U_low_init",
        "_osf_S_low_init",
        "_osf_V_low_init",
        "_osf_U_high",
        "_osf_V_high",
        "effective_rank",
    )

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.effective_rank = {}
        # Map adapter_name -> ParameterDict{"U_low", "S_low", "V_low"}
        self.osf_svd_params = nn.ModuleDict({})
        # Frozen initial low-rank components (for delta computation and gradient projection)
        self._osf_U_low_init = BufferDict({})
        self._osf_S_low_init = BufferDict({})
        self._osf_V_low_init = BufferDict({})
        # Frozen high-rank subspace for gradient projection — only stored when not
        # exactly recoverable from U_low_init / V_low_init (i.e. the non-square factor).
        # When the factor IS square, the high-rank subspace can be recovered from the
        # low-rank init via the orthogonal complement identity:
        #   (I - U_high @ U_high^T) = U_low_init @ U_low_init^T
        # so the projection can use the smaller low-rank init instead, and U_high/V_high
        # do not need to be stored.
        self._osf_U_high = BufferDict({})
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

    def update_layer(self, adapter_name: str, effective_rank: int, config: OSFConfig, **kwargs):
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

        # Store frozen initial low-rank components (for delta computation).
        # Must clone() to avoid sharing storage with the trainable Parameter.
        self._osf_U_low_init[adapter_name] = svd_dict["U_low"].detach().clone()
        self._osf_S_low_init[adapter_name] = svd_dict["S_low"].detach().clone()
        self._osf_V_low_init[adapter_name] = svd_dict["V_low"].detach().clone()

        # Determine which high-rank factors can be exactly recovered from low-rank inits.
        # U is square (hence exactly recoverable) when out_features <= in_features.
        # V is square (hence exactly recoverable) when out_features >= in_features.
        #
        # I.e.: (20, 10) -> U: (20, 10) and V: (10, 10)
        #        (10, 20) -> U: (10, 10) and V: (10, 20)
        u_is_square = self.out_features <= self.in_features
        v_is_square = self.out_features >= self.in_features

        if not u_is_square:
            # U_high is not recoverable from U_low_init; store it for gradient projection.
            self._osf_U_high[adapter_name] = svd_dict["U_high"].detach()
        if not v_is_square:
            # V_high is not recoverable from V_low_init; store it for gradient projection.
            self._osf_V_high[adapter_name] = svd_dict["V_high"].detach()

        # Create ParameterDict for trainable low-rank components.
        # When init_weights is False, randomly initialize the trainable parameters so the
        # adapter is not an identity at init (used by tests). The frozen init components
        # remain from the SVD decomposition, so delta computation and gradient projection
        # are unaffected.
        if config.init_weights is False:
            U_low = torch.randn_like(svd_dict["U_low"])
            S_low = torch.randn_like(svd_dict["S_low"])
            V_low = torch.randn_like(svd_dict["V_low"])
        else:
            U_low = svd_dict["U_low"]
            S_low = svd_dict["S_low"]
            V_low = svd_dict["V_low"]

        svd_params = nn.ParameterDict(
            {
                "U_low": U_low,
                "S_low": S_low,
                "V_low": V_low,
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
            # Project gradient to be orthogonal to high-rank subspace.
            #
            # When the SVD factor is square, the orthogonal complement of U_high is
            # exactly spanned by U_low_init:
            #   (I - U_high @ U_high^T) = U_low_init @ U_low_init^T
            # so we can project using U_low_init instead of U_high.
            #
            # When the factor is NOT square, the orthogonal complement has a null-space
            # component that U_low_init cannot capture, so we fall back to U_high.
            if name == "U_low":
                if adapter in layer._osf_U_high:
                    # Non-square case: use stored U_high
                    U_high = layer._osf_U_high[adapter]
                    proj = U_high @ (U_high.transpose(0, 1) @ grad)
                    return grad - proj
                else:
                    # Square case: (I - U_high @ U_high^T) = U_low_init @ U_low_init^T,
                    # so the projection is simply U_low_init @ (U_low_init^T @ grad).
                    U_low_init = layer._osf_U_low_init[adapter]
                    return U_low_init @ (U_low_init.transpose(0, 1) @ grad)
            elif name == "V_low":
                if adapter in layer._osf_V_high:
                    # Non-square case: use stored V_high
                    V_high = layer._osf_V_high[adapter]
                    proj = (grad @ V_high.transpose(0, 1)) @ V_high
                    return grad - proj
                else:
                    # Square case: grad @ (I - V_high^T @ V_high) = grad @ V_low_init^T @ V_low_init
                    V_low_init = layer._osf_V_low_init[adapter]
                    return (grad @ V_low_init.transpose(0, 1)) @ V_low_init
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

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        """Compute the weight delta: (U_low*S_low*V_low - U_low_init*S_low_init*V_low_init).

        - Based on: W = U_high_init*S_high_init*V_high_init + U_low_init*S_low_init*V_low_init
        - Applying delta: W_new = W + (U_low*S_low*V_low - U_low_init*S_low_init*V_low_init)
        - Resulting update: W_new = U_high_init*S_high_init*V_high_init + U_low*S_low*V_low
        """
        svd_module = self.osf_svd_params[adapter_name]
        U_low = svd_module["U_low"]
        S_low = svd_module["S_low"]
        V_low = svd_module["V_low"]

        U_low_init = self._osf_U_low_init[adapter_name]
        S_low_init = self._osf_S_low_init[adapter_name]
        V_low_init = self._osf_V_low_init[adapter_name]

        # Current low-rank component
        current = torch.mm(U_low * S_low.unsqueeze(0), V_low)
        # Initial low-rank component (frozen)
        initial = torch.mm(U_low_init * S_low_init.unsqueeze(0), V_low_init)
        return current - initial

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        if adapter_names is None:
            adapter_names = self.active_adapters

        for active_adapter in adapter_names:
            if active_adapter in self.osf_svd_params.keys():
                base_layer = self.get_base_layer()
                delta = self.get_delta_weight(active_adapter)

                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weight = base_layer.weight.data.clone()
                    new_weight = orig_weight + delta

                    if not torch.isfinite(new_weight).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = new_weight.to(orig_weight.dtype)
                else:
                    base_layer.weight.data = (base_layer.weight.data + delta).to(base_layer.weight.data.dtype)

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
        config: OSFConfig,
        effective_rank: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        OSFLayer.__init__(self, base_layer, **kwargs)

        # Set default effective_rank if not provided
        if effective_rank is None:
            # Default to 50% of min dimension
            effective_rank = min(self.in_features, self.out_features) // 2

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, effective_rank, config=config, **kwargs)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters or self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            # Delta-based forward: output = base_layer(x) + x @ delta^T
            # This avoids materializing the full reconstructed weight.
            active_adapter = self.active_adapters[0] if self.active_adapters else None
            if active_adapter and active_adapter in self.osf_svd_params:
                orig_dtype = x.dtype
                # Base output (may run in different precision)
                result = self.base_layer(x, *args, **kwargs)

                # Compute delta as a low-rank product
                delta = self.get_delta_weight(active_adapter)
                # Apply delta as a low-rank update to the output
                # delta is [out_features, in_features], x is [batch, ..., in_features]
                # We compute x @ delta^T, which is [batch, ..., out_features]
                x_cast = self._cast_input_dtype(x, delta.dtype)
                delta_out = F.linear(x_cast, delta)
                result = result + delta_out.to(orig_dtype)
            else:
                result = self.base_layer(x, *args, **kwargs)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "osf." + rep


def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    osf_config: OSFConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Linear):
        new_module = Linear(target, adapter_name, config=osf_config, **kwargs)

    return new_module
