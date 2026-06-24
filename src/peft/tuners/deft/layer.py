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

import warnings
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

from .config import DeftConfig


class DeftLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("deft_P", "deft_R", "deft_gate")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("deft_r", "deft_decomposition", "deft_use_gating", "deft_init_scale")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.deft_r = {}
        self.deft_decomposition = {}
        self.deft_use_gating = {}
        self.deft_init_scale = {}
        self.deft_P = nn.ParameterDict({})
        self.deft_R = nn.ParameterDict({})
        self.deft_gate = nn.ParameterDict({})
        self.deft_dropout = nn.ModuleDict({})
        # The residual projection (I - P_proj) @ W is not invertible, so the delta applied during merge is cached
        # here per adapter to allow an exact unmerge.
        self._cached_merge_delta = {}
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        # flag to enable/disable casting of input to weight dtype during forward call
        self.cast_input_dtype_enabled = True
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        else:
            raise TypeError(f"Unsupported layer type {type(base_layer)}")

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        config: DeftConfig,
        **kwargs,
    ) -> None:
        """Internal function to create the DEFT adapter.

        Args:
            adapter_name (`str`): Name for the adapter to add.
            r (`int`): Rank for the added adapter.
            config (`DeftConfig`): The adapter configuration for this layer.
        """
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        # The projection P_proj has rank at most out_features, so the effective rank is capped here (relevant for small
        # output dimensions, e.g. r=8 on a layer with out_features=2). This keeps P and R shapes consistent with the
        # qr decomposition, which can return at most out_features orthonormal columns.
        r = min(r, self.out_features)
        self.deft_r[adapter_name] = r
        self.deft_decomposition[adapter_name] = config.decomposition_method
        self.deft_use_gating[adapter_name] = config.use_gating
        self.deft_init_scale[adapter_name] = config.init_scale

        if config.deft_dropout > 0.0:
            self.deft_dropout[adapter_name] = nn.Dropout(p=config.deft_dropout)
        else:
            self.deft_dropout[adapter_name] = nn.Identity()

        # P: projection direction (out_features x r); R: injection matrix (r x in_features)
        self.deft_P[adapter_name] = nn.Parameter(torch.empty(self.out_features, r))
        self.deft_R[adapter_name] = nn.Parameter(torch.empty(r, self.in_features))
        if config.use_gating:
            self.deft_gate[adapter_name] = nn.Parameter(torch.full((1,), 0.5))

        self.reset_deft_parameters(adapter_name, init_weights=config.init_weights)

        # Move new weights to device
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=config.inference_mode)

    def reset_deft_parameters(self, adapter_name: str, init_weights: bool = True) -> None:
        if adapter_name not in self.deft_P.keys():
            return

        nn.init.normal_(self.deft_P[adapter_name], mean=0.0, std=0.02)
        if adapter_name in self.deft_gate.keys():
            nn.init.constant_(self.deft_gate[adapter_name], 0.5)

        base_weight = self.get_base_layer().weight
        R_param = self.deft_R[adapter_name]
        if self.deft_P[adapter_name].is_meta or base_weight.is_meta or R_param.is_meta:
            # tensors are on meta (e.g. low_cpu_mem_usage loading); the real values are loaded afterwards, so any
            # (data-dependent) initialization is skipped here.
            return

        if init_weights:
            # Identity initialization: choose R so that the delta is exactly zero at init, i.e. the adapted weight
            # equals the base weight. With delta = Q_P @ (g * R - right.T @ W) (see `get_delta_weight`), delta == 0 holds
            # for R = (right.T @ W) / g. This starts training from the pretrained weights and learns the injection,
            # avoiding the immediate "forgetting" caused by removing a sub-space of W.
            P = self.deft_P[adapter_name].detach().to(base_weight.device)
            _, right = self._project(P, adapter_name)
            R_init = right.transpose(0, 1) @ base_weight.detach().to(torch.float32)
            if adapter_name in self.deft_gate.keys():
                # the gate is moved to the base-layer device only after reset, so move it explicitly here
                gate = self.deft_gate[adapter_name].detach().to(device=base_weight.device, dtype=torch.float32)
                R_init = R_init / torch.sigmoid(gate)
            with torch.no_grad():
                R_param.copy_(R_init.to(dtype=R_param.dtype, device=R_param.device))
        else:
            # Random initialization: reproduces the original DEFT behavior where a sub-space of W is replaced from the
            # start (non-identity at init).
            nn.init.normal_(R_param, mean=0.0, std=0.02 * self.deft_init_scale[adapter_name])

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return
        for active_adapter in self.active_adapters:
            if active_adapter not in self.deft_P.keys():
                continue
            warnings.warn("Scaling operation for DEFT not supported! Automatically set scale to 1.")

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.deft_P.keys():
                continue
            warnings.warn("Unscaling operation for DEFT not supported! Keeping scale at 1.")


class DeftLinear(nn.Module, DeftLayer):
    """DEFT implemented in a dense layer."""

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        config: DeftConfig,
        r: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        DeftLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, config=config, **kwargs)

    def _project(self, P: torch.Tensor, adapter_name: str):
        """Return `(Q_P, right)` (both float32) such that the projector is `P_proj = Q_P @ right.T` and the injection
        uses `Q_P`.

          - `relu`: `Q_P = P`, `right = relu(P)` -> `P_proj = P @ relu(P).T` (non-orthogonal)
          - `qr`: `Q_P = qr(P)`, `right = Q_P` -> `P_proj = Q_P @ Q_P.T` (orthogonal)
        """
        # float32 for numerical stability: the QR decomposition (and the projection math) are unreliable in
        # half precision, so the projection is always computed in float32 and cast back by the caller.
        P = P.to(torch.float32)
        method = self.deft_decomposition[adapter_name]
        if method == "relu":
            Q_P = P
            right = torch.relu(P)
        elif method == "qr":
            Q_P, _ = torch.linalg.qr(P)
            right = Q_P
        else:
            raise ValueError(f"Unknown decomposition_method '{method}'.")
        return Q_P, right

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        """Return the additive delta such that `W + delta` equals the DEFT-adapted weight.

        The adapted weight is `(I - P_proj) @ W + g * Q_P @ R` (`g` is the optional gate), so the delta is `-P_proj @ W
        + g * Q_P @ R`. Using `P_proj = Q_P @ right.T` this factors into

            delta = Q_P @ (g * R - right.T @ W)

        which only uses rank-`r` matmuls and never materializes the `out x out` projection matrix. This is used by
        `merge`; the forward pass computes the equivalent update directly on the activations instead.
        """
        base_layer = self.get_base_layer()
        weight = base_layer.weight
        orig_dtype = weight.dtype

        # computed in float32 for numerical stability (see `_project`), then cast back to the base weight dtype
        Q_P, right = self._project(self.deft_P[adapter_name], adapter_name)
        W = weight.to(torch.float32)
        R = self.deft_R[adapter_name].to(torch.float32)
        if self.deft_use_gating[adapter_name]:
            R = R * torch.sigmoid(self.deft_gate[adapter_name].to(torch.float32))

        delta = Q_P @ (R - right.transpose(0, 1) @ W)
        return delta.to(orig_dtype)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights.

        Args:
            safe_merge (`bool`, *optional*):
                If `True`, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If `None`, all active adapters will be merged.
                Defaults to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        base_layer = self.get_base_layer()
        # Each delta's removal term depends on the base weight, so all deltas are computed from the original (unmerged)
        # weight *before* mutating it. This makes merging a set of adapters equivalent to summing their independent
        # deltas, matching what `forward` does, and keeps `unmerge` exact via the cached deltas.
        deltas = {a: self.get_delta_weight(a) for a in adapter_names if a in self.deft_P.keys()}
        for active_adapter, delta_weight in deltas.items():
            if safe_merge:
                new_weight = base_layer.weight.data.clone() + delta_weight
                if not torch.isfinite(new_weight).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )
                base_layer.weight.data = new_weight
            else:
                base_layer.weight.data = base_layer.weight.data + delta_weight
            # cache the exact delta applied so unmerge can reverse the (non-invertible) projection
            self._cached_merge_delta[active_adapter] = delta_weight
            self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """Unmerge all merged adapter layers from the base weights."""
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter not in self.deft_P.keys():
                continue
            delta_weight = self._cached_merge_delta.pop(active_adapter, None)
            if delta_weight is None:
                # fall back to recomputation (only exact if the base weight is unchanged since merge)
                delta_weight = self.get_delta_weight(active_adapter)
            base_layer = self.get_base_layer()
            base_layer.weight.data = base_layer.weight.data - delta_weight

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        elif not any(active_adapter in self.deft_P.keys() for active_adapter in self.active_adapters):
            # no active DEFT adapter on this layer
            result = self.base_layer(x, *args, **kwargs)
        else:
            # LoRA-style forward: instead of materializing W' = W + delta and calling F.linear, apply the update
            # directly to the activations. This preserves any hooks on the base layer's forward (e.g. sharding) and
            # keeps the path compatible with quantized base layers.
            #
            # For one adapter, delta = Q_P @ (g*R - right.T @ W), so
            #     x @ delta.T = [ (x @ R.T) - (x @ W.T) @ right ] @ Q_P.T  =  inject - correction
            # where (x @ W.T) is the bias-free base product.
            base_layer = self.get_base_layer()
            bias = base_layer.bias

            active_adapters = [a for a in self.active_adapters if a in self.deft_P.keys()]
            base_out = self.base_layer(x, *args, **kwargs)
            # the DEFT update is computed in float32 for numerical stability (see `_project`)
            compute_dtype = torch.float32
            result = base_out.to(compute_dtype)
            # bias-free base product (x @ W.T), needed by the subspace-removal (correction) term
            base_product = result if bias is None else result - bias.to(compute_dtype)

            x = self._cast_input_dtype(x, compute_dtype)
            for active_adapter in active_adapters:
                Q_P, right = self._project(self.deft_P[active_adapter], active_adapter)
                R = self.deft_R[active_adapter].to(compute_dtype)
                if self.deft_use_gating[active_adapter]:
                    R = R * torch.sigmoid(self.deft_gate[active_adapter].to(compute_dtype))

                x_drop = self.deft_dropout[active_adapter](x)
                inject = F.linear(x_drop, R) @ Q_P.transpose(0, 1)
                correction = (base_product @ right) @ Q_P.transpose(0, 1)
                result = result + inject - correction

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "deft." + rep
