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
import torch.nn.functional as F
from torch import nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .config import DeftConfig


class DeftLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("deft_P", "deft_R")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = (
        "deft_r",
        "deft_decomposition",
        "deft_init_scale",
        "deft_para",
        "deft_scaling",
    )

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.deft_r = {}
        self.deft_decomposition = {}
        self.deft_init_scale = {}
        self.deft_para = {}
        self.deft_scaling = {}
        self.deft_P = nn.ParameterDict({})
        self.deft_R = nn.ParameterDict({})
        self.deft_dropout = nn.ModuleDict({})
        # whether the base weight is stored transposed, i.e. (in_features, out_features) as in `Conv1D` (gpt-2)
        self.fan_in_fan_out = False
        # The residual projection (I - P_proj) @ W is not invertible, so unmerge cannot recover the delta from the
        # merged weight. Rather than cache the full out x in delta, cache only its base-weight-dependent factor
        # right.T @ W (shape r x in_features) per adapter; unmerge recomputes the exact delta from it (~r/out_features
        # of the memory).
        self._cached_merge_factor = {}
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        # flag to enable/disable casting of input to weight dtype during forward call
        self.cast_input_dtype_enabled = True
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            self.in_features, self.out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            # Conv1D (e.g. gpt-2) stores its weight transposed as (in_features, out_features)
            self.in_features, self.out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
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
        self.deft_init_scale[adapter_name] = config.init_scale
        self.deft_para[adapter_name] = config.para
        self.fan_in_fan_out = config.fan_in_fan_out
        # injection scaling (analogous to LoRA's alpha/r); 1.0 = no scaling (backward compatible)
        self.deft_scaling[adapter_name] = (config.alpha / r) if getattr(config, "alpha", None) else 1.0

        if config.deft_dropout > 0.0:
            self.deft_dropout[adapter_name] = nn.Dropout(p=config.deft_dropout)
        else:
            self.deft_dropout[adapter_name] = nn.Identity()

        # P: projection direction (out_features x r); R: injection matrix (r x in_features, full DEFT only).
        # With `para=True` the update is pure subspace removal `-P_proj @ W` (the PaRa method), and R is unused.
        self.deft_P[adapter_name] = nn.Parameter(torch.empty(self.out_features, r))
        if not config.para:
            self.deft_R[adapter_name] = nn.Parameter(torch.empty(r, self.in_features))

        self.reset_deft_parameters(adapter_name, init_weights=config.init_weights)

        # Move new weights to device
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=config.inference_mode)

    def reset_deft_parameters(self, adapter_name: str, init_weights: bool = True) -> None:
        if adapter_name not in self.deft_P.keys():
            return

        nn.init.normal_(self.deft_P[adapter_name], mean=0.0, std=0.02)

        if self.deft_para[adapter_name]:
            # PaRa (para=True): no injection matrix R to initialize; the update is pure subspace removal and
            # cannot be made an identity at init.
            return

        base_weight = self.get_base_layer().weight
        R_param = self.deft_R[adapter_name]
        if self.deft_P[adapter_name].is_meta or base_weight.is_meta or R_param.is_meta:
            # tensors are on meta (e.g. low_cpu_mem_usage loading); the real values are loaded afterwards, so any
            # (data-dependent) initialization is skipped here.
            return

        if init_weights:
            # Identity initialization: choose R so that the delta is exactly zero at init, i.e. the adapted weight
            # equals the base weight. With delta = Q_P @ (R - right.T @ W) (see `get_delta_weight`), delta == 0 holds
            # for R = right.T @ W. This starts training from the pretrained weights and learns the injection,
            # avoiding the immediate "forgetting" caused by removing a sub-space of W.
            P = self.deft_P[adapter_name].detach().to(base_weight.device)
            _, right = self._project(P, adapter_name)
            # transpose for `Conv1D`, whose weight is stored as (in_features, out_features), to the logical (out, in)
            W = transpose(base_weight.detach().to(torch.float32), self.fan_in_fan_out)
            R_init = right.transpose(0, 1) @ W
            # divide by the injection scaling so that (scaling * R_init) == right.T @ W, keeping delta == 0 at init
            R_init = R_init / self.deft_scaling[adapter_name]
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

    def _merge_factor(self, adapter_name: str) -> torch.Tensor:
        """Return the only base-weight-dependent part of the merge delta: `right.T @ W` (shape `r x in_features`), in
        float32. Caching this instead of the full `out x in` delta lets `unmerge` recompute the exact delta while
        storing roughly `r / out_features` as much memory.
        """
        # computed in float32 for numerical stability (see `_project`); transpose handles `Conv1D` (fan_in_fan_out),
        # whose weight is stored as (in_features, out_features), so `W` is always logical (out_features, in_features)
        weight = transpose(self.get_base_layer().weight.to(torch.float32), self.fan_in_fan_out)
        _, right = self._project(self.deft_P[adapter_name], adapter_name)
        return right.transpose(0, 1) @ weight

    def _delta_from_factor(self, adapter_name: str, factor: torch.Tensor) -> torch.Tensor:
        """Reconstruct the additive delta from the cached factor `M = right.T @ W` (float32): `delta = Q_P @ (R - M)`,
        or `-Q_P @ M` for PaRa. Exact even after the base weight changed, since `M` is taken from the original weight.
        """
        orig_dtype = self.get_base_layer().weight.dtype
        Q_P, _ = self._project(self.deft_P[adapter_name], adapter_name)
        if self.deft_para[adapter_name]:
            # PaRa: pure subspace removal, delta = -P_proj @ W = -Q_P @ (right.T @ W)
            delta = -(Q_P @ factor)
        else:
            R = self.deft_R[adapter_name].to(torch.float32) * self.deft_scaling[adapter_name]
            delta = Q_P @ (R - factor)
        # delta is logical (out_features, in_features); transpose back to the base layer's storage order (Conv1D)
        return transpose(delta, self.fan_in_fan_out).to(orig_dtype)

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        """Return the additive delta such that `W + delta` equals the DEFT-adapted weight.

        The adapted weight is `(I - P_proj) @ W + Q_P @ R`, so the delta is `-P_proj @ W + Q_P @ R`. Using `P_proj =
        Q_P @ right.T` this factors into `delta = Q_P @ (R - right.T @ W)` (`-Q_P @ (right.T @ W)` for PaRa), which
        only uses rank-`r` matmuls and never materializes the `out x out` projection matrix. The base-weight-dependent
        part `right.T @ W` is produced by `_merge_factor`. Used by `merge`; the forward pass computes the equivalent
        update directly on the activations instead.
        """
        return self._delta_from_factor(adapter_name, self._merge_factor(adapter_name))

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
        # The delta's removal term depends on the base weight, so every adapter's base-weight-dependent factor
        # (right.T @ W) is computed from the original (unmerged) weight *before* mutating it. This makes merging a set
        # of adapters equivalent to summing their independent deltas (matching `forward`) and keeps `unmerge` exact.
        factors = {a: self._merge_factor(a) for a in adapter_names if a in self.deft_P.keys()}
        for active_adapter, factor in factors.items():
            delta_weight = self._delta_from_factor(active_adapter, factor)
            if safe_merge:
                new_weight = base_layer.weight.data.clone() + delta_weight
                if not torch.isfinite(new_weight).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )
                base_layer.weight.data = new_weight
            else:
                base_layer.weight.data = base_layer.weight.data + delta_weight
            # cache only the small r x in_features factor (not the full out x in delta) for an exact unmerge
            self._cached_merge_factor[active_adapter] = factor.to(base_layer.weight.dtype)
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
            factor = self._cached_merge_factor.pop(active_adapter, None)
            if factor is not None:
                delta_weight = self._delta_from_factor(active_adapter, factor.to(torch.float32))
            else:
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
            # Apply the update on the activations instead of materializing W + delta. For one adapter,
            # delta = Q_P @ (R - right.T @ W), so x @ delta.T = [(x @ R.T) - (x @ W.T) @ right] @ Q_P.T
            # (injection minus subspace-removal), where (x @ W.T) is the bias-free base product.
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
                # subspace-removal (correction) term: (x @ W.T @ right) @ Q_P.T = P_proj @ W @ x
                correction = (base_product @ right) @ Q_P.transpose(0, 1)
                result = result - correction
                if not self.deft_para[active_adapter]:
                    R = self.deft_R[active_adapter].to(compute_dtype)
                    R = R * self.deft_scaling[active_adapter]
                    x_drop = self.deft_dropout[active_adapter](x)
                    result = result + F.linear(x_drop, R) @ Q_P.transpose(0, 1)

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "deft." + rep
