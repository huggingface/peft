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
from functools import lru_cache
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from peft.tuners._buffer_dict import BufferDict
from peft.tuners.tuners_utils import BaseTunerLayer, _get_in_out_features, check_adapters_to_merge
from peft.utils import quantization_extra_repr, resolve_quantization_backend

from . import _ops  # noqa: F401  # registers the custom ops used under torch.compile


# use a LRU_cache so that the warning is only ever called once and not repeated for every layer/step/epoch
@lru_cache(None)
def _warn_once_about_module_hooks(paca_layer):
    # PaCA runs the matmul on the base weight itself instead of calling the base layer's forward. This ignores any hook
    # set on the base layer. Inform the user about this so that they can register the hooks on the PEFT module instead.
    base_layer = paca_layer.get_base_layer()
    if any(
        [
            base_layer._forward_hooks,
            base_layer._forward_pre_hooks,
            base_layer._backward_hooks,
            base_layer._backward_pre_hooks,
        ]
    ):
        warnings.warn(
            "One of the base layers adapted with PaCA has backward/forward (pre) hooks set which will be ignored "
            "by the adapter's forward implementation. Please set the hooks on the adapted layer instead (i.e., "
            "apply the hooks on the same path but after applying the PEFT config)."
        )


def _swap_in_columns(weight, indices, scalings, deltas) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Write the adapted columns into `weight` in place and return the original and the adapted columns."""
    originals, adapted = [], []
    for idx, scaling, delta in zip(indices, scalings, deltas):
        original = weight.index_select(1, idx)
        # Type promotion computes the sum in the higher precision of the two (usually the fp32 adapter weight); writing
        # into a tensor of the weight's dtype casts the result in the same kernel.
        columns = torch.empty_like(original)
        torch.add(original, delta, alpha=scaling, out=columns)
        weight.index_copy_(1, idx, columns)
        originals.append(original)
        adapted.append(columns)
    return originals, adapted


_MM_OUT_DTYPE_SUPPORTED: dict[str, bool] = {}


def _mm_to_dtype(a: torch.Tensor, b: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """a @ b with the result in `dtype`.

    When the inputs are half precision and the result should be fp32 (the usual case for the adapter gradient), use
    `torch.mm(..., out_dtype=...)` where available: it saves the extra cast kernel and keeps the fp32 accumulator
    instead of rounding it to half precision first.
    """
    if a.dtype != dtype and dtype == torch.float32 and a.dtype in (torch.float16, torch.bfloat16):
        device_type = a.device.type
        if _MM_OUT_DTYPE_SUPPORTED.get(device_type, True):
            try:
                return torch.mm(a, b, out_dtype=dtype)
            except (TypeError, RuntimeError):
                _MM_OUT_DTYPE_SUPPORTED[device_type] = False
    return torch.mm(a, b).to(dtype)


def _write_columns(weight, indices, columns, reverse=False) -> None:
    pairs = list(zip(indices, columns))
    if reverse:
        # restore in reverse order so that overlapping columns of several adapters end up with the original values
        pairs = pairs[::-1]
    for idx, cols in pairs:
        weight.index_copy_(1, idx, cols)


class _PacaLinearFunction(torch.autograd.Function):
    """Linear layer whose weight has some of its input columns replaced by trainable values.

    This is the core of PaCA: instead of running an adapter branch next to the pretrained layer, the adapted columns
    are written into the pretrained weight right before the matmul and restored right after it. Forward and backward
    thus run a single GEMM each, like the frozen base layer, and the only activation kept for the backward pass is the
    slice of the input that corresponds to the selected columns (`r` instead of `in_features` channels).

    The swap costs a few copies of `out_features * r` elements, which does not depend on the number of tokens. The
    original and adapted columns are kept for the backward pass (again `out_features * r` elements) so that they are
    not recomputed. Restoring the columns after every matmul keeps the base weight unchanged outside of this function,
    so that saving, disabling, deleting and unloading adapters work the same as for any other PEFT method.
    """

    @staticmethod
    def forward(ctx, x, bias, layer, indices, scalings, *deltas):
        weight = layer._get_weight_for_matmul()
        originals, adapted = _swap_in_columns(weight, indices, scalings, deltas)
        try:
            result = F.linear(x, weight, bias)
        finally:
            _write_columns(weight, indices, originals, reverse=True)

        # Only the selected input channels are needed to compute the gradient of the adapted columns. Store them in
        # the dtype of the matmul (which differs from x.dtype under autocast).
        x_parts = [x.index_select(-1, idx).to(result.dtype) for idx in indices]
        ctx.save_for_backward(*x_parts, *originals, *adapted)
        ctx.layer = layer
        ctx.indices = indices
        ctx.scalings = scalings
        ctx.delta_dtypes = [delta.dtype for delta in deltas]
        ctx.x_dtype = x.dtype
        ctx.bias_dtype = bias.dtype if bias is not None else None
        return result

    @staticmethod
    def backward(ctx, grad_output):
        n = len(ctx.indices)
        saved = ctx.saved_tensors
        x_parts, originals, adapted = saved[:n], saved[n : 2 * n], saved[2 * n :]
        grad_x = grad_bias = None

        if ctx.needs_input_grad[0]:
            weight = ctx.layer._get_weight_for_matmul()
            _write_columns(weight, ctx.indices, adapted)
            try:
                grad_x = grad_output.matmul(weight.to(grad_output.dtype))
            finally:
                _write_columns(weight, ctx.indices, originals, reverse=True)
            grad_x = grad_x.to(ctx.x_dtype)

        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
        if ctx.needs_input_grad[1]:
            grad_bias = grad_output_2d.sum(0).to(ctx.bias_dtype)

        grad_deltas = []
        for i, (x_part, scaling, delta_dtype) in enumerate(zip(x_parts, ctx.scalings, ctx.delta_dtypes)):
            if not ctx.needs_input_grad[5 + i]:
                grad_deltas.append(None)
                continue
            x_part_2d = x_part.reshape(-1, x_part.shape[-1]).to(grad_output_2d.dtype)
            grad_delta = _mm_to_dtype(grad_output_2d.t(), x_part_2d, delta_dtype)
            if scaling != 1:
                grad_delta.mul_(scaling)
            grad_deltas.append(grad_delta)

        return (grad_x, grad_bias, None, None, None, *grad_deltas)


class PacaLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("paca_delta",)
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "paca_alpha", "scaling", "paca_indices")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.paca_alpha = {}
        self.scaling = {}
        # update of the selected columns, shape (out_features, r)
        self.paca_delta = nn.ParameterDict({})
        # indices of the selected input columns, shape (r,); saved with the adapter
        self.paca_indices = BufferDict({}, persistent=True)
        self.quantization_backend = resolve_quantization_backend(
            self.get_base_layer(), get_apply_tensor_subclass=kwargs.get("get_apply_tensor_subclass")
        )
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        self.in_features, self.out_features = _get_in_out_features(base_layer)
        if None in (self.in_features, self.out_features):
            raise TypeError("Only nn.Linear layers are supported by PaCA currently.")
        self.kwargs = kwargs

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        paca_alpha: int,
        init_weights: bool = True,
        random_seed: Optional[int] = None,
        inference_mode: bool = False,
        **kwargs,
    ) -> None:
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        if r > self.in_features:
            raise ValueError(f"`r` ({r}) cannot be larger than in_features ({self.in_features}) of the base layer.")

        self.r[adapter_name] = r
        self.paca_alpha[adapter_name] = paca_alpha
        self.scaling[adapter_name] = paca_alpha / r

        generator = torch.Generator().manual_seed(random_seed) if random_seed is not None else None
        indices = torch.randperm(self.in_features, generator=generator)[:r]
        # Sorting does not change which connections are trained, but gives a more regular memory access pattern.
        self.paca_indices[adapter_name] = torch.sort(indices).values
        self.paca_delta[adapter_name] = nn.Parameter(torch.zeros(self.out_features, r))
        self.reset_paca_parameters(adapter_name, init_weights)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=inference_mode)

    def reset_paca_parameters(self, adapter_name: str, init_weights: bool = True) -> None:
        if adapter_name not in self.paca_delta.keys():
            return
        if init_weights:
            nn.init.zeros_(self.paca_delta[adapter_name])
        else:
            # only used for testing, so that the adapter is not a no-op
            nn.init.normal_(self.paca_delta[adapter_name], std=0.1)

    def set_scale(self, adapter: str, scale: float) -> None:
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale * self.paca_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return
        for active_adapter in self.active_adapters:
            if active_adapter not in self.paca_delta.keys():
                continue
            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale: Optional[float] = None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.paca_delta.keys():
                continue
            if scale is None:
                self.scaling[active_adapter] = self.paca_alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale

    def _get_weight_for_matmul(self) -> torch.Tensor:
        """Return the tensor that the adapted columns are swapped into.

        For a regular layer, this is the storage of the base weight itself; the caller restores the columns after the
        matmul. For a quantized layer, this is a dequantized copy.
        """
        if self.quantization_backend is not None:
            return self.get_base_weight()
        return self.get_base_layer().weight.data


class Linear(nn.Module, PacaLayer):
    # PaCA implemented in a dense layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 8,
        paca_alpha: int = 8,
        init_weights: bool = True,
        random_seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        PacaLayer.__init__(self, base_layer, **kwargs)
        if self.base_layer is not self.get_base_layer():
            raise ValueError("PaCA does not support nested base layers")

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, paca_alpha=paca_alpha, init_weights=init_weights, random_seed=random_seed)

    def _adapted_columns(self, weight: torch.Tensor, adapter: str, sign: int) -> torch.Tensor:
        idx = self.paca_indices[adapter].to(weight.device)
        delta = self.paca_delta[adapter].detach().to(weight.device)
        compute_dtype = torch.promote_types(weight.dtype, delta.dtype)
        columns = weight.index_select(1, idx).to(compute_dtype) + sign * self.scaling[adapter] * delta.to(
            compute_dtype
        )
        return columns.to(weight.dtype)

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

        with torch.no_grad():
            for active_adapter in adapter_names:
                if active_adapter not in self.paca_delta.keys():
                    continue
                base_weight = self.get_base_weight()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge because of the copy operation.
                    base_weight = base_weight.clone()
                columns = self._adapted_columns(base_weight, active_adapter, sign=1)
                if safe_merge and not torch.isfinite(columns).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )
                base_weight.index_copy_(1, self.paca_indices[active_adapter].to(base_weight.device), columns)
                self.set_base_weight(base_weight)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        with torch.no_grad():
            while len(self.merged_adapters) > 0:
                active_adapter = self.merged_adapters.pop()
                if active_adapter not in self.paca_delta.keys():
                    continue
                base_weight = self.get_base_weight()
                columns = self._adapted_columns(base_weight, active_adapter, sign=-1)
                base_weight.index_copy_(1, self.paca_indices[active_adapter].to(base_weight.device), columns)
                self.set_base_weight(base_weight)

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter. It is non-zero only in the `r` selected columns, i.e. its rank
        is at most `r`.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        delta = self.paca_delta[adapter]
        idx = self.paca_indices[adapter].to(delta.device)
        delta_weight = torch.zeros(self.out_features, self.in_features, device=delta.device, dtype=delta.dtype)
        return delta_weight.index_copy(1, idx, delta * self.scaling[adapter])

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)
        if self.merged:
            return self.base_layer(x, *args, **kwargs)

        active_adapters = [a for a in self.active_adapters if a in self.paca_delta.keys()]
        if not active_adapters:
            return self.base_layer(x, *args, **kwargs)

        _warn_once_about_module_hooks(self)
        indices = tuple(self.paca_indices[a] for a in active_adapters)
        scalings = tuple(self.scaling[a] for a in active_adapters)
        deltas = [self.paca_delta[a] for a in active_adapters]
        bias = self.get_base_layer().bias
        if self.quantization_backend is None and torch.compiler.is_compiling():
            # Under torch.compile, use opaque custom ops so that the in-place column swap is not functionalized into a
            # copy of the full weight (see _ops.py).
            if torch.is_autocast_enabled(x.device.type):
                x = x.to(torch.get_autocast_dtype(x.device.type))
            out = torch.ops.peft.paca_linear_forward(
                x, self.get_base_layer().weight, bias, list(indices), [float(s) for s in scalings], deltas
            )
            return out[0]
        return _PacaLinearFunction.apply(x, bias, self, indices, scalings, *deltas)

    def supports_lora_conversion(self, adapter_name: str = "default") -> bool:
        # the delta weight is non-zero in r columns only, so a LoRA adapter of rank r represents it exactly
        return True

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "paca." + rep

    def extra_repr(self) -> str:
        return quantization_extra_repr(self)
