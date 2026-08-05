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

import math
import warnings
from typing import Any, Optional

import torch
from torch import nn

from peft.tuners._buffer_dict import BufferDict
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

from .config import SupertuningConfig


class DensePlusSparseLinear(torch.autograd.Function):
    """
    Linear layer whose effective weight is the frozen base weight plus a sparse update.

    The sparse update is described by ``indices`` (flat positions into the weight) and ``values`` (the trainable
    quantities scatter-added at those positions). The backward pass only propagates a gradient to ``values`` (and to
    the input/bias); the base ``weight`` receives none, which is what keeps the frozen support unchanged during
    optimization. This mirrors the reference Super-Tuning implementation and, unlike masking a dense weight gradient,
    behaves correctly under stateful optimizers such as AdamW.
    """

    @staticmethod
    def forward(ctx, input, weight, indices, values, bias=None):
        ctx.save_for_backward(input, weight, indices, values, bias)

        dense_plus_sparse = weight.reshape(-1).scatter_add(0, indices.to(torch.int64), values.to(weight.dtype))
        dense_plus_sparse = dense_plus_sparse.reshape_as(weight)

        return torch.nn.functional.linear(input, dense_plus_sparse, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, indices, values, bias = ctx.saved_tensors
        grad_input = grad_values = grad_bias = None

        dense_plus_sparse = weight.reshape(-1).scatter_add(0, indices.to(torch.int64), values.to(weight.dtype))
        dense_plus_sparse = dense_plus_sparse.reshape_as(weight)

        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output, dense_plus_sparse)

        if ctx.needs_input_grad[3] or (bias is not None and ctx.needs_input_grad[4]):
            grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
            if ctx.needs_input_grad[3]:
                input_2d = input.reshape(-1, input.shape[-1])
                grad_matrix = grad_output_2d.t().mm(input_2d)
                grad_values = grad_matrix.reshape(-1).gather(0, indices.to(torch.int64)).to(values.dtype)
            if bias is not None and ctx.needs_input_grad[4]:
                grad_bias = grad_output_2d.sum(dim=0)

        # No gradient flows to the (frozen) base weight or to the integer indices.
        return grad_input, None, None, grad_values, grad_bias


class SupertuningLayer(BaseTunerLayer):
    """
    Supertuning layer implementing weight-decomposed sparse fine-tuning.

    The base weight is frozen; a compact ``(indices, values)`` pair encodes the trainable sparse support:
    ``indices`` are flat positions into the weight (selected by magnitude at ``update_layer`` time, or supplied
    externally via [`SupertuningModel.set_precomputed_indices`]), and ``values`` are the trainable scalars
    scatter-added onto the base weight in the forward pass.

    When ``config.r`` is set, additionally allocates LoRA ``A`` and ``B`` low-rank parameters (the paper's Supra
    hybrid); their contribution is added to the sparse-composed output in ``forward`` and folded into the base
    weight during ``merge``. Standard LoRA init is used (Kaiming-uniform for ``A``, zeros for ``B``).
    """

    adapter_layer_names = ("supertuning_values", "lora_A", "lora_B")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        # Trainable sparse quantities, one 1-D parameter per adapter.
        self.supertuning_values = nn.ParameterDict({})
        # Flat positions of the sparse support inside the weight. Persistent so it is saved alongside
        # ``supertuning_values`` in the adapter checkpoint.
        self.supertuning_indices = BufferDict(persistent=True)
        # Per-adapter Supra rank / scaling. Rank 0 (falsy) → pure Super for that adapter.
        self.supertuning_rank: dict[str, int] = {}
        self.supertuning_lora_alpha: dict[str, float] = {}
        # LoRA parameters only populated for adapters with r > 0.
        self.lora_A = nn.ParameterDict({})
        self.lora_B = nn.ParameterDict({})
        self.lora_dropout = nn.ModuleDict({})

        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        # Expose base-layer shape attributes so downstream code and tests can query the wrapper directly.
        if hasattr(base_layer, "in_features"):
            self.in_features = base_layer.in_features
        if hasattr(base_layer, "out_features"):
            self.out_features = base_layer.out_features

    def _num_trainable(self, num_params: int, sparsity: float) -> int:
        """Number of trainable sparse entries at the given sparsity ratio."""
        return int(num_params * (1 - sparsity))

    def update_layer(self, adapter_name: str, config: SupertuningConfig, **kwargs):
        """Allocate the sparse support and (optionally) LoRA parameters for a new adapter.

        Indices are selected by weight magnitude (paper's best single-mechanism config; data-free). External callers
        can override the magnitude-computed indices via ``set_precomputed_indices`` on the outer ``SupertuningModel``.
        """
        base_layer = self.get_base_layer()
        weight = base_layer.weight.data  # [out_features, in_features]

        # Magnitude scoring — data-free, matches the paper's best-reported single-mechanism configuration.
        scores = weight.abs().flatten()
        num_trainable = self._num_trainable(scores.numel(), config.sparsity)
        largest = config.selection_direction == "top"
        _, indices = torch.topk(scores, k=num_trainable, largest=largest)
        indices = indices.to(dtype=torch.int32, device=weight.device)

        self.supertuning_indices[adapter_name] = indices
        if config.init_weights:
            values = torch.zeros(num_trainable, dtype=torch.float32, device=weight.device)
        else:
            # Non-identity init — used by tests to exercise a non-trivial adapter.
            values = torch.randn(num_trainable, dtype=torch.float32, device=weight.device)
        self.supertuning_values[adapter_name] = nn.Parameter(values)

        # Supra: optionally allocate LoRA A/B parameters.
        if config.r is not None:
            in_features, out_features = base_layer.in_features, base_layer.out_features
            self.supertuning_rank[adapter_name] = config.r
            self.supertuning_lora_alpha[adapter_name] = float(config.lora_alpha)
            self.lora_A[adapter_name] = nn.Parameter(
                torch.empty(config.r, in_features, dtype=torch.float32, device=weight.device)
            )
            self.lora_B[adapter_name] = nn.Parameter(
                torch.zeros(out_features, config.r, dtype=torch.float32, device=weight.device)
            )
            # Standard LoRA init: Kaiming-uniform for A, zeros for B (zero contribution at step 0).
            nn.init.kaiming_uniform_(self.lora_A[adapter_name], a=math.sqrt(5))
            self.lora_dropout[adapter_name] = (
                nn.Dropout(p=config.lora_dropout) if config.lora_dropout > 0.0 else nn.Identity()
            )
        else:
            self.supertuning_rank[adapter_name] = 0
            self.supertuning_lora_alpha[adapter_name] = 0.0

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=config.inference_mode)

    def set_precomputed_indices(self, adapter_name: str, indices: torch.Tensor) -> None:
        """Override the magnitude-computed indices with a user-supplied tensor.

        The user takes responsibility for the number of indices matching the layer's sparsity budget. The trainable
        ``values`` are re-initialized to zero (identity update) to preserve the invariant that a freshly-swapped
        support starts with no effect on the forward pass.
        """
        if adapter_name not in self.supertuning_values:
            raise KeyError(f"Adapter {adapter_name!r} not found on this layer.")

        expected = self.supertuning_values[adapter_name].numel()
        if indices.numel() != expected:
            raise ValueError(
                f"Adapter {adapter_name!r}: expected {expected} indices to match the sparse budget, "
                f"got {indices.numel()}."
            )

        weight = self.get_base_layer().weight
        self.supertuning_indices[adapter_name] = indices.to(dtype=torch.int32, device=weight.device)
        # Reset values to zero — the new support may point to different weight entries, so any previously
        # trained ``values`` no longer correspond to the intended parameters.
        with torch.no_grad():
            self.supertuning_values[adapter_name].zero_()


class Linear(nn.Module, SupertuningLayer):
    """Supertuning applied to a ``torch.nn.Linear`` base layer."""

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        config: SupertuningConfig,
        **kwargs,
    ) -> None:
        super().__init__()
        SupertuningLayer.__init__(self, base_layer)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, config=config)

    def _lora_delta(self, adapter_name: str, dtype: torch.dtype) -> torch.Tensor:
        """Fold LoRA A/B into a dense ``[out_features, in_features]`` weight delta scaled by ``alpha/r``."""
        r = self.supertuning_rank[adapter_name]
        alpha = self.supertuning_lora_alpha[adapter_name]
        A = self.lora_A[adapter_name]
        B = self.lora_B[adapter_name]
        return ((alpha / r) * (B @ A)).to(dtype)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """Fold each active adapter's (sparse + LoRA) contribution into the base weight in place."""
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        base_layer = self.get_base_layer()
        for active_adapter in adapter_names:
            if active_adapter not in self.supertuning_values.keys():
                continue
            weight = base_layer.weight
            indices = self.supertuning_indices[active_adapter].to(torch.int64)
            values = self.supertuning_values[active_adapter].to(weight.dtype)

            if safe_merge:
                merged = weight.data.reshape(-1).scatter_add(0, indices, values).reshape_as(weight)
                if self.supertuning_rank[active_adapter] > 0:
                    merged = merged + self._lora_delta(active_adapter, weight.dtype)
                if not torch.isfinite(merged).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )
                weight.data = merged
            else:
                weight.data.reshape(-1).scatter_add_(0, indices, values)
                if self.supertuning_rank[active_adapter] > 0:
                    weight.data.add_(self._lora_delta(active_adapter, weight.dtype))
            self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """Reverse [`merge`] for all currently-merged adapters."""
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        base_layer = self.get_base_layer()
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter not in self.supertuning_values.keys():
                continue
            weight = base_layer.weight
            indices = self.supertuning_indices[active_adapter].to(torch.int64)
            values = self.supertuning_values[active_adapter].to(weight.dtype)
            if self.supertuning_rank[active_adapter] > 0:
                weight.data.sub_(self._lora_delta(active_adapter, weight.dtype))
            weight.data.reshape(-1).scatter_add_(0, indices, -values)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Frozen base weight + sparse support + (optional) LoRA delta, applied as a single linear op.

        The sparse composition goes through :class:`DensePlusSparseLinear` so the backward pass only reaches the
        trainable ``values``. In Supra mode the LoRA output is added on top of the sparse-composed linear output
        (equivalent by linearity to composing all three deltas into the effective weight).
        """
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)

        if self.merged:
            return self.base_layer(x, *args, **kwargs)

        active_adapters = [a for a in self.active_adapters if a in self.supertuning_values.keys()]
        if not active_adapters:
            return self.base_layer(x, *args, **kwargs)

        base_layer = self.get_base_layer()
        weight = base_layer.weight
        bias = base_layer.bias

        if len(active_adapters) == 1:
            adapter_name = active_adapters[0]
            result = DensePlusSparseLinear.apply(
                x, weight, self.supertuning_indices[adapter_name], self.supertuning_values[adapter_name], bias
            )
            if self.supertuning_rank[adapter_name] > 0:
                r = self.supertuning_rank[adapter_name]
                alpha = self.supertuning_lora_alpha[adapter_name]
                x_dropped = self.lora_dropout[adapter_name](x)
                lora_out = torch.nn.functional.linear(x_dropped, self.lora_A[adapter_name])
                lora_out = torch.nn.functional.linear(lora_out, self.lora_B[adapter_name])
                result = result + (alpha / r) * lora_out.to(result.dtype)
            return result

        # Multiple active adapters: combine their sparse supports on top of the frozen weight. The frozen weight
        # receives no gradient, so the native ``scatter_add`` autograd is sufficient here.
        dense_plus_sparse = weight.reshape(-1)
        for adapter_name in active_adapters:
            indices = self.supertuning_indices[adapter_name].to(torch.int64)
            values = self.supertuning_values[adapter_name].to(weight.dtype)
            dense_plus_sparse = dense_plus_sparse.scatter_add(0, indices, values)
        dense_plus_sparse = dense_plus_sparse.reshape_as(weight)
        result = torch.nn.functional.linear(x, dense_plus_sparse, bias)

        for adapter_name in active_adapters:
            if self.supertuning_rank[adapter_name] > 0:
                r = self.supertuning_rank[adapter_name]
                alpha = self.supertuning_lora_alpha[adapter_name]
                x_dropped = self.lora_dropout[adapter_name](x)
                lora_out = torch.nn.functional.linear(x_dropped, self.lora_A[adapter_name])
                lora_out = torch.nn.functional.linear(lora_out, self.lora_B[adapter_name])
                result = result + (alpha / r) * lora_out.to(result.dtype)

        return result
