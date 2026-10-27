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

"""Initialization shared by dense, shared, and routed FFN branches."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from torch import nn

from peft.utils.integrations import gather_params_ctx

from .structure import logical_shape, read_components


if TYPE_CHECKING:
    from .config import CortexConfig
    from .layer import CortexLayer


def _require_materialized(parameters: Iterable[nn.Parameter]) -> None:
    if any(p.is_meta or tuple(p.shape) != logical_shape(p) for p in parameters):
        raise ValueError(
            "Initialization requires materialized parameters. For ZeRO-3, configure the Transformers "
            "DeepSpeed integration before loading the model so PEFT can gather its parameters."
        )


def _indices(source_width: int, width: int, device: torch.device) -> torch.Tensor:
    if width <= source_width:
        return torch.randperm(source_width, device=device)[:width]
    return torch.randint(source_width, (width,), device=device)


def _sample_branch(layer: CortexLayer, projections: dict[str, nn.Module], per_expert: bool, noise: float) -> None:
    base = layer.get_base_layer()
    # Only gather gate/up parameters. In particular, do not gather a large down
    # projection whose values are not used by sampling.
    source_params = [p for name, p in base.named_parameters() if not name.startswith("down_proj")]
    with gather_params_ctx(source_params, modifier_rank=None):
        _require_materialized(source_params)
        source = read_components(base, include_down=False)
        spec = layer.structure
        width = projections["up"].weight.shape[-2]
        for expert in range(spec.num_experts if per_expert else 1):
            pool_size = spec.intermediate_dim * (spec.num_experts if spec.packed and not per_expert else 1)
            idx = _indices(pool_size, width, source["up"].device)
            for kind in ("gate", "up") if spec.has_gate else ("up",):
                projection = projections[kind]
                for suffix, destination in (("", projection.weight), ("_bias", projection.bias)):
                    key = kind + suffix
                    if destination is None or key not in source:
                        continue
                    value = source[key]
                    dim = -1 if suffix else -2
                    if per_expert:
                        value = value[expert].index_select(dim, idx)
                        destination = destination[expert]
                    elif spec.packed:
                        # A shared channel inherits a real paired gate/up neuron,
                        # not the mean of unrelated channels across experts.
                        value = value[idx // spec.intermediate_dim, idx % spec.intermediate_dim]
                    else:
                        value = value.index_select(dim, idx)
                    destination.copy_(value)
                    if not suffix and noise:
                        destination.add_(torch.randn_like(destination), alpha=noise)


@torch.no_grad()
def initialize_branch(
    layer: CortexLayer, name: str, config: CortexConfig, prefix: str = "", per_expert: bool = False
) -> None:
    """Initialize an additive branch; normal initialization preserves the output for finite gate/up features.

    Except in the testing mode `init_weights=False`, down weights and down bias are zero for the initial no-op. Gate/up
    initialization still affects the first down-weight gradient, even when the initial output is unchanged.
    """
    kinds = ("gate", "up", "down") if layer.structure.has_gate else ("up", "down")
    projections = {kind: getattr(layer, f"cortex_{prefix}{kind}")[name] for kind in kinds}
    parameters = [p for projection in projections.values() for p in projection.parameters()]
    # low_cpu_mem_usage creates adapter tensors on meta, then loads checkpoint
    # values. Do not read base weights or consume RNG in that loading path.
    if any(p.is_meta for p in parameters):
        return
    with gather_params_ctx(parameters, modifier_rank=0):
        _require_materialized(parameters)
        for kind, projection in projections.items():
            if kind == "down" and config.init_weights is not False:
                nn.init.zeros_(projection.weight)
            elif config.init_weights != "sample":
                # A packed [E, out, in] tensor must use fan_in=in, not out*in.
                weights = projection.weight.unbind(0) if per_expert else (projection.weight,)
                for weight in weights:
                    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            if projection.bias is not None:
                if (
                    kind != "down"
                    and config.init_weights in ("random", True)
                    and config.adapter_bias_init == "uniform"
                ):
                    bound = 1 / math.sqrt(projection.weight.shape[-1])
                    nn.init.uniform_(projection.bias, -bound, bound)
                else:
                    # Down/testing biases and missing source biases during sampling stay zero.
                    nn.init.zeros_(projection.bias)
        if config.init_weights == "sample":
            _sample_branch(layer, projections, per_expert, config.init_noise_std)
