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

"""Audited Transformers FFN layouts; routing stays in the native model.

The registry describes arithmetic and module boundaries, not model forwards. Remote implementations and unknown layouts
must be admitted with numerical tests.
"""

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PretrainedConfig, activations

from peft.tuners.tuners_utils import BaseTunerLayer


@dataclass(frozen=True)
class _FfnSpec:
    """Static arithmetic and export fields; dimensions and layouts remain instance-validated."""

    packed: bool = False
    has_gate: bool = True
    activation: str = "standard"
    width_field: str = "intermediate_size"
    shared_width_field: str | None = None
    shared_multiplier: str | None = None


@dataclass(frozen=True)
class _MoeSpec:
    """Native module boundaries and the audited rule for proving routing mass."""

    routing: str | None = None
    # None means discover optional shared_expert/shared_experts modules on the instance.
    shared_modules: tuple[str, ...] | None = None
    enabled_attribute: str | None = None


_FFN_SPECS = MappingProxyType(
    {
        "LlamaMLP": _FfnSpec(),
        "Qwen2MLP": _FfnSpec(),
        "Qwen3MLP": _FfnSpec(),
        "Qwen3MoeMLP": _FfnSpec(),
        "Qwen3NextMLP": _FfnSpec(shared_width_field="shared_expert_intermediate_size"),
        "Qwen3_5MLP": _FfnSpec(),
        "Qwen3_5MoeMLP": _FfnSpec(shared_width_field="shared_expert_intermediate_size"),
        "OlmoeMLP": _FfnSpec(),
        "DeepseekV2MLP": _FfnSpec(shared_width_field="moe_intermediate_size", shared_multiplier="n_shared_experts"),
        "DeepseekV3MLP": _FfnSpec(shared_width_field="moe_intermediate_size", shared_multiplier="n_shared_experts"),
        "DeepseekV32MLP": _FfnSpec(shared_width_field="moe_intermediate_size", shared_multiplier="n_shared_experts"),
        "DeepseekV4MLP": _FfnSpec(activation="clipped"),
        "KimiLinearMLP": _FfnSpec(shared_width_field="moe_intermediate_size", shared_multiplier="n_shared_experts"),
        "Glm4MoeMLP": _FfnSpec(shared_width_field="moe_intermediate_size", shared_multiplier="n_shared_experts"),
        "Glm4MoeLiteMLP": _FfnSpec(shared_width_field="moe_intermediate_size", shared_multiplier="n_shared_experts"),
        "GlmMoeDsaMLP": _FfnSpec(shared_width_field="moe_intermediate_size", shared_multiplier="n_shared_experts"),
        "Glm5NextTextMLP": _FfnSpec(
            activation="clipped", shared_width_field="moe_intermediate_size", shared_multiplier="n_shared_experts"
        ),
        "MiniMaxM3VLDenseMLP": _FfnSpec(
            activation="offset_gated",
            width_field="dense_intermediate_size",
            shared_width_field="shared_intermediate_size",
        ),
        "Gemma3MLP": _FfnSpec(),
        "Gemma4TextMLP": _FfnSpec(),
        "NemotronHMLP": _FfnSpec(has_gate=False, shared_width_field="moe_shared_expert_intermediate_size"),
        "GptOssExperts": _FfnSpec(packed=True, activation="offset_gated"),
        "Qwen3MoeExperts": _FfnSpec(packed=True, width_field="moe_intermediate_size"),
        "MixtralExperts": _FfnSpec(packed=True),
        "OlmoeExperts": _FfnSpec(packed=True),
        "MiniMaxExperts": _FfnSpec(packed=True),
        "MiniMaxM2Experts": _FfnSpec(packed=True),
        "MiniMaxM3VLExperts": _FfnSpec(packed=True, activation="offset_gated"),
        "DeepseekV2Experts": _FfnSpec(packed=True, width_field="moe_intermediate_size"),
        "DeepseekV3Experts": _FfnSpec(packed=True, width_field="moe_intermediate_size"),
        "DeepseekV32Experts": _FfnSpec(packed=True, width_field="moe_intermediate_size"),
        "DeepseekV4Experts": _FfnSpec(packed=True, activation="clipped"),
        "KimiLinearExperts": _FfnSpec(packed=True, width_field="moe_intermediate_size"),
        "Glm4MoeExperts": _FfnSpec(packed=True, width_field="moe_intermediate_size"),
        "Glm4MoeLiteExperts": _FfnSpec(packed=True, width_field="moe_intermediate_size"),
        "GlmMoeDsaExperts": _FfnSpec(packed=True, width_field="moe_intermediate_size"),
        "Glm5NextTextExperts": _FfnSpec(packed=True, activation="clipped", width_field="moe_intermediate_size"),
        "Qwen3NextExperts": _FfnSpec(packed=True, width_field="moe_intermediate_size"),
        "Qwen3_5MoeExperts": _FfnSpec(packed=True, width_field="moe_intermediate_size"),
        "Gemma4TextExperts": _FfnSpec(packed=True, width_field="moe_intermediate_size"),
        "NemotronHExperts": _FfnSpec(packed=True, has_gate=False, width_field="moe_intermediate_size"),
    }
)
_MOE_SPECS = MappingProxyType(
    {
        "GptOssMLP": _MoeSpec(routing="unit"),
        "Qwen3MoeSparseMoeBlock": _MoeSpec(routing="topk"),
        "MixtralSparseMoeBlock": _MoeSpec(routing="unit"),
        "OlmoeSparseMoeBlock": _MoeSpec(routing="topk"),
        "MiniMaxSparseMoeBlock": _MoeSpec(routing="unit"),
        "MiniMaxM2SparseMoeBlock": _MoeSpec(routing="unit"),
        "MiniMaxM3VLSparseMoeBlock": _MoeSpec(),
        "DeepseekV2Moe": _MoeSpec(),
        "DeepseekV3MoE": _MoeSpec(),
        "DeepseekV32MoE": _MoeSpec(),
        "DeepseekV4SparseMoeBlock": _MoeSpec(),
        "KimiLinearMoE": _MoeSpec(),
        "Glm4MoeMoE": _MoeSpec(),
        "Glm4MoeLiteMoE": _MoeSpec(),
        "GlmMoeDsaMoE": _MoeSpec(),
        "Glm5NextTextMoE": _MoeSpec(),
        "Qwen3NextSparseMoeBlock": _MoeSpec(),
        "Qwen3_5MoeSparseMoeBlock": _MoeSpec(),
        "NemotronHMoE": _MoeSpec(),
        "Gemma4TextDecoderLayer": _MoeSpec(shared_modules=("mlp",), enabled_attribute="enable_moe_block"),
    }
)
# Preserve the classification helpers without maintaining separate architecture lists.
DENSE = {name for name, spec in _FFN_SPECS.items() if not spec.packed}
EXPERTS = {name for name, spec in _FFN_SPECS.items() if spec.packed}
FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
ELEMENTWISE_ACTIVATIONS = {
    nn.CELU,
    nn.ELU,
    nn.GELU,
    nn.Hardshrink,
    nn.Hardsigmoid,
    nn.Hardswish,
    nn.Hardtanh,
    nn.Identity,
    nn.LeakyReLU,
    nn.Mish,
    nn.ReLU,
    nn.ReLU6,
    nn.SELU,
    nn.Sigmoid,
    nn.SiLU,
    nn.Softplus,
    nn.Softshrink,
    nn.Softsign,
    nn.Tanh,
    nn.Tanhshrink,
    nn.Threshold,
}
TRANSFORMERS_ELEMENTWISE_ACTIVATIONS = (
    "AccurateGELUActivation",
    "ClippedGELUActivation",
    "FastGELUActivation",
    "GELUActivation",
    "GELUTanh",
    "LaplaceActivation",
    "LinearActivation",
    "MishActivation",
    "NewGELUActivation",
    "PytorchGELUTanh",
    "QuickGELUActivation",
    "ReLUSquaredActivation",
    "SiLUActivation",
    "SqrtSoftplusActivation",
)


def is_elementwise_activation(activation: Callable[..., torch.Tensor]) -> bool:
    """Admit audited implementations; names, subclasses and probes are not proofs.

    Structural merging requires act(cat(a, b)) == cat(act(a), act(b)). Parameter-free softmax, normalization and
    stochastic activations violate that contract. An arbitrary callable must be audited before it is admitted.
    """
    if type(activation) in ELEMENTWISE_ACTIVATIONS or any(
        type(activation) is getattr(activations, name, None) for name in TRANSFORMERS_ELEMENTWISE_ACTIVATIONS
    ):
        return True
    return any(
        activation is function
        for function in (
            F.celu,
            F.elu,
            F.gelu,
            F.hardshrink,
            F.hardsigmoid,
            F.hardswish,
            F.hardtanh,
            F.leaky_relu,
            F.mish,
            F.relu,
            F.relu6,
            F.selu,
            F.sigmoid,
            F.silu,
            F.softplus,
            F.softshrink,
            F.softsign,
            F.tanh,
            F.tanhshrink,
            F.threshold,
            torch.relu,
            torch.sigmoid,
            torch.tanh,
        )
    )


def unwrap(module: nn.Module) -> nn.Module:
    return module.get_base_layer() if isinstance(module, BaseTunerLayer) else module


def native_name(module: nn.Module) -> str:
    cls = type(unwrap(module))
    return cls.__name__ if cls.__module__.startswith("transformers.models.") else ""


def is_moe(module: nn.Module) -> bool:
    spec = _MOE_SPECS.get(native_name(module))
    return spec is not None and (spec.enabled_attribute is None or getattr(module, spec.enabled_attribute))


def shared_names(module: nn.Module) -> list[str]:
    spec = _MOE_SPECS.get(native_name(module))
    if spec is not None and spec.shared_modules is not None:
        return list(spec.shared_modules)
    return [name for name in ("shared_expert", "shared_experts") if isinstance(getattr(module, name, None), nn.Module)]


def public_merge_scale(block: nn.Module) -> float | None:
    """Prove constant routing mass; never infer it from one batch of tokens."""
    spec = _MOE_SPECS.get(native_name(block))
    if spec is None:
        return None
    if spec.routing == "unit":
        return 1.0
    if spec.routing == "topk":
        router = block.gate
        if router.norm_topk_prob or router.top_k == router.num_experts:
            return 1.0
    return None


@dataclass(frozen=True)
class MlpStructure:
    input_dim: int
    output_dim: int
    intermediate_dim: int
    num_experts: int = 1
    packed: bool = False
    fused: bool = False
    has_gate: bool = True
    transposed: bool = False
    interleaved: bool = False
    kind: str = "standard"
    act_fn: Callable[[torch.Tensor], torch.Tensor] | None = None
    alpha: float = 1.0
    limit: float | None = None

    def activate(self, gate: torch.Tensor | None, up: torch.Tensor) -> torch.Tensor:
        if not self.has_gate:
            return self.act_fn(up)
        if self.limit is not None:
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
        if self.kind == "offset_gated":
            return gate * torch.sigmoid(self.alpha * gate) * (up + 1)
        return self.act_fn(gate) * up


def logical_shape(parameter: torch.Tensor) -> tuple[int, ...]:
    """Read the unpartitioned shape without gathering ZeRO-3 parameters."""
    return tuple(getattr(parameter, "ds_shape", parameter.shape))


def describe_mlp(module: nn.Module) -> MlpStructure:
    module = unwrap(module)
    if is_moe(module):
        return describe_mlp(module.experts)
    name = native_name(module)
    architecture = _FFN_SPECS.get(name)
    if architecture is None:
        # Explicit custom targets opt into the standard FFN arithmetic contract.
        # Unknown Transformers classes can have extra operations; do not guess.
        if name or not all(hasattr(module, key) for key in ("up_proj", "down_proj", "act_fn")):
            raise ValueError(f"Unsupported Cortex target: {type(module).__name__}.")
        architecture = _FfnSpec(has_gate=hasattr(module, "gate_proj"))
    packed = architecture.packed
    if packed and any(type(getattr(module, attr, None)) is not bool for attr in ("is_transposed", "is_concatenated")):
        raise ValueError(
            f"{name} lacks the supported packed expert layout metadata (is_transposed/is_concatenated). "
            "Older Transformers expert implementations are unsupported; see the Cortex version compatibility notes."
        )
    if packed and getattr(module, "has_post_expert_norm", False):
        raise ValueError("Cortex does not support normalization inside individual expert outputs.")
    has_gate = architecture.has_gate
    fused = hasattr(module, "gate_up_proj")
    transposed = packed and getattr(module, "is_transposed", False)
    interleaved = packed and not getattr(module, "is_concatenated", True)
    projection_names = ["gate_up_proj" if fused else "up_proj", "down_proj"]
    if has_gate and not fused:
        projection_names.append("gate_proj")
    weights = {}
    for attr in projection_names:
        projection = getattr(module, attr)
        if packed:
            if type(projection) is not nn.Parameter or len(logical_shape(projection)) != 3:
                raise ValueError("Expected three-dimensional packed expert parameters.")
            weight = projection
        else:
            if type(projection) is not nn.Linear:
                raise ValueError("Only ordinary floating point nn.Linear projections are supported.")
            weight = projection.weight
        if weight.dtype not in FLOAT_DTYPES or type(weight.data) is not torch.Tensor:
            raise ValueError(
                "Only ordinary floating point tensors are supported; "
                "quantized and tensor-parallel layouts are unsupported."
            )
        shape = logical_shape(weight)
        if len(shape) != (3 if packed else 2):
            raise ValueError("Unexpected projection shape.")
        weights[attr] = (*shape[:-2], shape[-1], shape[-2]) if transposed else shape
    up, down = weights[projection_names[0]], weights["down_proj"]
    intermediate, input_dim, output_dim = down[-1], up[-1], down[-2]
    count = up[0] if packed else 1
    if up[-2] != intermediate * (2 if fused else 1):
        raise ValueError("Inconsistent up/down intermediate dimensions.")
    if has_gate and not fused and weights["gate_proj"] != up:
        raise ValueError("Inconsistent gate/up dimensions.")
    if packed and (count != module.num_experts or down[0] != count):
        raise ValueError("Sharded expert parameters are not supported.")
    act_fn = getattr(module, "act_fn", None)
    limit, alpha = None, 1.0
    kind = "offset_gated" if architecture.activation == "offset_gated" else "standard"
    if architecture.activation in {"clipped", "offset_gated"}:
        limit = getattr(module, "swiglu_limit", getattr(module, "limit", None))
        if not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit <= 0:
            raise ValueError(f"{name} must expose a finite positive SwiGLU clipping limit.")
        if architecture.activation == "offset_gated":
            alpha = getattr(module, "swiglu_alpha", getattr(module, "alpha", None))
            if not isinstance(alpha, (int, float)) or not math.isfinite(alpha):
                raise ValueError(f"{name} must expose a finite SwiGLU alpha.")
        act_fn = act_fn or F.silu
    if not callable(act_fn) or (isinstance(act_fn, nn.Module) and next(act_fn.parameters(), None) is not None):
        raise ValueError("The MLP must expose a parameter-free activation.")
    if not is_elementwise_activation(act_fn):
        raise ValueError(
            "The MLP must expose an audited deterministic elementwise activation, such as nn.SiLU or nn.GELU. "
            "Cross-neuron operations, stochastic activations and unaudited custom callables cannot be "
            "structurally merged. Add a validated structure implementation before targeting this FFN."
        )
    return MlpStructure(
        input_dim=input_dim,
        output_dim=output_dim,
        intermediate_dim=intermediate,
        num_experts=count,
        packed=packed,
        fused=fused,
        has_gate=has_gate,
        transposed=transposed,
        interleaved=interleaved,
        kind=kind,
        act_fn=act_fn,
        alpha=alpha,
        limit=limit,
    )


def read_components(
    module: nn.Module, include_down: bool = True, *, spec: MlpStructure | None = None
) -> dict[str, torch.Tensor]:
    """Read native FFN parameters in a common layout without copying or detaching them.

    Args:
        module (`torch.nn.Module`):
            The unwrapped dense FFN or packed expert module.
        include_down (`bool`, *optional*, defaults to `True`):
            Include the down projection and its bias, when present.
        spec (`MlpStructure`, *optional*):
            A structure already validated for this module in the same operation. Otherwise, call `describe_mlp`.

    Returns:
        `dict[str, torch.Tensor]`: Contains `up`, `gate` for gated FFNs, and `down` when `include_down=True`. Dense
        `up`/`gate` weights have shape `[intermediate, input]`; `down` has shape `[output, intermediate]`. Packed
        experts add a leading `num_experts` dimension. Present biases use the corresponding `*_bias` key and omit the
        weight's last dimension: `[intermediate]` for `up_bias`/`gate_bias`, `[output]` for `down_bias`, with the same
        leading expert dimension for packed layouts. Missing biases are omitted. Transposed weights are normalized to
        this layout, and fused or interleaved gate/up projections are split into separate `gate` and `up` entries.
        Values share storage with the native parameters and may be non-contiguous views; clone them before modifying
        them if the native parameters must remain unchanged.
    """
    spec = describe_mlp(module) if spec is None else spec
    if spec.fused:
        projection_names = ("gate_up", "down")
    elif spec.has_gate:
        projection_names = ("gate", "up", "down")
    else:
        projection_names = ("up", "down")
    components = {}
    for kind in projection_names:
        if kind == "down" and not include_down:
            continue
        projection = getattr(module, kind + "_proj")
        weight = projection if spec.packed else projection.weight
        weight = weight.transpose(-1, -2) if spec.transposed else weight
        bias = getattr(module, kind + "_proj_bias", None) if spec.packed else projection.bias
        if kind == "gate_up":
            gate, up = (weight[..., ::2, :], weight[..., 1::2, :]) if spec.interleaved else weight.chunk(2, dim=-2)
            components.update(gate=gate, up=up)
            if bias is not None:
                gb, ub = (bias[..., ::2], bias[..., 1::2]) if spec.interleaved else bias.chunk(2, dim=-1)
                components.update(gate_bias=gb, up_bias=ub)
        else:
            components[kind] = weight
            if bias is not None:
                components[kind + "_bias"] = bias
    return components


def width_binding(module: nn.Module, path: str, config: PretrainedConfig | None) -> tuple[PretrainedConfig, str, int]:
    """Return the field and multiplier within the owning serialized config."""
    module = unwrap(module)
    name = native_name(module)
    config = config.get_text_config() if hasattr(config, "get_text_config") else config
    shared = path.endswith(("shared_expert", "shared_experts"))
    architecture = _FFN_SPECS.get(name, _FfnSpec())
    field, multiplier = architecture.width_field, 1
    if shared and architecture.shared_width_field is not None:
        field = architecture.shared_width_field
        if architecture.shared_multiplier is not None:
            multiplier = getattr(config, architecture.shared_multiplier)
    if name == "Gemma4TextMLP":
        if config.use_double_wide_mlp:
            match = re.search(r"(?:^|\.)layers\.(\d+)\.mlp$", path)
            if match is None:
                raise ValueError(f"Cannot determine Gemma4 layer index for {path}.")
            start = config.num_hidden_layers - config.num_kv_shared_layers
            multiplier = 2 if int(match[1]) >= start > 0 else 1
    if getattr(config, field, None) is None or not multiplier:
        raise ValueError(f"Cannot determine export width for {path}: {field}.")
    return config, field, multiplier
