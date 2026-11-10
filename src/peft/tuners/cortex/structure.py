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

The registry describes known arithmetic and module boundaries. Native modules outside it may use the standard FFN
conventions below; their inferred layouts need numerical validation.
"""

import math
import operator
import re
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import torch
from torch import fx, nn
from torch.nn import functional as F
from transformers import PretrainedConfig, activations

from peft.tuners.tuners_utils import BaseTunerLayer, _is_key_under_prefixes

from .config import CortexFfnSpec, CortexMoeSpec


_FFN_SPECS = MappingProxyType(
    {
        "LlamaMLP": CortexFfnSpec(),
        "Qwen2MLP": CortexFfnSpec(),
        "Qwen3MLP": CortexFfnSpec(),
        "Qwen3MoeMLP": CortexFfnSpec(),
        "Qwen3NextMLP": CortexFfnSpec(shared_width_field="shared_expert_intermediate_size"),
        "Qwen3_5MLP": CortexFfnSpec(),
        "Qwen3_5MoeMLP": CortexFfnSpec(shared_width_field="shared_expert_intermediate_size"),
        "OlmoeMLP": CortexFfnSpec(),
        "DeepseekV2MLP": CortexFfnSpec(
            shared_width_field="moe_intermediate_size", shared_multiplier="n_shared_experts"
        ),
        "DeepseekV3MLP": CortexFfnSpec(
            shared_width_field="moe_intermediate_size", shared_multiplier="n_shared_experts"
        ),
        "DeepseekV32MLP": CortexFfnSpec(
            shared_width_field="moe_intermediate_size", shared_multiplier="n_shared_experts"
        ),
        "DeepseekV4MLP": CortexFfnSpec(activation="clipped"),
        "KimiLinearMLP": CortexFfnSpec(
            shared_width_field="moe_intermediate_size", shared_multiplier="n_shared_experts"
        ),
        "Glm4MoeMLP": CortexFfnSpec(shared_width_field="moe_intermediate_size", shared_multiplier="n_shared_experts"),
        "Glm4MoeLiteMLP": CortexFfnSpec(
            shared_width_field="moe_intermediate_size", shared_multiplier="n_shared_experts"
        ),
        "GlmMoeDsaMLP": CortexFfnSpec(
            shared_width_field="moe_intermediate_size", shared_multiplier="n_shared_experts"
        ),
        "Glm5NextTextMLP": CortexFfnSpec(
            activation="clipped", shared_width_field="moe_intermediate_size", shared_multiplier="n_shared_experts"
        ),
        "MiniMaxM3VLDenseMLP": CortexFfnSpec(
            activation="offset_gated",
            width_field="dense_intermediate_size",
            shared_width_field="shared_intermediate_size",
        ),
        "Gemma3MLP": CortexFfnSpec(),
        "Gemma4TextMLP": CortexFfnSpec(),
        "NemotronHMLP": CortexFfnSpec(has_gate=False, shared_width_field="moe_shared_expert_intermediate_size"),
        "GptOssExperts": CortexFfnSpec(packed=True, activation="offset_gated"),
        "Qwen3MoeExperts": CortexFfnSpec(packed=True, width_field="moe_intermediate_size"),
        "MixtralExperts": CortexFfnSpec(packed=True),
        "OlmoeExperts": CortexFfnSpec(packed=True),
        "MiniMaxExperts": CortexFfnSpec(packed=True),
        "MiniMaxM2Experts": CortexFfnSpec(packed=True),
        "MiniMaxM3VLExperts": CortexFfnSpec(packed=True, activation="offset_gated"),
        "DeepseekV2Experts": CortexFfnSpec(packed=True, width_field="moe_intermediate_size"),
        "DeepseekV3Experts": CortexFfnSpec(packed=True, width_field="moe_intermediate_size"),
        "DeepseekV32Experts": CortexFfnSpec(packed=True, width_field="moe_intermediate_size"),
        "DeepseekV4Experts": CortexFfnSpec(packed=True, activation="clipped"),
        "KimiLinearExperts": CortexFfnSpec(packed=True, width_field="moe_intermediate_size"),
        "Glm4MoeExperts": CortexFfnSpec(packed=True, width_field="moe_intermediate_size"),
        "Glm4MoeLiteExperts": CortexFfnSpec(packed=True, width_field="moe_intermediate_size"),
        "GlmMoeDsaExperts": CortexFfnSpec(packed=True, width_field="moe_intermediate_size"),
        "Glm5NextTextExperts": CortexFfnSpec(packed=True, activation="clipped", width_field="moe_intermediate_size"),
        "Qwen3NextExperts": CortexFfnSpec(packed=True, width_field="moe_intermediate_size"),
        "Qwen3_5MoeExperts": CortexFfnSpec(packed=True, width_field="moe_intermediate_size"),
        "Gemma4TextExperts": CortexFfnSpec(packed=True, width_field="moe_intermediate_size"),
        "NemotronHExperts": CortexFfnSpec(packed=True, has_gate=False, width_field="moe_intermediate_size"),
    }
)
_MOE_SPECS = MappingProxyType(
    {
        "GptOssMLP": CortexMoeSpec(),
        "Qwen3MoeSparseMoeBlock": CortexMoeSpec(),
        "MixtralSparseMoeBlock": CortexMoeSpec(),
        "OlmoeSparseMoeBlock": CortexMoeSpec(),
        "MiniMaxSparseMoeBlock": CortexMoeSpec(),
        "MiniMaxM2SparseMoeBlock": CortexMoeSpec(),
        "MiniMaxM3VLSparseMoeBlock": CortexMoeSpec(),
        "DeepseekV2Moe": CortexMoeSpec(),
        "DeepseekV3MoE": CortexMoeSpec(),
        "DeepseekV32MoE": CortexMoeSpec(),
        "DeepseekV4SparseMoeBlock": CortexMoeSpec(),
        "KimiLinearMoE": CortexMoeSpec(),
        "Glm4MoeMoE": CortexMoeSpec(),
        "Glm4MoeLiteMoE": CortexMoeSpec(),
        "GlmMoeDsaMoE": CortexMoeSpec(),
        "Glm5NextTextMoE": CortexMoeSpec(),
        "Qwen3NextSparseMoeBlock": CortexMoeSpec(),
        "Qwen3_5MoeSparseMoeBlock": CortexMoeSpec(),
        "NemotronHMoE": CortexMoeSpec(),
        # With MoE enabled, the decoder layer owns both the shared .mlp and the routed .experts.
        "Gemma4TextDecoderLayer": CortexMoeSpec(shared_modules=("mlp",), enabled_attribute="enable_moe_block"),
    }
)
# These rules describe audited native router implementations. They are deliberately
# separate from user-provided module names: a structural hint cannot prove merge equivalence.
_ROUTING_RULES = MappingProxyType(
    {
        "GptOssMLP": "unit",
        "Qwen3MoeSparseMoeBlock": "topk",
        "MixtralSparseMoeBlock": "unit",
        "OlmoeSparseMoeBlock": "topk",
        "MiniMaxSparseMoeBlock": "unit",
        "MiniMaxM2SparseMoeBlock": "unit",
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


def text_backbone_prefixes(model: nn.Module) -> set[str] | None:
    """Limit inference on composite models to the subtree that owns the text configuration."""
    config = getattr(model, "config", None)
    if not hasattr(config, "get_text_config"):
        return None
    try:
        text_config = config.get_text_config()
    except ValueError:
        # A composite config may own several text backbones with no unambiguous default.
        return set()
    if text_config is config:
        return None

    roots = set()
    for path, module in model.named_modules():
        if getattr(unwrap(module), "config", None) is text_config and not _is_key_under_prefixes(path, roots):
            roots.add(path)
    return roots


def is_registered_layout(module: nn.Module) -> bool:
    name = native_name(module)
    return name in _FFN_SPECS or name in _MOE_SPECS


def explicit_spec(module: nn.Module, specs: dict | None):
    """Find a class description, preferring its fully qualified name over its short name."""
    if not specs:
        return None
    cls = type(unwrap(module))
    return specs.get(f"{cls.__module__}.{cls.__qualname__}", specs.get(cls.__name__))


def is_expert_candidate(module: nn.Module, ffn_specs: dict[str, CortexFfnSpec] | None = None) -> bool:
    spec = explicit_spec(module, ffn_specs)
    if spec is not None:
        return spec.packed
    name = native_name(module)
    if name in EXPERTS or name.endswith("Experts"):
        return True
    base = unwrap(module)
    return bool(name) and hasattr(base, "num_experts") and isinstance(getattr(base, "down_proj", None), nn.Parameter)


def is_ffn_candidate(module: nn.Module, ffn_specs: dict[str, CortexFfnSpec] | None = None) -> bool:
    """Recognize a native or explicitly described FFN boundary before validating its tensors."""
    if explicit_spec(module, ffn_specs) is not None:
        return True
    name = native_name(module)
    if name in _FFN_SPECS or is_expert_candidate(module):
        return True
    base = unwrap(module)
    return bool(name) and hasattr(base, "down_proj") and hasattr(base, "up_proj") and hasattr(base, "act_fn")


def is_moe(module: nn.Module, moe_specs: dict[str, CortexMoeSpec] | None = None) -> bool:
    module = unwrap(module)
    name = native_name(module)
    spec = explicit_spec(module, moe_specs) or _MOE_SPECS.get(name)
    if spec is not None:
        if spec.enabled_attribute is None:
            return True
        enabled = getattr(module, spec.enabled_attribute, None)
        if type(enabled) is not bool:
            raise ValueError(f"{type(module).__name__}.{spec.enabled_attribute} must be a boolean MoE enable flag.")
        return enabled

    # Keep an unfamiliar MoE intact during discovery, even if its expert layout will be rejected.
    # Descending into it could silently select a shared dense FFN for a routed-only adapter.
    return bool(name) and ("experts" in module._modules or name.lower().endswith(("moe", "moeblock", "sparsemlp")))


class UnresolvedSharedExpertsError(ValueError):
    """The native shared FFN could live outside the selected MoE block."""


def _validate_moe_children(module: nn.Module, spec: CortexMoeSpec) -> CortexMoeSpec:
    children = dict(module.named_children())
    experts = getattr(module, spec.experts, None)
    if isinstance(experts, nn.Module) and children.get(spec.experts) is not experts:
        raise ValueError(f"{type(module).__name__}.{spec.experts} must name a registered child, not an alias.")
    seen = set()
    for name in spec.shared_modules or ():
        shared = children.get(name)
        if name not in children:
            raise ValueError(f"{type(module).__name__}.{name} must name a registered shared FFN, not an alias.")
        if shared is experts or id(shared) in seen:
            raise ValueError("Routed and shared Cortex targets must refer to distinct physical modules.")
        seen.add(id(shared))
    return spec


def resolve_moe_spec(module: nn.Module, moe_specs: dict[str, CortexMoeSpec] | None = None) -> CortexMoeSpec | None:
    """Resolve branch ownership without turning unknown shared experts into a known absence."""
    module = unwrap(module)
    if not is_moe(module, moe_specs):
        return None
    declared = explicit_spec(module, moe_specs)
    registered = _MOE_SPECS.get(native_name(module))
    spec = declared or registered
    if spec is not None and spec.shared_modules is not None:
        return _validate_moe_children(module, spec)

    shared = tuple(
        name for name in ("shared_expert", "shared_experts") if isinstance(getattr(module, name, None), nn.Module)
    )
    if spec is not None:
        # The built-in table audits the complete native boundary. A user description
        # with shared_modules=None still needs an explicit () to establish absence.
        names = shared if declared is None else shared or None
        return _validate_moe_children(module, replace(spec, shared_modules=names))

    children = set(dict(module.named_children()))
    routers = {"gate", "router"}
    unknown = children - routers - {"experts", "shared_expert", "shared_experts", "shared_expert_gate"}
    if not children.intersection(routers) or unknown:
        raise ValueError(f"Cannot infer router and shared FFNs for {native_name(module)} from {sorted(children)}.")

    # Some decoders add a shared MLP next to this block. Seeing only a router
    # and experts here establishes the routed target, not the absence of a shared FFN.
    return _validate_moe_children(module, CortexMoeSpec(shared_modules=shared or None))


def shared_names(module: nn.Module, moe_specs: dict[str, CortexMoeSpec] | None = None) -> list[str]:
    spec = resolve_moe_spec(module, moe_specs)
    return list(spec.shared_modules or ()) if spec is not None else []


def resolve_target_branches(
    model: nn.Module,
    path: str,
    ffn_specs: dict[str, CortexFfnSpec] | None = None,
    moe_specs: dict[str, CortexMoeSpec] | None = None,
) -> tuple[str, str | None, tuple[str, ...] | None]:
    """Classify a target and locate its native branches, without applying exclusions or runtime checks.

    Return the kind (moe, routed, shared, or dense), the routed path, and the shared paths. For a MoE parent, shared
    paths of None mean unknown ownership; an empty tuple means no native shared FFN. Only parent targets depend on
    moe_target. Explicit leaves retain their own role, independently of any saved adapter roles.
    """
    module = model.get_submodule(path)
    spec = resolve_moe_spec(module, moe_specs)
    if spec is not None:
        prefix = path + "." if path else ""
        shared = None if spec.shared_modules is None else tuple(prefix + name for name in spec.shared_modules)
        return "moe", prefix + spec.experts, shared
    if is_expert_candidate(module, ffn_specs):
        return "routed", path, ()

    parent_path, _, child = path.rpartition(".")
    parent_spec = resolve_moe_spec(model.get_submodule(parent_path), moe_specs)
    if parent_spec is not None and child in (parent_spec.shared_modules or ()):
        return "shared", None, (path,)
    return "dense", None, ()


def public_merge_scale(block: nn.Module, moe_specs: dict[str, CortexMoeSpec] | None = None) -> float | None:
    """Prove constant routing mass from audited native rules, never from user field names."""
    block = unwrap(block)
    spec = resolve_moe_spec(block, moe_specs)
    if spec is None:
        return None
    name = native_name(block)
    declared = explicit_spec(block, moe_specs)
    if declared is not None and declared != _MOE_SPECS.get(name):
        return None
    routing = _ROUTING_RULES.get(name)
    if routing == "unit":
        return 1.0
    if routing == "topk":
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
    ffn_spec: CortexFfnSpec = field(default_factory=CortexFfnSpec)

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


class _FfnTracer(fx.Tracer):
    def is_leaf_module(self, module, qualname):
        # Keep projections and the activation as named calls. Never trace into their weights.
        return True


def _standard_dense_forward(module: nn.Module) -> bool:
    """Check the native forward symbolically, without reading weights or running example tokens."""
    if set(module._modules) - {"gate_proj", "up_proj", "down_proj", "act_fn"} or module._parameters or module._buffers:
        return False

    # A model loaded in eval mode may switch to a different formula during training.
    # Shallow copies let us trace both paths without copying weights or changing live modules.
    probe = copy(module)
    probe._modules = {name: copy(child) for name, child in module._modules.items()}
    for training in (False, True):
        for submodule in (probe, *probe.children()):
            submodule.training = training
        try:
            nodes = list(_FfnTracer().trace(probe).nodes)
        except (AttributeError, RuntimeError, TypeError, ValueError, fx.proxy.TraceError):
            return False
        if not _matches_dense_graph(nodes, has_gate=hasattr(module, "gate_proj")):
            return False
    return True


def _matches_dense_graph(nodes: list[fx.Node], has_gate: bool) -> bool:
    # Only admit down(act(up(x))) or down(act(gate(x)) * up(x)). The operand
    # order of the multiplication is immaterial, but which projection is activated matters.
    calls = {node.target: node for node in nodes if node.op == "call_module"}
    if not {"up_proj", "down_proj", "act_fn"} <= calls.keys():
        return False
    x, out = nodes[0], nodes[-1]
    up, down, act = calls["up_proj"], calls["down_proj"], calls["act_fn"]
    if x.op != "placeholder" or up.args != (x,) or out.args != (down,):
        return False
    if not has_gate:
        return len(nodes) == 5 and act.args == (up,) and down.args == (act,)

    gate = calls.get("gate_proj")
    product = next((node for node in nodes if node.op == "call_function" and node.target is operator.mul), None)
    return (
        len(nodes) == 7
        and gate is not None
        and gate.args == (x,)
        and act.args == (gate,)
        and product is not None
        and set(product.args) == {act, up}
        and down.args == (product,)
    )


def _infer_ffn_spec(module: nn.Module) -> CortexFfnSpec:
    name = native_name(module)
    if not name:
        # Explicit custom targets opt into the standard FFN arithmetic contract.
        if all(hasattr(module, attr) for attr in ("up_proj", "down_proj", "act_fn")):
            return CortexFfnSpec(has_gate=hasattr(module, "gate_proj"))
    else:
        # FX treats projections/activations as leaves and won't see their local hooks.
        # An extra operation on the native output could change the merged function.
        for child in module.modules():
            if child._forward_pre_hooks or child._forward_hooks or "forward" in vars(child):
                raise ValueError(f"Cannot infer {name} with forward hooks or a replaced forward.")

    if is_expert_candidate(module):
        # The native Transformers packed-expert decorator supplies layout flags and a default
        # gate function. A custom gate (e.g. clipped SwiGLU) needs a dedicated arithmetic rule.
        gate = getattr(getattr(module, "_apply_gate", None), "__func__", None)
        if (
            gate is getattr(type(module), "_apply_gate", None)
            and getattr(gate, "__module__", None) == "transformers.integrations.moe"
            and getattr(gate, "__name__", None) == "_default_apply_gate"
            and type(getattr(module, "has_gate", None)) is bool
            and isinstance(getattr(module, "down_proj", None), nn.Parameter)
            and not any(hasattr(module, attr) for attr in ("limit", "alpha", "swiglu_limit", "swiglu_alpha"))
        ):
            weights = {"gate_up_proj" if module.has_gate else "up_proj", "down_proj"}
            allowed = weights | {attr + "_bias" for attr in weights}
            if (
                weights <= set(module._parameters) <= allowed
                and set(module._modules) <= {"act_fn"}
                and not module._buffers
            ):
                return CortexFfnSpec(packed=True, has_gate=module.has_gate)
    elif is_ffn_candidate(module) and _standard_dense_forward(module):
        return CortexFfnSpec(has_gate=hasattr(module, "gate_proj"))

    raise ValueError(f"Unsupported Cortex target: {type(module).__name__}. Cannot infer a standard FFN layout.")


def resolve_ffn_spec(module: nn.Module, ffn_specs: dict[str, CortexFfnSpec] | None = None) -> CortexFfnSpec:
    module = unwrap(module)
    spec = explicit_spec(module, ffn_specs) or _FFN_SPECS.get(native_name(module))
    return _infer_ffn_spec(module) if spec is None else spec


def describe_mlp(
    module: nn.Module,
    ffn_specs: dict[str, CortexFfnSpec] | None = None,
    *,
    spec: CortexFfnSpec | None = None,
) -> MlpStructure:
    module = unwrap(module)
    if spec is None and is_moe(module):
        return describe_mlp(module.experts, ffn_specs)
    name = native_name(module) or type(module).__name__
    architecture = resolve_ffn_spec(module, ffn_specs) if spec is None else spec
    packed = architecture.packed
    if packed and any(type(getattr(module, attr, None)) is not bool for attr in ("is_transposed", "is_concatenated")):
        raise ValueError(
            f"{name} lacks the supported packed expert layout metadata (is_transposed/is_concatenated). "
            "Older Transformers expert implementations are unsupported; see the Cortex version compatibility notes."
        )
    if packed and getattr(module, "has_post_expert_norm", False):
        raise ValueError("Cortex does not support normalization inside individual expert outputs.")
    has_gate = architecture.has_gate
    fused = architecture.gate_up_proj is not None and hasattr(module, architecture.gate_up_proj)
    if fused and not has_gate:
        raise ValueError("A fused gate/up projection requires has_gate=True.")
    transposed = packed and getattr(module, "is_transposed", False)
    interleaved = packed and not getattr(module, "is_concatenated", True)
    kinds = ["gate_up" if fused else "up", "down"]
    if has_gate and not fused:
        kinds.append("gate")
    weights = {}
    parameters = set()
    for kind in kinds:
        attr = getattr(architecture, kind + "_proj")
        projection = getattr(module, attr, None)
        if packed:
            if type(projection) is not nn.Parameter or len(logical_shape(projection)) != 3:
                raise ValueError(f"Expected a three-dimensional packed expert parameter at {name}.{attr}.")
            weight = projection
        else:
            if type(projection) is not nn.Linear:
                raise ValueError(
                    f"Only ordinary floating point nn.Linear projections are supported; found {name}.{attr}."
                )
            weight = projection.weight
        if id(weight) in parameters:
            raise ValueError(
                "Cortex projections must have distinct weight parameters; tied projections cannot be merged."
            )
        parameters.add(id(weight))
        if weight.dtype not in FLOAT_DTYPES or type(weight.data) is not torch.Tensor:
            raise ValueError(
                "Only ordinary floating point tensors are supported; "
                "quantized and tensor-parallel layouts are unsupported."
            )
        shape = logical_shape(weight)
        if len(shape) != (3 if packed else 2):
            raise ValueError("Unexpected projection shape.")
        weights[kind] = (*shape[:-2], shape[-1], shape[-2]) if transposed else shape
    up, down = weights[kinds[0]], weights["down"]
    intermediate, input_dim, output_dim = down[-1], up[-1], down[-2]
    count = up[0] if packed else 1
    if up[-2] != intermediate * (2 if fused else 1):
        raise ValueError("Inconsistent up/down intermediate dimensions.")
    if has_gate and not fused and weights["gate"] != up:
        raise ValueError("Inconsistent gate/up dimensions.")
    if packed and (count != getattr(module, "num_experts", None) or down[0] != count):
        raise ValueError("Sharded expert parameters are not supported.")
    act_fn = getattr(module, architecture.act_fn, None)
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
        ffn_spec=architecture,
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
        attr = getattr(spec.ffn_spec, kind + "_proj")
        projection = getattr(module, attr)
        weight = projection if spec.packed else projection.weight
        weight = weight.transpose(-1, -2) if spec.transposed else weight
        bias = getattr(module, attr + "_bias", None) if spec.packed else projection.bias
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


def width_binding(
    module: nn.Module,
    path: str,
    config: PretrainedConfig | None,
    ffn_specs: dict[str, CortexFfnSpec] | None = None,
    *,
    shared: bool | None = None,
) -> tuple[PretrainedConfig, tuple[str, ...], int]:
    """Find width fields on the owning config. Inferred layouts may have an ambiguous binding."""
    module = unwrap(module)
    name = native_name(module)
    config = config.get_text_config() if hasattr(config, "get_text_config") else config
    if shared is None:
        shared = path.rsplit(".", 1)[-1] in ("shared_expert", "shared_experts")
    architecture = explicit_spec(module, ffn_specs) or _FFN_SPECS.get(name)
    if architecture is None:
        # Use constructor conventions, not a search for an attribute with the same numeric value.
        if shared:
            names = (
                "shared_expert_intermediate_size",
                "shared_intermediate_size",
                "moe_shared_expert_intermediate_size",
            )
        elif is_expert_candidate(module):
            names = ("moe_intermediate_size", "expert_intermediate_size")
            if not any(hasattr(config, field) for field in names):
                names = ("intermediate_size",)
        else:
            names = ("intermediate_size", "dense_intermediate_size", "ffn_dim")
        return config, tuple(field for field in names if getattr(config, field, None) is not None), 1

    field, multiplier = architecture.width_field, 1
    if shared and architecture.shared_width_field is not None:
        field = architecture.shared_width_field
        if architecture.shared_multiplier is not None:
            multiplier = getattr(config, architecture.shared_multiplier, None)
    if name == "Gemma4TextMLP" and explicit_spec(module, ffn_specs) is None:
        if config.use_double_wide_mlp:
            match = re.search(r"(?:^|\.)layers\.(\d+)\.mlp$", path)
            if match is None:
                raise ValueError(f"Cannot determine Gemma4 layer index for {path}.")
            start = config.num_hidden_layers - config.num_kv_shared_layers
            multiplier = 2 if int(match[1]) >= start > 0 else 1
    width = getattr(config, field, None)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (width, multiplier)):
        raise ValueError(
            f"Cannot determine export width for {path}: {field} and its multiplier must be positive integers."
        )
    return config, (field,), multiplier
