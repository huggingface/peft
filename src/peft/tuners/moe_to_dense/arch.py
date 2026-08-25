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

"""
Architecture-specific knowledge for the MoE-to-dense method.

Everything in this module is *best effort*: Transformers MoE implementations share a common structure (a "router"
module that selects experts per token and an "experts" module holding the weights of all experts as 3D tensors) but
differ in details such as attribute names, config field names, and whether the router returns logits or probabilities.
The `MoeArchSpec` registry captures these details for known architectures; unknown architectures fall back to
duck-typing heuristics which may or may not work.
"""

import copy
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

from peft.tuners.tuners_utils import BaseTunerLayer


# attributes set by transformers' `use_experts_implementation` decorator, describing the layout of the expert tensors
EXPERTS_LAYOUT_FLAGS: tuple[str, ...] = ("is_transposed", "is_concatenated", "has_bias", "has_gate")


@dataclass
class MoeArchSpec:
    """
    Best-effort description of how an MoE architecture is implemented in transformers.

    Args:
        router_attr (`str`, *optional*):
            Attribute name of the router module, which must be a sibling of the experts module. If `None`, the router
            is detected heuristically.
        router_returns_probs (`bool`, *optional*):
            Whether the first output of the router's forward call contains routing probabilities (`True`) or logits
            (`False`). If `None`, this is detected heuristically from the values.
        num_experts_fields (`tuple[str, ...]`):
            Config fields holding the number of (routed) experts, tried in order.
        top_k_fields (`tuple[str, ...]`):
            Config fields holding the number of experts activated per token, tried in order.
        expert_intermediate_fields (`tuple[str, ...]`):
            Config fields holding the intermediate size of a single expert, tried in order.
        expert_scale_fn (`Callable`, *optional*):
            Function that takes the router module and returns a per-expert scale (shape `[num_experts]`) that the
            router applies on top of the normalized routing weights, or `None` if there is no such scale.
        export (`str`):
            Export strategy used by `compress_and_unload`, either `"one_expert"` (generic, rebuilds the MoE layer with
            a single expert) or `"dense_mlp"` (replaces the whole MoE block by the architecture's dense MLP class).
        dense_mlp_class_name (`str`, *optional*):
            For `export="dense_mlp"`: name of the dense MLP class, resolved from the module defining the experts class.
            If provided, must be a Transformers module name. Its `__init__` must have the signature `(config,
            intermediate_size=None)`.
        dense_layers_field (`str`, *optional*):
            For `export="dense_mlp"`: config field holding the list of layer indices that use the dense MLP.
        dense_intermediate_field (`str`, *optional*):
            For `export="dense_mlp"`: config field holding the intermediate size of the dense MLP.
    """

    router_attr: str | None = None
    router_returns_probs: bool | None = None
    num_experts_fields: tuple[str, ...] = ("num_experts", "num_local_experts", "n_routed_experts")
    top_k_fields: tuple[str, ...] = ("num_experts_per_tok", "top_k_experts", "top_k", "experts_per_token")
    expert_intermediate_fields: tuple[str, ...] = ("moe_intermediate_size", "intermediate_size")
    expert_scale_fn: Callable[[nn.Module], torch.Tensor | None] | None = None
    export: Literal["one_expert", "dense_mlp"] = "one_expert"
    dense_mlp_class_name: str | None = None
    dense_layers_field: str | None = None
    dense_intermediate_field: str | None = None

    def __post_init__(self):
        if (self.dense_mlp_class_name is not None) and not self.dense_mlp_class_name.startswith("transformers."):
            raise ValueError(f"Provided MLP class name is not a Transformers class, got {self.dense_mlp_class_name}.")


def _gemma4_expert_scale(router: nn.Module) -> torch.Tensor | None:
    scale = getattr(router, "per_expert_scale", None)
    return None if scale is None else scale.detach()


GENERIC_SPEC = MoeArchSpec()

ARCH_SPECS: dict[str, MoeArchSpec] = {
    "qwen3_moe": MoeArchSpec(
        router_attr="gate",
        router_returns_probs=False,
        num_experts_fields=("num_experts",),
        top_k_fields=("num_experts_per_tok",),
        expert_intermediate_fields=("moe_intermediate_size",),
        export="dense_mlp",
        dense_mlp_class_name="transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeMLP",
        dense_layers_field="mlp_only_layers",
        dense_intermediate_field="intermediate_size",
    ),
    "gpt_oss": MoeArchSpec(
        router_attr="router",
        router_returns_probs=False,
        num_experts_fields=("num_local_experts",),
        top_k_fields=("num_experts_per_tok",),
        expert_intermediate_fields=("intermediate_size",),
    ),
    "gemma4_text": MoeArchSpec(
        router_attr="router",
        router_returns_probs=True,
        num_experts_fields=("num_experts",),
        top_k_fields=("top_k_experts",),
        expert_intermediate_fields=("moe_intermediate_size",),
        expert_scale_fn=_gemma4_expert_scale,
    ),
}


def get_arch_spec(config: Any) -> MoeArchSpec:
    """Return the spec for the model type of the given (sub-)config, falling back to the generic spec."""
    model_type = getattr(config, "model_type", None)
    return ARCH_SPECS.get(model_type, GENERIC_SPEC)


def get_config_value(config: Any, fields: tuple[str, ...]) -> Any:
    """Return the value of the first of the given fields that exists on the config and is not `None`."""
    for name in fields:
        value = getattr(config, name, None)
        if value is not None:
            return value
    raise AttributeError(f"None of the config fields {fields} could be found on the config of type {type(config)}.")


def set_config_value(config: Any, fields: tuple[str, ...], value: Any) -> str:
    """Set the first of the given fields that exists on the config, return the name of the field that was set."""
    for name in fields:
        if getattr(config, name, None) is not None:
            setattr(config, name, value)
            return name
    raise AttributeError(f"None of the config fields {fields} could be found on the config of type {type(config)}.")


def is_experts_module(module: nn.Module) -> bool:
    """
    Check whether a module looks like a transformers MoE experts module.

    Such a module holds the weights of all experts as 3D tensors (`gate_up_proj`/`up_proj` and `down_proj`), has no
    submodules with parameters (only e.g. activation functions), and is decorated with `use_experts_implementation`,
    which sets the attributes describing the tensor layout.
    """
    if any(next(child.parameters(), None) is not None for child in module.children()):
        return False
    if not all(hasattr(module, flag) for flag in EXPERTS_LAYOUT_FLAGS):
        return False
    down_proj = getattr(module, "down_proj", None)
    up_proj = getattr(module, "gate_up_proj", None) if module.has_gate else getattr(module, "up_proj", None)
    return (
        isinstance(down_proj, nn.Parameter)
        and isinstance(up_proj, nn.Parameter)
        and (down_proj.ndim == 3)
        and (up_proj.ndim == 3)
    )


def find_experts_modules(model: nn.Module) -> list[str]:
    """
    Return the names of all modules of the model that look like MoE experts modules.

    Experts modules that are already wrapped by a tuner layer are returned under the name of the tuner layer (this is
    relevant when adding multiple adapters).
    """
    names = []
    wrapped_prefixes: list[str] = []
    for name, module in model.named_modules():
        if not name or any(name.startswith(prefix) for prefix in wrapped_prefixes):
            continue
        if isinstance(module, BaseTunerLayer):
            wrapped_prefixes.append(name + ".")
            if is_experts_module(module.get_base_layer()):
                names.append(name)
        elif is_experts_module(module):
            names.append(name)
    return names


def find_router(parent: nn.Module, experts_name: str, spec: MoeArchSpec) -> tuple[str, nn.Module]:
    """
    Find the router module belonging to the experts module, which must be a sibling of the experts module.

    Returns the attribute name and the module.
    """
    if spec.router_attr is not None:
        router = getattr(parent, spec.router_attr, None)
        if router is None:
            raise ValueError(
                f"Expected to find the router module at attribute '{spec.router_attr}' of the parent of the experts "
                f"module but it does not exist. Found children: {[n for n, _ in parent.named_children()]}."
            )
        return spec.router_attr, router

    candidates = [(name, module) for name, module in parent.named_children() if name != experts_name]
    for name, module in candidates:
        if name in ("router", "gate"):
            return name, module
    for name, module in candidates:
        if hasattr(module, "top_k") or hasattr(module, "num_experts"):
            return name, module
    raise ValueError(
        "Could not find the router module belonging to the experts module. Found these sibling modules: "
        f"{[n for n, _ in candidates]}. This architecture is probably not supported."
    )


@dataclass
class ExpertsLayout:
    """Shape information of the expert tensors of an experts module."""

    num_experts: int
    hidden_size: int
    intermediate_size: int
    is_transposed: bool
    is_concatenated: bool
    has_bias: bool
    has_gate: bool

    @classmethod
    def from_module(cls, experts: nn.Module) -> "ExpertsLayout":
        if not is_experts_module(experts):
            raise ValueError(
                f"The targeted module of type {type(experts).__name__} does not look like a transformers MoE experts "
                "module (expected 3D expert weights and the layout attributes set by `use_experts_implementation`)."
            )
        down_proj = experts.down_proj
        if experts.is_transposed:  # [E, I, H]
            num_experts, intermediate_size, hidden_size = down_proj.shape
        else:  # [E, H, I]
            num_experts, hidden_size, intermediate_size = down_proj.shape
        return cls(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            is_transposed=experts.is_transposed,
            is_concatenated=experts.is_concatenated,
            has_bias=experts.has_bias,
            has_gate=experts.has_gate,
        )

    @property
    def up_proj_name(self) -> str:
        return "gate_up_proj" if self.has_gate else "up_proj"

    def expected_param_shapes(self, num_experts: int, intermediate_size: int) -> dict[str, tuple[int, ...]]:
        """The shapes of all expert parameters for the given number of experts and intermediate size."""
        num_exp, hidden, inter = num_experts, self.hidden_size, intermediate_size
        up_dim = 2 * inter if self.has_gate else inter
        if self.is_transposed:
            shapes = {self.up_proj_name: (num_exp, hidden, up_dim), "down_proj": (num_exp, inter, hidden)}
        else:
            shapes = {self.up_proj_name: (num_exp, up_dim, hidden), "down_proj": (num_exp, hidden, inter)}
        if self.has_bias:
            shapes[f"{self.up_proj_name}_bias"] = (num_exp, up_dim)
            shapes["down_proj_bias"] = (num_exp, hidden)
        return shapes

    def validate_module(self, experts: nn.Module, num_experts: int, intermediate_size: int) -> None:
        """Check that the experts module holds exactly the expected parameters with the expected shapes."""
        expected = self.expected_param_shapes(num_experts, intermediate_size)
        actual = {name: tuple(param.shape) for name, param in experts.named_parameters(recurse=False)}
        if actual != expected:
            raise ValueError(
                f"The experts module of type {type(experts).__name__} has unexpected parameters. Expected {expected} "
                f"but found {actual}. This architecture is probably not supported."
            )


def make_dense_config(config: Any, spec: MoeArchSpec, intermediate_size: int) -> Any:
    """Create a copy of the MoE (sub-)config describing an MoE layer with a single expert of the given size."""
    dense_config = copy.deepcopy(config)
    set_config_value(dense_config, spec.num_experts_fields, 1)
    set_config_value(dense_config, spec.top_k_fields, 1)
    set_config_value(dense_config, spec.expert_intermediate_fields, intermediate_size)
    return dense_config


def _split_gate_up(tensor: torch.Tensor, dim: int, layout: ExpertsLayout) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a gate_up tensor of a single expert into its gate and up part along the given dimension."""
    if layout.is_concatenated:  # [gate; up]
        return tensor.chunk(2, dim=dim)
    # interleaved: [gate0, up0, gate1, up1, ...]
    moved = tensor.movedim(dim, -1)
    return moved[..., ::2].movedim(-1, dim), moved[..., 1::2].movedim(-1, dim)


def _concat_gate_up(gates: list[torch.Tensor], ups: list[torch.Tensor], dim: int, layout: ExpertsLayout):
    """Inverse of `_split_gate_up` for multiple experts: build a single gate_up tensor from the gate and up parts."""
    gate = torch.cat(gates, dim=dim)
    up = torch.cat(ups, dim=dim)
    if layout.is_concatenated:
        return torch.cat([gate, up], dim=dim)
    # interleaved
    return torch.stack([gate, up], dim=dim + 1).flatten(dim, dim + 1)


def build_dense_expert_tensors(
    experts: nn.Module, layout: ExpertsLayout, selected: torch.Tensor, scales: torch.Tensor
) -> dict[str, torch.Tensor]:
    """
    Build the tensors of a single "dense" expert by concatenating the selected experts of the experts module.

    The intermediate activations of the selected experts are preserved exactly by the concatenation, the down
    projections (and their biases) are scaled by the given per-expert scales, which replace the token-dependent routing
    weights of the original MoE layer (see Appendix A of the paper).

    Args:
        experts (`nn.Module`):
            The experts module holding the weights of all experts.
        layout (`ExpertsLayout`):
            The layout of the expert tensors.
        selected (`torch.Tensor`):
            1D tensor of the indices of the experts to concatenate, in the order of concatenation.
        scales (`torch.Tensor`):
            1D tensor of the same length as `selected` with the scale applied to each expert's down projection.

    Returns:
        `dict[str, torch.Tensor]`: The parameter names of the experts module mapped to the concatenated tensors, with a
        leading expert dimension of size 1.
    """
    if not experts.down_proj.is_floating_point():
        raise NotImplementedError(
            "MoE-to-dense requires the expert weights to be floating point tensors, quantized experts are not "
            "supported. Please load the model with dequantized experts."
        )

    selected = selected.tolist()
    scales = scales.to(device=experts.down_proj.device, dtype=torch.float32).tolist()
    # dimension along which the intermediate size lives, for 2D (per-expert) tensors
    up_dim = 1 if layout.is_transposed else 0  # gate_up: [H, 2I] if transposed, else [2I, H]
    down_dim = 0 if layout.is_transposed else 1  # down: [I, H] if transposed, else [H, I]
    up_proj = getattr(experts, layout.up_proj_name).detach()
    down_proj = experts.down_proj.detach()

    tensors = {}
    if layout.has_gate:
        gates, ups = zip(*[_split_gate_up(up_proj[i], up_dim, layout) for i in selected])
        tensors[layout.up_proj_name] = _concat_gate_up(list(gates), list(ups), up_dim, layout)
    else:
        tensors[layout.up_proj_name] = torch.cat([up_proj[i] for i in selected], dim=up_dim)
    tensors["down_proj"] = torch.cat([down_proj[i] * scale for i, scale in zip(selected, scales)], dim=down_dim)

    if layout.has_bias:
        up_bias = getattr(experts, f"{layout.up_proj_name}_bias").detach()
        if layout.has_gate:
            gates, ups = zip(*[_split_gate_up(up_bias[i], 0, layout) for i in selected])
            tensors[f"{layout.up_proj_name}_bias"] = _concat_gate_up(list(gates), list(ups), 0, layout)
        else:
            tensors[f"{layout.up_proj_name}_bias"] = torch.cat([up_bias[i] for i in selected], dim=0)
        down_bias = experts.down_proj_bias.detach()
        # the biases of the down projections are summed (with scales), as the expert outputs are summed
        tensors["down_proj_bias"] = sum(down_bias[i] * scale for i, scale in zip(selected, scales))

    return {name: tensor.unsqueeze(0) for name, tensor in tensors.items()}


def infer_layer_index(module_name: str) -> int | None:
    """Infer the index of the decoder layer from a module name like `model.layers.3.mlp.experts`."""
    match = re.search(r"\.layers\.(\d+)\.", f".{module_name}.")
    return int(match.group(1)) if match else None
