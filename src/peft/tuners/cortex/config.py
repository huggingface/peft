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
from dataclasses import dataclass, field
from typing import Literal

from peft.config import PeftConfig
from peft.utils import PeftType


def _check_attribute(name: str, value: str | None, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or not value.isidentifier():
        raise ValueError(f"{name} must be a direct attribute name, without dots.")


@dataclass(frozen=True)
class CortexFfnSpec:
    """Describe a dense FFN or packed experts using Cortex's supported arithmetic.

    This same description is used by the built-in registry and `CortexConfig.ffn_specs`. Attribute names refer to
    direct children or parameters of the FFN. Shapes, dtypes and activation functions are checked on the actual module.
    Changing names does not add support for a different forward computation.

    Args:
        packed (`bool`, *optional*, defaults to `False`):
            Whether projections are three-dimensional expert parameters rather than `nn.Linear` modules.
        has_gate (`bool`, *optional*, defaults to `True`):
            Whether the FFN multiplies activated gate channels by up channels.
        activation (`str`, *optional*, defaults to `"standard"`):
            Supported arithmetic: `"standard"`, `"clipped"`, or `"offset_gated"`.
        gate_proj (`str`, *optional*, defaults to `"gate_proj"`):
            Gate projection attribute for separate gated projections.
        up_proj (`str`, *optional*, defaults to `"up_proj"`):
            Up projection attribute.
        down_proj (`str`, *optional*, defaults to `"down_proj"`):
            Down projection attribute.
        gate_up_proj (`str`, *optional*, defaults to `"gate_up_proj"`):
            Fused gate/up projection attribute, used when present. Set to `None` to require separate projections.
            Packed projection biases use the corresponding projection name followed by `"_bias"`.
        act_fn (`str`, *optional*, defaults to `"act_fn"`):
            Parameter-free elementwise activation attribute.
        width_field (`str`, *optional*, defaults to `"intermediate_size"`):
            Field on the owning model configuration used to reconstruct this FFN's intermediate width.
        shared_width_field (`str`, *optional*):
            Alternative configuration field when this FFN is a native shared expert.
        shared_multiplier (`str`, *optional*):
            Configuration field multiplying `shared_width_field` to obtain the shared FFN's actual width.
        width_attributes (`tuple[str, ...]`, *optional*, defaults to `("intermediate_dim", "intermediate_size")`):
            Width attributes on the FFN itself to update during merging, when present.
    """

    packed: bool = False
    has_gate: bool = True
    activation: str = "standard"
    gate_proj: str = "gate_proj"
    up_proj: str = "up_proj"
    down_proj: str = "down_proj"
    gate_up_proj: str | None = "gate_up_proj"
    act_fn: str = "act_fn"
    width_field: str = "intermediate_size"
    shared_width_field: str | None = None
    shared_multiplier: str | None = None
    width_attributes: tuple[str, ...] = ("intermediate_dim", "intermediate_size")

    def __post_init__(self):
        if type(self.packed) is not bool or type(self.has_gate) is not bool:
            raise TypeError("packed and has_gate must be bools.")
        if self.activation not in ("standard", "clipped", "offset_gated"):
            raise ValueError("activation must be 'standard', 'clipped', or 'offset_gated'.")
        if not self.has_gate and self.activation != "standard":
            raise ValueError("Clipped and offset-gated arithmetic require has_gate=True.")
        for name in ("gate_proj", "up_proj", "down_proj", "act_fn", "width_field"):
            _check_attribute(name, getattr(self, name))
        for name in ("gate_up_proj", "shared_width_field", "shared_multiplier"):
            _check_attribute(name, getattr(self, name), optional=True)
        if self.shared_multiplier is not None and self.shared_width_field is None:
            raise ValueError("shared_multiplier requires shared_width_field.")
        projections = [self.up_proj, self.down_proj]
        if self.has_gate:
            projections.append(self.gate_proj)
            if self.gate_up_proj is not None:
                projections.append(self.gate_up_proj)
        if len(set(projections)) != len(projections) or self.act_fn in projections:
            raise ValueError("Projection and activation attribute names must be distinct.")
        if not isinstance(self.width_attributes, (list, tuple)):
            raise TypeError("width_attributes must be a list or tuple of attribute names.")
        for name in self.width_attributes:
            _check_attribute("width_attributes", name)
        if len(set(self.width_attributes)) != len(self.width_attributes):
            raise ValueError("width_attributes must not contain duplicate names.")
        if set(self.width_attributes).intersection(projections + [self.act_fn]):
            raise ValueError("width_attributes must not overwrite projections or the activation.")
        object.__setattr__(self, "width_attributes", tuple(self.width_attributes))


@dataclass(frozen=True)
class CortexMoeSpec:
    """Describe the native MoE boundaries without replacing its router.

    Args:
        experts (`str`, *optional*, defaults to `"experts"`):
            Direct child containing packed routed expert parameters. Describe nonstandard projection names with a
            `CortexFfnSpec` for that child's class.
        shared_modules (`tuple[str, ...]`, *optional*):
            Direct child names of native shared FFNs. `None` discovers conventional names but does not establish
            absence on a user-described block; an empty tuple explicitly declares no shared FFNs in this block.
        enabled_attribute (`str`, *optional*):
            Boolean attribute on the parent indicating whether its MoE path is enabled.
        routing_indices (`str`, *optional*):
            Expert forward argument containing selected expert IDs. Specify together with `routing_weights` when the
            routing arguments use nonstandard names. The first argument must be named `hidden_states`, followed by
            indices and weights, and all three must accept positional calls. This does not rename the input argument.
        routing_weights (`str`, *optional*):
            Expert forward argument containing weights for the selected experts. This does not declare normalization or
            authorize merging an unweighted public branch.
    """

    experts: str = "experts"
    shared_modules: tuple[str, ...] | None = None
    enabled_attribute: str | None = None
    routing_indices: str | None = None
    routing_weights: str | None = None

    def __post_init__(self):
        _check_attribute("experts", self.experts)
        for name in ("enabled_attribute", "routing_indices", "routing_weights"):
            _check_attribute(name, getattr(self, name), optional=True)
        if (self.routing_indices is None) != (self.routing_weights is None):
            raise ValueError("Specify routing_indices and routing_weights together.")
        if self.routing_indices is not None and self.routing_indices == self.routing_weights:
            raise ValueError("routing_indices and routing_weights must name different arguments.")
        if self.shared_modules is not None:
            if not isinstance(self.shared_modules, (list, tuple)):
                raise TypeError("shared_modules must be a list or tuple of child names, or None.")
            for name in self.shared_modules:
                _check_attribute("shared_modules", name)
            if len(set(self.shared_modules)) != len(self.shared_modules) or self.experts in self.shared_modules:
                raise ValueError("Routed and shared expert names must be distinct.")
            object.__setattr__(self, "shared_modules", tuple(self.shared_modules))


@dataclass
class CortexConfig(PeftConfig):
    """Configuration for Cortex's additive FFN channels.

    Args:
        k (`int`, *optional*, defaults to 16):
            Intermediate width of each added branch. Dense FFNs always use this width. This is a channel count, not a
            low-rank matrix factorization rank.
        target_modules (`Union[list[str], set[str], str]`, *optional*):
            Complete FFN or MoE blocks to adapt. A list or set matches exact names or suffixes; a string is a full
            regular expression. If unset, discover native Transformers FFNs using registered layouts or standard FFN
            and expert conventions. Cortex warns when it infers an unregistered layout. Individual projections and
            `"all-linear"` are not supported. Custom PyTorch FFNs require explicit selection or a user-provided spec.
            Explicitly selecting an experts module selects routed branches, regardless of `moe_target`. Select its
            parent MoE block to choose branches with `moe_target`. Auto-discovery skips existing auxiliary training
            wrappers, including `modules_to_save`, and their descendants. Explicit targets cannot enter these wrappers.
        exclude_modules (`Union[list[str], set[str], str]`, *optional*):
            Blocks or resolved FFNs to exclude, using the same name-matching rules as `target_modules`.
        ffn_specs (`dict[str, CortexFfnSpec]`, *optional*):
            Explicit FFN descriptions keyed by class name or fully qualified class name. A qualified name takes
            precedence. Values may also be dictionaries restored from JSON. Each value is a complete description,
            replacing the built-in default for that class. These classes participate in automatic discovery, and their
            descriptions also apply to unadapted FFNs that share an export width. Provide the complete model structure
            descriptions in every adapter configuration, even when targeting different layers.
        moe_specs (`dict[str, CortexMoeSpec]`, *optional*):
            Explicit MoE descriptions, with the same class-name matching as `ffn_specs`. These locate routed and shared
            branches; they do not prove routing normalization for public-branch merging.
        moe_target (`Union[str, list[str]]`, *optional*, defaults to `"shared"`):
            Which MoE branches to extend: `"routed"` adds channels to each routed expert; `"shared"` extends native
            shared FFNs, or adds a public branch when the layout is known to have none. If an inferred layout cannot
            establish where the shared FFN lives, select `"routed"` or add a layout rule. Use `["shared", "routed"]`
            for both. Native shared gates and normalization remain in place. Dense FFNs are unaffected. Adapters
            sharing a physical FFN must resolve to the same branch role.
        public_weighting (`str`, *optional*, defaults to `"unweighted"`):
            How to weight an added public branch. `"unweighted"` adds its output directly. `"routing"` multiplies it by
            the sum of the selected native router weights, enabling merging with unnormalized top-k routing. This
            changes the function and must be chosen before training. Dense, native shared, and routed branches are
            unaffected. The router weights are never renormalized.
        k_shared (`int`, *optional*):
            Width per native shared FFN or public branch. Defaults to `k`.
        k_routed (`int`, *optional*):
            Width per routed expert. Defaults to `k`; the total parameter count scales with the number of experts.
        init_weights (`Union[bool, str]`, *optional*, defaults to `"random"`):
            `"random"` uses Kaiming-uniform gate/up weights. `"sample"` copies paired gate/up channels from the base
            FFN, with optional noise. Routed branches sample their corresponding expert; public branches sample paired
            channels from the pool of all experts. Both zero the down weights and bias, preserving the initial output
            for finite features. `True` aliases `"random"`. `False` keeps random down weights for testing and does not
            preserve the initial output.
        init_noise_std (`float`, *optional*, defaults to 0.01):
            Standard deviation of additive Gaussian noise on sampled gate/up weights. Set to zero for exact copying.
            Ignored by random initialization.
        target_roles (`dict[str, str]`, *optional*):
            Resolved physical module paths and branch roles stored in adapter checkpoints. Populated by the tuner;
            users should leave this empty when creating a new adapter. Create a fresh configuration for a different
            model; reusing a runtime configuration that would resolve to different targets raises an error.
        adapter_bias (`Optional[bool]`, *optional*, defaults to `None`):
            `None` follows the presence of bias in each corresponding base FFN projection. Public branches follow the
            native expert projections. `True` creates biases in every added projection; `False` creates none. Adapter
            biases cannot be merged into native projections without corresponding biases.
        adapter_bias_init (`str`, *optional*, defaults to `"uniform"`):
            For random initialization, initialize existing gate/up biases uniformly in `[-1/sqrt(fan_in),
            1/sqrt(fan_in)]`, or use `"zeros"` to start them at zero. This does not create biases. Sample
            initialization instead copies available base biases and zeros missing ones. Down biases always start at
            zero. With `init_weights=False`, all biases start at zero.
        modules_to_save (`list[str]`, *optional*):
            Additional modules, such as a task head, to train fully and include in the adapter checkpoint. Names match
            exact paths or dot-delimited suffixes. These modules and their descendants are excluded from Cortex
            targets. Existing copies owned by another adapter are also excluded from auto-discovery; use
            `modules_to_save` to request a separate full module copy. A new saved module cannot contain existing Cortex
            layers.

    See the method documentation for routing and export limitations.
    """

    k: int = field(default=16, metadata={"help": "Intermediate width of each Cortex branch."})
    target_modules: list[str] | set[str] | str | None = field(default=None)
    exclude_modules: list[str] | set[str] | str | None = field(default=None)
    ffn_specs: dict[str, CortexFfnSpec] = field(default_factory=dict)
    moe_specs: dict[str, CortexMoeSpec] = field(default_factory=dict)
    moe_target: Literal["shared", "routed"] | list[str] = field(default="shared")
    public_weighting: Literal["unweighted", "routing"] = field(
        default="unweighted", metadata={"help": "Weight an added public branch by the selected routing mass."}
    )
    k_shared: int | None = field(default=None)
    k_routed: int | None = field(default=None)
    init_weights: bool | Literal["random", "sample"] = field(default="random")
    init_noise_std: float = field(default=0.01, metadata={"help": "Additive weight noise for sample initialization."})
    target_roles: dict[str, str] = field(default_factory=dict, metadata={"help": "Resolved checkpoint target roles."})
    adapter_bias: bool | None = field(
        default=None, metadata={"help": "Follow base projection biases, or explicitly enable/disable adapter biases."}
    )
    adapter_bias_init: Literal["uniform", "zeros"] = field(
        default="uniform", metadata={"help": "Gate/up bias initialization for random initialization."}
    )
    modules_to_save: list[str] | None = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.CORTEX
        for name, spec_type in (("ffn_specs", CortexFfnSpec), ("moe_specs", CortexMoeSpec)):
            values = getattr(self, name)
            if not isinstance(values, dict):
                raise TypeError(f"{name} must map class names to {spec_type.__name__} descriptions.")
            normalized = {}
            for key, value in values.items():
                if not isinstance(key, str) or not key or any(not part.isidentifier() for part in key.split(".")):
                    raise ValueError(f"{name} keys must be class names or fully qualified class names.")
                if isinstance(value, dict):
                    value = spec_type(**value)
                if not isinstance(value, spec_type):
                    raise TypeError(f"{name}[{key!r}] must be a {spec_type.__name__} or a dictionary.")
                normalized[key] = value
            setattr(self, name, normalized)
        for name in ("k", "k_shared", "k_routed"):
            value = getattr(self, name)
            if value is None and name != "k":
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        targets = self.moe_target
        if isinstance(targets, str):
            targets = [targets]
        if (
            not isinstance(targets, list)
            or not targets
            or any(not isinstance(target, str) or target not in ("shared", "routed") for target in targets)
            or len(set(targets)) != len(targets)
        ):
            raise ValueError("moe_target must be 'shared', 'routed', or a nonempty list of those unique names.")
        # Canonicalize order and singleton lists for adapter comparison and JSON.
        self.moe_target = targets[0] if len(targets) == 1 else ["shared", "routed"]
        if self.public_weighting not in ("unweighted", "routing"):
            raise ValueError("public_weighting must be 'unweighted' or 'routing'.")
        if not isinstance(self.init_weights, bool) and self.init_weights not in ("random", "sample"):
            raise ValueError("init_weights must be 'random', 'sample', or a bool (False is for testing only).")
        if (
            isinstance(self.init_noise_std, bool)
            or not isinstance(self.init_noise_std, (int, float))
            or not math.isfinite(self.init_noise_std)
            or self.init_noise_std < 0
        ):
            raise ValueError("init_noise_std must be finite and nonnegative.")
        roles = {"dense", "shared", "routed", "public", "routed_public"}
        if not isinstance(self.target_roles, dict) or any(
            not isinstance(path, str) or not path or role not in roles for path, role in self.target_roles.items()
        ):
            raise ValueError("target_roles must map nonempty module paths to supported FFN roles.")
        if self.adapter_bias is not None and not isinstance(self.adapter_bias, bool):
            raise TypeError("adapter_bias must be a bool or None.")
        if self.adapter_bias_init not in ("uniform", "zeros"):
            raise ValueError("adapter_bias_init must be 'uniform' or 'zeros'.")
        for name in ("target_modules", "exclude_modules"):
            value = getattr(self, name)
            if isinstance(value, (list, set)):
                if not all(isinstance(item, str) and item for item in value):
                    raise ValueError(f"{name} must contain nonempty module names.")
                setattr(self, name, set(value))
            elif value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a list/set of names or a nonempty regex.")
        if self.target_modules == "all-linear":
            raise ValueError("Cortex targets whole MLP blocks; 'all-linear' is not supported.")
        if self.target_modules == set():
            raise ValueError("target_modules must not be empty.")
