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


@dataclass
class CortexConfig(PeftConfig):
    """Configuration for Cortex's additive FFN channels.

    Args:
        k (`int`, *optional*, defaults to 16):
            Intermediate width of each added branch. Dense FFNs always use this width. This is a channel count, not a
            low-rank matrix factorization rank.
        target_modules (`Union[list[str], set[str], str]`, *optional*):
            Complete FFN or MoE blocks to adapt. A list or set matches exact names or suffixes; a string is a full
            regular expression. If unset, discover registered native FFN structures. Individual projections and
            `"all-linear"` are not supported. Custom FFNs require explicit selection and a supported structure.
            Explicitly selecting an experts module selects routed branches, regardless of `moe_target`. Select its
            parent MoE block to choose branches with `moe_target`. Auto-discovery skips existing auxiliary training
            wrappers, including `modules_to_save`, and their descendants. Explicit targets cannot enter these wrappers.
        exclude_modules (`Union[list[str], set[str], str]`, *optional*):
            Blocks or resolved FFNs to exclude, using the same name-matching rules as `target_modules`.
        moe_target (`Union[str, list[str]]`, *optional*, defaults to `"shared"`):
            Which MoE branches to extend: `"routed"` adds channels to each routed expert; `"shared"` extends native
            shared FFNs, or adds one public branch if none exist. Use `["shared", "routed"]` for both. Native shared
            gates and normalization remain in place. Dense FFNs are unaffected. Adapters may use different selections
            on different physical FFNs; adapters sharing a physical FFN must resolve to the same branch role.
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
