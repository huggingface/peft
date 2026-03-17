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
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from transformers.activations import ACT2FN

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class PeanutConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`PeanutModel`].

    Args:
        r (`int`):
            PEANuT rank. This is the hidden dimension used by the adapters. Similar to LoRA rank, larger `r` increases
            adapter capacity and trainable parameters.
        depth (`int`):
            Number of hidden adapter layers per encoder/decoder side in PEANuT. The input projection `A` and output
            projection `B` are always present in addition to these hidden layers. Therefore, `depth` must be a
            non-negative integer.

            - `depth=0`: `A`, `B`.
            - `depth=1`: `A`, one encoder, one decoder, `B`.
            - `depth=2`: `A`, two encoders, two decoders, `B`.
            - `depth=3`: `A`, three encoders, three decoders, `B`, etc.
        act_fn (`str`):
            Non-linear activation applied in the PEANuT network. This corresponds to `non_linear` in the vanilla
            PyTorch implementation. Default is `"relu"`. Any activation key available in
            `transformers.activations.ACT2FN` is supported and may perform better on different tasks.
        scaling (`float`):
            A scalar multiplier applied to the PEANuT output before adding it to the frozen base layer output. The
            final adapter contribution is `scaling * (x @ delta_w)`.
        target_modules (`Union[List[str], str]`, *optional*):
            The names of the modules to apply PEANuT to. Can be a list of module name strings (e.g. `['q_proj',
            'v_proj']`) or a regex pattern.
        modules_to_save (`List[str]`, *optional*):
            List of modules apart from PEANuT layers to be set as trainable and saved in the final checkpoint.
        exclude_modules (`Union[List[str], str]`, *optional*):
            The names of the modules to not apply the adapter. When passing a string, a regex match will be performed.
            When passing a list of strings, either an exact match will be performed or it is checked if the name of the
            module ends with any of the passed strings.
        layers_to_transform (`Union[list[int], int]`, *optional*):
            The layer indexes to transform. If this argument is specified, PEFT will transform only the layer indexes
            that are specified in this list. If a single integer is passed, PEFT will transform only the layer at this
            index.
        layers_pattern (`Optional[Union[List[str], str]]`, *optional*):
            The layer pattern name, used only if `layers_to_transform` is not None and if the layer pattern is not in
            the common layers pattern.
        init_weights (`bool`):
            Whether to initialize PEANuT adapter weights using the default initialization scheme:

            - If `True`: all weights except `B` are initialized with Kaiming uniform, and `B` is initialized to zero.
            - If `False`: all weights (including `B`) are initialized with Kaiming uniform.

            Initializing `B` to zero makes the adapter start as an exact no-op.

    Notes:
        PEANuT uses a weight-aware pathway, where the delta weight is conditioned on the base weight. The `A` adapter
        is applied over the base weight's output dimension, so `A` has shape `(out_dim -> r)` rather than the usual
        `(in_dim -> r)` used by LoRA-like methods.
    """

    r: int = field(
        default=32,
        metadata={
            "help": (
                "PEANuT rank. This is the hidden dimension used by the adapter stack. Similar to LoRA rank, larger "
                "`r` increases adapter capacity and trainable parameters."
            )
        },
    )
    depth: int = field(
        default=0,
        metadata={
            "help": (
                "Number of hidden adapter layers per encoder/decoder side in PEANuT. The input projection `A` and "
                "output projection `B` are added automatically, so `depth` must be a non-negative integer."
            )
        },
    )
    act_fn: str = field(
        default="relu",
        metadata={
            "help": (
                "Non-linear activation applied in the PEANuT pathway. This corresponds to `non_linear` in the "
                "vanilla implementation. Must be a key in `transformers.activations.ACT2FN`."
            )
        },
    )
    scaling: float = field(
        default=1.0,
        metadata={
            "help": (
                "A scalar multiplier applied to the PEANuT output before adding it to the frozen base layer output. "
                "The final adapter contribution is `scaling * (x @ delta_w)`."
            )
        },
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with PEANuT. "
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. "
                "If not specified, PEANuT will use architecture-specific default target modules."
            )
        },
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to exclude from PEANuT. "
                "When passing a string, a regex match will be performed. When passing a list of strings, "
                "either an exact match will be performed or it is checked if the name of the module ends "
                "with any of the passed strings."
            )
        },
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from PEANuT layers to be set as trainable and saved in the final checkpoint."
            )
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "The layer indexes to transform, if this argument is specified, PEFT will transform only the layers "
                "indexes that are specified inside this list. If a single integer is passed, PEFT will transform only "
                "the layer at this index."
            )
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer "
                "pattern is not in the common layers pattern. This should target the `nn.ModuleList` of the model, "
                "which is often called `'layers'` or `'h'`."
            )
        },
    )
    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize PEANuT adapter weights using the default initialization scheme: if `True`, "
                "all weights except `B` are initialized with Kaiming uniform and `B` is initialized to zero; if "
                "`False`, all weights including `B` are initialized with Kaiming uniform."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.PEANUT
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified.")
        if self.r <= 0:
            raise ValueError("`r` must be a positive integer.")
        if self.depth < 0:
            raise ValueError("`depth` can only be a non-negative integer.")
        if self.act_fn not in ACT2FN:
            raise ValueError(f"Unsupported `act_fn`: {self.act_fn}. Must be one of {sorted(ACT2FN.keys())}.")
