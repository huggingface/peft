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
from typing import Literal, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class DeftConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`DeftModel`].

    DEFT (Decompositional Efficient Fine-Tuning) performs knowledge injection through a residual-projection update. For
    a frozen base weight `W`, a low-rank projection direction `P` (shape `out_features x r`) and an injection matrix
    `R` (shape `r x in_features`) are learned. The adapted weight is `W' = (I - P_proj) @ W + Q_P @ R`, where the
    projector `P_proj` is derived from `P` according to `decomposition_method`:

      - `"relu"` (default): `Q_P = P`, `P_proj = P @ relu(P).T` (non-orthogonal projection)
      - `"qr"`: `Q_P = qr(P)`, `P_proj = Q_P @ Q_P.T` (orthogonal projection)

    By default (`init_weights=True`) `R` is initialized so the update is an exact identity at init (the adapted weight
    equals `W`), so training starts from the pretrained weights and learns the injection.

    Args:
        r (`int`):
            The rank of the DEFT projection/injection across layers.
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the specified
            names will be replaced. When passing a string, a regex match will be performed. When passing a list of
            strings, either an exact match will be performed or it is checked if the name of the module ends with any
            of the passed strings. If this is specified as 'all-linear', then all linear modules are chosen, excluding
            the output layer. If this is not specified, modules will be chosen according to the model architecture. If
            the architecture is not known, an error will be raised -- in this case, you should specify the target
            modules manually.
        exclude_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to not apply the adapter. When passing a string, a regex match will be performed.
            When passing a list of strings, either an exact match will be performed or it is checked if the name of the
            module ends with any of the passed strings.
        decomposition_method (`str`):
            How the projector `P_proj` is derived from `P`. Either `"relu"` (default, non-orthogonal `P @ relu(P).T`)
            or `"qr"` (orthogonal `Q_P @ Q_P.T`).
        init_scale (`float`):
            Scaling applied to the standard deviation used to initialize the injection matrix `R` (only used when
            `init_weights=False`). Smaller values keep the injected update closer to zero at initialization. Defaults
            to `1.0`.
        alpha (`Optional[int]`):
            The scaling factor for the injection term, which is scaled by `alpha / r` (analogous to LoRA's alpha). If
            `None`, no scaling is applied (factor `1.0`). The subspace-removal term is unaffected.
        para (`bool`):
            Whether to use the PaRa method: pure subspace removal (`delta = -P_proj @ W`) with no injection term. When
            `True`, `R` is not created, `P` is the only trainable matrix, and the adapter cannot be an identity at
            init. Defaults to `False` (full DEFT).
        fan_in_fan_out (`bool`):
            Set this to `True` if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.
        deft_dropout (`float`):
            The dropout probability applied to the layer input. Defaults to `0.0`.
        init_weights (`bool`):
            Whether to use DEFT's default (identity) initialization for the adapter weights, so the adapter is a no-op
            at the start of training. Don't change this setting, except if you know exactly what you're doing. Defaults
            to `True`.
        layers_to_transform (`Union[List[int], int]`):
            The layer indices to transform. If a list of ints is passed, it will apply the adapter to the layer indices
            that are specified in this list. If a single integer is passed, it will apply the transformations on the
            layer at this index.
        layers_pattern (`Optional[Union[List[str], str]]`):
            The layer pattern name, used only if `layers_to_transform` is different from `None`. This should target the
            `nn.ModuleList` of the model, which is often called `'layers'` or `'h'`.
        bias (`str`):
            Bias type for DEFT. Can be `'none'`, `'all'` or `'deft_only'`.
        modules_to_save (`List[str]`):
            List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
    """

    r: int = field(
        default=8,
        metadata={"help": "The rank of the DEFT projection/injection across layers."},
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with DEFT.",
            "example": "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' ",
        },
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude from DEFT."},
    )
    decomposition_method: Literal["relu", "qr"] = field(
        default="relu",
        metadata={
            "help": (
                "How the projector P_proj is derived from P. Either 'relu' (default, non-orthogonal P @ relu(P).T) or "
                "'qr' (orthogonal Q_P @ Q_P.T)."
            ),
        },
    )
    init_scale: float = field(
        default=1.0,
        metadata={"help": "Scaling applied to the std used to initialize the injection matrix R."},
    )
    alpha: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Scaling factor for the injection term, scaled by `alpha / r` (analogous to LoRA's alpha). If None, no "
                "scaling is applied (factor 1.0). The subspace-removal term is unaffected."
            )
        },
    )
    para: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use the PaRa method: pure subspace removal `delta = -P_proj @ W` with no injection term. "
                "When True, R is not created, P is the only trainable matrix, and the adapter cannot be an identity at "
                "init. Defaults to False (full DEFT)."
            )
        },
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={
            "help": (
                "Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 "
                "uses `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to True."
            )
        },
    )
    deft_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout probability applied to the layer input."},
    )
    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the DEFT layers with their default initialization. Don't change "
                "this setting, except if you know exactly what you're doing."
            ),
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, if this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern. "
            "This should target the `nn.ModuleList` of the model, which is often called `'layers'` or `'h'`."
        },
    )
    bias: str = field(default="none", metadata={"help": "Bias type for DEFT. Can be 'none', 'all' or 'deft_only'"})
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from DEFT layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.DEFT
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )

        valid_methods = {"qr", "relu"}
        if self.decomposition_method not in valid_methods:
            raise ValueError(
                f"Unknown decomposition_method '{self.decomposition_method}', must be one of {sorted(valid_methods)}."
            )

        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # if target_modules is a regex expression, then layers_pattern should be None
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")

        # check for layers_to_transform and layers_pattern
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified. ")
