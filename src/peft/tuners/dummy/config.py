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

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class DummyConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`DummyModel`].

    The Dummy method is not a research method. It is a minimal, deliberately trivial tuner that serves as a template
    for contributors who want to add a new PEFT method and as a test vehicle for PEFT features that any tuner should
    support (saving, loading, merging, quantization, mixed adapters, ...).

    Its implementation is essentially a plain LoRA without any of the optimizations or extras that LoRA offers: an
    adapter consists of a down-projection `A` and an up-projection `B`, and the effective weight of an adapted layer is
    `W + B @ A * (lora_alpha / r)`. This makes the effect of the adapter easy to predict and reason about.

    Args:
        target_modules (`Union[List[str], str]`):
            The names of the modules to apply the adapter to. Supports `nn.Linear`,
            `transformers.pytorch_utils.Conv1D`, `nn.Conv1d`/`nn.Conv2d`/`nn.Conv3d` (including groups), and
            `nn.Embedding`.
        exclude_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to not apply the adapter.
        r (`int`):
            The rank of the update matrices, expressed in `int`. A smaller rank results in fewer trainable parameters
            and less capacity. Must be divisible by `groups` when targeting grouped convolutional layers.
        lora_alpha (`int`):
            The scaling factor of the update matrices, expressed in `int`. The effective scaling is `lora_alpha / r`.
        lora_dropout (`float`):
            The dropout probability for the update matrices, expressed in `float`.
        init_weights (`bool`):
            Whether to initialize the update matrices such that the adapter starts out as an identity transform (`B` is
            zero). Set this to False to randomly initialize both `A` and `B`, which gives the adapter a non-zero effect
            right away.
        fan_in_fan_out (`bool`):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.
        modules_to_save (`List[str]`):
            List of modules apart from Dummy layers to be set as trainable and saved in the final checkpoint.
        layers_to_transform (`Union[List[int],int]`):
            The layer indexes to transform, if this argument is specified, it will apply the Dummy transformations on
            the layer indexes that are specified in this list. If a single integer is passed, it will apply the Dummy
            transformations on the layer at this index.
        layers_pattern (`Optional[Union[List[str], str]]`):
            The layer pattern name, used only if `layers_to_transform` is different from `None`. This should target the
            `nn.ModuleList` of the model, which is often called `'layers'` or `'h'`.
    """

    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with Dummy."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. "
                "Supported modules are nn.Linear, transformers.pytorch_utils.Conv1D, nn.Conv1d/2d/3d, and nn.Embedding."
            )
        },
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude from Dummy."},
    )
    r: int = field(default=8, metadata={"help": "The rank of the low-rank update matrices"})
    lora_alpha: int = field(default=8, metadata={"help": "The scaling factor of the low-rank update matrices"})
    lora_dropout: float = field(default=0.0, metadata={"help": "The dropout probability for the update matrices"})
    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the update matrices such that the adapter starts out as an identity transform"
                " (B is zero). Set to False to randomly initialize both A and B, which gives the adapter a non-zero"
                " effect right away."
            ),
        },
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from Dummy layers to be set as trainable and saved in the final checkpoint. For"
                " example, in Sequence Classification or Token Classification tasks, the final layer"
                " `classifier/score` are randomly initialized and as such need to be trainable and saved."
            )
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers"
                " indexes that are specified inside this list. If a single integer is passed, PEFT will transform only"
                " the layer at this index."
            )
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer "
                "pattern is not in the common layers pattern. This should target the `nn.ModuleList` of the "
                "model, which is often called `'layers'` or `'h'`."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.DUMMY
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )

        # check for layers_to_transform and layers_pattern
        if self.layers_pattern and self.layers_to_transform is None:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified. ")
        # if target_modules is a regex expression, then layers_pattern should be None
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")

        if self.r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {self.r}")
