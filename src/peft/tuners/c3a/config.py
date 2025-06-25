# Copyright 2025-present the HuggingFace Inc. team.
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
class C3AConfig(PeftConfig):
    """This is the configuration class to store the configuration of a [`C3AModel`].

    Args:
        block_size (`int`):
            block size for C3A, must be divisible by both the input size and the output size of the target layer. If
            you have no idea what block_size you should use, set it to the greatest common divisor of all input &
            output sizes of your target layers. Increasing this would result in less parameters.
        target_modules (`Union[list[str],str]`): The names of the modules to apply C3A to.
        bias (`str`): Bias type for C3A. Can be 'none', 'all' or 'c3a_only'. If 'all' or 'c3a_only', the
            corresponding biases will be updated during training. Be aware that this means that, even when disabling
            the adapters, the model will not produce the same output as the base model would have without adaptation.
        modules_to_save (`list[str]`):list of modules apart from C3A layers to be set as trainable
            and saved in the final checkpoint.
        layers_to_transform (`Union[list[int],int]`):
            The layer indexes to transform, if this argument is specified, it will apply C3A on the layer indexes that
            are specified in this list. If a single integer is passed, it will apply C3A on the layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None` and if the layer
            pattern is not in the common layers pattern.
        block_size_pattern (`dict`):
            The mapping from layer names or regexp expression to block_size which are different from the default
            specified. For example, `{"model.decoder.layers.0.encoder_attn.k_proj": 1280`}
        init_weights (`Union[bool, Literal["gaussian", "kaiming_uniform", "xavier_uniform"]]`):
            The initialization of the C3A weights. Set this to False if the weights should be initialized to a commonly
            used distribution. Set this to True if the weights should be initialized to zeros.
    """

    block_size: int = field(
        default=256,
        metadata={
            "help": (
                "block size for C3A, must be divisible by both the input size and the output size of the target layer."
                " If you have no idea what block_size you should use, set it to the greatest common divisor of all"
                " input & output sizes of your target layers. Increasing this would result in less parameters."
            )
        },
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "list of module names or regex expression of the module names to replace with C3A."
                " For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
            )
        },
    )
    bias: str = field(default="none", metadata={"help": "Bias type for C3A. Can be 'none', 'all' or 'c3a_only'"})
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "list of modules apart from C3A layers to be set as trainable and saved in the final checkpoint."
                " For example, in Sequence Classification or Token Classification tasks,"
                " the final layer `classifier/score` are randomly initialized"
                " and as such need to be trainable and saved."
            )
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "The layer indexes to transform, is this argument is specified,"
                " PEFT will transform only the layers indexes that are specified inside this list."
                " If a single integer is passed, PEFT will transform only the layer at this index."
                " This only works when target_modules is a list of str."
            )
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "The layer pattern name, used only if `layers_to_transform` is different to None"
                " and if the layer pattern is not in the common layers pattern."
                " This only works when target_modules is a list of str."
            )
        },
    )
    block_size_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to block_size"
                " which are different from the default specified."
                " For example, `{model.decoder.layers.0.encoder_attn.k_proj: 1280`}"
            )
        },
    )
    init_weights: Optional[Union[bool, Literal["gaussian", "kaiming_uniform", "xavier_uniform"]]] = field(
        default="xavier_uniform",
        metadata={
            "help": (
                "The initialization of the C3A weights. Leave it as default or"
                " set it to False if the weights should be initialized with Xavier uniform,"
                " which is experimentally suitable for C3A."
                " Set this to True if the weights should be initialized to zeros."
            )
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.C3A
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # if target_modules is a regex expression, then layers_pattern should be None
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")
