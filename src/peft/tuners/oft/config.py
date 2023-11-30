# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
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

from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.tuners.lycoris_utils import LycorisConfig
from peft.utils import PeftType


@dataclass
class OFTConfig(LycorisConfig):
    """
    This is the configuration class to store the configuration of a [`OFTModel`].

    Args:
        r (`int`): OFT rank.
        module_dropout (`int`): The dropout probability for disabling OFT modules during training.
        target_modules (`Union[List[str],str]`): The names of the modules to apply OFT to.
        init_weights (`bool`): Whether to perform initialization of OFT weights.
        layers_to_transform (`Union[List[int],int]`):
            The layer indexes to transform, if this argument is specified, it will apply the OFT transformations on the
            layer indexes that are specified in this list. If a single integer is passed, it will apply the OFT
            transformations on the layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None` and if the layer
            pattern is not in the common layers pattern.
        rank_pattern (`dict`):
            The mapping from layer names or regexp expression to ranks which are different from the default rank
            specified by `r`.
        modules_to_save (`List[str]`): The names of modules to be set as trainable except OFT parameters.
        coft (`bool`): Whether to use the constrainted variant of OFT or not.
        eps (`float`):
            The control strength of COFT. The freedom of rotation. Only has an effect if `coft` is set to True.
        block_share (`bool`): Whether to share the OFT parameters between blocks or not.
    """

    r: int = field(default=8, metadata={"help": "OFT rank"})
    module_dropout: float = field(
        default=0.0, metadata={"help": "The dropout probability for disabling OFT modules during training"}
    )
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with OFT."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the OFT layers with their default initialization. Don't change "
                "this setting, except if you know exactly what you're doing."
            ),
        },
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
        },
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
        },
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from OFT layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    coft: bool = field(
        default=False,
        metadata={"help": "Whether to use the constrainted variant of OFT or not."},
    )
    eps: float = field(
        default=6e-5,
        metadata={
            "help": "The control strength of COFT. The freedom of rotation. Only has an effect if `coft` is set to True."
        },
    )
    block_share: bool = field(
        default=False,
        metadata={"help": "Whether to share the OFT parameters between blocks or not."},
    )

    def __post_init__(self):
        self.peft_type = PeftType.OFT
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
