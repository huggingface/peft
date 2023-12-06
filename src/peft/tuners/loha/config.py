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
class LoHaConfig(LycorisConfig):
    """
    This is the configuration class to store the configuration of a [`LoHaModel`].

    Args:
        r (`int`): LoHa rank.
        alpha (`int`): The alpha parameter for LoHa scaling.
        rank_dropout (`int`): The dropout probability for rank dimension during training.
        module_dropout (`int`): The dropout probability for disabling LoHa modules during training.
        use_effective_conv2d (`bool`):
            Use parameter effective decomposition for Conv2d with ksize > 1 ("Proposition 3" from FedPara paper).
        target_modules (`Union[List[str],str]`): The names of the modules to apply LoHa to.
        init_weights (`bool`): Whether to perform initialization of LoHa weights.
        layers_to_transform (`Union[List[int],int]`):
            The layer indexes to transform, if this argument is specified, it will apply the LoHa transformations on
            the layer indexes that are specified in this list. If a single integer is passed, it will apply the LoHa
            transformations on the layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None` and if the layer
            pattern is not in the common layers pattern.
        rank_pattern (`dict`):
            The mapping from layer names or regexp expression to ranks which are different from the default rank
            specified by `r`.
        alpha_pattern (`dict`):
            The mapping from layer names or regexp expression to alphas which are different from the default alpha
            specified by `alpha`.
        modules_to_save (`List[str]`): The names of modules to be set as trainable except LoHa parameters.
    """

    r: int = field(default=8, metadata={"help": "LoHa rank"})
    alpha: int = field(default=8, metadata={"help": "LoHa alpha"})
    rank_dropout: float = field(
        default=0.0, metadata={"help": "The dropout probability for rank dimension during training"}
    )
    module_dropout: float = field(
        default=0.0, metadata={"help": "The dropout probability for disabling LoHa modules during training"}
    )
    use_effective_conv2d: bool = field(
        default=False,
        metadata={
            "help": 'Use parameter effective decomposition for Conv2d 3x3 with ksize > 1 ("Proposition 3" from FedPara paper)'
        },
    )
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with LoHa."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the LoHa layers with their default initialization. Don't change "
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
            "help": "List of modules apart from LoHA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.LOHA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
