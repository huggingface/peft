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

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class IA3Config(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`IA3Model`].

    Args:
        target_modules (`Union[List[str],str]`):
            The names of the modules to apply (IA)^3 to.
        feedforward_modules (`Union[List[str],str]`):
            The names of the modules to be treated as feedforward modules, as in the original paper.
        fan_in_fan_out (`bool`):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.
        modules_to_save (`List[str]`):
            List of modules apart from (IA)^3 layers to be set as trainable and saved in the final checkpoint.
        init_ia3_weights (`bool`):
            Whether to initialize the vectors in the (IA)^3 layers, defaults to `True`.
    """

    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with ia3."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    feedforward_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or a regex expression of module names which are feedforward"
            "For example, ['output.dense']"
        },
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from (IA)^3 layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_ia3_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the vectors in the (IA)^3 layers."},
    )

    def __post_init__(self):
        self.peft_type = PeftType.IA3
