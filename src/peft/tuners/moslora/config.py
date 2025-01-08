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

import warnings
from dataclasses import dataclass, field
from typing import Optional, Literal

from peft.tuners.lora import LoraConfig
from peft.utils import PeftType


@dataclass
class MoSLoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.MoSLoRA`].

    Args:
        use_moslora: (`bool` | `Literal["kai", "orth"]`):

    """
    use_moslora: (bool | Literal["kai", "orth"]) = field(
        default=True,
        metadata={
            "help": (
                "Whether to enable 'Mixture-of-Subspaces in Low-Rank Adaptation' (MoSLoRA)."
                "This technique employs a learnable mixer to fuse more subspaces in vanilla LoRA and more flexibly."
                "In code implement, MoSLoRA inserts a mixer of r*r size between lora_A and lora_B."
                "For more information, see https://arxiv.org/pdf/2406.11909."
                "Passing `'False'` to disable mixer and thus it would be same as vanilla LoRA"
                "Passing `'kai'` results in Kaiming Uniform initialization for Mixer."
                "Passing `'orth'` results in Orthogonal initialization for Mixer."
                "Passing `'True'` would enable Kaiming Uniform initialization which is default"
            ),
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.MOSLORA

        if isinstance(self.use_moslora, str) or self.use_moslora is True:
            if self.use_dora:
                raise ValueError(f"{self.peft_type} does not support DoRA. You can set use_moslora=False to enable this.")

            if self.loftq_config:
                raise ValueError(f"{self.peft_type} does not support LOFTQ.")

            if isinstance(self.init_lora_weights, str):
                raise ValueError(f"{self.peft_type} does not support other initilization methods. You can set use_moslora=False to enable this.")

        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )
        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # check for layers_to_transform and layers_pattern
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified. ")

        
        
