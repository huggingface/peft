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

# Author: Yixiao Li
import warnings
from dataclasses import dataclass, field

from peft.tuners.lora import LoraConfig
from peft.utils import PeftType


@dataclass
class LoftQConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`LoftQModel`].

    Args:
        bits (`int`): Quantization bits, choose from [2, 4, 8].
        num_iters (`int`): Number of alternating steps for LoftQ.
        fake_quant (`bool`): Whether to use fake quantization.
    """

    bits: int = field(
        default=4,
        metadata={"help": "Quantization bits, choose from [2, 4, 8]"}
    )
    num_iter: int = field(
        default=1,
        metadata={"help": "Number of alternating steps for LoftQ"}
    )
    fake_quant: bool = field(
        default=False,
        metadata={"help": "Whether to use fake quantization"}
    )
    loftq_init: bool = field(
        default=False,
        metadata={"help": "True if you want to apply LoftQ to your model; "
                          "False if you download it from the HuggingFace model hub"}
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.LOFTQ
        if self.lora_alpha != self.r:
            warnings.warn(f"`lora_alpha` is usually the same as rank `r`,"
                          "but got lora_alpha={self.lora_alpha} and r={self.r}")
