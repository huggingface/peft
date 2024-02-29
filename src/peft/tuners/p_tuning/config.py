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

import enum
from dataclasses import dataclass, field
from typing import Union

from peft.config import PromptLearningConfig
from peft.utils import PeftType


class PromptEncoderReparameterizationType(str, enum.Enum):
    MLP = "MLP"
    LSTM = "LSTM"


@dataclass
class PromptEncoderConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`PromptEncoder`].

    Args:
        encoder_reparameterization_type (Union[[`PromptEncoderReparameterizationType`], `str`]):
            The type of reparameterization to use.
        encoder_hidden_size (`int`): The hidden size of the prompt encoder.
        encoder_num_layers (`int`): The number of layers of the prompt encoder.
        encoder_dropout (`float`): The dropout probability of the prompt encoder.
    """

    encoder_reparameterization_type: Union[str, PromptEncoderReparameterizationType] = field(
        default=PromptEncoderReparameterizationType.MLP,
        metadata={"help": "How to reparameterize the prompt encoder"},
    )
    encoder_hidden_size: int = field(
        default=None,
        metadata={"help": "The hidden size of the prompt encoder"},
    )
    encoder_num_layers: int = field(
        default=2,
        metadata={"help": "The number of layers of the prompt encoder"},
    )
    encoder_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout of the prompt encoder"},
    )

    def __post_init__(self):
        self.peft_type = PeftType.P_TUNING
