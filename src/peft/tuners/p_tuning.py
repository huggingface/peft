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

import enum
import warnings
from dataclasses import dataclass, field
from typing import Union

import torch

from ..utils import PeftType, PromptLearningConfig


class PromptEncoderReparameterizationType(str, enum.Enum):
    MLP = "MLP"
    LSTM = "LSTM"


@dataclass
class PromptEncoderConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.PromptEncoder`].

    Args:
        encoder_reparameterization_type
            (Union[[`PromptEncoderReparameterizationType`], `str`]): The type of reparameterization to use.
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


# Based on https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/modules/common/prompt_encoder.py
# with some refactor
class PromptEncoder(torch.nn.Module):
    """
    The prompt encoder network that is used to generate the virtual token embeddings for p-tuning.

    Args:
        config ([`PromptEncoderConfig`]): The configuration of the prompt encoder.

    Example::

        >>> from peft import PromptEncoder, PromptEncoderConfig >>> config = PromptEncoderConfig(
                peft_type="P_TUNING", task_type="SEQ_2_SEQ_LM", num_virtual_tokens=20, token_dim=768,
                num_transformer_submodules=1, num_attention_heads=12, num_layers=12,
                encoder_reparameterization_type="MLP", encoder_hidden_size=768
            )
        >>> prompt_encoder = PromptEncoder(config)

    **Attributes**:
        - **embedding** ([`~torch.nn.Embedding`]) -- The embedding layer of the prompt encoder.
        - **mlp_head** ([`~torch.nn.Sequential`]) -- The MLP head of the prompt encoder if `inference_mode=False`.
        - **lstm_head** ([`~torch.nn.LSTM`]) -- The LSTM head of the prompt encoder if `inference_mode=False` and
        `encoder_reparameterization_type="LSTM"`.
        - **token_dim** (`int`) -- The hidden embedding dimension of the base transformer model.
        - **input_size** (`int`) -- The input size of the prompt encoder.
        - **output_size** (`int`) -- The output size of the prompt encoder.
        - **hidden_size** (`int`) -- The hidden size of the prompt encoder.
        - **total_virtual_tokens** (`int`): The total number of virtual tokens of the
        prompt encoder.
        - **encoder_type** (Union[[`PromptEncoderReparameterizationType`], `str`]):
            The encoder type of the prompt encoder.


    Input shape: (batch_size, total_virtual_tokens)

    Output shape: (batch_size, total_virtual_tokens, token_dim)
    """

    def __init__(self, config):
        super().__init__()
        self.token_dim = config.token_dim
        self.input_size = self.token_dim
        self.output_size = self.token_dim
        self.hidden_size = config.encoder_hidden_size
        self.total_virtual_tokens = config.num_virtual_tokens * config.num_transformer_submodules
        self.encoder_type = config.encoder_reparameterization_type

        # embedding
        self.embedding = torch.nn.Embedding(self.total_virtual_tokens, self.token_dim)
        if not config.inference_mode:
            if self.encoder_type == PromptEncoderReparameterizationType.LSTM:
                lstm_dropout = config.encoder_dropout
                num_layers = config.encoder_num_layers
                # LSTM
                self.lstm_head = torch.nn.LSTM(
                    input_size=self.input_size,
                    hidden_size=self.hidden_size,
                    num_layers=num_layers,
                    dropout=lstm_dropout,
                    bidirectional=True,
                    batch_first=True,
                )

                self.mlp_head = torch.nn.Sequential(
                    torch.nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_size * 2, self.output_size),
                )

            elif self.encoder_type == PromptEncoderReparameterizationType.MLP:
                warnings.warn(
                    f"for {self.encoder_type}, the `encoder_num_layers` is ignored. Exactly 2 MLP layers are used."
                )
                layers = [
                    torch.nn.Linear(self.input_size, self.hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_size, self.hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(self.hidden_size, self.output_size),
                ]
                self.mlp_head = torch.nn.Sequential(*layers)

            else:
                raise ValueError("Prompt encoder type not recognized. Please use one of MLP (recommended) or LSTM.")

    def forward(self, indices):
        input_embeds = self.embedding(indices)
        if self.encoder_type == PromptEncoderReparameterizationType.LSTM:
            output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0])
        elif self.encoder_type == PromptEncoderReparameterizationType.MLP:
            output_embeds = self.mlp_head(input_embeds)
        else:
            raise ValueError("Prompt encoder type not recognized. Please use one of MLP (recommended) or LSTM.")

        return output_embeds
