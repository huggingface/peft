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
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class TrainableTokensConfig(PeftConfig):
    """
    Configuration for the `TrainableTokens` method.

    Allows for training new tokens (and re-training existing ones) without training the full embedding matrix. By
    marking a few select tokens (identified by their indices) trainable and leaving the rest untouched, this method can
    be used to add new tokens or changing the embedding of existing tokens while saving on memory. Both storage as well
    as working memory usage are reduced in contrast to training the embedding matrix fully.

    Note that training with FSDP/DeepSpeed might not yet be fully supported. Also note that models using weight tying
    are currently not supported and will raise an error.

    Args:
        token_indices (`list[int]`):
            List of integers, signifying the indices of the tokens you want to be trainable. To find the index of a
            token with a tokenizer, you can tokenize the string and look at the returned `input_ids`. The closer the
            amount of indices is to the total amount of tokens, the less efficient this method gets.
        target_modules (`Optional[Union[list[str], str]]`):
            List of module names or regex expression of the module names to replace with our `TrainableTokensLayer`.
            This is by default the `embed_tokens` layer. But could be multiple embedding-like layers, such as
            `embedding`, `encoder.embeddings` or `decoder.embeddings`.
        init_weights (`bool`):
            By default the new token weights are initialized to be the same as the respective token embeddings. This
            makes TrainableTokens a no-op when not trained. If set to `False` the weights will be random values. Do not
            change this setting unless you know exactly what you're doing.
    """

    token_indices: list[int] = field(
        default_factory=list,
        metadata={
            "help": (
                "List of integers, signifying the indices of the tokens you want to be trainable. "
                "To find the index of a token with a tokenizer, you can tokenize the string and "
                "look at the returned `input_ids`. The closer the amount of indices is to the total amount of "
                "tokens, the less efficient this method gets."
            )
        },
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default_factory=lambda: ["embed_tokens"],
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with our "
                "`TrainableTokensLayer`. This is by default the `embed_tokens` layer. "
                "But could be multiple embedding-like layers, such as `embedding`, `encoder.embeddings` or "
                "`decoder.embeddings`."
            ),
        },
    )

    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "By default the new token weights are initialized to be the same as the respective token embeddings. "
                "This makes TrainableTokens a no-op when not trained. If set to `False` the weights will be random "
                "values. Do not change this setting unless you know exactly what you're doing. "
            )
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.TRAINABLE_TOKENS
