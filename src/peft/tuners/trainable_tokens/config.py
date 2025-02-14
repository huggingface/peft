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

from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class TrainableTokensConfig(PeftConfig):
    """
    Configuration for the `TrainableTokens` method.

    Since this method attempts to make certain tokens (identified by their indices) trainable without as large a memory
    footprint as training the target matrix fully the following arguments are need.

    Args:
        token_indices (`list[int]`):
            Indices of the tokens that are supposed to be trainable. The closer the amount of indices is to the total
            amount of tokens, the less efficient this method gets.
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to target. Defaults to 'embedding'.
    """

    token_indices: List[int] = field(
        default_factory=list,
        metadata={
            "help": (
                "List of integers, signifying the indices of the tokens you want to be trainable. "
                "To find the index of a token with a tokenizer, you can tokenize the string and "
                "look at the returned `input_ids`."
            )
        },
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default_factory=lambda: ["embedding"],
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with our "
                "`TrainableTokensLayer`. This is by default the `embedding` layer. "
                "But could be multiple embedding-like layers, such as `embed_tokens`, `encoder.embeddings` or "
                "`decoder.embeddings`."
            ),
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.TRAINABLE_TOKENS
