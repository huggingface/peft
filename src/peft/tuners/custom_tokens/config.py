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
class CustomTokensConfig(PeftConfig):
    token_indices: List[int] = field(default_factory=list)
    target_modules: Optional[Union[list[str], str]] = field(
        default_factory=lambda: ['embedding'],
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with our CustomTokensLayer."
                "This is by default the `embedding` layer.",
                "But could be multiple embedding-like layers, such as `embed_tokens`, `encoder.embeddings` or "
                "`decoder.embeddings`."
            ),
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.CUSTOM_TOKENS
