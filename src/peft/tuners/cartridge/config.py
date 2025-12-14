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

from peft.config import PromptLearningConfig
from peft.utils import PeftType


@dataclass
class CartridgeConfig(PromptLearningConfig):
    """
    Configuration for CARTRIDGE, a KV-cache-parameterized prefix adapter.

    This is similar to prefix-tuning in how it is served (as `past_key_values`), but it stores the KV cache directly as
    trainable parameters instead of learning it via an MLP projection.

    Args:
        num_frozen_tokens (`int`, defaults to 0):
            Number of *prefix* tokens at the start of the cartridge to keep frozen (no gradients).
    """

    num_frozen_tokens: int = field(
        default=0,
        metadata={"help": "Number of initial virtual tokens to freeze (no gradients)."},
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.CARTRIDGE
