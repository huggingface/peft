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

    Initialization:
        The Cartridges paper discusses multiple initialization options. In PEFT, initialization is a *separate* step
        from constructing the adapter config:

        - **Random KV initialization (paper option 2)**: Create the adapter via `get_peft_model(...)`. The CARTRIDGE
          prompt encoder parameters are randomly initialized by PyTorch.

        - **KV derived from the first tokens of a prompt/corpus (paper option 3)**: Run a no-grad prefill on the *base
          model* and copy the first `num_virtual_tokens` cached KV tokens into the adapter. PEFT provides utilities for
          this (importable from `peft` or from `peft.tuners.cartridge.utils`):

          - `initialize_kv_prefix_from_text(model, tokenizer, text=...)`
          - `initialize_kv_prefix_from_past_key_values(model, past_key_values=...)`

          If you already have a flattened KV-prefix tensor, you can load it directly via the prompt encoder’s
          `load_prompt_embeddings(...)` method.

    Args:
        num_frozen_tokens (`int`, defaults to 1):
            Number of *prefix* tokens at the start of the cartridge to keep frozen (no gradients). The Cartridges paper
            recommends freezing the first token as an attention sink for stability (set this to `1`), as many LLMs use
            early tokens as attention sinks and changing them can harm training.
    """

    num_frozen_tokens: int = field(
        default=1,
        metadata={
            "help": (
                "Number of initial virtual tokens to freeze (no gradients). The paper recommends freezing the first "
                "token as an attention sink for stability."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        if self.num_frozen_tokens < 0:
            raise ValueError(f"`num_frozen_tokens` must be >= 0, got {self.num_frozen_tokens}.")
        # `num_virtual_tokens` is required for prompt-learning configs. Validate the relationship early for a clearer
        # error, even if the encoder also checks it.
        if (self.num_virtual_tokens is not None) and (self.num_frozen_tokens > self.num_virtual_tokens):
            raise ValueError(
                f"`num_frozen_tokens` must be <= `num_virtual_tokens`, got num_frozen_tokens={self.num_frozen_tokens} "
                f"and num_virtual_tokens={self.num_virtual_tokens}."
            )
        self.peft_type = PeftType.CARTRIDGE
