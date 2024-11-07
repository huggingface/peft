# Copyright 2024-present the HuggingFace Inc. team.
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
from typing import Optional

from peft.config import PromptLearningConfig
from peft.utils import PeftType


class CPTPromptInit(str, enum.Enum):
    """Enum for specifying the initialization method for CPT."""

    TEXT = "TEXT"  # Initialize using text-based embeddings.
    RANDOM = "RANDOM"  # Initialize randomly.


@dataclass
class CPTConfig(PromptLearningConfig):
    """
    CPT Configuration class extending PeftConfig for Context-aware Prompt Tuning (CPT).

    This class introduces additional parameters required for CPT, such as:
    - Token type masks
    - Prompt tuning initialization
    - Loss weighting
    - Projection settings

    For more details, see the paper: https://arxiv.org/abs/2410.17222
    """

    # Token-related configurations
    cpt_token_ids: Optional[list[int]] = field(
        default=None, metadata={"help": "Tensor of token IDs used for CPT prompts."}
    )
    cpt_mask: Optional[list[int]] = field(default=None, metadata={"help": "Tensor mask applied to CPT tokens."})
    cpt_tokens_type_mask: Optional[list[int]] = field(
        default=None, metadata={"help": "Mask indicating the type of each CPT token."}
    )

    # Prompt tuning initialization method
    cpt_prompt_init: Optional[str] = field(
        default="TEXT", metadata={"help": "Initialization method: 'TEXT' for embedding-based, 'RANDOM' for random."}
    )

    # Loss-related configurations
    opt_weighted_loss_type: Optional[str] = field(
        default="none", metadata={"help": "Type of weighted loss: 'none' or 'decay'."}
    )
    opt_loss_decay_factor: Optional[float] = field(
        default=1.0, metadata={"help": "Factor for exponential decay in loss weighting."}
    )

    # Projection-related configurations
    opt_projection_epsilon: Optional[float] = field(
        default=0.1, metadata={"help": "Epsilon value for input projection."}
    )
    opt_projection_format_epsilon: Optional[float] = field(
        default=0.1, metadata={"help": "Epsilon value for format projection."}
    )

    # Tokenizer configuration
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer to use for prompt tuning initialization. Only used if prompt_tuning_init is `TEXT`"
        },
    )

    # Virtual token configurations
    num_virtual_tokens: int = field(default=0, metadata={"help": "Number of virtual tokens used in the prompt."})

    # CPT-specific static attributes
    is_prompt_learning = True  # Indicates that CPT is a prompt-learning method.
    num_layers = None  # Number of layers (optional, not always required).
    token_dim = None  # Dimension of token embeddings.
    num_attention_heads = None  # Number of attention heads (if applicable).
    task_type = "CAUSAL_LM"  # Specifies that CPT is used for causal language modeling.
    num_transformer_submodules = 1  # Number of transformer submodules used.

    def __post_init__(self):
        """
        Post-initialization hook to set additional attributes after the config is initialized.
        """
        self.peft_type = PeftType.CPT  # Specifies that the PEFT type is CPT.
        self.task_type = "CAUSAL_LM"  # Ensures task type is causal language modeling.

        if (self.cpt_prompt_init == CPTPromptInit.TEXT) and self.cpt_token_ids is None:
            raise ValueError(f"When prompt_tuning_init='{CPTPromptInit.TEXT.value}', " f"cpt_token_ids can't be None.")
        if (self.cpt_prompt_init == CPTPromptInit.TEXT) and self.cpt_mask is None:
            raise ValueError(f"When prompt_tuning_init='{CPTPromptInit.TEXT.value}', " f"cpt_mask can't be None.")
        if (self.cpt_prompt_init == CPTPromptInit.TEXT) and self.cpt_tokens_type_mask is None:
            raise ValueError(
                f"When prompt_tuning_init='{CPTPromptInit.TEXT.value}', " f"cpt_tokens_type_mask can't be None."
            )
        if (self.cpt_prompt_init == CPTPromptInit.TEXT) and not (
            len(self.cpt_token_ids) == len(self.cpt_mask) == len(self.cpt_tokens_type_mask) == self.num_virtual_tokens
        ):
            raise ValueError(
                f"When prompt_tuning_init='{CPTPromptInit.TEXT.value}', "
                f"cpt_token_ids, cpt_mask and cpt_tokens_type_mask must have the same length."
            )

        if (self.cpt_prompt_init == CPTPromptInit.RANDOM) and self.cpt_token_ids is not None:
            raise ValueError(
                f"When prompt_tuning_init='{CPTPromptInit.RANDOM.value}', " f"cpt_token_ids must be None."
            )
        if (self.cpt_prompt_init == CPTPromptInit.RANDOM) and self.cpt_mask is not None:
            raise ValueError(f"When prompt_tuning_init='{CPTPromptInit.RANDOM.value}', " f"cpt_mask must be None.")
        if (self.cpt_prompt_init == CPTPromptInit.RANDOM) and self.cpt_tokens_type_mask is not None:
            raise ValueError(
                f"When prompt_tuning_init='{CPTPromptInit.RANDOM.value}', " f"cpt_tokens_type_mask must be None."
            )
        if (self.cpt_prompt_init == CPTPromptInit.RANDOM) and self.num_virtual_tokens == 0:
            raise ValueError(
                f"When prompt_tuning_init='{CPTPromptInit.RANDOM.value}', "
                f"num_virtual_tokens must be greater than zero."
            )
        if (self.cpt_prompt_init != CPTPromptInit.RANDOM) and (self.cpt_prompt_init != CPTPromptInit.TEXT):
            raise ValueError("prompt_tuning_init must be 'RANDOM' or 'TEXT'")
