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

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional

from peft.config import PromptLearningConfig
from peft.utils import PeftType, TaskType


@dataclass
class CPTConfig(PromptLearningConfig):
    """
    CPT Configuration class extending PeftConfig for Context-aware Prompt Tuning (CPT).

    This class introduces additional parameters required for CPT, such as:
    - Token type masks
    - Prompt tuning initialization
    - Loss weighting
    - Projection settings

    For more details, see the paper: https://huggingface.co/papers/2410.17222
    """

    # Token-related configurations
    cpt_token_ids: Optional[list[int]] = field(
        default=None, metadata={"help": "Tensor of token IDs used for CPT prompts."}
    )
    cpt_mask: Optional[list[int]] = field(default=None, metadata={"help": "Tensor mask applied to CPT tokens."})
    cpt_tokens_type_mask: Optional[list[int]] = field(
        default=None, metadata={"help": "Mask indicating the type of each CPT token."}
    )

    # Loss-related configurations
    opt_weighted_loss_type: Optional[Literal["none", "decay"]] = field(
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
    # Neet to define CPT-specific static attributes
    is_prompt_learning = True  # Indicates that CPT is a prompt-learning method.

    def __post_init__(self):
        """
        Post-initialization hook to set additional attributes after the config is initialized.
        """
        # CPT-specific static attributes
        self.is_prompt_learning = True  # Indicates that CPT is a prompt-learning method.
        self.num_layers = None  # Number of layers (optional, not always required).
        self.token_dim = None  # Dimension of token embeddings.
        self.num_attention_heads = None  # Number of attention heads (if applicable).
        self.num_transformer_submodules = 1  # Number of transformer submodules used.
        self.peft_type = PeftType.CPT  # Specifies that the PEFT type is CPT.
        if self.task_type != TaskType.CAUSAL_LM:
            # TODO: adjust this to raise an error with PEFT v0.18.0
            warnings.warn(
                f"{self.__class__.__name__} only supports task_type = {TaskType.CAUSAL_LM.value}, "
                "setting it automatically. This will raise an error starting from PEFT v0.18.0.",
                FutureWarning,
            )
            self.task_type = TaskType.CAUSAL_LM  # Ensures task type is causal language modeling.

        if self.cpt_token_ids is None:
            self.cpt_token_ids = [0]

        self.num_virtual_tokens = len(self.cpt_token_ids)

        if self.cpt_mask is None:
            self.cpt_mask = [1 for _ in self.cpt_token_ids]

        if self.cpt_tokens_type_mask is None:
            self.cpt_tokens_type_mask = [1 for _ in self.cpt_token_ids]

        if not (
            len(self.cpt_token_ids) == len(self.cpt_mask) == len(self.cpt_tokens_type_mask) == self.num_virtual_tokens
        ):
            raise ValueError("cpt_token_ids, cpt_mask and cpt_tokens_type_mask must have the same length.")
