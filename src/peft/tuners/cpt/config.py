import enum
from dataclasses import dataclass, field
from typing import Optional

import torch

from peft.config import PeftConfig
from peft.utils import PeftType


class PromptTuningInit(str, enum.Enum):
    """Enum for specifying the initialization method for prompt tuning."""

    TEXT = "TEXT"  # Initialize using text-based embeddings.
    RANDOM = "RANDOM"  # Initialize randomly.


@dataclass
class CPTConfig(PeftConfig):
    """
    CPT Configuration class extending PeftConfig for Context-aware Prompt Tuning (CPT).

    This class introduces additional parameters required for CPT, such as token type masks,
    prompt tuning initialization, loss weighting, and projection settings.
    """

    # Token-related configurations
    CPT_token_ids: Optional[torch.Tensor] = field(
        default=None, metadata={"help": "Tensor of token IDs used for CPT prompts."}
    )
    CPT_mask: Optional[torch.Tensor] = field(default=None, metadata={"help": "Tensor mask applied to CPT tokens."})
    CPT_tokens_type_mask: Optional[bool] = field(
        default=None, metadata={"help": "Mask indicating the type of each CPT token."}
    )

    # Prompt tuning initialization method
    CPT_prompt_tuning_init: Optional[str] = field(
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
        self.target_modules = None  # Placeholder for target modules in CPT.
        self.task_type = "CAUSAL_LM"  # Ensures task type is causal language modeling.
