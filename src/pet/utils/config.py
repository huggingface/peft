import enum
from dataclasses import dataclass, field
from typing import Optional, Union


class PETType(str, enum.Enum):
    PROMPT_TUNING = "PROMPT_TUNING"
    P_TUNING = "P_TUNING"
    PREFIX_TUNING = "PREFIX_TUNING"
    LORA = "LORA"


class TaskType(str, enum.Enum):
    SEQ_CLS = "SEQ_CLS"
    SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
    CAUSAL_LM = "CAUSAL_LM"


@dataclass
class PETConfig:
    """
    This is the configuration class to store the configuration of a :class:`~pet.PETModel`.
    """

    pet_type: Union[str, PETType] = field(default=None, metadata={"help": "PET type"})
    task_type: Union[str, TaskType] = field(default=None, metadata={"help": "Task type"})
    inference_mode: bool = field(default=False, metadata={"help": "Whether to use inference mode"})


@dataclass
class PromptLearningConfig(PETConfig):
    num_virtual_tokens: int = field(default=None, metadata={"help": "Number of virtual tokens"})
    token_dim: int = field(default=None, metadata={"help": "Dimension of virtual tokens"})
    num_transformer_submodules: Optional[int] = field(default=1, metadata={"help": "Number of transformer submodules"})
    num_attention_heads: Optional[int] = field(default=None, metadata={"help": "Number of attention heads"})
    num_layers: Optional[int] = field(default=None, metadata={"help": "Number of transformer layers"})
