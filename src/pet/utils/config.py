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
    TOKEN_CLS = "TOKEN_CLS"


@dataclass
class PETConfig:
    """
    This is the base configuration class to store the configuration of a :class:`~pet.PETModel`.

    Args:
        pet_type (:obj:Union[:class:`~pet.utils.config.PETType`, :obj:`str`]): The type of PET method to use.
        task_type (:obj:Union[:class:`~pet.utils.config.TaskType`, :obj:`str`]): The type of task to perform.
        inference_mode (:obj:`bool`, defaults to :obj:`False`): Whether to use the PET model in inference mode.
    """

    pet_type: Union[str, PETType] = field(default=None, metadata={"help": "PET type"})
    task_type: Union[str, TaskType] = field(default=None, metadata={"help": "Task type"})
    inference_mode: bool = field(default=False, metadata={"help": "Whether to use inference mode"})


@dataclass
class PromptLearningConfig(PETConfig):
    """
    This is the base configuration class to store the configuration of a :obj:Union[:class:`~pet.PrefixTuning`,
    :class:`~pet.PromptEncoder`, :class:`~pet.PromptTuning`].

    Args:
        num_virtual_tokens (:obj:`int`): The number of virtual tokens to use.
        token_dim (:obj:`int`): The hidden embedding dimension of the base transformer model.
        num_transformer_submodules (:obj:`int`): The number of transformer submodules in the base transformer model.
        num_attention_heads (:obj:`int`): The number of attention heads in the base transformer model.
        num_layers (:obj:`int`): The number of layers in the base transformer model.
    """

    num_virtual_tokens: int = field(default=None, metadata={"help": "Number of virtual tokens"})
    token_dim: int = field(
        default=None, metadata={"help": "The hidden embedding dimension of the base transformer model"}
    )
    num_transformer_submodules: Optional[int] = field(default=1, metadata={"help": "Number of transformer submodules"})
    num_attention_heads: Optional[int] = field(default=None, metadata={"help": "Number of attention heads"})
    num_layers: Optional[int] = field(default=None, metadata={"help": "Number of transformer layers"})
