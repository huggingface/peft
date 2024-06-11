from dataclasses import dataclass, field
from typing import List, Optional, Union
from peft.utils import PeftType
from peft.config import PeftConfig


@dataclass
class GLoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`GLoraModel`].

    Args:
        r (`int`): GLora attention dimension.
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
        layers_to_transform (`Union[List[int],int]`):
            The layer indexes to transform, if this argument is specified, it will apply the LoRA transformations on
            the layer indexes that are specified in this list. If a single integer is passed, it will apply the LoRA
            transformations on the layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None` and if the layer
            pattern is not in the common layers pattern.
    """

    r: int = field(default=4, metadata={"help": "Lora attention dimension"})
    megatron_config: Optional[dict] = field(
        default=None,
        metadata={
            "help": (
                "The TransformerConfig from Megatron. It is used to create LoRA's parallel linear layer."
                "You can get it like this, `core_transformer_config_from_args(get_args())`, "
                "these two functions being from Megatron."
                "You need to specify this parameter when you want to apply LoRA to the ColumnParallelLinear and "
                "RowParallelLinear layers of megatron."
                "It should be noted that we may not be able to use the `save_pretrained` and `from_pretrained` "
                "functions, because TransformerConfig may not necessarily be serialized."
                "But when using megatron, we can use `get_peft_model_state_dict` function and "
                "megatron's framework, they can also save and load models and configurations."
            )
        },
    )
    megatron_core: Optional[str] = field(
        default="megatron.core",
        metadata={
            "help": (
                "The core module from Megatron, it is used to create LoRA's parallel linear layer. "
                "It only needs to be passed in when you need to use your own modified megatron core module. "
                "Otherwise, it will use the default value `megatron.core`. "
            )
        },
    )    
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )

    def __post_init__(self):
        self.peft_type = PeftType.GLORA