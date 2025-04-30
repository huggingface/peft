# https://github.com/IBM/activated-lora/blob/main/alora/config.py
from dataclasses import dataclass, field
import warnings

from peft.utils.peft_types import PeftType, TaskType

from ..lora.config import LoraConfig


@dataclass
class ALoraConfig(LoraConfig):
    """
    aLORA configuration class. Inherits from LoraConfig.
        It subclasses PEFT's LoraConfig, modifies the default rank r to 32 (often best), and adds an additional parameter:
        r (`int`):
            aLora attention dimension (the "rank").
            Typically needs to be higher than used for standard Lora. Default=32.
        invocation_string (str):
            String intended to activate the aLoRA. The aLoRA adapted weights will activate
            1 token after the first token in this string. This string must be present in all input data.
    """

    r: int = field(default=32, metadata={"help": "Lora attention dimension"})

    invocation_string: str = field(
        default=None,
        metadata={
            "help": (
                "aLoRA invocation string. The aLoRA adapted weights will activate 1 token after the first token in "
                "this string. This string must be present in all input data."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.ALORA

        # Check if the invocation string is provided
        if (
            self.invocation_string is None
            or not isinstance(self.invocation_string, str)
            or len(self.invocation_string) == 0
        ):
            raise ValueError("invocation_string must be a non-empty string.")

        # aLora is only supported for causal LM
        if (isinstance(self.task_type, TaskType) and self.task_type != TaskType.CAUSAL_LM) or (
            self.task_type.lower() != "causal_lm"
        ):
            raise ValueError(
                f"task_type {self.task_type} is not supported for aLora. " "only use TaskType.CAUSAL_LM for aLora."
            )

        # Reusing attention layers only
        allowed_target_modules = ("q_proj", "k_proj", "v_proj")
        if any(not target_module.lower().endswith(allowed_target_modules) for target_module in self.target_modules):
            raise ValueError(
                f"One or more target_modules are not supported for aLora. "
                f"Only use modules ending with {allowed_target_modules} for aLora."
            )

        # DORA is not supported
        if self.use_dora:
            warnings.warn(
                "DORA is not supported for aLoRA. Please set use_dora=False in the config to avoid this warning.",
                UserWarning,
            )
