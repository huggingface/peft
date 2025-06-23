from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from torch import nn
from peft.utils import PeftType
from peft.tuners.lora import LoraConfig


@dataclass
class aLoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of an [`aLoraModel`].

    It subclasses PEFT's LoraConfig, modifies the default rank r to 32 (often best), and adds an additional parameter:
        r (`int`): aLora attention dimension (the "rank"). Typically needs to be higher than used for standard Lora. Default=32.
        invocation_string (str): String intended to activate the aLoRA. The aLoRA adapted weights will activate
                                 1 token after the first token in this string. This string must be present in all input data.
    """
    r: int = field(default=32, metadata={"help": "aLora attention dimension. Typically needs to be higher than used for standard Lora. Default=32."})
    invocation_string: str = field(
        default=None,
        metadata={
            "help": (
                "aLoRA invocation string. The aLoRA adapted weights will activate 1 token after the first token in "
                "this string. This string must be present in all input data."
            )
        }
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.ALORA
        if self.invocation_string is None:
            warnings.warn("invocation_string cannot be None for aLoRA.", UserWarning)
        # The r field with default=32 is handled by the dataclass field definition.
        # LoraConfig's __post_init__ does not modify self.r.

