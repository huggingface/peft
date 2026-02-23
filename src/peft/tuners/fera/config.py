# config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PeftConfig


@dataclass
class FeRAConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`FeRAModel`].
    """

    rank: int = field(
        default=4,
        metadata={"help": "The rank of the LoRA experts."},
    )
    lora_alpha: float = field(
        default=8.0,
        metadata={"help": "The scaling factor for the LoRA experts."},
    )
    dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout probability for the LoRA experts."},
    )
    num_bands: int = field(
        default=3,
        metadata={"help": "Num of frequency bands for the Frequency-Energy Indicator."},
    )
    num_experts: int = field(
        default=3,
        metadata={"help": "Number of LoRA experts to route between."},
    )
    router_tau: float = field(
        default=0.7,
        metadata={"help": "Temperature for the Softmax in the router."},
    )
    fecl_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for the Frequency-Energy Consistency Loss."},
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to replace with FeRA."},
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude from FeRA."},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for FeRA. Can be 'none', 'all' or 'lora_only'."})
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from FeRA layers to be set as trainable and saved in the final checkpoint."
        },
    )
    init_lora_weights: bool = field(
        default=True,
        metadata={"help": "Whether to initialize the weights of the LoRA experts."},
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list."
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "The layer pattern name, used only if `layers_to_transform` is different to None."},
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = "FeRA"
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )
