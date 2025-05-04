from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union, List

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class DiagConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`DiagModel`].

    Args:
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the specified
            names will be replaced. When passing a string, a regex match will be performed. When passing a list of
            strings, either an exact match will be performed or it is checked if the name of the module ends with any
            of the passed strings. If this is specified as 'all-linear', then all linear/Conv1D modules are chosen (if
            the model is a PreTrainedModel, the output layer excluded). If this is not specified, modules will be
            chosen according to the model architecture. If the architecture is not known, an error will be raised -- in
            this case, you should specify the target modules manually.
        diag_alpha (`float`):
            The alpha parameter for row-based scaling.
        diag_dropout (`float`):
            The dropout probability for row-based layers.
        fan_in_fan_out (`bool`):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.
        bias (`str`):
            Bias type for row-based adapter. Can be 'none', 'all' or 'row_only'. If 'all' or 'row_only', the corresponding biases
            will be updated during training. Be aware that this means that, even when disabling the adapters, the model
            will not produce the same output as the base model would have without adaptation.
        modules_to_save (`List[str]`):
            List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
        init_diag_weights (`bool`):
            Whether to initialize the weights of the row-based adapter. If True, the first row will be initialized
            to 1.0, meaning the adapter will initially be a no-op. If False, the weights will be randomly initialized.
    """

    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with row-based adapter."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'."
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D "
                "(if the model is a PreTrainedModel, the output layer excluded)."
                "If not specified, modules will be chosen according to the model architecture, If the architecture is "
                "not known, an error will be raised -- in this case, you should specify the target modules manually."
            ),
        },
    )
    diag_alpha: float = field(default=1.0, metadata={"help": "Row-based alpha"})
    diag_dropout: float = field(default=0.0, metadata={"help": "Row-based dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(
        default="none", metadata={"help": "Bias type for row-based adapter. Can be 'none', 'all' or 'row_only'"}
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from row-based layers to be set as trainable and saved in the final checkpoint."
        },
    )
    init_diag_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the row-based adapter. If True, the first row will be "
                "initialized to 1.0, meaning the adapter will initially be a no-op. If False, the weights will be "
                "randomly initialized."
            )
        },
    )

    def __post_init__(self):
        print("[DiagConfig] Initializing configuration")
        print(f"[DiagConfig] Target modules: {self.target_modules}")
        print(f"[DiagConfig] Alpha: {self.diag_alpha}, Dropout: {self.diag_dropout}")
        print(f"[DiagConfig] Fan in/out: {self.fan_in_fan_out}, Bias: {self.bias}")
        print(f"[DiagConfig] Init weights: {self.init_diag_weights}")
        
        super().__post_init__()
        self.peft_type = PeftType.DIAG
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        ) 