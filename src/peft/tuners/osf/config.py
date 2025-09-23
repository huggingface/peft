from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class OSFConfig(PeftConfig):
    """
    Configuration for Orthogonal Subspace Fine-tuning (OSF).
    
    Args:
        effective_rank (`int`, *optional*):
            The effective rank for OSF decomposition. If None, defaults to 50% of min(weight.shape).
        target_modules (`Union[list[str], str]`, *optional*):
            The names of the modules to apply OSF to. Can be a list of module names or 'all-linear'.
        rank_pattern (`dict[str, int]`, *optional*):
            A dictionary of regex patterns to override effective_rank for specific modules.
    """

    effective_rank: Optional[int] = field(
        default=None,
        metadata={"help": "The effective rank for OSF decomposition. If None, defaults to 50% of min(weight.shape)."}
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "The names of the modules to apply OSF to. Can be a list of module names or 'all-linear'."}
    )
    rank_pattern: Optional[dict[str, int]] = field(
        default=None,
        metadata={"help": "A dictionary of regex patterns to override effective_rank for specific modules."}
    )

    def __post_init__(self):
        self.peft_type = PeftType.OSF