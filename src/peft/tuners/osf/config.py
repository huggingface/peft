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
        effective_rank (`int` or `float`, *optional*):
            Preserved SVD rank ("high" subspace). The top-``effective_rank`` singular directions are frozen and
            retained across tasks; the remaining dimensions form the trainable low-rank subspace. If `None`, defaults
            to 50% of the smaller weight dimension per target module. Note: This differs from LoRA's `r` (trainable
            rank). In OSF, the trainable rank is `min(weight.shape) - effective_rank`.
        target_modules (`Union[list[str], str]`, *optional*):
            The names of the modules to apply OSF to. Can be a list of module names or `"all-linear"`.
        rank_pattern (`dict[str, int|float]`, *optional*):
            A dictionary of regex patterns to override `effective_rank` for specific modules. Values can be absolute
            integers or fractions in (0, 1], interpreted as a fraction of the smaller matrix dimension per target.
    """

    effective_rank: Optional[Union[int, float]] = field(
        default=None,
        metadata={
            "help": (
                "Preserved SVD rank (frozen). Trainable rank equals min(weight.shape) - effective_rank. "
                "If None, defaults to 50% of the smaller weight dimension."
            )
        },
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "The names of the modules to apply OSF to. Can be a list of module names or 'all-linear'."},
    )
    rank_pattern: Optional[dict[str, Union[int, float]]] = field(
        default=None,
        metadata={
            "help": (
                "A dictionary of regex patterns to override effective_rank per module. Values can be absolute "
                "integers or fractions in (0, 1], interpreted as a fraction of the smaller matrix dimension."
            )
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.OSF
