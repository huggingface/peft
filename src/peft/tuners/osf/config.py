from __future__ import annotations

from dataclasses import dataclass, field

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class OSFConfig(PeftConfig):
    """Configuration for Orthogonal Subspace Fine-tuning (OSF)."""

    target_svd_config: dict[str, int] | None = field(
        default=None,
        metadata={"help": "Mapping from parameter names to top_k rank."},
    )

    def __post_init__(self):
        self.peft_type = PeftType.OSF