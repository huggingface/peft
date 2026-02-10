# config.py
from dataclasses import dataclass, field
from peft.config import PeftConfig
from peft.utils import PeftType

@dataclass
class NonlinearLoraConfig(PeftConfig):
    r: int = field(default=8)
    alpha: int = field(default=16)
    dropout: float = field(default=0.0)
    activation_fn: str = field(default="relu")
    target_modules: list[str] | str | None = field(default=None)  # same semantics as other tuners
    bias: str = field(default="none")

    def __post_init__(self):
        # IMPORTANT: requires PeftType entry exists in your PEFT install
        self.peft_type = PeftType.NLORA
        super().__post_init__()
