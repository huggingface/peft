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

    # consolidation params
    consolidate_lambda: float = field(default=1e-3)   # regularization strength for consolidation (0: no consolidation, inf: L2 penalty)
    consolidate_lr: float = field(default=1e-3)       # learning rate for consolidation
    consolidate_offload_cpu: bool = field(default=True) # accumulate on CPU to save VRAM
    consolidate_scale_lambda_by_trace: bool = field(default=True) # scale lambda by trace of X^T X to make it more invariant to feature scale and count

    def __post_init__(self):
        # IMPORTANT: requires PeftType entry exists in your PEFT install
        self.peft_type = PeftType.NLORA
        super().__post_init__()
