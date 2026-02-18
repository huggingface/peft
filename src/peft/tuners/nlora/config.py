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
    consolidate_batches: int | None = field(default=None) # number of batches to use for consolidation (None = all)
    consolidate_rls: bool = field(default=True) # use recursive least squares for consolidation instead of direct ridge regression (can be more efficient for large d or many batches)
    consolidate_layer_update_frequency: int | None = field(default=None) # frequency (in number of batches) to update consolidation stats per layer (None = every layer)
    consolidate_zero_shift: bool = field(default=False) # whether to perform zero-shift consolidation (i.e. directly fit adapter outputs instead of deltas, which can be more stable but less efficient and may require more careful tuning of lambda)

    def __post_init__(self):
        # IMPORTANT: requires PeftType entry exists in your PEFT install
        self.peft_type = PeftType.NLORA
        super().__post_init__()
