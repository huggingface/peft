from dataclasses import dataclass, field
from peft.tuners.lora.config import LoraConfig
from peft.utils.peft_types import PeftType

@dataclass
class PCLoraConfig(LoraConfig):
    decay_schedule: str = field(default="linear", metadata={"help": "Decay schedule for LoRA."})
    q: int = field(default=None, metadata={"help": "Hyperparameter for decay schedule. Steps after q will have lambda=0."})
    task_loss_alpha: float =field(default=0.1, metadata={"help": "Weighting of the task loss. Feature distillation and task loss will be convexly combined"})
    keep_constant_for_k_steps: int  = field(default=10, metadata={"help": "Keep the lambda constant for k steps."})
    
    def __post_init__(self):
        self.peft_type = PeftType.PCLORA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )