from dataclasses import dataclass, field
from peft.tuners.lora.config import LoraConfig

@dataclass
class PCLoraConfig(LoraConfig):
    decay_schedule: str = field(default="linear", metadata={"help": "Decay schedule for LoRA."})
    total_steps: int = field(default=None, metadata={"help": "Total number of steps for LoRA."})
    distillation_loss_lambda: float =field(default=0.1, metadata={"help": "Distillation loss lambda for LoRA."})
    