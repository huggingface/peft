from dataclasses import dataclass, field
from peft.tuners.lora.config import LoraConfig
from peft.utils.peft_types import PeftType

from logging import getLogger

logger = getLogger(__name__)
@dataclass
class PCLoraConfig(LoraConfig):
    decay_schedule: str = field(default="linear", metadata={"help": "Decay schedule for LoRA."})
    total_steps: int = field(default=None, metadata={"help": "Total number of steps for LoRA."})
    distillation_loss_lambda: float =field(default=0.1, metadata={"help": "Distillation loss lambda for LoRA."})
    
    def __post_init__(self):
        self.peft_type = PeftType.PCLORA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        logger.info(f"PEFT after post_init in {PCLoraConfig} type: {self.peft_type}")