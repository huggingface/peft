from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class PolyConfig(PeftConfig):
    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Poly."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from Poly layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_poly_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the Poly layers with their default initialization. Don't change "
                "this setting, except if you know exactly what you're doing."
            ),
        },
    )

    poly_type: str = field(
        default="poly",
        metadata={"help": "Type of Poly module to be used."},
    )
    n_tasks: int = field(
        default=None,
        metadata={"help": "Number of tasks in multitasking scenario."},
    )
    n_skills: int = field(
        default=4,
        metadata={"help": "Number of skills (experts) in each Poly layer."},
    )
    n_splits: int = field(
        default=1,
        metadata={"help": "Number of splits (splits) in each LoRA in the Poly layer."},
    )

    def __post_init__(self):
        self.peft_type = PeftType.POLY
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
