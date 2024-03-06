from dataclasses import asdict, dataclass, field
from typing import List, Optional, Union
from peft import PeftConfig
from peft.utils import PeftType


@dataclass
class CustomTokensConfig(PeftConfig):
    token_indices: List[int] = field(default_factory=list)
    target_modules: Optional[Union[list[str], str]] = field(
        default='embedding',
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with our CustomTokensLayer."
                "This is by default the `embedding` layer.",
                "But could be multiple embedding-like layers, such as `encoder.embeddings` or `decoder.embeddings`."
            ),
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.CUSTOM_TOKENS