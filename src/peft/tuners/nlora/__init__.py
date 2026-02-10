from peft.utils.peft_types import register_peft_method
from .config import NonlinearLoraConfig
from .model import NonlinearLoraModel

register_peft_method(
    name="nlora",
    config_cls=NonlinearLoraConfig,
    model_cls=NonlinearLoraModel,
    prefix="nlora_",
)

__all__ = ["NonlinearLoraConfig", "NonlinearLoraModel"]
