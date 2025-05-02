# duplicated from src/peft/tuners/lora/__init__.py

from peft.import_utils import is_bnb_4bit_available, is_bnb_available, is_eetq_available
from peft.utils.peft_types import register_peft_method

from .config import ALoraConfig
from .model import ALoraModel


__all__ = ["ALoraConfig", "ALoraModel", "is_eetq_available"]

register_peft_method(
    name="alora", config_cls=ALoraConfig, model_cls=ALoraModel, prefix="lora_", is_mixed_compatible=True
)


def __getattr__(name):
    # interested in bitsandbytes 4bit and 8bit only atm
    if (name == "Linear8bitLt") and is_bnb_available():
        from ..lora.bnb import Linear8bitLt

        return Linear8bitLt

    if (name == "Linear4bit") and is_bnb_4bit_available():
        from ..lora.bnb import Linear4bit

        return Linear4bit

    # if (name == "EetqLoraLinear") and is_eetq_available():
    #     from .eetq import EetqLoraLinear

    #     return EetqLoraLinear

    raise AttributeError(f"module {__name__} has no attribute {name}")
