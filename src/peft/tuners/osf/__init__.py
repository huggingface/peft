from peft.utils import register_peft_method

from .config import OSFConfig
from .layer import OSFLayer, Linear
from .model import OSFModel

__all__ = ["OSFConfig", "OSFModel", "OSFLayer", "Linear"]

register_peft_method(
    name="osf",
    config_cls=OSFConfig,
    model_cls=OSFModel,
    is_mixed_compatible=False,
)