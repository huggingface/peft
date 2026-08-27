from peft.utils import register_peft_method

from .config import OSFConfig
from .layer import Linear, OSFLayer
from .model import OSFModel


__all__ = ["Linear", "OSFConfig", "OSFLayer", "OSFModel"]

register_peft_method(
    name="osf",
    config_cls=OSFConfig,
    model_cls=OSFModel,
    is_mixed_compatible=False,
)
