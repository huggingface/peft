from peft.utils import register_peft_method

from .config import FRODConfig
from .layer import FRODLayer, Linear
from .model import FRODModel


__all__ = ["FRODConfig", "FRODLayer", "FRODModel", "Linear"]

register_peft_method(name="frod", config_cls=FRODConfig, model_cls=FRODModel, prefix="frod_")
