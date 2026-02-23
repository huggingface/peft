# __init__.py
from peft.utils import register_peft_method

from .config import FeRAConfig
from .layer import FeRALayer, FeRALinear
from .model import FeRAModel


__all__ = ["FeRAConfig", "FeRALayer", "FeRALinear", "FeRAModel"]

register_peft_method(name="FeRA", model_cls=FeRAModel, config_cls=FeRAConfig)
