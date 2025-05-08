from peft.tuners.uilinlora.config import UILinLoRAConfig
from peft.tuners.uilinlora.model import UILinLoRAModel
from peft.tuners.uilinlora.bnb import Linear8bitLt, Linear4bit
from peft.utils import register_peft_method

register_peft_method(name="uilinlora", config_cls=UILinLoRAConfig, model_cls=UILinLoRAModel)

__all__ = ["UILinLoRAConfig", "UILinLoRAModel", "Linear8bitLt", "Linear4bit"] 