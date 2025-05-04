from peft.tuners.diag.config import DiagConfig
from peft.tuners.diag.model import DiagModel
from peft.tuners.diag.bnb import Linear8bitLt, Linear4bit
from peft.utils import register_peft_method

register_peft_method(name="diag", config_cls=DiagConfig, model_cls=DiagModel)

__all__ = ["DiagConfig", "DiagModel", "Linear8bitLt", "Linear4bit"] 