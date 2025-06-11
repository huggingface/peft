#from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.utils import register_peft_method

from .config import ShiraConfig
from .layer import Linear, ShiraLayer
from .model import ShiraModel


__all__ = ["Linear", "ShiraConfig", "ShiraLayer", "ShiraModel"]


register_peft_method(name="shira", config_cls=ShiraConfig, model_cls=ShiraModel, prefix="shira_", is_mixed_compatible=True)


def __getattr__(name):
    #TODO: remove
    # if (name == "Linear8bitLt") and is_bnb_available():
    #     from .bnb import Linear8bitLt

    #     return Linear8bitLt

    # if (name == "Linear4bit") and is_bnb_4bit_available():
    #     from .bnb import Linear4bit

    #     return Linear4bit

    raise AttributeError(f"module {__name__} has no attribute {name}")
