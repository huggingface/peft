from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union, Callable, Any
import torch
import torch.nn as nn

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class ShiraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`ShiraModel`].

    Args:
        r (`int`, *optional*, defaults to `32`):
            As SHiRA is high rank, it does not really make sense to choose a "rank".  However, we would like a way to be consistent
            with LoRA adapter types in terms of parameter count for direct comparison.  Thus, for a given target module, the number
            of SHiRA parameters will be set to r(m+d) where the original tensor dimensions are m x d.  This means the number of
            additional parameters is the same as a LoRA adapter.
        mask_fn_tag (`str`, defaults to `random_mask`):
            Name of the mask function. Since the mask_fn is not serializable, this field can be used to keep a record of mask_fn. This string is not used in the implementation.
        mask_fn (`function`, defaults to `None`):
            An optional user-defined mask_fn to compute the mask value from base_weights. For pretrained weight with shape m, n, return only one mask (shape: m, n) 
            which must be binary 0 or 1 with num_shira_parameters = r(m+n) for linear layers. Device and dtype of mask must be same as base layer's weight's device and dtype. 
            If mask_fn is None, then it will generate a random sparse mask of size r(m + n). This function can be found inside mask_functions.py."
        random_seed (`int`, *optional*, defaults to `42`):
            random seed for the torch generator for random_mask.
        target_modules (`Union[List[str], str]`):
            The names of the modules to apply SHiRA to. Only linear layers are supported.
        fan_in_fan_out (`bool`):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.
        modules_to_save (`List[str]`):
            List of modules apart from SHiRA layers to be set as trainable and saved in the final checkpoint.
    """

    r: int = field(default=32, metadata={"help": (
                                                    "As SHiRA is high rank, it does not really make sense to choose a 'rank'."  
                                                    "However, we would like a way to be consistent with LoRA adapter types in "
                                                    "terms of parameter count for direct comparison.  Thus, for a given target " 
                                                    "module, the number of SHiRA parameters will be set to r(m+d) where the original " 
                                                    "tensor dimensions are m x d.  This means the number of additional parameters " 
                                                    "is the same as a LoRA adapter."
                                                    )
                                                })
    mask_fn_tag: str = field(default='random_mask', metadata={"help": ("Name of the mask function. Since the mask_fn is not serializable, this field can be used to keep a record of mask_fn. This string is not used in the implementation.")})
    mask_fn: Callable[[nn.Module, int, Any], torch.tensor] = field(default=None, metadata={"help": (
                                                                "An optional user-defined mask_fn to compute the mask value from base_weights. " 
                                                                "For pretrained weight with shape m, n, return only one mask (shape: m, n) which must be binary 0 or 1 with num_shira_parameters = r(m+n) for linear layers. " 
                                                                "Device and dtype of mask must be same as base layer's weight's device and dtype. " 
                                                                "If mask_fn is None, then it will generate a random sparse mask of size r(m + n). "
                                                                "This function can be found inside mask_functions.py. "
                                                        )
                                                    }) 
    random_seed: int = field(default=42, metadata={"help": "random seed for the torch generator for random_mask"})
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with SHiRA."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. "
                "Only linear layers are supported."
            )
        },
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from SHiRA layers to be set as trainable and saved in the final checkpoint. For"
                " example, in Sequence Classification or Token Classification tasks, the final layer"
                " `classifier/score` are randomly initialized and as such need to be trainable and saved."
            )
        },
    )

    def to_dict(self):
        """
        Returns the config for your adapter model as a dictionary. Removes mask function callable.
        """
        rv = super().to_dict()
        rv.pop("mask_fn")
        return rv

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.SHIRA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
