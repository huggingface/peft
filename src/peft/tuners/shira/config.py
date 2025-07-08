# Copyright 2025-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType

from .mask_functions import random_mask


@dataclass
class ShiraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`ShiraModel`].

    Args:
        r (`int`, *optional*, defaults to `32`):
            For a given target module, the number of SHiRA parameters is computed as r(m+n), where the original tensor
            dimensions are m x n. This means the number of SHiRA parameters is the same as that for a LoRA adapter.
            SHiRA is a high rank adapter. Setting this r parameter does not restrict the rank to this value.
        mask_type (`str`, defaults to `random`):
            Type of mask function. Defaults to a random sparse mask. An optional user-defined mask_fn to compute the
            mask value can also be supplied by instantiating `config = ShiraConfig(...)` and then setting
            `config.mask_fn = <your custom mask function>`. For a pretrained weight with shape m x n, the custom mask
            function must return only one mask (shape: m x n) which must be binary 0 or 1 with num_shira_parameters =
            r(m + n) for linear layers. Device and dtype of mask must be same as base layer's weight's device and
            dtype. Please see mask_functions.py for more details and to see the default random sparse mask
            implementation.
        random_seed (`int`, *optional*, defaults to `None`):
            random seed for the torch generator for random_mask.
        target_modules (`Union[List[str], str]`):
            List of module names or regex expression of the module names to replace with SHiRA. For example, ['q', 'v']
            or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. Only linear layers are supported.
        fan_in_fan_out (`bool`):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.
        init_weights (`bool`, defaults to `True`):
            Initialize SHiRA weight to have zero values. If set to False, SHiRA weights are initialized to randn values
            instead of zeros and this is used only for testing.
        modules_to_save (`List[str]`):
            List of modules apart from SHiRA layers to be set as trainable and saved in the final checkpoint.
    """

    r: int = field(
        default=32,
        metadata={
            "help": (
                "For a given target module, the number of SHiRA parameters is computed as r(m+n), where the original "
                "tensor dimensions are m x n. This means the number of SHiRA parameters is the same as that for a LoRA adapter. "
                "SHiRA is a high rank adapter. Setting this r parameter does not restrict the rank to this value."
            )
        },
    )
    mask_type: Literal["random"] = field(
        default="random",
        metadata={
            "help": (
                "Type of mask function. Defaults to a random sparse mask. "
                "An optional user-defined mask_fn to compute the mask value can also be supplied by instantiating `config = ShiraConfig(...)` and then setting "
                "`config.mask_fn = <your custom mask function>`. For a pretrained weight with shape m x n, the custom mask function must return only one mask (shape: m x n) "
                "which must be binary 0 or 1 with num_shira_parameters = r(m + n) for linear layers. Device and dtype of mask must be same as base layer's weight's device and dtype. "
                "Please see mask_functions.py for more details and to see the default random sparse mask implementation."
            )
        },
    )
    random_seed: Optional[int] = field(
        default=None, metadata={"help": "random seed for the torch generator for random_mask"}
    )
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
    init_weights: bool = field(
        default=True,
        metadata={
            "help": "Initialize SHiRA weight to have zero values. If set to False, SHiRA weights are initialized to randn values instead of zeros and this is used only for testing."
        },
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

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.SHIRA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        if self.mask_type == "random":
            self.mask_fn = random_mask
        else:
            if not self.inference_mode:
                warnings.warn(
                    f"Argument {self.mask_type=} is not recognized, please supply your own masking function by calling `config.mask_fn = my_mask_fn`."
                )
            self.mask_fn = None
