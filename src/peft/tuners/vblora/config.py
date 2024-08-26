# Copyright 2023-present the HuggingFace Inc. team.
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

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class VBLoRAConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`VBLoRAConfig`].

    Paper: https://arxiv.org/abs/.

    Args:
        r (`int`, *optional*, defaults to `4`):
            VBLoRA parameter dimension ("rank").
        target_modules (`Union[List[str], str]`):
            The names of the modules to apply Vera to. Only linear layers are supported.
        vblora_dropout (`float`):
            The dropout probability for Vera layers.
        fan_in_fan_out (`bool`):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.
        bias (`str`):
            Bias type for Vera. Can be 'none', 'all' or 'vera_only'. If 'all' or 'vera_only', the corresponding biases
            will be updated during training. Be aware that this means that, even when disabling the adapters, the model
            will not produce the same output as the base model would have without adaptation.
        modules_to_save (`List[str]`):
            List of modules apart from Vera layers to be set as trainable and saved in the final checkpoint.
        init_weights (`bool`):
            Whether to initialize the weights of the vblora layers with their default initialization. Don't change this
            setting, except if you know exactly what you're doing.
        layers_to_transform (`Union[List[int],int]`):
            The layer indices to transform. If a list of ints is passed, it will apply the adapter to the layer indices
            that are specified in this list. If a single integer is passed, it will apply the transformations on the
            layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None`.
    """

    r: int = field(default=4, metadata={"help": "Rank."})
    num_vectors: int = field(default=256, metadata={"help": "Number of vectors in the vector bank."})
    vector_length: int = field(default=256, metadata={"help": "The lentgh of the vectors in the vector bank."})
    topk: int = field(default=2, metadata={"help": "K value for topk selection."})

    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with VBLoRA."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. "
                "Only linear layers are supported."
            )
        },
    )
    save_topk_weights: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only save the topk weights.  Models saved in this mode can be used for merging"
                " or inference only, not for resuming training."
            )
        },
    )
    vblora_dropout: float = field(default=0.0, metadata={"help": "VBLoRA dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for Vera. Can be 'none', 'all' or 'vera_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from VBLoRA layers to be set as trainable and saved in the final checkpoint. For"
                " example, in Sequence Classification or Token Classification tasks, the final layer"
                " `classifier/score` are randomly initialized and as such need to be trainable and saved."
            )
        },
    )
    init_vector_bank_bound: float = field(
        default=0.02,
        metadata={
            "help": (
                "The vector bank is initialized with a uniform distribution between -init_vector_bank_bound and"
                " init_vector_bank_bound."
            ),
        },
    )
    init_logits_std: float = field(
        default=0.1,
        metadata={
            "help": (
                "The logits are initialized with a normal distribution with a standard deviation of init_logits_std."
            ),
        },
    )
    layers_to_transform: Optional[Union[List[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index. "
            "This only works when target_modules is a list of str."
        },
    )
    layers_pattern: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
            "This only works when target_modules is a list of str."
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.VBLORA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        if self.save_topk_weights:
            warnings.warn(
                "Warning: The `save_topk_weights` mode is enabled. Models saved in this mode can be used for merging"
                " or inference only, not for resuming training.",
                UserWarning,
            )
