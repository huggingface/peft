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

from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class GraloraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`GraloraModel`].

    Args:
        r (`int`):
            GraLoRA attention dimension determines the rank of the GraLoRA adapter. The total parameter count of the
            GraLoRA adapter is same as LoRA with same rank r, while the expressivitiy is multiplied by gralora_k.
        hybrid_r (`int`):
            Hybrid GraLoRA rank determines the rank allocated to vanilla LoRA method when using Hybrid GraLoRA method.
            Hybrid GraLoRA, a combination of GraLoRA and vanilla LoRA, becomes available when hybrid_r > 0. The
            parameter count of the GraLoRA adapter is r + hybrid_r.
        target_modules (`Union[List[str], str]`):
            List of module names or regex expression of the module names to replace with GraLoRA. " For example, ['q',
            'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. " This can also be a wildcard 'all-linear'
            which matches all linear/Conv1D " "(if the model is a PreTrainedModel, the output layer excluded). " If not
            specified, modules will be chosen according to the model architecture, If the architecture is " not known,
            an error will be raised -- in this case, you should specify the target modules manually. " To avoid
            targeting any modules (because you want to apply `target_parameters`), set " `target_modules=[]`.
        alpha (`int`): GraLoRA alpha.
            GraLoRA alpha is the scaling factor for the GraLoRA adapter. Scale becomes alpha / (r + hybrid_r).
        gralora_dropout (`float`):
            GraLoRA dropout is the dropout probability for the GraLoRA adapter. It is used to prevent overfitting and
            improve the generalization of the GraLoRA adapter.
        gralora_k (`int`):
            GraLoRA k determines the number of subblocks in the GraLoRA adapter. The rank r must be divisible by
            gralora_k for the GraLoRA adapter to be valid. The total parameter count is preserved regardles of
            gralora_k. The entire rank of the GraLoRA adapter is increased by gralora_k, while the rank of each
            subblock is reduced by gralora_k. gralora_k=2 is recommended for rank 32 or lower, and gralora_k=4 is
            recommended for rank 64 or higher.
        fan_in_fan_out (`bool`):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.
        bias (`str`):
            Bias type for gralora. Can be 'none', 'all' or 'gralora_only'. If 'all' or 'gralora_only', the
            corresponding biases will be updated during training. Be aware that this means that, even when disabling
            the adapters, the model will not produce the same output as the base model would have without adaptation.
        init_weights (`bool`):
            Whether to initialize the weights of the GraLoRA layers with their default initialization. Don't change
            this setting, except if you know exactly what you're doing.
        layers_to_transform (`Union[List[int], int]`):
            The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes
            that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at
            this index. This only works when target_modules is a list of str.
        layers_pattern (`Optional[Union[List[str], str]]`):
            The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is
            not in the common layers pattern. This only works when target_modules is a list of str. This should target
            the `nn.ModuleList` of the model, which is often called `'layers'` or `'h'`.
    """

    r: int = field(
        default=32,
        metadata={
            "help": (
                "GraLoRA attention dimension determines the rank of the GraLoRA adapter. "
                "The total parameter count of the GraLoRA adapter is same as LoRA with same rank r, while the expressivitiy is multiplied by gralora_k."
            )
        },
    )
    hybrid_r: int = field(
        default=0,
        metadata={
            "help": (
                "hybrid_r is the rank allocated to vanilla LoRA method when using Hybrid GraLoRA method. "
                "Hybrid GraLoRA, a combination of GraLoRA and vanilla LoRA, becomes available when hybrid_r > 0. "
                "r + hybrid_r determines the parameter count of the GraLoRA adapter."
            )
        },
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with LoRA. "
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. "
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D "
                "(if the model is a PreTrainedModel, the output layer excluded). "
                "If not specified, modules will be chosen according to the model architecture, If the architecture is "
                "not known, an error will be raised -- in this case, you should specify the target modules manually. "
                "To avoid targeting any modules (because you want to apply `target_parameters`), set "
                "`target_modules=[]`."
            )
        },
    )
    alpha: int = field(
        default=64,
        metadata={
            "help": (
                "gralora alpha is the scaling factor for the GraLoRA adapter. Scale becomes alpha / (r + hybrid_r). "
            )
        },
    )
    gralora_dropout: float = field(default=0.0, metadata={"help": "gralora dropout"})
    gralora_k: int = field(
        default=2,
        metadata={
            "help": (
                "gralora_k determines the number of subblocks in the GraLoRA adapter. "
                "The rank r must be divisible by gralora_k for the GraLoRA adapter to be valid. "
                "The total parameter count is preserved regardles of gralora_k. "
                "The entire rank of the GraLoRA adapter is increased by gralora_k, while the rank of each subblock is reduced by gralora_k. "
                "gralora_k=2 is recommended for rank 32 or lower, and gralora_k=4 is recommended for rank 64 or higher. "
            )
        },
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(
        default="none", metadata={"help": "Bias type for gralora. Can be 'none', 'all' or 'gralora_only'"}
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from gralora layers to be set as trainable and saved in the final checkpoint. For"
                " example, in Sequence Classification or Token Classification tasks, the final layer"
                " `classifier/score` are randomly initialized and as such need to be trainable and saved."
            )
        },
    )
    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the GraLoRA layers with their default initialization. "
                "Don't change this setting, except if you know exactly what you're doing."
            )
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. "
                "If a single integer is passed, PEFT will transform only the layer at this index. "
                "This only works when target_modules is a list of str."
            )
        },
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern. "
                "This only works when target_modules is a list of str. This should target the `nn.ModuleList` of the "
                "model, which is often called `'layers'` or `'h'`."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.GRALORA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        if self.r % self.gralora_k != 0:
            raise ValueError(f"r should be divisible by gralora_k, but got {self.r} and {self.gralora_k}")
