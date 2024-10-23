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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class LycorisLoKrConfig(PeftConfig):
    """
    Configuration class of [`LoKrModel`].

    Args:
        r (`int`):
            LoKr rank.
        alpha (`int`):
            The alpha parameter for LoKr scaling.
        rank_dropout (`float`):
            The dropout probability for rank dimension during training.
        module_dropout (`float`):
            The dropout probability for disabling LoKr modules during training.
        use_effective_conv2d (`bool`):
            Use parameter effective decomposition for Conv2d with ksize > 1 ("Proposition 3" from FedPara paper).
        decompose_both (`bool`):
            Perform rank decomposition of left kronecker product matrix.
        decompose_factor (`int`):
            Kronecker product decomposition factor.
        rank_dropout_scale ('bool)
            Scale the rank dropout while training.
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the specified
            names will be replaced. When passing a string, a regex match will be performed. When passing a list of
            strings, either an exact match will be performed or it is checked if the name of the module ends with any
            of the passed strings. If this is specified as 'all-linear', then all linear/Conv1D modules are chosen,
            excluding the output layer. If this is not specified, modules will be chosen according to the model
            architecture. If the architecture is not known, an error will be raised -- in this case, you should specify
            the target modules manually.
        bias (`str`):
            Bias type for LoKr. Can be 'none', 'all' or 'lokr_only'. If 'all' or 'lora_only', the corresponding biases
            will be updated during training. Be aware that this means that, even when disabling the adapters, the model
            will not produce the same output as the base model would have without adaptation.
        init_weights (`bool`):
            Whether to perform initialization of adapter weights. This defaults to `True`, passing `False` is
            discouraged.
        layers_to_transform (`Union[List[int], int]`):
            The layer indices to transform. If a list of ints is passed, it will apply the adapter to the layer indices
            that are specified in this list. If a single integer is passed, it will apply the transformations on the
            layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None`.
        rank_pattern (`dict`):
            The mapping from layer names or regexp expression to ranks which are different from the default rank
            specified by `r`.
        alpha_pattern (`dict`):
            The mapping from layer names or regexp expression to alphas which are different from the default alpha
            specified by `alpha`.
        modules_to_save (`Optional[List[str]]`):
            List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
        use_scalar (`Optional[bool]`):
            Whether to use scalar multiplication for LoKR. If `True`, a scalar value will be learned and multiplied
            with the adapter weights.
        use_full_matrix (`Optional[bool]`):
            Whether to use the full matrix instead of performing Low-Rank Decomposition for the LoKR layers.
        use_upstream (`Optional[bool]`):
            Whether to use the latest version of the LoKR module from the `Lycoris` repository.
    """

    r: int = field(default=8, metadata={"help": "LoKr rank"})
    alpha: int = field(default=8, metadata={"help": "LoKr alpha"})
    rank_dropout: float = field(
        default=0.0, metadata={"help": "The dropout probability for rank dimension during training"}
    )
    module_dropout: float = field(
        default=0.0, metadata={"help": "The dropout probability for disabling LoKr modules during training"}
    )
    use_effective_conv2d: bool = field(
        default=False,
        metadata={
            "help": 'Use parameter effective decomposition for Conv2d 3x3 with ksize > 1 ("Proposition 3" from FedPara paper)'
        },
    )
    decompose_both: bool = field(
        default=False,
        metadata={"help": "Perform rank decomposition of left kronecker product matrix."},
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    decompose_factor: int = field(default=-1, metadata={"help": "Kronecker product decomposition factor."})
    rank_dropout_scale: bool = field(default=False, metadata={"help": "Rank dropout scale"})
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with LoKr."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
            "This can also be a wildcard 'all-linear' which matches all linear/Conv1D layers except the output layer."
        },
    )
    bias: str = field(default="none", metadata={"help": "Bias type for LoKr. Can be 'none', 'all' or 'lokr_only'"})
    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the LoKr layers with their default initialization. Don't change "
                "this setting, except if you know exactly what you're doing."
            ),
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
        },
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
        },
    )
    rank_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to ranks which are different from the default rank specified by `r`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 8`}"
                "Important: the rank pattern won't be applied to the layers after 0.12.1.dev0!"
            )
        },
    )
    alpha_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to alphas which are different from the default alpha specified by `alpha`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 32`}"
                "Important: the alpha pattern won't be applied to the layers after 0.12.1.dev0!"
            )
        },
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoKr layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    use_scalar: Optional[bool] = field(
        default=False,
        metadata={"help": "Use scalar multiplication for LoKR."},
    )
    use_full_matrix: Optional[bool] = field(
        default=False,
        metadata={"help": "Use full matrix instead of Low-Rank Decomposition."},
    )
    use_upstream: Optional[bool] = field(
        default=False,
        metadata={"help": "Use the latest version of LoKr module from `Lycoris` repository."},
    )

    def __post_init__(self):
        self.peft_type = PeftType.LOKRv2
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # if target_modules is a regex expression, then layers_pattern should be None
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")
