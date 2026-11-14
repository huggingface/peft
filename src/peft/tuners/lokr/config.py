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
from typing import Literal, Optional, Union

from peft.tuners.lycoris_utils import LycorisConfig
from peft.utils import PeftType


@dataclass
class LoKrConfig(LycorisConfig):
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
            Use parameter effective decomposition for Conv2d (and Conv1d) with ksize > 1 ("Proposition 3" from FedPara
            paper).
        decompose_both (`bool`):
            Perform rank decomposition of left kronecker product matrix.
        decompose_factor (`int`):
            Kronecker product decomposition factor.
        rank_dropout_scale ('bool)
            Whether to scale the rank dropout while training, defaults to `False`.
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the specified
            names will be replaced. When passing a string, a regex match will be performed. When passing a list of
            strings, either an exact match will be performed or it is checked if the name of the module ends with any
            of the passed strings. If this is specified as 'all-linear', then all linear/Conv1D modules are chosen,
            excluding the output layer. If this is not specified, modules will be chosen according to the model
            architecture. If the architecture is not known, an error will be raised -- in this case, you should specify
            the target modules manually.
        exclude_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to not apply the adapter. When passing a string, a regex match will be performed.
            When passing a list of strings, either an exact match will be performed or it is checked if the name of the
            module ends with any of the passed strings.
        init_weights (`bool`):
            Whether to perform initialization of adapter weights. This defaults to `True`. Use "lycoris" to initialize
            weights in the style of the LYCORIS repository. Passing `False` is discouraged.
        layers_to_transform (`Union[list[int], int]`, *optional*):
            The indices of the layers to transform. If a list of ints is passed, only the layers at these indices are
            transformed. If a single int is passed, only the layer at this index is transformed. If not specified, all
            modules matched by `target_modules` are transformed. The layer index of a module is read from its name,
            e.g. `1` for `model.layers.1.self_attn.q_proj`, see `layers_pattern` for details. This can only be used
            when `target_modules` is a list of str.
        layers_pattern (`Union[list[str], str]`, *optional*):
            The name of the layer container (typically an `nn.ModuleList`, often called `'layers'` or `'h'`), used
            together with `layers_to_transform`. The index following this name in the module name is the layer index,
            e.g. `layers_pattern="layers"` reads the index `1` from `model.layers.1.self_attn.q_proj`. If a list is
            passed, the first name that matches is used. If not specified, the first index in the module name is used,
            which is the layer index for most models, e.g. `1` for `model.layers.1.mlp.experts.0.up_proj` and for
            `layers.1.mlp.experts.0.up_proj`. Specify `layers_pattern` when another numbered module comes before the
            layers, e.g. for `encoder.layers.0.blocks.1.attention`, `layers_pattern="blocks"` reads the index `1`
            instead of `0`.
        rank_pattern (`dict`):
            The mapping from layer names or regexp expression to ranks which are different from the default rank
            specified by `r`. For example, `{'^model.decoder.layers.0.encoder_attn.k_proj': 16}`.
        alpha_pattern (`dict`):
            The mapping from layer names or regexp expression to alphas which are different from the default alpha
            specified by `alpha`. For example, `{'^model.decoder.layers.0.encoder_attn.k_proj': 16}`.
        modules_to_save (`Optional[List[str]]`):
            List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
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
            "help": (
                "Use parameter effective decomposition for Conv2d (and Conv1d) with ksize > 1 "
                '("Proposition 3" from FedPara paper)'
            )
        },
    )
    decompose_both: bool = field(
        default=False,
        metadata={"help": "Perform rank decomposition of left kronecker product matrix."},
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
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude from LoKr."},
    )
    init_weights: Union[bool, Literal["lycoris"]] = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the LoKr layers with their default initialization. Can be True, False or 'lycoris'."
                "Default is True. Don't change this setting to False, except if you know exactly what you're doing."
            ),
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "The indices of the layers to transform. If a list of ints is passed, only the layers at these indices"
                " are transformed. If a single int is passed, only the layer at this index is transformed. If not "
                "specified, all modules matched by `target_modules` are transformed. This can only be used when "
                "`target_modules` is a list of str."
            )
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "The name of the layer container (typically an `nn.ModuleList`, often called `'layers'` or `'h'`), "
                "used together with `layers_to_transform`. The index following this name in the module name is the "
                "layer index. If not specified, the first index in the module name is used, which is the layer index "
                "for most models. Specify it when another numbered module comes before the layers."
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

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.LOKR
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )
        # check for layers_to_transform and layers_pattern
        if self.layers_pattern and self.layers_to_transform is None:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified. ")
