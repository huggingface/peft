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

import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class RandLoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`RandLoraModel`].

    Paper: https://huggingface.co/papers/2502.00987.

    Args:
        r (`int`, *optional*, defaults to `32`):
            RandLora's random basis rank dimension. Contrary to Lora, this parameter is inversely proportional to the
            amount of trainable parameters as reducing it increases trainable parameters.
        target_modules (`Union[list[str], str]`):
            The names of the modules to apply RandLora to. Only linear layers are supported.
        projection_prng_key (`int`):
            RandLora PRNG init key. Used for initialising basis_A and basis_B for new models or when loading a
            checkpoint that did not include these projections. Defaults to `0`.
        save_projection (`bool`):
            Whether to save the global basis_A / basis_B random basis in the state dict alongside per layer lambda /
            gamma diagonal matrices. This will increase the size of the checkpoint, but guarantee that we can reload
            the checkpoint on all system configurations. Defaults to `True`.
        sparse (`bool`):
            Whether to use sparse random bases as described in the RandLora paper. The bases are ternary sparse bases
            (only containing -1, 0 and 1) where the attribution probability is 1/6 for -1 and 1 and 2/3 for 0. These
            sparse matrices aim to be used for matmul free computation in the future, see
            https://huggingface.co/papers/2406.02528v1 The current implementation is a proof of concept however where
            the sparseness is not used to improve speed or memory usage. Using sparse matrices typically does not
            reduce performance and can even help reduce overfitting. Defaults to `False`.
        very_sparse (`bool`):
            Whether to use highly sparse random bases as described in the RandLora paper. The very sparse bases are
            ternary sparse bases (only containing -1, 0 and 1) given a matrix with smallest dimension d, the
            attribution probability is 1/√D for -1 and 1 and 1- 2/√D for 0. Using these sparse matrices can further
            reduce overfitting over the `sparse` alternatives but will most likely decrease performance as a results.
            Use carefully. Defaults to `False`.
        randlora_dropout (`float`):
            The dropout probability for RandLora layers.
        randlora_alpha (`float`):
            The scaling coefficient for RandLora layers, this would typically be 20 times the rank. Because the
            `randlora_alpha` coefficient is large by default, it can lead to numerical instabilities especially when
            learning rates are high. If training is unstable, consider reducing the learning rate or the
            `randlora_alpha` coefficient.
        fan_in_fan_out (`bool`):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.
        bias (`str`):
            Bias type. Can be 'none', 'all' or 'randlora_only'. If 'all' or 'randlora_only', the corresponding biases
            will be updated during training. Be aware that this means that, even when disabling the adapters, the model
            will not produce the same output as the base model would have without adaptation.
        modules_to_save (`list[str]`):
            list of modules apart from RandLora layers to be set as trainable and saved in the final checkpoint.
        init_weights (`bool`):
            Whether to initialize the weights of the RandLora layers with their default initialization. Don't change
            this setting, except if you know exactly what you're doing.
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
    """

    r: int = field(default=32, metadata={"help": "RandLora random basis rank"})

    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "list of module names or regex expression of the module names to replace with RandLora."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. "
                "Only linear layers are supported."
            )
        },
    )
    projection_prng_key: int = field(
        default=0,
        metadata={
            "help": (
                "RandLora PRNG init key. Used for initialising basis_A and basis_B for new models or when loading a "
                "checkpoint that did not include these projections."
            )
        },
    )
    save_projection: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to save the basis_A / basis_B projections in the state dict alongside per layer lambda / "
                "gamma weights. This will increase the size of the checkpoint, but guarantee that we can reload "
                "the checkpoint on all system configurations."
            )
        },
    )
    sparse: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use sparse random bases as described in the RandLora paper."
                "The current implementation is a proof of concept where the sparseness"
                "is not used to improve speed or memory usage."
            )
        },
    )
    very_sparse: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use very sparse random bases."
                "The current implementation is a proof of concept where the sparseness"
                "is not used to improve speed or memory usage."
            )
        },
    )
    randlora_dropout: float = field(default=0.0, metadata={"help": "Dropout in the adapter layers"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    randlora_alpha: int = field(
        default=640,
        metadata={
            "help": "Scaling coefficient in the adapter layers, typically 20 times the rank of the random bases."
        },
    )
    bias: str = field(
        default="none", metadata={"help": "Bias type for RandLora. Can be 'none', 'all' or 'randlora_only'"}
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "list of modules apart from RandLora layers to be set as trainable and saved in the final checkpoint. For"
                " example, in Sequence Classification or Token Classification tasks, the final layer"
                " `classifier/score` are randomly initialized and as such need to be trainable and saved."
            )
        },
    )
    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the RandLora layers with their default initialization. Don't change "
                "this setting, except if you know exactly what you're doing."
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
    layers_pattern: Optional[str] = field(
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

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.RANDLORA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )

        if not self.save_projection:
            warnings.warn(
                "Specified to not save basis_A and basis_B within the state dictionary, instead they will be restored "
                "using the PRNG key store in `config.projection_prng_key`. Consider setting `config.save_projection` "
                "to `True` to guarantee restoring the checkpoint correctly on all system configurations."
            )
