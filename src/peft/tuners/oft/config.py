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

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class OFTConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`OFTModel`].

    Args:
        r (`int`): OFT rank, number of OFT blocks per injected layer.
        oft_block_size (`int`): OFT block size across different layers.
        module_dropout (`float`):
            The multiplicative dropout probability, by setting OFT blocks to identity during training, similar to the
            dropout layer in LoRA.
        target_modules (`Optional[Union[list[str], str]]`):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the specified
            names will be replaced. When passing a string, a regex match will be performed. When passing a list of
            strings, either an exact match will be performed or it is checked if the name of the module ends with any
            of the passed strings. If this is specified as 'all-linear', then all linear modules are chosen, excluding
            the output layer. If this is not specified, modules will be chosen according to the model architecture. If
            the architecture is not known, an error will be raised -- in this case, you should specify the target
            modules manually.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
        bias (`str`): Bias type for OFT. Can be 'none', 'all' or 'oft_only'. If 'all' or 'oft_only', the
            corresponding biases will be updated during training. Be aware that this means that, even when disabling
            the adapters, the model will not produce the same output as the base model would have without adaptation.
        exclude_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to not apply the adapter. When passing a string, a regex match will be performed.
            When passing a list of strings, either an exact match will be performed or it is checked if the name of the
            module ends with any of the passed strings.
        init_weights (`bool`):
            Whether to perform initialization of OFT weights.
        layers_to_transform (`Union[List[int], int]`):
            The layer indices to transform. If a list of ints is passed, it will apply the adapter to the layer indices
            that are specified in this list. If a single integer is passed, it will apply the transformations on the
            layer at this index.
        layers_pattern (`Optional[Union[List[str], str]]`):
            The layer pattern name, used only if `layers_to_transform` is different from `None`. This should target the
            `nn.ModuleList` of the model, which is often called `'layers'` or `'h'`.
        modules_to_save (`List[str]`):
            List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
        coft (`bool`):
            Whether to use the constrained variant of OFT or not, off by default.
        eps (`float`):
            The control strength of COFT. The freedom of rotation. Only has an effect if `coft` is set to True.
        block_share (`bool`):
            Whether to share the OFT parameters between blocks or not. This is `False` by default.
    """

    r: int = field(default=0, metadata={"help": "OFT rank, number of OFT blocks per injected layer."})
    oft_block_size: int = field(
        default=32,
        metadata={
            "help": "OFT block size across different layers.",
            "note": "You can only specify either r or oft_block_size, but not both simultaneously, because r x oft_block_size = layer dimension.",
        },
    )
    module_dropout: float = field(
        default=0.0,
        metadata={
            "help": "OFT multiplicative dropout, randomly setting blocks of OFT to be identity matrix, similar to the dropout layer in LoRA."
        },
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with OFT."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
            "This can also be a wildcard 'all-linear' which matches all linear/Conv1D layers except the output layer."
        },
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: Literal["none", "all", "oft_only"] = field(
        default="none", metadata={"help": "Bias type for OFT. Can be 'none', 'all' or 'oft_only'"}
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude from OFT."},
    )
    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the OFT layers with their default initialization. Don't change "
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
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern. "
            "This should target the `nn.ModuleList` of the model, which is often called `'layers'` or `'h'`."
        },
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from OFT layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    coft: bool = field(
        default=False,
        metadata={"help": "Whether to use the constrained variant of OFT or not."},
    )
    eps: float = field(
        default=6e-5,
        metadata={
            "help": "The control strength of COFT. The freedom of rotation. Only has an effect if `coft` is set to True."
        },
    )
    block_share: bool = field(
        default=False,
        metadata={"help": "Whether to share the OFT parameters between blocks or not."},
    )
    use_cayley_neumann: bool = field(
        default=True,
        metadata={
            "help": "Whether to use the Cayley-Neumann Formulation of OFT or not. Set to True to improve computational efficiency but comes at costs of bigger approximation error for orthogonality."
        },
    )
    num_cayley_neumann_terms: int = field(
        default=5,
        metadata={
            "help": "Number of Cayley-Neumann terms to use. Higher number results in less approximation error for orthogonality."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.OFT
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )
        # check for layers_to_transform and layers_pattern
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified. ")
        if self.r == 0 and self.oft_block_size == 0:
            raise ValueError(
                f"Either `r` or `oft_block_size` must be non-zero. Currently, r = {self.r} and oft_block_size = {self.oft_block_size}."
            )
        if not (self.r != 0) ^ (self.oft_block_size != 0):
            raise ValueError(
                f"You can only specify either r ({self.r}) or oft_block_size ({self.oft_block_size}), but not both simultaneously, because r x oft_block_size == in_features."
            )

    @classmethod
    def check_kwargs(cls, **kwargs):
        r"""
        Check if the kwargs are valid for the configuration.

        Args:
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the child class initialization.
        """
        if "oft_block_size" not in kwargs:
            raise ValueError(
                "OFT has been updated since PEFT 0.14.0. Your trained adapter weights are incompatible "
                "with the latest version of OFT. Please retrain your adapter weights with newer PEFT versions. "
                "Alternatively, downgrade PEFT to version 0.13.0 to use the old adapter weights."
            )
        return super().check_kwargs(**kwargs)
