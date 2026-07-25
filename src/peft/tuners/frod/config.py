# Copyright 2026-present the HuggingFace Inc. team.
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
class FrodConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`FrodModel`].

    Paper: https://doi.org/10.1609/aaai.v40i31.39813.

    Args:
        target_modules (`Union[List[str], str]`):
            The names of the modules to apply FRoD to. Only linear layers are supported.
        projection_prng_key (`int`):
            Random seed used when initializing the sparse FRoD COO pattern.
        save_projection (`bool`):
            Whether to save the FRoD projection tensors in the state dict. This increases checkpoint size but makes
            adapter reloading independent of local cache regeneration. Defaults to `True`.
        frod_dropout (`float`):
            The dropout probability for FRoD layers.
        fan_in_fan_out (`bool`):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.
        bias (`str`):
            Bias type for FRoD. Can be 'none', 'all' or 'frod_only'. If 'all' or 'frod_only', the corresponding biases
            will be updated during training. Be aware that this means that, even when disabling the adapters, the model
            will not produce the same output as the base model would have without adaptation.
        modules_to_save (`List[str]`):
            List of modules apart from FRoD layers to be set as trainable and saved in the final checkpoint.
        init_weights (`bool`):
            Whether to initialize the weights of the FRoD layers with their default initialization. Don't change this
            setting, except if you know exactly what you're doing.
        layers_to_transform (`Union[List[int],int]`):
            The layer indexes to transform, if this argument is specified, it will apply the FRoD transformations on
            the layer indexes that are specified in this list. If a single integer is passed, it will apply the FRoD
            transformations on the layer at this index.
        layers_pattern (`Optional[Union[List[str], str]]`):
            The layer pattern name, used only if `layers_to_transform` is different from `None`. This should target the
            `nn.ModuleList` of the model, which is often called `'layers'` or `'h'`.
        sparse_rate (`float`):
            Fraction of off-diagonal entries in the sparse trainable rotation matrix. Higher values increase capacity
            and trainable parameters; lower values are cheaper. Defaults to `0.01`.
        regularization_alpha (`float`):
            Small positive value used while building the shared basis from base weights. It stabilizes the matrix
            inverse when layers in the same category have correlated weights. Defaults to `1e-3`.
        progressbar (`bool`):
            Whether to show a progress bar while building the FRoD projections. Projection initialization can be slow
            on large models because it runs matrix decompositions over the target module categories. Defaults to
            `True`.
        runtime_offload_base_weight (`bool`):
            Whether to keep target base weights on CPU when the active FRoD path does not need them. This can reduce
            GPU memory because FRoD reconstructs the adapted weight directly, but it changes the usual PEFT convention
            that all base parameters stay on the accelerator after moving the model or running forward. Defaults to
            `False`.
    """

    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with FRoD."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. "
                "Only linear layers are supported."
            )
        },
    )
    projection_prng_key: int = field(
        default=0,
        metadata={"help": "Random seed used when initializing the FRoD sparse COO structure."},
    )
    save_projection: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to save the FRoD projection tensors in the state dict. This increases checkpoint size but "
                "guarantees that we can reload the adapter on all system configurations."
            )
        },
    )
    frod_dropout: float = field(default=0.0, metadata={"help": "Dropout in the FRoD adapter layers"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for FRoD. Can be 'none', 'all' or 'frod_only'"})
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from FRoD layers to be set as trainable and saved in the final checkpoint. For"
                " example, in Sequence Classification or Token Classification tasks, the final layer"
                " `classifier/score` are randomly initialized and as such need to be trainable and saved."
            )
        },
    )
    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the FRoD layers with their default initialization. Don't change "
                "this setting, except if you know exactly what you're doing."
            ),
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers"
                " indexes that are specified inside this list. If a single integer is passed, PEFT will transform only"
                " the layer at this index."
            )
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer "
                "pattern is not in the common layers pattern. This should target the `nn.ModuleList` of the "
                "model, which is often called `'layers'` or `'h'`."
            )
        },
    )
    sparse_rate: float = field(
        default=0.01,
        metadata={
            "help": (
                "Fraction of off-diagonal entries in the sparse trainable rotation matrix. Higher values increase "
                "capacity and trainable parameters; lower values are cheaper."
            )
        },
    )
    regularization_alpha: float = field(
        default=1e-3,
        metadata={
            "help": (
                "Small positive value used while building the shared basis from base weights. It stabilizes matrix "
                "inverses for correlated layers."
            ),
        },
    )
    progressbar: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to show a progress bar while building FRoD projections. Initialization can be slow on large "
                "models because it runs matrix decompositions over the target module categories."
            )
        },
    )
    runtime_offload_base_weight: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to keep target base weights on CPU when the active FRoD path does not need them. This reduces "
                "GPU memory but does not preserve the usual PEFT invariant that all base parameters stay on the "
                "accelerator after moving the model or running forward."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.FROD
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        # check for layers_to_transform and layers_pattern
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified. ")
        if self.sparse_rate < 0 or self.sparse_rate > 1:
            raise ValueError(f"`sparse_rate` should be between 0 and 1, got {self.sparse_rate}.")
