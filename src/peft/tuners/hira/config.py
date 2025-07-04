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

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from torch import nn

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class HiRARuntimeConfig:
    """
    This is the sub-configuration class to store the runtime configurations for the model.

    Args:
        ephemeral_gpu_offload (`bool`):
            Whether to use ephemeral GPU offloading for models partially kept in CPU memory.
    """

    ephemeral_gpu_offload: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use ephemeral GPU offloading for models partially kept in CPU memory. Ephemeral GPU offloading result in "
                "the data involved in intense operations being momentarily copied over to the GPU, and the results copied "
                "back to CPU. There is a momentary VRAM overhead, but operations are generally orders of magnitude faster "
                "compared to performing them on the CPU. This is useful when parts of the model and/or components (such "
                "as adapters) are kept in CPU memory until they are needed. Rather than perform expensive operations on "
                "small data, the data is transferred to the GPU on-demand, the operation(s) performed, and the results "
                "moved back to CPU memory. Currently only affects DoRA initialization."
            )
        },
    )


@dataclass
class HiRAConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`HiRAModel`].

    Args:
        r (`int`):
            HiRA r configuration (the "r").
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the specified
            names will be replaced. When passing a string, a regex match will be performed. When passing a list of
            strings, either an exact match will be performed or it is checked if the name of the module ends with any
            of the passed strings. If this is specified as 'all-linear', then all linear/Conv1D modules are chosen (if
            the model is a PreTrainedModel, the output layer excluded). If this is not specified, modules will be
            chosen according to the model architecture. If the architecture is not known, an error will be raised -- in
            this case, you should specify the target modules manually.
        exclude_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to not apply the adapter. When passing a string, a regex match will be performed.
            When passing a list of strings, either an exact match will be performed or it is checked if the name of the
            module ends with any of the passed strings.
        hira_dropout (`float`):
            The dropout probability for HiRA layers.
        fan_in_fan_out (`bool`):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.
        modules_to_save (`List[str]`):
            List of modules apart from adapter layers to be set as trainable and saved in the final checkpoint.
        init_hira_weights (`bool` | `Literal["gaussian"]`):
            How to initialize the weights of the adapter layers. Passing True (default) results in the default
            initialization from the reference implementation from Microsoft, with the HiRA B weight being set to 0.
        layers_to_transform (`Union[List[int], int]`):
            The layer indices to transform. If a list of ints is passed, it will apply the adapter to the layer indices
            that are specified in this list. If a single integer is passed, it will apply the transformations on the
            layer at this index.
        layers_pattern (`Optional[Union[List[str], str]]`):
            The layer pattern name, used only if `layers_to_transform` is different from `None`. This should target the
            `nn.ModuleList` of the model, which is often called `'layers'` or `'h'`.
        r_pattern (`dict`):
            The mapping from layer names or regexp expression to ranks which are different from the default r
            specified by `r`. For example, `{'^model.decoder.layers.0.encoder_attn.k_proj': 16}`.
        layer_replication (`List[Tuple[int, int]]`):
            Build a new stack of layers by stacking the original model layers according to the ranges specified. This
            allows expanding (or shrinking) the model without duplicating the base model weights. The new layers will
            all have separate HiRA adapters attached to them.
        runtime_config (`HiRARuntimeConfig`):
            Runtime configurations (which are not saved or restored).
    """

    r: int = field(default=32, metadata={"help": "HiRA intermediate r configuration"})
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with HiRA."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'."
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D "
                "(if the model is a PreTrainedModel, the output layer excluded)."
                "If not specified, modules will be chosen according to the model architecture, If the architecture is "
                "not known, an error will be raised -- in this case, you should specify the target modules manually."
            ),
        },
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "List of module names or regex expression of the module names to exclude from HiRA."},
    )
    hira_dropout: float = field(default=0.0, metadata={"help": "HiRA dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from HiRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_hira_weights: (
        bool | Literal["gaussian"]
    ) = field(
        default=True,
        metadata={
            "help": (
                "How to initialize the weights of the HiRA layers. "
                "Passing True (default) results in the default initialization"
                ", with the HiRA B weight being set to 0. This means that without further training, the HiRA "
                "adapter will be a no-op. "
                "Setting the initialization to False leads to random initialization of HiRA A and B, meaning that HiRA "
                "is not a no-op before training; this setting is intended for debugging purposes. "
                "Passing `'gaussian'` results in Gaussian initialization scaled by the HiRA rank for linear and layers. "
            ),
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index. "
            "This only works when target_modules is a list of str."
        },
    )
    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
            "This only works when target_modules is a list of str. This should target the `nn.ModuleList` of the "
            "model, which is often called `'layers'` or `'h'`."
        },
    )
    r_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to r which are different from the default rank specified by `r`. "
                "For example, `{'^model.decoder.layers.0.encoder_attn.k_proj': 16}`."
            )
        },
    )

    # Enables replicating layers in a model to expand it to a larger model.
    layer_replication: Optional[list[tuple[int, int]]] = field(
        default=None,
        metadata={
            "help": (
                "This enables using HiRA to effectively expand a transformer model to a larger size by repeating some layers. "
                "The transformation handles models (currently Llama, Bert or Falcon compatible architectures) with "
                "a module list in the model which it modifies to expand the number of modules. "
                "Base weights are shared so the memory usage is close to the original model. The intended use is these base weights "
                "remain fixed during finetuning but each layer has a separate HiRA adapter so the layers can be specialed via "
                "the adapter layers fit during fine tuning."
                "The format is a list of [start, end) pairs which specify the layer ranges to stack. For example:\n"
                "   Original model has 5 layers labelled by their position in the model: `[0, 1, 2, 3, 4]`\n"
                "   layer_replication: `[[0, 4], [2, 5]]`\n"
                "   Final model will have this arrangement of original layers: `[0, 1, 2, 3, 2, 3, 4]`\n"
                "This format is based on what is used for pass-through merges in mergekit. It makes it simple to select sequential "
                "ranges of a model and stack them while reusing layers at either end of each sequence."
            )
        },
    )
    runtime_config: HiRARuntimeConfig = field(
        default_factory=HiRARuntimeConfig, metadata={"help": "Runtime configurations"}
    )

    def to_dict(self):
        """
        Returns the configuration for your adapter model as a dictionary. Removes runtime configurations.
        """
        rv = super().to_dict()
        rv.pop("runtime_config")
        return rv

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.HiRA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )

        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # if target_modules is a regex expression, then layers_pattern should be None
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")

        # check for layers_to_transform and layers_pattern
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified. ")

        self._custom_modules: Optional[dict[type[nn.Module], type[nn.Module]]] = None

    def _register_custom_module(self, mapping: dict[type[nn.Module], type[nn.Module]]) -> None:
        """
        Experimental API to support providing custom HiRA layers.

        This API is subject to change, you should carefully read the docs before deciding to use it:

        https://huggingface.co/docs/peft/developer_guides/custom_models

        To register custom HiRA module types, call this method with a `mapping` argument that is a dict that maps from
        the target layer type to the custom HiRA layer type. The dict can contain multiple items if you wish to target
        multiple layer types. The target layer type can be any nn.Module that we currently don't support in PEFT,
        whether that is an official PyTorch layer type or a custom layer type. The custom HiRA module class has to be
        implemented by the user and follow the PEFT conventions for HiRA layers.

        """
        if self._custom_modules is None:
            self._custom_modules = {}
        self._custom_modules.update(mapping)
