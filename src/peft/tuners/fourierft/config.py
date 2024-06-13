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
class FourierConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`FourierModel`].

    Args:
        n_frequency (`int`): Discrete Fourier Transform frequency dimension.
        scaling (`float`): The scaling value for the delta W matrix.
        target_modules (`Union[list[str],str]`): The names of the modules to apply FourierFT to.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
            For example, gpt-2 uses `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set
            to `True`.
        bias (`str`): Bias type for FourierFT. Can be 'none', 'all' or 'fourier_only'. If 'all' or 'fourier_only', the
            corresponding biases will be updated during training. Be aware that this means that, even when disabling
            the adapters, the model will not produce the same output as the base model would have without adaptation.
        modules_to_save (`list[str]`):list of modules apart from Fourier layers to be set as trainable
            and saved in the final checkpoint.
        layers_to_transform (`Union[list[int],int]`):
            The layer indexes to transform, if this argument is specified, it will apply the Fourier transformations on
            the layer indexes that are specified in this list. If a single integer is passed, it will apply the Fourier
            transformations on the layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None` and if the layer
            pattern is not in the common layers pattern.
        n_frequency_pattern (`dict`):
            The mapping from layer names or regexp expression to n_frequency which are different from the default.
    """

    n_frequency: int = field(
        default=1000, metadata={"help": "Num of learnable frequencies for the Discrete Fourier Transform."}
    )
    scaling: float = field(default=150.0, metadata={"help": "The scaling value for the delta W matrix."})
    random_loc_seed: Optional[int] = field(
        default=777, metadata={"help": "Seed for the random location of the frequencies."}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "list of module names or regex expression of the module names to replace with FourierFT."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    bias: str = field(
        default="none", metadata={"help": "Bias type for FourierFT. Can be 'none', 'all' or 'fourier_only'"}
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "list of modules apart from FourierFT layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
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
            "This only works when target_modules is a list of str."
        },
    )
    n_frequency_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to n_frequency which are different from the default specified. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 1000`}"
            )
        },
    )
    init_fourier_weights: Optional[str] = field(
        default="gaussian",
        metadata={"help": "The initialization of the Fourier weights. Can be 'xavier_dense' or 'gaussian'."},
    )

    def __post_init__(self):
        self.peft_type = PeftType.FOURIERFT
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # if target_modules is a regex expression, then layers_pattern should be None
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")
