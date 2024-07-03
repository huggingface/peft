# Copyright 2024-present the HuggingFace Inc. team.
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
class FourierFTConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`FourierFTModel`].

    Args:
        n_frequency (`int`):
            Num of learnable frequencies for the Discrete Fourier Transform. 'n_frequency' is an integer that is
            greater than 0 and less than or equal to d^2 (assuming the weight W has dimensions of d by d).
            Additionally, it is the number of trainable parameters required to update each delta W weight.
            'n_frequency' will affect the performance and efficiency for PEFT. Specifically, it has little impact on
            training speed, but higher values of it (typically) result in larger GPU memory costs and better accuracy.
            With the same `target_modules`, the number of parameters of LoRA is (2*d*r/n_frequency) times that of
            FourierFT. The following examples of settings regarding 'n_frequency' can be used as reference for users.
            For NLU tasks with the RoBERTa-large model, adopting 'n_frequency': 1000 can almost achieve similar results
            as 'r': 8 in LoRA. At this time, the number of parameters of LoRA is about 16 times that of FourierFT. For
            image classification tasks with Vit-large models, adopting 'n_frequency': 3000 can almost achieve similar
            results as 'r': 16 in LoRA, where the number of parameters of LoRA is about 11 times that of FourierFT.
        scaling (`float`):
            The scaling value for the delta W matrix. This is an important hyperparameter used for scaling, similar to
            the 'lora_alpha' parameter in the LoRA method. 'scaling' can be determined during the hyperparameter search
            process. However, if users want to skip this process, one can refer to the settings in the following
            scenarios. This parameter can be set to 100.0 or 150.0 for both RoBERTa-base and RoBERTa-large models
            across all NLU (GLUE) tasks. This parameter can be set to 300.0 for both LLaMA family models for all
            instruction tuning. This parameter can be set to 300.0 for both ViT-base and ViT-large models across all
            image classification tasks.
        random_loc_seed (`int`):
            Seed for the random location of the frequencies, i.e., the spectral entry matrix.
        target_modules (`Union[list[str],str]`):
            List of module names or regex expression of the module names to replace with FourierFT. For example, ['q',
            'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. Only linear layers are supported.
        fan_in_fan_out (`bool`):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out).
        bias (`str`):
            Bias type for FourierFT. Can be 'none', 'all' or 'fourier_only'.
        modules_to_save (`list[str]`):
            List of modules apart from FourierFT layers to be set as trainable and saved in the final checkpoint. For
            example, in Sequence Classification or Token Classification tasks, the final layer `classifier/score` are
            randomly initialized and as such need to be trainable and saved.
        layers_to_transform (`Union[list[int],int]`):
            The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes
            that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at
            this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is
            not in the common layers pattern.
        n_frequency_pattern (`dict`):
            The mapping from layer names or regexp expression to n_frequency which are different from the default
            specified. For example, `{model.decoder.layers.0.encoder_attn.k_proj: 1000`}.
        init_weights (`bool`):
            The initialization of the Fourier weights. Set this to False if the spectrum are initialized to a standard
            normal distribution. Set this to True if the spectrum are initialized to zeros.
    """

    n_frequency: int = field(
        default=1000,
        metadata={
            "help": (
                "Num of learnable frequencies for the Discrete Fourier Transform. 'n_frequency' is an integer that is"
                "greater than 0 and less than or equal to d^2 (assuming the weight W has dimensions of d by d)."
                "Additionally, it is the number of trainable parameters required to update each delta W weight."
                "'n_frequency' will affect the performance and efficiency for PEFT. Specifically, it has little impact on"
                "training speed, but higher values of it (typically) result in larger GPU memory costs and better accuracy."
                "With the same `target_modules`, the number of parameters of LoRA is (2*d*r/n_frequency) times that of FourierFT."
                "The following examples of settings regarding 'n_frequency' can be used as reference for users. For NLU"
                "tasks with the RoBERTa-large model, adopting 'n_frequency': 1000 can almost achieve similar results as"
                "'r': 8 in LoRA. At this time, the number of parameters of LoRA is about 16 times that of FourierFT."
                "For image classification tasks with Vit-large models, adopting 'n_frequency': 3000 can almost achieve"
                "similar results as 'r': 16 in LoRA, where the number of parameters of LoRA is about 11 times that of FourierFT."
            )
        },
    )
    scaling: float = field(
        default=150.0,
        metadata={
            "help": (
                "The scaling value for the delta W matrix. This is an important hyperparameter used for scaling, similar to the"
                "'lora_alpha' parameter in the LoRA method. 'scaling' can be determined during the hyperparameter search process."
                "However, if users want to skip this process, one can refer to the settings in the following scenarios."
                "This parameter can be set to 100.0 or 150.0 for both RoBERTa-base and RoBERTa-large models across all NLU (GLUE) tasks."
                "This parameter can be set to 300.0 for both LLaMA family models for all instruction tuning."
                "This parameter can be set to 300.0 for both ViT-base and ViT-large models across all image classification tasks."
            )
        },
    )
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
            "help": (
                "List of module names or regex expression of the module names to replace with FourierFT."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. "
                "Only linear layers are supported."
            )
        },
    )
    bias: str = field(
        default="none", metadata={"help": "Bias type for FourierFT. Can be 'none', 'all' or 'fourier_only'."}
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from FourierFT layers to be set as trainable and saved in the final checkpoint. For"
                " example, in Sequence Classification or Token Classification tasks, the final layer"
                " `classifier/score` are randomly initialized and as such need to be trainable and saved."
            )
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
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer"
                " pattern is not in the common layers pattern."
            )
        },
    )
    n_frequency_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to n_frequency which are different from the default specified."
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 500`}."
            )
        },
    )
    init_weights: bool = field(
        default=False,
        metadata={
            "help": (
                "The initialization of the Fourier weights. Set this to False if the spectrum should be initialized to a standard normal distribution."
                "Set this to True if the spectrum should be initialized to zeros."
            )
        },
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
