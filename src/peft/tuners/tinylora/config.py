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
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class TinyLoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`TinyLoraModel`].

    TinyLoRA is an extremely parameter-efficient fine-tuning method based on the paper "Learning to Reason in 13
    Parameters" (arXiv:2602.04118). It uses SVD decomposition of frozen weights and projects a tiny trainable vector
    through fixed random tensors.

    Paper: https://arxiv.org/abs/2602.04118

    Args:
        r (`int`, *optional*, defaults to `2`):
            SVD rank for the frozen U, Sigma, V decomposition. The paper recommends r=2.
        u (`int`, *optional*, defaults to `64`):
            Trainable vector dimension per group. This controls the expressivity of the adaptation. Can be as low as
            1-13 for extreme parameter efficiency.
        ntie (`int`, *optional*, defaults to `1`):
            Weight tying factor. Controls how many modules share the same trainable vector v. ntie=1 means each layer
            has its own v. Higher values reduce parameters further.
        projection_seed (`int`, *optional*, defaults to `42`):
            Random seed for generating the fixed projection matrices P.
        save_projection (`bool`, *optional*, defaults to `True`):
            Whether to save the projection tensors P in the state dict. If False, they will be regenerated from the
            seed when loading.
        init_v_bound (`float`, *optional*, defaults to `0.02`):
            Uniform initialization bound for the trainable vector v. Values are initialized in [-init_v_bound,
            init_v_bound].
        target_modules (`Union[List[str], str]`, *optional*):
            The names of the modules to apply TinyLoRA to. Only `nn.Linear`, `nn.Embedding`, and
            `transformers.pytorch_utils.Conv1D` layers are supported.
        tinylora_dropout (`float`, *optional*, defaults to `0.0`):
            The dropout probability for TinyLoRA layers.
        fan_in_fan_out (`bool`, *optional*, defaults to `False`):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). For example, gpt-2 uses
            `Conv1D` which stores weights like (fan_in, fan_out).
        bias (`str`, *optional*, defaults to `"none"`):
            Bias type for TinyLoRA. Can be 'none', 'all' or 'tinylora_only'.
        modules_to_save (`List[str]`, *optional*):
            List of modules apart from TinyLoRA layers to be set as trainable and saved.
        init_weights (`bool`, *optional*, defaults to `True`):
            Whether to initialize the trainable vector v with random values. If `True`, v is initialized with
            uniform random values. If `False`, v is initialized to zeros, making the adapter an identity operation
            (no change to base model output).
        layers_to_transform (`Union[List[int], int]`, *optional*):
            The layer indexes to transform. If specified, only these layers will be adapted.
        layers_pattern (`Optional[Union[List[str], str]]`, *optional*):
            The layer pattern name, used only if `layers_to_transform` is different from `None`.

    Example:
        ```python
        from peft import get_peft_model, TinyLoraConfig

        config = TinyLoraConfig(
            r=2,  # SVD rank (paper recommends 2)
            u=64,  # Trainable vector dimension
            ntie=1,  # Weight tying (1 = no tying)
            target_modules=["q_proj", "v_proj"],
            projection_seed=42,
        )
        model = get_peft_model(base_model, config)
        ```
    """

    r: int = field(default=2, metadata={"help": "TinyLoRA SVD rank (frozen)"})
    u: int = field(default=64, metadata={"help": "Trainable vector dimension per group"})
    ntie: int = field(default=1, metadata={"help": "Weight tying factor (modules sharing same v)"})
    projection_seed: int = field(
        default=42,
        metadata={
            "help": (
                "Random seed for generating the fixed projection matrices P. Used for initialising "
                "projections for new models or when loading a checkpoint that did not include these projections."
            )
        },
    )
    save_projection: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to save the projection tensors P in the state dict. If False, they will be "
                "regenerated from the seed when loading. Setting to True increases checkpoint size but "
                "guarantees reproducibility across system configurations."
            )
        },
    )
    init_v_bound: float = field(default=0.02, metadata={"help": "Uniform init bound for v in [-bound, bound]"})
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with TinyLoRA. "
                "For example, ['q_proj', 'v_proj'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'. "
                "Only nn.Linear, nn.Embedding, and transformers.pytorch_utils.Conv1D layers are supported."
            )
        },
    )
    tinylora_dropout: float = field(default=0.0, metadata={"help": "TinyLoRA dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(
        default="none", metadata={"help": "Bias type for TinyLoRA. Can be 'none', 'all' or 'tinylora_only'"}
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from TinyLoRA layers to be set as trainable and saved in the final checkpoint. "
                "For example, in Sequence Classification or Token Classification tasks, the final layer "
                "`classifier/score` are randomly initialized and as such need to be trainable and saved."
            )
        },
    )
    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the trainable vector v with random values. If True, v is initialized "
                "with uniform random values. If False, v is initialized to zeros, making the adapter an "
                "identity operation (no change to base model output)."
            ),
        },
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "The layer indexes to transform. If this argument is specified, PEFT will transform only the layers "
                "indexes that are specified inside this list. If a single integer is passed, PEFT will transform only "
                "the layer at this index."
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

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.TINYLORA
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        # check for layers_to_transform and layers_pattern
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError("When `layers_pattern` is specified, `layers_to_transform` must also be specified.")
        if not self.save_projection:
            warnings.warn(
                "Specified to not save projection tensors P within the state dictionary. They will be restored "
                "using the PRNG key stored in `config.projection_seed`. Consider setting `config.save_projection` "
                "to `True` to guarantee restoring the checkpoint correctly on all system configurations."
            )
        if self.r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {self.r}")
        if self.u <= 0:
            raise ValueError(f"`u` should be a positive integer value but the value passed is {self.u}")
        if self.ntie <= 0:
            raise ValueError(f"`ntie` should be a positive integer value but the value passed is {self.ntie}")
