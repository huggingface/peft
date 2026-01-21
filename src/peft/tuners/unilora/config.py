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

from dataclasses import dataclass, field
from typing import List, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType

@dataclass
class UniLoraConfig(PeftConfig):
    """
    Configuration class for UniLoRA adapters.

    UniLoRA replaces per-layer LoRA matrices with a single shared vector theta_d.
    A fixed projection matrix maps theta_d from a low-dimensional subspace to full LoRA parameters, 
    allowing cross-layer parameter sharing and significantly reducing trainable parameters 
    while maintaining accuracy.

    Reference:
        Uni-LoRA: One Vector Is All You Need (2025)
        https://arxiv.org/abs/2506.00799

    Args:
        r (`int`):
            Effective rank of the UniLoRA adaptation.

        proj_seed (`int`):
            Random seed for initialization of projection matrices, ensuring reproducibility.

        theta_d_length (`int`):
            Length of the shared trainable vector (`theta_d`). 

        target_modules (`Union[List[str], str]`, *optional*):
            Specifies which modules UniLoRA should be applied to.
            - If a string, interpreted as a regex pattern.
            - If a list, matched by exact name or name suffix.
            - The special value `'all-linear'` applies UniLoRA to all Linear/Conv1D
              layers except the output head.
            If not provided, the target modules are inferred from the architecture.

        unilora_dropout (`float`):
            Dropout probability inside UniLoRA layers.

        fan_in_fan_out (`bool`):
            Whether weight matrices are stored in (fan_in, fan_out) format. Set to
            `True` for GPT-2 `Conv1D` layers.

        bias (`str`):
            Bias handling strategy. One of:
            - `'none'`: do not update biases
            - `'all'`: update all bias parameters
            - `'unilora_only'`: update only UniLoRA-internal biases

        modules_to_save (`List[str]`, *optional*):
            Modules to keep trainable and save in addition to UniLoRA parameters.
            Useful for classifier heads or task-specific layers.

        init_theta_d_bound (`float`):
            Uniform initialization bound for the UniLoRA vector bank. Vectors are
            sampled from `[-bound, bound]`. Small values (e.g. `0.02`) are recommended
            for stable training.

        layers_to_transform (`Union[List[int], int]`, *optional*):
            Transformer layer indices to which UniLoRA is applied. Effective only
            when `target_modules` is a list.

        layers_pattern (`Union[List[str], str]`, *optional*):
            Custom naming pattern for locating transformer blocks when layer naming
            does not follow a standard convention. Used together with
            `layers_to_transform`.

        init_weights (`bool`):
            Whether to initialize UniLoRA-specific weights (e.g. vector bank,
            scale parameters). Set to `False` for loading adapters from checkpoints.
    """

    r: int = field(
        default=4,
        metadata={"help": "Rank of the UniLoRA adaptation."},
    )

    proj_seed: int = field(
        default=42,
        metadata={"help": "Random seed for UniLoRA projection matrices."},
    )

    theta_d_length: int = field(
        default=256,
        metadata={"help": "Length of the trainable vector `theta_d`)."},
    )

    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={"help": "Module names or patterns to apply UniLoRA to."},
    )

    unilora_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout probability used within UniLoRA layers."},
    )

    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Use (fan_in, fan_out) layout, e.g. GPT-2 Conv1D layers."},
    )

    bias: str = field(
        default="none",
        metadata={"help": "Bias strategy: 'none', 'all', or 'unilora_only'."},
    )

    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Additional non-UniLoRA modules to keep trainable."},
    )

    init_theta_d_bound: float = field(
        default=0.02,
        metadata={"help": "Uniform initialization bound for the vector `theta_d`."},
    )

    layers_to_transform: Optional[Union[List[int], int]] = field(
        default=None,
        metadata={"help": "Indices of transformer layers to apply UniLoRA to."},
    )

    layers_pattern: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={"help": "Custom pattern to locate transformer blocks."},
    )

    init_weights: bool = field(
        default=False,
        metadata={"help": "Whether to initialize UniLoRA adapter weights."},
    )

    def __post_init__(self):
        self.peft_type = PeftType.UNILORA
        self.base_model_prefix = "unilora_"
        if isinstance(self.target_modules, list):
            self.target_modules = set(self.target_modules)
