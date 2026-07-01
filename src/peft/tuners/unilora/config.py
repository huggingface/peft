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

from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class UniLoraConfig(PeftConfig):
    """
    Configuration class for UniLora adapters.

    This class defines all hyperparameters required to initialize and apply UniLora layers within the PEFT framework.
    The configuration is intentionally minimal and only includes parameters that are actively used by the current
    UniLora implementation.

    Reference:
        Uni-LoRA: One Vector Is All You Need https://arxiv.org/abs/2506.00799

    Args:
        r (`int`):
            Rank of the low-rank adaptation. This controls the expressive capacity of the UniLora update.
        proj_seed (`int`):
            Random seed used to generate the fixed index assignment. This ensures reproducibility across runs.
        theta_d_length (`int`):
            Length of the shared UniLora vector `theta_d`.
        target_modules (`Union[list[str], str]`, *optional*):
            Names or patterns of modules to which UniLora adapters are applied.
            - If a string is provided, it is treated as a regular expression.
            - If a list is provided, modules are matched by exact name or suffix.
            - The special value 'all-linear' applies UniLora to all Linear / Conv1D layers except the output layer.
            If not specified, modules are inferred from the model architecture. An error is raised if the architecture
            is unsupported.
        unilora_dropout (`float`):
            Dropout probability applied within UniLora layers.
        fan_in_fan_out (`bool`):
            Whether the replaced layer stores weights in (fan_in, fan_out) format. This should be set to True for
            models such as GPT-2 that use Conv1D layers.
        bias (`str`):
            Specifies which bias terms are trainable:
            - 'none': no bias parameters are updated
            - 'all': all bias parameters are updated
            - 'unilora_only': only biases inside UniLora layers are updated
            Note: enabling bias updates changes model outputs even when adapters are disabled.
        modules_to_save (`list[str]`, *optional*):
            Additional modules (outside UniLora layers) that should remain trainable and be saved in the final
            checkpoint. This is commonly used for task-specific heads such as classifiers.
        init_theta_d_bound (`float`):
            Initialization bound for the UniLora vector bank. Vectors are sampled uniformly from [-init_theta_d_bound,
            init_theta_d_bound]. Initializing with zeros is avoided to prevent vanishing gradients. Small values (e.g.,
            0.02) are recommended for stable training.
        init_weights (`bool`):
            Whether to initialize `theta_d` with the default UniLora initialization. If set to `False`, `theta_d` keeps
            a random initialization.
        save_indices (`bool`):
            Whether to save the generated UniLora index and scale buffers alongside `theta_d`. This increases
            checkpoint size, but makes saved adapters independent from future changes to the index generation routine.
        layers_to_transform (`Union[list[int], int]`, *optional*):
            Indices of transformer layers to which UniLora is applied. If specified, only these layers are modified.
            This option is valid only when `target_modules` is a list.
        layers_pattern (`Union[list[str], str]`, *optional*):
            Custom layer name pattern used together with `layers_to_transform` when the model does not follow standard
            layer naming conventions. This option is valid only when `target_modules` is a list.
    """

    r: int = field(
        default=4,
        metadata={
            "help": "Rank of the low-rank adaptation. This controls the expressive capacity of the UniLora update."
        },
    )

    proj_seed: int = field(
        default=42,
        metadata={
            "help": "Random seed used to generate the fixed index assignment. This ensures reproducibility across runs."
        },
    )

    theta_d_length: int = field(
        default=256,
        metadata={
            "help": (
                "Length of the shared UniLora vector `theta_d`. Larger values increase trainable parameters and "
                "reduce collisions in the shared parameter space."
            )
        },
    )

    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "Names or patterns of modules to apply UniLora to. Accepts a list of "
                "module name suffixes, a regex string, or the special value "
                "'all-linear' to match all Linear/Conv1D layers except the output layer."
            )
        },
    )

    unilora_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout probability used inside UniLora layers."},
    )

    fan_in_fan_out: bool = field(
        default=False,
        metadata={
            "help": (
                "Set to True if the target layer stores weights in (fan_in, fan_out) "
                "format, e.g., GPT-2 Conv1D layers."
            )
        },
    )

    bias: str = field(
        default="none",
        metadata={"help": ("Bias handling strategy for UniLora. Options are 'none', 'all', or 'unilora_only'.")},
    )

    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "Additional non-UniLora modules to keep trainable and save in the "
                "final checkpoint, such as task-specific classification heads."
            )
        },
    )

    init_theta_d_bound: float = field(
        default=0.02,
        metadata={
            "help": (
                "Initialization bound for UniLora vectors. Vectors are sampled "
                "uniformly from [-bound, bound]. Small values are recommended "
                "for stable training."
            )
        },
    )

    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize `theta_d` with the default UniLora initialization. If set to `False`, "
                "`theta_d` keeps a random initialization."
            )
        },
    )

    save_indices: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to save the generated UniLora index and scale buffers alongside `theta_d`. This increases "
                "checkpoint size, but makes saved adapters independent from future changes to the index generation "
                "routine."
            )
        },
    )

    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "Indices of transformer layers to apply UniLora to. If specified, "
                "only these layers are modified. Valid only when target_modules "
                "is a list."
            )
        },
    )

    layers_pattern: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "Custom layer name pattern used together with layers_to_transform "
                "when standard layer naming conventions are not followed. "
                "Valid only when target_modules is a list."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.UNILORA

        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )

        if self.r <= 0:
            raise ValueError(f"`r` should be a positive integer, got {self.r}.")
        if self.theta_d_length <= 0:
            raise ValueError(f"`theta_d_length` should be a positive integer, got {self.theta_d_length}.")
        if not 0.0 <= self.unilora_dropout <= 1.0:
            raise ValueError(f"`unilora_dropout` should be between 0.0 and 1.0, got {self.unilora_dropout}.")
        if self.init_theta_d_bound <= 0:
            raise ValueError(f"`init_theta_d_bound` should be a positive float, got {self.init_theta_d_bound}.")
        if self.bias not in {"none", "all", "unilora_only"}:
            raise ValueError(f"`bias` should be 'none', 'all', or 'unilora_only', got {self.bias}.")
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")
