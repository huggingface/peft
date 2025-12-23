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
class UniLoRAConfig(PeftConfig):
    """
    Configuration class for UniLoRA adapters.

    This class defines all hyperparameters required to initialize and apply
    UniLoRA layers within the PEFT framework. The configuration is intentionally
    minimal and only includes parameters that are actively used by the current
    UniLoRA implementation.

    Reference:
        Uni-LoRA: One Vector Is All You Need
        https://arxiv.org/abs/2506.00799

    Args:
        r (int):
            Rank of the low-rank adaptation. This controls the expressive capacity
            of the UniLoRA update.

        proj_seed (int):
            Random seed used to initialize the fixed projection matrix. This ensures
            reproducibility across runs.

        theta_d_length (int):
            Length of the shared UniLoRA vectors (theta_d). This value must be
            divisible by the hidden dimension of the target model, as vectors are
            partitioned across hidden units.

        target_modules (Union[List[str], str], optional):
            Names or patterns of modules to which UniLoRA adapters are applied.
            - If a string is provided, it is treated as a regular expression.
            - If a list is provided, modules are matched by exact name or suffix.
            - The special value 'all-linear' applies UniLoRA to all Linear / Conv1D
              layers except the output layer.
            If not specified, modules are inferred from the model architecture.
            An error is raised if the architecture is unsupported.

        unilora_dropout (float):
            Dropout probability applied within UniLoRA layers.

        fan_in_fan_out (bool):
            Whether the replaced layer stores weights in (fan_in, fan_out) format.
            This should be set to True for models such as GPT-2 that use Conv1D layers.

        bias (str):
            Specifies which bias terms are trainable:
            - 'none': no bias parameters are updated
            - 'all': all bias parameters are updated
            - 'unilora_only': only biases inside UniLoRA layers are updated
            Note: enabling bias updates changes model outputs even when adapters
            are disabled.

        modules_to_save (List[str], optional):
            Additional modules (outside UniLoRA layers) that should remain trainable
            and be saved in the final checkpoint. This is commonly used for
            task-specific heads such as classifiers.

        init_theta_d_bound (float):
            Initialization bound for the UniLoRA vector bank. Vectors are sampled
            uniformly from [-init_theta_d_bound, init_theta_d_bound]. Initializing
            with zeros is avoided to prevent vanishing gradients. Small values
            (e.g., 0.02) are recommended for stable training.

        layers_to_transform (Union[List[int], int], optional):
            Indices of transformer layers to which UniLoRA is applied. If specified,
            only these layers are modified. This option is valid only when
            `target_modules` is a list.

        layers_pattern (Union[List[str], str], optional):
            Custom layer name pattern used together with `layers_to_transform` when
            the model does not follow standard layer naming conventions. This option
            is valid only when `target_modules` is a list.
    """

    r: int = field(
        default=4,
        metadata={"help": "Rank of the UniLoRA low-rank adaptation."},
    )

    proj_seed: int = field(
        default=42,
        metadata={"help": "Random seed for initializing the fixed projection matrix."},
    )

    theta_d_length: int = field(
        default=256,
        metadata={
            "help": (
                "Length of the shared UniLoRA vectors (theta_d). Must be divisible "
                "by the hidden dimension of the target model."
            )
        },
    )

    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "Names or patterns of modules to apply UniLoRA to. Accepts a list of "
                "module name suffixes, a regex string, or the special value "
                "'all-linear' to match all Linear/Conv1D layers except the output layer."
            )
        },
    )

    unilora_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout probability used inside UniLoRA layers."},
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
        metadata={
            "help": (
                "Bias handling strategy for UniLoRA. Options are 'none', 'all', "
                "or 'unilora_only'."
            )
        },
    )

    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": (
                "Additional non-UniLoRA modules to keep trainable and save in the "
                "final checkpoint, such as task-specific classification heads."
            )
        },
    )

    init_theta_d_bound: float = field(
        default=0.02,
        metadata={
            "help": (
                "Initialization bound for UniLoRA vectors. Vectors are sampled "
                "uniformly from [-bound, bound]. Small values are recommended "
                "for stable training."
            )
        },
    )

    layers_to_transform: Optional[Union[List[int], int]] = field(
        default=None,
        metadata={
            "help": (
                "Indices of transformer layers to apply UniLoRA to. If specified, "
                "only these layers are modified. Valid only when target_modules "
                "is a list."
            )
        },
    )

    layers_pattern: Optional[Union[List[str], str]] = field(
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
        # Register UniLoRA as a PEFT adapter type
        self.peft_type = PeftType.UNILORA

        # Normalize target_modules to a set when provided as a list
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
