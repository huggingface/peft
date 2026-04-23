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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


RoadVariant = Literal["road_1", "road_2", "road_4"]


@dataclass
class RoadConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`RoadModel`]. RoAd adapter is proposed in
    https://huggingface.co/papers/2409.00119.

    Args:
        variant (Union[`RoadVariant`, `str`]):
            The variant of the Road model to use. It can be one of road_1, road_2, or road_4. Refer to the paper for
            more details.
            - road_1: Uses the same scale and angle for all pairs of elements.
            This variant has lowest number of parameters, it stores a number equal to the output hidden size of
            parameters for each layer that RoAd is applied to.
            - road_2: Uses the same scale and angle for each element.
            This variant has 2x the number of parameters compared to road_1.
            - road_4: Uses two different scales and angles for each ellement.
            This variant has 4x the number of parameters compared to road_1.
        group_size (`int`):
            Group size defines how elements are grouped together into 2D vectors for rotation. Within each group
            element 0 is paired with element group_size/2, then element 1 is paired with element group_size/2+1 and so
            on. This has no effect on the model performance, since elements are unordered, however it has some effect
            on inference speed when used in e.g. VLLM. For best speed group size of at least 32 or 64 (the default) is
            recommended. Note that model hidden size (or hidden size per partition when used with tensor parallelism)
            must be divisible by group_size, so for very small models you might need to reduce this parameter.
        init_weights (`bool`):
            Whether to perform initialization of RoAd weights.
        target_modules (`Optional[Union[List[str], str]]`):
            The names of the modules to apply the adapter to. If this is specified, only the modules with the specified
            names will be replaced. When passing a string, a regex match will be performed. When passing a list of
            strings, either an exact match will be performed or it is checked if the name of the module ends with any
            of the passed strings. If this is specified as 'all-linear', then all linear/Conv1D modules are chosen (if
            the model is a PreTrainedModel, the output layer excluded). If this is not specified, modules will be
            chosen according to the model architecture. If the architecture is not known, an error will be raised -- in
            this case, you should specify the target modules manually.
        modules_to_save (`List[str]`):
            List of modules apart from Road layers to be set as trainable and saved in the final checkpoint.
    """

    variant: Union[str, RoadVariant] = field(
        default="road_1",
        metadata={"help": ("Variant of the Road model to use.")},
    )
    group_size: int = field(
        default=64,
        metadata={
            "help": (
                "Group size defines how elements are grouped together into 2D vectors for rotation. "
                "Within each group element 0 is paired with element group_size/2, "
                "then element 1 is paired with element group_size/2+1 and so on. "
                "This has no effect on the model performance, since elements are unordered, "
                "however it has some effect on inference speed when used in e.g. VLLM. "
                "For best speed group size of at least 64 is recommended. "
                "Note that model hidden size (or hidden size per partition when used with tensor parallelism) "
                "must be divisible by group_size, so for very small models you might need to reduce this parameter."
            )
        },
    )
    init_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the RoAd layers with their default initialization. Don't change "
                "this setting, except if you know exactly what you're doing."
            ),
        },
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with Road."
                "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'."
                "This can also be a wildcard 'all-linear' which matches all linear/Conv1D "
                "(if the model is a PreTrainedModel, the output layer excluded)."
                "If not specified, modules will be chosen according to the model architecture, If the architecture is "
                "not known, an error will be raised -- in this case, you should specify the target modules manually."
            ),
        },
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "List of modules apart from RoAd layers to be set as trainable and saved in the final checkpoint. For"
                " example, in Sequence Classification or Token Classification tasks, the final layer"
                " `classifier/score` are randomly initialized and as such need to be trainable and saved."
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.ROAD
        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        if self.variant not in ["road_1", "road_2", "road_4"]:
            raise ValueError(f"Invalid variant {self.variant} specified. Please choose from road_1, road_2 or road_4")
        if self.group_size <= 0 or self.group_size % 2 != 0:
            raise ValueError(f"The group_size must be divisible by 2 when using RoadLayer, but got {self.group_size}.")
