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
class LNTuningConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a :class:`~peft.tuners.LNTuningModel`.

    Args:
        target_modules (`Optional[Union[List[str], str]]`):
            List of module names or regex expression of the module names to replace with LNTuning. For example,
            '.*decoder.*' or '.*encoder.*'. If this is not specified, modules will be chosen according to the model
            architecture. If the architecture is not known, an error will be raised -- in this case, you should specify
            the target modules manually.
        modules_to_save (`Optional[Union[List[str], str]]`):
            List of modules to be set as trainable and saved in the final checkpoint. For example, in Sequence
            Classification or Token Classification tasks, the final layer `classifier/score` are randomly initialized
            and as such need to be trainable and saved.
    """

    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": (
                "List of module names or regex expression of the module names to replace with LNTuning."
                "For example, '.*decoder.*' or '.*encoder.*'. "
                "If not specified, modules will be chosen according to the model architecture, If the architecture is "
                "not known, an error will be raised -- in this case, you shoud specify the target modules manually."
            ),
        },
    )
    modules_to_save: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={
            "help": "List of modules to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.LN_TUNING
