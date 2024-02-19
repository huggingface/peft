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

import enum
from dataclasses import dataclass, field
from typing import Optional, Union

from peft.tuners.prompt_tuning import PromptTuningConfig
from peft.utils import PeftType


class MultitaskPromptTuningInit(str, enum.Enum):
    # initialize prompt with text
    TEXT = "TEXT"
    # initialize prompt with random matrix
    RANDOM = "RANDOM"
    # average the prefix and column matrices obtained during source training
    AVERAGE_SOURCE_TASKS = "AVERAGE_SOURCE_TASKS"
    # pick prefix and column matrices for a particular task obtained during source training
    EXACT_SOURCE_TASK = "EXACT_SOURCE_TASK"
    # only use the prompt embeddings trained during source training
    ONLY_SOURCE_SHARED = "ONLY_SOURCE_SHARED"


@dataclass
class MultitaskPromptTuningConfig(PromptTuningConfig):
    prompt_tuning_init: Union[MultitaskPromptTuningInit, str] = field(
        default=MultitaskPromptTuningInit.RANDOM,
        metadata={
            "help": (
                "How to initialize the prompt tuning parameters. Can be one of TEXT, RANDOM, AVERAGE_SOURCE_TASKS, "
                "EXACT_SOURCE_TASK, ONLY_SOURCE_SHARED."
            ),
        },
    )
    prompt_tuning_init_state_dict_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The path of source state dict. This is required when training the downstream target prompt from "
                "the pretrained source prompt"
            ),
        },
    )
    prompt_tuning_init_task: Optional[int] = field(default=0, metadata={"help": "source task id for initialization"})
    num_ranks: Optional[int] = field(default=1, metadata={"help": "ranks"})
    num_tasks: Optional[int] = field(default=1, metadata={"help": "number of tasks"})

    def __post_init__(self):
        self.peft_type = PeftType.MULTITASK_PROMPT_TUNING
