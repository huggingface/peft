"""This code is adapted for the paper: https://arxiv.org/abs/2303.02861 and constitutes the work done at MIT-IBM Watson
Research Lab."""
# coding=utf-8
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

import torch

from ..utils import PeftType, TaskType
from .prompt_tuning import PromptEmbedding, PromptTuningConfig


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
            "help": "How to initialize the prompt tuning parameters. Can be one of TEXT, RANDOM, AVERAGE_SOURCE_TASKS, EXACT_SOURCE_TASK, ONLY_SOURCE_SHARED."
        },
    )
    prompt_tuning_init_state_dict_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path of source state dict. This is required when training the downstream target prompt from the pretrained source prompt"
        },
    )
    prompt_tuning_init_task: Optional[int] = field(default=0, metadata={"help": "source task id for initialization"})
    num_ranks: Optional[int] = field(default=1, metadata={"help": "ranks"})
    num_tasks: Optional[int] = field(default=1, metadata={"help": "number of tasks"})

    def __post_init__(self):
        self.peft_type = PeftType.MULTITASK_PROMPT_TUNING


class MultitaskPromptEmbedding(PromptEmbedding):
    def __init__(self, config: MultitaskPromptTuningConfig, word_embeddings):
        super().__init__(config, word_embeddings)

        self.num_tasks = config.num_tasks
        self.num_ranks = config.num_ranks
        self.num_virtual_tokens = config.num_virtual_tokens

        self.num_transformer_submodules = config.num_transformer_submodules
        if self.num_transformer_submodules is None:
            self.num_transformer_submodules = 2 if config.task_type == TaskType.SEQ_2_SEQ_LM else 1

        self.token_dim = config.token_dim

        total_virtual_tokens = self.num_virtual_tokens * self.num_transformer_submodules

        self.prefix_task_cols = torch.nn.Parameter(
            torch.normal(
                mean=0,
                std=0.02,
                size=(self.num_tasks, total_virtual_tokens, self.num_ranks),
            )
        )
        self.prefix_task_rows = torch.nn.Parameter(
            torch.normal(
                mean=0,
                std=0.02,
                size=(self.num_tasks, self.num_ranks, self.token_dim),
            )
        )

        if config.prompt_tuning_init in [
            MultitaskPromptTuningInit.AVERAGE_SOURCE_TASKS,
            MultitaskPromptTuningInit.EXACT_SOURCE_TASK,
            MultitaskPromptTuningInit.ONLY_SOURCE_SHARED,
        ]:
            if config.prompt_tuning_init_state_dict_path is None:
                raise ValueError(
                    f"prompt_tuning_init_state_dict_path needs to be specified with {config.prompt_tuning_init} init method"
                )

            state_dict: dict = torch.load(
                config.prompt_tuning_init_state_dict_path,
                map_location=word_embeddings.device,
            )

        if config.prompt_tuning_init in [
            MultitaskPromptTuningInit.AVERAGE_SOURCE_TASKS,
            MultitaskPromptTuningInit.EXACT_SOURCE_TASK,
        ]:
            prefix_task_cols_: torch.Tensor = state_dict["prefix_task_cols"]
            prefix_task_rows_: torch.Tensor = state_dict["prefix_task_rows"]

            if config.prompt_tuning_init == MultitaskPromptTuningInit.AVERAGE_SOURCE_TASKS:
                prefix_task_cols_ = prefix_task_cols_.mean(0, keepdim=True)
                prefix_task_rows_ = prefix_task_rows_.mean(0, keepdim=True)
            elif config.prompt_tuning_init == MultitaskPromptTuningInit.EXACT_SOURCE_TASK:
                prefix_task_cols_ = prefix_task_cols_[config.prompt_tuning_init_task, ...].unsqueeze(0)
                prefix_task_rows_ = prefix_task_rows_[config.prompt_tuning_init_task, ...].unsqueeze(0)

            state_dict = {
                "embedding.weight": state_dict["prompt_embeddings"],
                "prefix_task_cols": prefix_task_cols_,
                "prefix_task_rows": prefix_task_rows_,
            }

            self.load_state_dict(state_dict, strict=True)
        elif config.prompt_tuning_init == MultitaskPromptTuningInit.ONLY_SOURCE_SHARED:
            state_dict = {
                "embedding.weight": state_dict["prompt_embeddings"],
            }

            self.load_state_dict(state_dict, strict=False)

    def forward(self, indices, task_ids):
        if task_ids is None:
            raise ValueError("task_ids cannot be None")

        prompt_embeddings = self.embedding(indices)

        task_cols = torch.index_select(self.prefix_task_cols, 0, task_ids)
        task_rows = torch.index_select(self.prefix_task_rows, 0, task_ids)
        task_prompts = torch.matmul(task_cols, task_rows)

        prompt_embeddings *= task_prompts

        return prompt_embeddings
