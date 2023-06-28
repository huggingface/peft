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
from .text_generation import PeftTextGenerationPipeline


PEFT_PIPELINE_MAPPING = {"text-generation": PeftTextGenerationPipeline}


def peft_pipeline(task_type, *args, **kwargs):
    if task_type not in PEFT_PIPELINE_MAPPING:
        raise ValueError(
            f"`task_type` {task_type} not supported. Supported values are {list(PEFT_PIPELINE_MAPPING.keys())}"
        )
    return PEFT_PIPELINE_MAPPING[task_type](*args, **kwargs)
