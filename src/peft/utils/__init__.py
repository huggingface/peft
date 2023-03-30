# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all

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

from .adapters_utils import CONFIG_NAME, WEIGHTS_NAME
from .config import PeftConfig, PeftType, PromptLearningConfig, TaskType
from .other import (
    TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING,
    _set_trainable,
    bloom_model_postprocess_past_key_value,
    prepare_model_for_int8_training,
    shift_tokens_right,
    transpose,
)
from .save_and_load import get_peft_model_state_dict, set_peft_model_state_dict
