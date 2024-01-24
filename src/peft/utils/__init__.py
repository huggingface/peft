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

# from .config import PeftConfig, PeftType, PromptLearningConfig, TaskType
from .peft_types import PeftType, TaskType
from .other import (
    TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING,
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING,
    CONFIG_NAME,
    WEIGHTS_NAME,
    SAFETENSORS_WEIGHTS_NAME,
    INCLUDE_LINEAR_LAYERS_SHORTHAND,
    _set_trainable,
    bloom_model_postprocess_past_key_value,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    shift_tokens_right,
    transpose,
    _get_batch_size,
    _get_submodules,
    _set_adapter,
    _freeze_adapter,
    ModulesToSaveWrapper,
    _prepare_prompt_learning_config,
    _is_valid_match,
    infer_device,
    get_auto_gptq_quant_linear,
    get_quantization_config,
    id_tensor_storage,
    cast_mixed_precision_params,
)
from .save_and_load import get_peft_model_state_dict, set_peft_model_state_dict, load_peft_weights
