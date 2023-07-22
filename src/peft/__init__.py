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

__version__ = "0.5.0.dev0"

from .auto import (
    AutoPeftModel as AutoPeftModel,
)
from .auto import (
    AutoPeftModelForCausalLM as AutoPeftModelForCausalLM,
)
from .auto import (
    AutoPeftModelForFeatureExtraction as AutoPeftModelForFeatureExtraction,
)
from .auto import (
    AutoPeftModelForQuestionAnswering as AutoPeftModelForQuestionAnswering,
)
from .auto import (
    AutoPeftModelForSeq2SeqLM as AutoPeftModelForSeq2SeqLM,
)
from .auto import (
    AutoPeftModelForSequenceClassification as AutoPeftModelForSequenceClassification,
)
from .auto import (
    AutoPeftModelForTokenClassification as AutoPeftModelForTokenClassification,
)
from .mapping import (
    MODEL_TYPE_TO_PEFT_MODEL_MAPPING as MODEL_TYPE_TO_PEFT_MODEL_MAPPING,
)
from .mapping import (
    PEFT_TYPE_TO_CONFIG_MAPPING as PEFT_TYPE_TO_CONFIG_MAPPING,
)
from .mapping import (
    get_peft_config as get_peft_config,
)
from .mapping import (
    get_peft_model as get_peft_model,
)
from .peft_model import (
    PeftModel as PeftModel,
)
from .peft_model import (
    PeftModelForCausalLM as PeftModelForCausalLM,
)
from .peft_model import (
    PeftModelForFeatureExtraction as PeftModelForFeatureExtraction,
)
from .peft_model import (
    PeftModelForQuestionAnswering as PeftModelForQuestionAnswering,
)
from .peft_model import (
    PeftModelForSeq2SeqLM as PeftModelForSeq2SeqLM,
)
from .peft_model import (
    PeftModelForSequenceClassification as PeftModelForSequenceClassification,
)
from .peft_model import (
    PeftModelForTokenClassification as PeftModelForTokenClassification,
)
from .tuners import (
    AdaLoraConfig as AdaLoraConfig,
)
from .tuners import (
    AdaLoraModel as AdaLoraModel,
)
from .tuners import (
    AdaptionPromptConfig as AdaptionPromptConfig,
)
from .tuners import (
    AdaptionPromptModel as AdaptionPromptModel,
)
from .tuners import (
    IA3Config as IA3Config,
)
from .tuners import (
    IA3Model as IA3Model,
)
from .tuners import (
    LoraConfig as LoraConfig,
)
from .tuners import (
    LoraModel as LoraModel,
)
from .tuners import (
    PrefixEncoder as PrefixEncoder,
)
from .tuners import (
    PrefixTuningConfig as PrefixTuningConfig,
)
from .tuners import (
    PromptEmbedding as PromptEmbedding,
)
from .tuners import (
    PromptEncoder as PromptEncoder,
)
from .tuners import (
    PromptEncoderConfig as PromptEncoderConfig,
)
from .tuners import (
    PromptEncoderReparameterizationType as PromptEncoderReparameterizationType,
)
from .tuners import (
    PromptTuningConfig as PromptTuningConfig,
)
from .tuners import (
    PromptTuningInit as PromptTuningInit,
)
from .utils import (
    TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING as TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING,
)
from .utils import (
    PeftConfig as PeftConfig,
)
from .utils import (
    PeftType as PeftType,
)
from .utils import (
    PromptLearningConfig as PromptLearningConfig,
)
from .utils import (
    TaskType as TaskType,
)
from .utils import (
    bloom_model_postprocess_past_key_value as bloom_model_postprocess_past_key_value,
)
from .utils import (
    get_peft_model_state_dict as get_peft_model_state_dict,
)
from .utils import (
    prepare_model_for_int8_training as prepare_model_for_int8_training,
)
from .utils import (
    prepare_model_for_kbit_training as prepare_model_for_kbit_training,
)
from .utils import (
    set_peft_model_state_dict as set_peft_model_state_dict,
)
from .utils import (
    shift_tokens_right as shift_tokens_right,
)
