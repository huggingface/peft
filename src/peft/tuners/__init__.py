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

from typing import TYPE_CHECKING

from ..import_utils import _LazyModule

_import_structure = {
    "adaption_prompt": [
        "AdaptionPromptConfig",
        "AdaptionPromptModel",
    ],
    "lora": [
        "LoraConfig",
        "LoraModel",
    ],
    "ia3": [
        "IA3Config",
        "IA3Model",
    ],
    "adalora": [
        "AdaLoraConfig",
        "AdaLoraModel",
    ],
    "p_tuning": [
        "PromptEncoder",
        "PromptEncoderConfig",
        "PromptEncoderReparameterizationType",
    ],
    "prefix_tuning": [
        "PrefixEncoder",
        "PrefixTuningConfig",
    ],
    "prompt_tuning": [
        "PromptEmbedding",
        "PromptTuningConfig",
        "PromptTuningInit",
    ],
    "multitask_prompt_tuning": [
        "MultitaskPromptEmbedding",
        "MultitaskPromptTuningConfig",
        "MultitaskPromptTuningInit",
    ],
}

if TYPE_CHECKING:
    from .adaption_prompt import AdaptionPromptConfig, AdaptionPromptModel
    from .lora import LoraConfig, LoraModel
    from .ia3 import IA3Config, IA3Model
    from .adalora import AdaLoraConfig, AdaLoraModel
    from .p_tuning import PromptEncoder, PromptEncoderConfig, PromptEncoderReparameterizationType
    from .prefix_tuning import PrefixEncoder, PrefixTuningConfig
    from .prompt_tuning import PromptEmbedding, PromptTuningConfig, PromptTuningInit
    from .multitask_prompt_tuning import MultitaskPromptEmbedding, MultitaskPromptTuningConfig, MultitaskPromptTuningInit

    # Mapping of tuners that support direct plugging
    TUNERS_MAPPING = {
        "LORA": LoraModel,
        "IA3": IA3Model,
        "ADALORA": AdaLoraModel,
    }
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
