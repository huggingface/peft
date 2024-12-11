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

from .adaption_prompt import AdaptionPromptConfig, AdaptionPromptModel  # noqa: I001
from .lora import (
    LoraConfig,
    LoraModel,
    LoftQConfig,
    EvaConfig,
    LoraRuntimeConfig,
    get_eva_state_dict,
    initialize_lora_eva_weights,
)
from .loha import LoHaConfig, LoHaModel
from .lokr import LoKrConfig, LoKrModel
from .ia3 import IA3Config, IA3Model
from .adalora import AdaLoraConfig, AdaLoraModel
from .boft import BOFTConfig, BOFTModel
from .p_tuning import PromptEncoder, PromptEncoderConfig, PromptEncoderReparameterizationType
from .prefix_tuning import PrefixEncoder, PrefixTuningConfig
from .prompt_tuning import PromptEmbedding, PromptTuningConfig, PromptTuningInit
from .multitask_prompt_tuning import MultitaskPromptEmbedding, MultitaskPromptTuningConfig, MultitaskPromptTuningInit
from .oft import OFTConfig, OFTModel
from .mixed import MixedModel
from .poly import PolyConfig, PolyModel
from .ln_tuning import LNTuningConfig, LNTuningModel
from .vera import VeraConfig, VeraModel
from .fourierft import FourierFTConfig, FourierFTModel
from .xlora import XLoraConfig, XLoraModel
from .hra import HRAConfig, HRAModel
from .vblora import VBLoRAConfig, VBLoRAModel
from .cpt import CPTConfig, CPTEmbedding
from .bone import BoneConfig, BoneModel

__all__ = [
    "AdaLoraConfig",
    "AdaLoraModel",
    "AdaptionPromptConfig",
    "AdaptionPromptModel",
    "BOFTConfig",
    "BOFTModel",
    "BoneConfig",
    "BoneModel",
    "CPTConfig",
    "CPTEmbedding",
    "EvaConfig",
    "FourierFTConfig",
    "FourierFTModel",
    "HRAConfig",
    "HRAModel",
    "IA3Config",
    "IA3Model",
    "LNTuningConfig",
    "LNTuningModel",
    "LoHaConfig",
    "LoHaModel",
    "LoKrConfig",
    "LoKrModel",
    "LoftQConfig",
    "LoraConfig",
    "LoraModel",
    "LoraRuntimeConfig",
    "MixedModel",
    "MultitaskPromptEmbedding",
    "MultitaskPromptTuningConfig",
    "MultitaskPromptTuningInit",
    "OFTConfig",
    "OFTModel",
    "PolyConfig",
    "PolyModel",
    "PrefixEncoder",
    "PrefixTuningConfig",
    "PromptEmbedding",
    "PromptEncoder",
    "PromptEncoderConfig",
    "PromptEncoderReparameterizationType",
    "PromptTuningConfig",
    "PromptTuningInit",
    "VBLoRAConfig",
    "VBLoRAModel",
    "VeraConfig",
    "VeraModel",
    "XLoraConfig",
    "XLoraModel",
    "get_eva_state_dict",
    "initialize_lora_eva_weights",
]
