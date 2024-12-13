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


class PeftType(str, enum.Enum):
    """
    Enum class for the different types of adapters in PEFT.

    Supported PEFT types:
    - PROMPT_TUNING
    - MULTITASK_PROMPT_TUNING
    - P_TUNING
    - PREFIX_TUNING
    - LORA
    - ADALORA
    - BOFT
    - ADAPTION_PROMPT
    - IA3
    - LOHA
    - LOKR
    - OFT
    - XLORA
    - POLY
    - LN_TUNING
    - VERA
    - FOURIERFT
    - HRA
    - BONE
    """

    PROMPT_TUNING = "PROMPT_TUNING"
    MULTITASK_PROMPT_TUNING = "MULTITASK_PROMPT_TUNING"
    P_TUNING = "P_TUNING"
    PREFIX_TUNING = "PREFIX_TUNING"
    LORA = "LORA"
    ADALORA = "ADALORA"
    BOFT = "BOFT"
    ADAPTION_PROMPT = "ADAPTION_PROMPT"
    IA3 = "IA3"
    LOHA = "LOHA"
    LOKR = "LOKR"
    OFT = "OFT"
    POLY = "POLY"
    LN_TUNING = "LN_TUNING"
    VERA = "VERA"
    FOURIERFT = "FOURIERFT"
    XLORA = "XLORA"
    HRA = "HRA"
    VBLORA = "VBLORA"
    CPT = "CPT"
    BONE = "BONE"


class TaskType(str, enum.Enum):
    """
    Enum class for the different types of tasks supported by PEFT.

    Overview of the supported task types:
    - SEQ_CLS: Text classification.
    - SEQ_2_SEQ_LM: Sequence-to-sequence language modeling.
    - CAUSAL_LM: Causal language modeling.
    - TOKEN_CLS: Token classification.
    - QUESTION_ANS: Question answering.
    - FEATURE_EXTRACTION: Feature extraction. Provides the hidden states which can be used as embeddings or features
      for downstream tasks.
    """

    SEQ_CLS = "SEQ_CLS"
    SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
    CAUSAL_LM = "CAUSAL_LM"
    TOKEN_CLS = "TOKEN_CLS"
    QUESTION_ANS = "QUESTION_ANS"
    FEATURE_EXTRACTION = "FEATURE_EXTRACTION"


def register_peft_method(*, name, config_cls, model_cls=None) -> None:
    """TODO"""
    from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING, PEFT_TYPE_TO_TUNER_MAPPING
    from peft.utils.constants import PEFT_TYPE_TO_PREFIX_MAPPING

    if name.endswith("_"):
        raise ValueError(f"Please pass the name of the PEFT method without '_' suffix, got {name}.")

    if not name.islower():
        raise ValueError(f"The name of the PEFT method should be in lower case letters, got {name}.")

    if name.upper() not in list(PeftType):
        raise ValueError(f"Unknown PEFT type {name.upper()}, please add an entry to peft.utils.peft_types.PeftType.")

    peft_type = getattr(PeftType, name.upper())

    # model_cls can be None for prompt learning methods, which don't have dedicated model classes
    if model_cls:

        if model_cls.prefix != name + "_":
            raise ValueError(
                f"Inconsistent names: The method is called '{name}' but the prefix is called {model_cls.prefix} "
                "(they should be the same, except that the prefix ends with an '_')"
            )

    PEFT_TYPE_TO_PREFIX_MAPPING[peft_type] = name
    PEFT_TYPE_TO_CONFIG_MAPPING[peft_type] = config_cls
    if model_cls:
        PEFT_TYPE_TO_TUNER_MAPPING[peft_type] = model_cls
