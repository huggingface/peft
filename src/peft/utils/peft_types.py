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
from typing import Optional


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
    - RANDLORA
    - C3A
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
    RANDLORA = "RANDLORA"
    TRAINABLE_TOKENS = "TRAINABLE_TOKENS"
    C3A = "C3A"


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


def register_peft_method(
    *, name: str, config_cls, model_cls, prefix: Optional[str] = None, is_mixed_compatible=False
) -> None:
    """
    Function to register a finetuning method like LoRA to be available in PEFT.

    This method takes care of registering the PEFT method's configuration class, the model class, and optionally the
    prefix.

    Args:
        name (str):
            The name of the PEFT method. It must be unique.
        config_cls:
            The configuration class of the PEFT method.
        model_cls:
            The model class of the PEFT method.
        prefix (Optional[str], optional):
            The prefix of the PEFT method. It should be unique. If not provided, the name of the PEFT method is used as
            the prefix.
        is_mixed_compatible (bool, optional):
            Whether the PEFT method is compatible with `PeftMixedModel`. If you're not sure, leave it as False
            (default).

    Example:

        ```py
        # inside of peft/tuners/my_peft_method/__init__.py
        from peft.utils import register_peft_method

        register_peft_method(name="my_peft_method", config_cls=MyConfig, model_cls=MyModel)
        ```
    """
    from peft.mapping import (
        PEFT_TYPE_TO_CONFIG_MAPPING,
        PEFT_TYPE_TO_MIXED_MODEL_MAPPING,
        PEFT_TYPE_TO_PREFIX_MAPPING,
        PEFT_TYPE_TO_TUNER_MAPPING,
    )

    if name.endswith("_"):
        raise ValueError(f"Please pass the name of the PEFT method without '_' suffix, got {name}.")

    if not name.islower():
        raise ValueError(f"The name of the PEFT method should be in lower case letters, got {name}.")

    if name.upper() not in list(PeftType):
        raise ValueError(f"Unknown PEFT type {name.upper()}, please add an entry to peft.utils.peft_types.PeftType.")

    peft_type = getattr(PeftType, name.upper())

    # model_cls can be None for prompt learning methods, which don't have dedicated model classes
    if prefix is None:
        prefix = name + "_"

    if (
        (peft_type in PEFT_TYPE_TO_CONFIG_MAPPING)
        or (peft_type in PEFT_TYPE_TO_TUNER_MAPPING)
        or (peft_type in PEFT_TYPE_TO_MIXED_MODEL_MAPPING)
    ):
        raise KeyError(f"There is already PEFT method called '{name}', please choose a unique name.")

    if prefix in PEFT_TYPE_TO_PREFIX_MAPPING:
        raise KeyError(f"There is already a prefix called '{prefix}', please choose a unique prefix.")

    model_cls_prefix = getattr(model_cls, "prefix", None)
    if (model_cls_prefix is not None) and (model_cls_prefix != prefix):
        raise ValueError(
            f"Inconsistent prefixes found: '{prefix}' and '{model_cls_prefix}' (they should be the same)."
        )

    PEFT_TYPE_TO_PREFIX_MAPPING[peft_type] = prefix
    PEFT_TYPE_TO_CONFIG_MAPPING[peft_type] = config_cls
    PEFT_TYPE_TO_TUNER_MAPPING[peft_type] = model_cls
    if is_mixed_compatible:
        PEFT_TYPE_TO_MIXED_MODEL_MAPPING[peft_type] = model_cls
