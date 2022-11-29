from .pet_model import PETModelForCausalLM, PETModelForSeq2SeqLM, PETModelForSequenceClassification
from .tuners import PrefixTuningConfig, PromptEncoderConfig, PromptTuningConfig
from .utils import PETConfig


MODEL_TYPE_TO_PROMPT_MODEL_MAPPING = {
    "SEQ_CLS": PETModelForSequenceClassification,
    "SEQ_2_SEQ_LM": PETModelForSeq2SeqLM,
    "CAUSAL_LM": PETModelForCausalLM,
}

PET_TYPE_TO_CONFIG_MAPPING = {
    "PROMPT_TUNING": PromptTuningConfig,
    "PREFIX_TUNING": PrefixTuningConfig,
    "P_TUNING": PromptEncoderConfig,
    "LORA": PETConfig,
}
