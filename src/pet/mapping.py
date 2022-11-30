from .pet_model import PETModelForCausalLM, PETModelForSeq2SeqLM, PETModelForSequenceClassification
from .tuners import LoRAConfig, PrefixTuningConfig, PromptEncoderConfig, PromptTuningConfig
from .utils import PETType


MODEL_TYPE_TO_PET_MODEL_MAPPING = {
    "SEQ_CLS": PETModelForSequenceClassification,
    "SEQ_2_SEQ_LM": PETModelForSeq2SeqLM,
    "CAUSAL_LM": PETModelForCausalLM,
}

PET_TYPE_TO_CONFIG_MAPPING = {
    "PROMPT_TUNING": PromptTuningConfig,
    "PREFIX_TUNING": PrefixTuningConfig,
    "P_TUNING": PromptEncoderConfig,
    "LORA": LoRAConfig,
}

TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = {
    "t5": ["q", "v"],
    "bart": ["q_proj", "v_proj"],
    "gpt2": ["c_attn"],
    "bloom": ["query_key_value"],
    "opt": ["q_proj", "v_proj"],
    "gptj": ["q_proj", "v_proj"],
    "gpt_neox": ["query_key_value"],
    "gpt_neo": ["q_proj", "v_proj"],
    "bert": ["query", "value"],
    "roberta": ["query", "value"],
    "electra": ["query", "value"],
    "deberta-v2": ["query_proj", "value_proj"],
    "deberta": ["in_proj"],
}


def get_pet_config(config_dict):
    return PET_TYPE_TO_CONFIG_MAPPING[config_dict["pet_type"]](**config_dict)


def _prepare_prompt_learning_config(pet_config, config):
    if pet_config.num_layers is None:
        if "num_hidden_layers" in config:
            num_layers = config["num_hidden_layers"]
        elif "num_layers" in config:
            num_layers = config["num_layers"]
        elif "n_layer" in config:
            num_layers = config["n_layer"]
        else:
            raise ValueError("Please specify `num_layers` in `pet_config`")
        pet_config.num_layers = num_layers

    if pet_config.token_dim is None:
        if "hidden_size" in config:
            token_dim = config["hidden_size"]
        elif "n_embd" in config:
            token_dim = config["n_embd"]
        elif "d_model" in config:
            token_dim = config["d_model"]
        else:
            raise ValueError("Please specify `token_dim` in `pet_config`")
        pet_config.token_dim = token_dim

    if pet_config.num_attention_heads is None:
        if "num_attention_heads" in config:
            num_attention_heads = config["num_attention_heads"]
        elif "n_head" in config:
            num_attention_heads = config["n_head"]
        elif "num_heads" in config:
            num_attention_heads = config["num_heads"]
        elif "encoder_attention_heads" in config:
            num_attention_heads = config["encoder_attention_heads"]
        else:
            raise ValueError("Please specify `num_attention_heads` in `pet_config`")
        pet_config.num_attention_heads = num_attention_heads

    if pet_config.encoder_hidden_size is None:
        pet_config.encoder_hidden_size = token_dim

    return pet_config


def _prepare_lora_config(pet_config, config):
    if pet_config.target_modules is None:
        if config.model_type not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
            raise ValueError("Please specify `target_modules` in `pet_config`")
        pet_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[config.model_type]
    if len(pet_config.target_modules) == 1:
        pet_config.fan_in_fan_out = True
        pet_config.enable_lora = [True, False, True]
    if pet_config.inference_mode:
        pet_config.merge_weights = True
    return pet_config


def get_pet_model(model, pet_config):
    config = model.config.to_dict()
    if pet_config.pet_type != PETType.LORA:
        pet_config = _prepare_prompt_learning_config(pet_config, config)
    else:
        pet_config = _prepare_lora_config(pet_config, config)

    return MODEL_TYPE_TO_PET_MODEL_MAPPING[pet_config.task_type](model, pet_config)
