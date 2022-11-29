from .pet_model import PETModelForCausalLM, PETModelForSeq2SeqLM, PETModelForSequenceClassification
from .tuners import PrefixTuningConfig, PromptEncoderConfig, PromptTuningConfig
from .utils import PETConfig


MODEL_TYPE_TO_PET_MODEL_MAPPING = {
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


def get_pet_config(config_dict):
    return PET_TYPE_TO_CONFIG_MAPPING[config_dict["pet_type"]](**config_dict)


def get_pet_model(model, pet_config):
    config = model.config.to_dict()
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

    return MODEL_TYPE_TO_PET_MODEL_MAPPING[pet_config.task_type](model, pet_config)
