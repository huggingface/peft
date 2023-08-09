import inspect
import warnings
from copy import deepcopy
from functools import update_wrapper
from types import MethodType

import torch

from .peft_model import (PeftModel, PeftModelForCausalLM,
                         PeftModelForSeq2SeqLM, PeftType)


# Copied from peft.peft_model PeftModel.forward
def default_forward(self:PeftModel, *args, **kwargs):
    return self.get_base_model()(*args, **kwargs)

# Copied from peft.peft_model PeftModelForCausalLM.generate
def generate_causal_lm(self:PeftModelForCausalLM, **kwargs):
    self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
    if hasattr(self.base_model, "model"):
        self.base_model.model.generation_config = self.generation_config
    else:
        self.base_model.generation_config = self.generation_config
    try:
        outputs = self.base_model.generate(**kwargs)
    except:
        self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
        raise
    else:
        self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
        return outputs

# Copied from peft.peft_model PeftModelForSeq2SeqLM.generate
def generate_seq2seq(self:PeftModelForSeq2SeqLM, **kwargs):
    peft_config = self.active_peft_config
    self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
    self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
        self._prepare_encoder_decoder_kwargs_for_generation
    )
    try:
        if not peft_config.is_prompt_learning:
            outputs = self.base_model.generate(**kwargs)
        else:
            if "input_ids" not in kwargs:
                raise ValueError("input_ids must be provided for Peft model generation")
            if kwargs.get("position_ids", None) is not None:
                warnings.warn(
                    "Position ids are not supported for parameter efficient tuning. Ignoring position ids."
                )
                kwargs["position_ids"] = None
            if kwargs.get("token_type_ids", None) is not None:
                warnings.warn(
                    "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                )
                kwargs["token_type_ids"] = None

            if peft_config.peft_type == PeftType.PREFIX_TUNING:
                outputs = self.base_model.generate(**kwargs)
            elif peft_config.peft_type in [PeftType.PROMPT_TUNING, PeftType.P_TUNING]:
                kwargs = deepcopy(kwargs)

                if "encoder_outputs" in kwargs:
                    del kwargs["encoder_ouputs"]
                    warnings.warn(
                        "`encoder_outputs` should not be passed to `generate` when using prompt tuning. Ignoring it."
                    )

                input_ids = kwargs.pop("input_ids")
                inputs_embeds = self.word_embeddings(input_ids)
                batch_size = inputs_embeds.shape[0]
                prompts = self.get_prompt(batch_size=batch_size)
                prompts = prompts.to(inputs_embeds.dtype)

                inputs_embeds = torch.cat((prompts[:, : peft_config.num_virtual_tokens], inputs_embeds), dim=1)
                kwargs["inputs_embeds"] = inputs_embeds

                if "attention_mask" in kwargs:
                    prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(
                        kwargs["attention_mask"].device
                    )
                    kwargs["attention_mask"] = torch.cat((prefix_attention_mask, kwargs["attention_mask"]), dim=1)

                return self.base_model.generate(**kwargs)
            else:
                raise NotImplementedError
    except:
        self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
        self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
            self.base_model_prepare_encoder_decoder_kwargs_for_generation
        )
        raise
    else:
        self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
        self.base_model._prepare_encoder_decoder_kwargs_for_generation = (
            self.base_model_prepare_encoder_decoder_kwargs_for_generation
        )
        return outputs


def update_forward_signature(model:PeftModel):
    """
    Updates the forward signature of the PeftModel to include parents class signature
    Args:
        model (`PeftModel`): Peft model to update the forward signature
    Example:

    ```python
    >>> from transformers import  WhisperForConditionalGeneration
    >>> from peft import  get_peft_model, LoraConfig,  update_forward_signature

    >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    >>> peft_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "v_proj"])

    >>> peft_model = get_peft_model(model, peft_config)
    >>> update_forward_signature(peft_model)
    ```
    """

    # Only update signature when the current forward signature only has *args and **kwargs
    current_signature = inspect.signature(model.forward)
    if (
        len(current_signature.parameters) == 2
        and "args" in current_signature.parameters
        and "kwargs" in current_signature.parameters
    ):
        update_wrapper(
            default_forward, type(model.get_base_model()).forward, assigned=("__doc__", "__name__", "__annotations__")
        )
        model.forward = MethodType(default_forward, model)
        

def update_generate_signature(model:PeftModel):
    """
    Updates the generate signature of a PeftModel with overriding generate to include parents class signature
    Args:
        model (`PeftModel`): Peft model to update the generate signature
    Example:

    ```python
    >>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    >>> from peft import  get_peft_model, LoraConfig, TaskType,  update_generate_signature

    >>> model_name_or_path = "bigscience/mt0-large"
    >>> tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    >>> model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

    >>> peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    >>> peft_model = get_peft_model(model, peft_config)
    >>> update_generate_signature(peft_model)
    >>> help(peft_model.generate)
    ```
    """
    if isinstance(model, PeftModelForSeq2SeqLM):
        update_wrapper(
            generate_seq2seq, type(model.get_base_model()).generate, assigned=("__doc__", "__name__", "__annotations__")
        )
        model.generate = MethodType(generate_seq2seq, model)
    elif isinstance(model, PeftModelForCausalLM):
        update_wrapper(
            generate_causal_lm, type(model.get_base_model()).generate, assigned=("__doc__", "__name__", "__annotations__")
        )
        model.generate = MethodType(generate_causal_lm, model)
        
def update_signature(model:PeftModel, method:str = 'all'):
    """
    Updates the signature of a PeftModel include parents class signature for forward or generate method
    Args:
        model (`PeftModel`): Peft model to update generate or forward signature
        method (`str`): method to update signature choose one of "forward", "generate", "all"
    Example:
     ```python
    >>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    >>> from peft import  get_peft_model, LoraConfig, TaskType,  update_signature

    >>> model_name_or_path = "bigscience/mt0-large"
    >>> tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    >>> model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

    >>> peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    >>> peft_model = get_peft_model(model, peft_config)
    >>> update_signature(peft_model)
    >>> help(peft_model.generate)
    ```
    """
    if method == "forward":
        update_forward_signature(model)
    elif method == "generate":
        update_generate_signature(model)
    elif method == "all":
        update_forward_signature(model)
        update_generate_signature(model)
    else:
        raise ValueError(f"method {method} is not supported please choose one of ['forward', 'generate', 'all']")