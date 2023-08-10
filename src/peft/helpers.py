import inspect
from copy import deepcopy
from functools import update_wrapper
from types import MethodType

from .peft_model import PeftModel


def update_forward_signature(model: PeftModel):
    """
    Args:
    Updates the forward signature of the PeftModel to include parents class signature
        model (`PeftModel`): Peft model to update the forward signature
    Example:

    ```python
    >>> from transformers import WhisperForConditionalGeneration
    >>> from peft import get_peft_model, LoraConfig, update_forward_signature

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
        forward = deepcopy(model.forward.__func__)
        update_wrapper(
            forward, type(model.get_base_model()).forward, assigned=("__doc__", "__name__", "__annotations__")
        )
        model.forward = MethodType(forward, model)


def update_generate_signature(model: PeftModel):
    """
    Args:
    Updates the generate signature of a PeftModel with overriding generate to include parents class signature
        model (`PeftModel`): Peft model to update the generate signature
    Example:

    ```python
    >>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    >>> from peft import get_peft_model, LoraConfig, TaskType, update_generate_signature

    >>> model_name_or_path = "bigscience/mt0-large"
    >>> tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    >>> model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

    >>> peft_config = LoraConfig(
    ...     task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    ... )
    >>> peft_model = get_peft_model(model, peft_config)
    >>> update_generate_signature(peft_model)
    >>> help(peft_model.generate)
    ```
    """
    if not hasattr(model, "generate"):
        return
    current_signature = inspect.signature(model.generate)
    if (
        len(current_signature.parameters) == 2
        and "args" in current_signature.parameters
        and "kwargs" in current_signature.parameters
    ) or (len(current_signature.parameters) == 1 and "kwargs" in current_signature.parameters):
        generate = deepcopy(model.generate.__func__)
        update_wrapper(
            generate,
            type(model.get_base_model()).generate,
            assigned=("__doc__", "__name__", "__annotations__"),
        )
        model.generate = MethodType(generate, model)


def update_signature(model: PeftModel, method: str = "all"):
    """
    Args:
    Updates the signature of a PeftModel include parents class signature for forward or generate method
        model (`PeftModel`): Peft model to update generate or forward signature method (`str`): method to update
        signature choose one of "forward", "generate", "all"
    Example:
     ```python
    >>> from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    >>> from peft import get_peft_model, LoraConfig, TaskType, update_signature

    >>> model_name_or_path = "bigscience/mt0-large"
    >>> tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    >>> model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

    >>> peft_config = LoraConfig(
    ...     task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    ... )
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
