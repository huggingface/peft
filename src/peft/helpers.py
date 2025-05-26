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

import inspect
from contextlib import contextmanager
from copy import deepcopy
from functools import update_wrapper
from types import MethodType

from torch import nn

from .peft_model import PeftConfig, PeftModel
from .tuners.lora import LoraLayer
from .tuners.tuners_utils import BaseTunerLayer


def update_forward_signature(model: PeftModel) -> None:
    """
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


def update_generate_signature(model: PeftModel) -> None:
    """
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


def update_signature(model: PeftModel, method: str = "all") -> None:
    """
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


def check_if_peft_model(model_name_or_path: str) -> bool:
    """
    Check if the model is a PEFT model.

    Args:
        model_name_or_path (`str`):
            Model id to check, can be local or on the Hugging Face Hub.

    Returns:
        `bool`: True if the model is a PEFT model, False otherwise.
    """
    is_peft_model = True
    try:
        PeftConfig.from_pretrained(model_name_or_path)
    except Exception:
        # allow broad exceptions so that this works even if new exceptions are added on HF Hub side
        is_peft_model = False

    return is_peft_model


@contextmanager
def rescale_adapter_scale(model, multiplier):
    """
    Context manager to temporarily rescale the scaling of the LoRA adapter in a model.

    The original scaling values are restored when the context manager exits. This context manager works with the
    transformers and diffusers models that have directly loaded LoRA adapters.

    For LoRA, applying this context manager with multiplier in [0, 1] is strictly equivalent to applying
    [wise-ft](https://huggingface.co/papers/2109.01903) (see [#1940](https://github.com/huggingface/peft/issues/1940)
    for details). It can improve the performances of the model if there is a distribution shiftbetween the training
    data used for fine-tuning, and the test data used during inference.

    Warning: It has been reported that when using Apple's MPS backend for PyTorch, it is necessary to add a short sleep
        time after exiting the context before the scales are fully restored.

    Args:
        model: The model containing `LoraLayer` modules whose scaling is to be adjusted.
        multiplier (float or int):
            The multiplier that rescales the `scaling` attribute. Must be of type float or int.

    Raises:
        ValueError: If the model does not contain any `LoraLayer`
            instances, indicating that the model does not support scaling.

    Example:

    ```python
    >>> model = ModelWithLoraLayer()
    >>> multiplier = 0.5
    >>> with rescale_adapter_scale(model, multiplier):
    ...     outputs = model(**inputs)  # Perform operations with the scaled model
    >>> outputs = model(**inputs)  # The original scaling values are restored here
    ```
    """
    # check if multiplier has a valid data type
    if not isinstance(multiplier, (float, int)):
        raise TypeError(f"Argument multiplier should be of type float, got {type(multiplier)}")

    # iterate on the model's modules and grab the original scaling attribute
    # from the lora layers if present
    original_scaling = {}
    for module in model.modules():
        if isinstance(module, LoraLayer):
            original_scaling[module] = module.scaling.copy()
            module.scaling = {k: v * multiplier for k, v in module.scaling.items()}

    # check whether scaling is prohibited on model
    # the original scaling dictionary should be empty
    # if there were no lora layers
    if not original_scaling:
        raise ValueError("scaling is only supported for models with `LoraLayer`s")
    try:
        yield

    finally:
        # restore original scaling values after exiting the context
        for module, scaling in original_scaling.items():
            module.scaling = scaling


@contextmanager
def disable_input_dtype_casting(model: nn.Module, active: bool = True):
    """
    Context manager disables input dtype casting to the dtype of the weight.

    Parameters:
        model (nn.Module):
            The model containing PEFT modules whose input dtype casting is to be adjusted.
        active (bool):
            Whether the context manager is active (default) or inactive.

    """
    # Additional info: Normally, the dtype of the weight and input need to match, which is why the dtype is cast.
    # However, in certain circumustances, this is handled by forward hooks, e.g. when using layerwise casting in
    # diffusers. In that case, PEFT casting the dtype interferes with the layerwise casting, which is why the option to
    # disable it is given.
    if not active:
        yield
        return

    original_values = {}
    for name, module in model.named_modules():
        if not isinstance(module, BaseTunerLayer):
            continue
        original_values[name] = module.cast_input_dtype_enabled
        module.cast_input_dtype_enabled = False

    try:
        yield
    finally:
        for name, module in model.named_modules():
            if not isinstance(module, BaseTunerLayer):
                continue
            if name in original_values:
                module.cast_input_dtype_enabled = original_values[name]
