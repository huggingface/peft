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
from copy import deepcopy
from functools import reduce, update_wrapper
from operator import attrgetter
from types import MethodType

import bitsandbytes
import torch.nn as nn
from torch.optim import Optimizer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

from .peft_model import PeftConfig, PeftModel
from .tuners.lora.layer import Embedding


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


def get_module(name, opt_model):
    """
    Retrieve a module from a model using its parameter name.

    Args:
        name (str): Full name of the parameter, typically including module path.
        opt_model (torch.nn.Module): The model from which to retrieve the module.

    Returns:
        Module corresponding to the given name.
    """
    parent_idx = 2 if "lora" in name else 1
    module_names = name.split(sep=".")[:-parent_idx]
    module = reduce(getattr, module_names, opt_model)
    return module


def create_loraplus_optimizer(
    model: PeftModel,
    optimizer_cls: type[Optimizer],
    optimizer_kwargs: dict,
    loraplus_lr_ratio: float,
    loraplus_lr_embedding: float = 1e-6,
) -> Optimizer:
    """
    Creates a LoraPlus optimizer. Implementing LoRA+ https://arxiv.org/abs/2402.12354 Reference:
    https://github.com/nikhil-ghosh-berkeley/loraplus/

    Args:
        model (`torch.nn.Module`): The model to be optimized.
        optimizer_cls (`torch.optim.Optimizer`): The optimizer class to be used.
        optimizer_kwargs (`dict`): Additional keyword arguments to be passed to the optimizer.
        loraplus_lr_ratio (`float`): The ratio of the learning rate to be used for the embedding layer.
        loraplus_lr_embedding (`float`): The learning rate to be used for the embedding layer.
    """

    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    param_groups = {
        "groupA": {},
        "groupB": {},
        "groupB_no_decay": {},
        "embedding": {},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        module = attrgetter(name)(model)
        if isinstance(module, Embedding):
            param_groups["embedding"][name] = param
        elif "lora_B" in name or param.ndim == 1:
            if name in decay_parameters:
                param_groups["groupB"][name] = param
            else:
                param_groups["groupB_no_decay"][name] = param
        else:
            param_groups["groupA"][name] = param

    assigned_param_groups = ""
    for group in param_groups:
        assigned_param_groups += f"{group}\n {list(param_groups[group].keys())}\n\n"

    lr = optimizer_kwargs["lr"]
    weight_decay = optimizer_kwargs.get("weight_decay", 0.0)

    optimizer_grouped_parameters = [
        {
            "params": list(param_groups["groupA"].values()),
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": list(param_groups["embedding"].values()),
            "weight_decay": weight_decay,
            "lr": loraplus_lr_embedding,
        },
        {
            "params": list(param_groups["groupB"].values()),
            "weight_decay": weight_decay,
            "lr": lr * loraplus_lr_ratio,
        },
        {
            "params": list(param_groups["groupB_no_decay"].values()),
            "weight_decay": 0.0,
            "lr": lr * loraplus_lr_ratio,
        },
    ]

    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    if optimizer_cls.__name__ == "Adam8bit":
        manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                manager.register_module_override(module, "weight", {"optim_bits": 32})
    return optimizer
