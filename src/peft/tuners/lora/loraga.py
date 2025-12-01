# Copyright 2024-present the HuggingFace Inc. team.
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

# Reference code: https://github.com/Outsider565/LoRA-GA
# Reference paper: https://arxiv.org/abs/2407.05000

import os
from collections.abc import Callable
from typing import Any, Optional

import torch
import torch.nn as nn
from attr import dataclass

from peft.tuners.lora.config import LoraConfig
from peft.tuners.lora.model import LoraModel
from peft.utils.other import get_pattern_key


@dataclass
class LoraGAGradient:
    gradient: torch.Tensor


def target_modules(model: nn.Module, config: LoraConfig):
    """
    Iterate over LoRA-GA target name and modules of a model. A module is a target if its name is in
    `config.target_modules` and is `nn.Linear`.
    """
    for name, module in model.named_modules():
        if LoraModel._check_target_module_exists(config, name) and isinstance(module, nn.Linear):
            yield name, module


def get_model_device(model: nn.Module) -> str:
    if hasattr(model, "module"):  # Handle DeepSpeed/DataParallel
        model = model.module
    return next(iter(model.parameters())).device


@torch.no_grad()
def preprocess_loraga(
    model: nn.Module,
    lora_config: LoraConfig,
    train_step: Callable[[], None],
):
    """
    Build necessary LoRA-GA fields for a model by estimating gradients.

    For each linear layer, gradients will be estimated by running the provided train_step callback.
    These gradients are then attached to the modules and used during initialization.

    Args:
        model (`nn.Module`):
            Model to preprocess.
        lora_config (`LoraConfig`):
            Lora configuration of the model. `lora_config.lora_ga_config` should be set.
        train_step (`Callable[[], None]`):
            Callback to run gradient estimation. Typically you should run model forward and backward
            passes in this callback. The gradients will be accumulated across all calls within this
            callback.

    Upon completion, the following fields are set for each target module:
        loraga_grad (`torch.Tensor`):
            Accumulated gradient for the weight matrix.
    """
    if lora_config.lora_ga_config is None:
        raise ValueError("`lora_config.lora_ga_config` must be set when using LoRA-GA initialization.")

    cache_file = lora_config.lora_ga_config.cache_file

    # If cache exists, load from cache
    if cache_file is not None and os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
        cache = torch.load(cache_file, map_location=get_model_device(model))
        for name, module in target_modules(model, lora_config):
            module.loraga_grad = cache[f"{name}.loraga_grad"]
    else:
        # Estimate gradients by running train_step
        estimate_gradients(model, lora_config, train_step)

        # Save cache to disk if specified
        if cache_file is not None:
            cache: dict[str, Any] = {}
            for name, module in target_modules(model, lora_config):
                cache[f"{name}.loraga_grad"] = module.loraga_grad

            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            torch.save(cache, cache_file)

    # Attach lora_ga_config to each target module for layer initialization
    for name, module in target_modules(model, lora_config):
        module.lora_ga_config = lora_config.lora_ga_config


def estimate_gradients(
    model: nn.Module,
    lora_config: LoraConfig,
    train_step: Callable[[], None],
):
    """
    Estimate gradients for LoRA-GA initialization.

    This function enables gradient computation on model parameters and runs the train_step callback.
    After backward passes, gradients are accumulated from the .grad attribute of each module's weight.
    """
    model.train()

    # Ensure parameters require gradients
    for param in model.parameters():
        param.requires_grad_(True)

    # Initialize gradient storage for each target module
    for name, module in target_modules(model, lora_config):
        module.loraga_grad = torch.zeros_like(module.weight.data, dtype=torch.float32)
        module.grad_count = 0

    # Enable gradient computation
    with torch.enable_grad():
        # Run train_step to accumulate gradients
        train_step()

    # Accumulate gradients from each target module's weight.grad
    for name, module in target_modules(model, lora_config):
        if module.weight.grad is not None:
            module.loraga_grad += module.weight.grad.detach().float()
            module.grad_count += 1

    # Average gradients and clean up temporary fields
    for name, module in target_modules(model, lora_config):
        if module.grad_count > 0:
            module.loraga_grad /= module.grad_count
        del module.grad_count
