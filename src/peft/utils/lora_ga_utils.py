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

from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, Optional

import torch
import torch.distributed as dist
from tqdm import tqdm


def timer(func):
    """
    Decorator to measure and print function execution time.

    Args:
        func: Function to time

    Returns:
        Wrapped function that prints execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()
        duration = end - start
        print(f"{func.__name__} took {duration}")
        return result
    return wrapper


def get_record_gradient_hook(named_grad: Dict[str, torch.Tensor]):
    """
    Create a backward hook to record gradients for each module.

    The hook captures gradients during backward pass and accumulates them
    in the named_grad dictionary.

    Args:
        named_grad: Dictionary to store gradients by module name

    Returns:
        Hook function to attach to modules
    """
    def hook(module, grad_input, grad_output):
        module_name = getattr(module, "_lora_ga_name", None)
        if module_name is not None and hasattr(module, "weight"):
            if module.weight.grad is not None:
                if module_name not in named_grad:
                    named_grad[module_name] = module.weight.grad.clone().detach()
                else:
                    named_grad[module_name] += module.weight.grad.clone().detach()
    return hook


@timer
def estimate_gradient(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    accelerator,
    iters: int = 64,
    quant_flag: bool = False,
    module_offload_classes: Optional[tuple] = None,
) -> Dict[str, torch.Tensor]:
    """
    Estimate gradients over a dataset for LoRA-GA initialization.

    This function performs forward and backward passes over the provided data
    to accumulate gradients for each target module. These gradients are then
    used for gradient-aware initialization of LoRA adapters.

    Args:
        model: The model to estimate gradients for
        dataloader: DataLoader providing training data
        accelerator: Accelerator instance for distributed training
        iters: Number of batches to process for gradient estimation. Default: 64
        quant_flag: Whether the model is quantized (requires special handling). Default: False
        module_offload_classes: Tuple of module classes to offload when using quantization. Default: None

    Returns:
        Dictionary mapping module names to accumulated gradient tensors
    """
    model.train()
    named_grad = {}

    hooks = []
    for name, module in model.named_modules():
        if hasattr(module, "weight") and module.weight.requires_grad:
            module._lora_ga_name = name
            hook = module.register_full_backward_hook(get_record_gradient_hook(named_grad))
            hooks.append(hook)

    progress_bar = tqdm(range(min(iters, len(dataloader))), desc="Estimating gradients")

    for i, batch in enumerate(dataloader):
        if i >= iters:
            break

        outputs = model(**batch)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

        accelerator.backward(loss)

        model.zero_grad()

        progress_bar.update(1)

    for hook in hooks:
        hook.remove()

    for module in model.modules():
        if hasattr(module, "_lora_ga_name"):
            delattr(module, "_lora_ga_name")

    for name in named_grad:
        named_grad[name] /= min(iters, len(dataloader))

    if dist.is_initialized():
        world_size = dist.get_world_size()
        for name in named_grad:
            dist.all_reduce(named_grad[name], op=dist.ReduceOp.SUM)
            named_grad[name] /= world_size

    model.eval()
    return named_grad


@contextmanager
def LoraGAContext(model: torch.nn.Module, named_grad: Dict[str, torch.Tensor]):
    """
    Context manager to temporarily attach gradients to model for LoRA-GA initialization.

    Usage:
        with LoraGAContext(model=model, named_grad=named_grad):
            model = get_peft_model(model, lora_ga_config)

    Args:
        model: The model to attach gradients to
        named_grad: Dictionary of gradients by module name

    Yields:
        The model with gradients attached
    """
    model.named_grad = named_grad
    try:
        yield model
    finally:
        if hasattr(model, "named_grad"):
            delattr(model, "named_grad")


def save_loraga_model_init(model, save_dir: str):
    """
    Save initial LoRA adapter state before training.

    LoRA-GA modifies base weights during initialization, so we need to save
    the initial adapter state to compute deltas later.

    Args:
        model: PEFT model with LoRA-GA adapters
        save_dir: Directory to save initial state
    """
    from safetensors.torch import save_file

    os.makedirs(save_dir, exist_ok=True)

    adapter_state = {}
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            adapter_state[name] = param.detach().cpu().clone()

    save_path = os.path.join(save_dir, "adapter_model_init.safetensors")
    save_file(adapter_state, save_path)
    print(f"Initial adapter state saved to {save_path}")


def save_loraga_model_final(model, save_dir: str):
    """
    Save final LoRA adapter state after training as delta.

    Since LoRA-GA modifies base weights, the standard save only captures
    adapter matrices. To get the true training effect, we compute:
    delta = (B_final @ A_final) - (B_init @ A_init)

    Args:
        model: PEFT model after training
        save_dir: Directory containing initial state and to save final state
    """
    from safetensors.torch import load_file, save_file

    init_path = os.path.join(save_dir, "adapter_model_init.safetensors")
    if not os.path.exists(init_path):
        warnings.warn(
            f"Initial state not found at {init_path}. Saving final state without delta computation. "
            "For proper LoRA-GA workflow, call save_loraga_model_init before training."
        )
        model.save_pretrained(save_dir)
        return

    init_state = load_file(init_path)

    final_state = {}
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            final_state[name] = param.detach().cpu().clone()

    delta_state = {}
    for name in final_state:
        if name in init_state:
            delta_state[name] = final_state[name] - init_state[name]
        else:
            delta_state[name] = final_state[name]

    delta_path = os.path.join(save_dir, "adapter_model.safetensors")
    save_file(delta_state, delta_path)

    model.save_pretrained(save_dir)
    print(f"Final adapter delta saved to {delta_path}")
