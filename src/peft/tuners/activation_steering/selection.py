# Copyright 2026-present the HuggingFace Inc. team.
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

import math
from collections.abc import Callable

import torch
from torch import nn

from peft.tuners.tuners_utils import BaseTuner
from peft.utils.other import _get_submodules

from .layer import ActivationSteeringLayer
from .model import ActivationSteeringModel


def _gradient_scores(layers, train_step, adapter_name):
    # Each dose's gradient is the first-order effect of steering its neuron on the loss.
    values = [layer.activation_steering_values[adapter_name] for layer in layers.values()]
    states = [(value.requires_grad, value.grad) for value in values]
    try:
        for value in values:
            value.requires_grad_(True)
            value.grad = None
        with torch.enable_grad():
            train_step()
        grads = [value.grad for value in values]
    finally:
        for value, (requires_grad, grad) in zip(values, states):
            value.requires_grad_(requires_grad)
            value.grad = grad

    if any(grad is None for grad in grads):
        raise RuntimeError(
            "Some doses received no gradient. `train_step` must run forward and backward passes through every steered "
            "projection."
        )
    return [grad.float().cpu() for grad in grads]


@torch.no_grad()
def select_neurons(
    model: nn.Module,
    train_step: Callable[[], None],
    num_neurons: int,
    *,
    init_val: float,
    adapter_name: str = "default",
) -> None:
    """Select the neurons to steer by the gradient of a loss and remove all other doses in place.

    Start from an adapter whose config has no `steering_pattern`, so that every input channel of the targeted
    projections has a dose of zero, and calibrate it. `train_step` runs forward and backward passes on data showing the
    desired behavior, which gives every dose the gradient of the loss. Across all layers, the `num_neurons` doses with
    the largest absolute gradients are kept, each initialized to `init_val` with the sign that lowers the loss. All
    other doses are removed, projections without kept doses are restored to their base layers, and the kept neurons are
    written to the adapter's `steering_pattern`. The calibrated scales of the kept neurons are retained.

    The doses remain trainable, so they can be refined with a regular training loop, and `save_pretrained` saves only
    the kept doses and their scales. Call this function before creating an optimizer and before compiling the model,
    since it replaces the dose parameters.

    Args:
        model (`torch.nn.Module`):
            A PEFT model, or its tuner, with a single calibrated activation steering adapter.
        train_step (`Callable[[], None]`):
            Runs forward and backward passes of a loss whose descent direction is the desired behavior, e.g. the
            negative log-likelihood of example completions. Gradients accumulate over all passes.
        num_neurons (`int`):
            The number of neurons to keep across all targeted projections.
        init_val (`float`):
            The positive magnitude of each kept neuron's initial dose.
        adapter_name (`str`, defaults to `"default"`):
            The adapter to select neurons for.

    Example:
        ```python
        from peft import ActivationSteeringConfig, get_peft_model
        from peft.tuners.activation_steering import calibrate_activation_steering, select_neurons

        config = ActivationSteeringConfig(target_modules=["down_proj"], task_type="CAUSAL_LM")
        model = get_peft_model(base_model, config)
        calibrate_activation_steering(model, calibration_batches)


        def train_step():
            for batch in selection_batches:
                model(**batch).loss.backward()


        select_neurons(model, train_step, num_neurons=8, init_val=0.5)
        model.save_pretrained("steering-adapter")
        ```
    """
    tuner = model if isinstance(model, BaseTuner) else getattr(model, "base_model", None)
    if not isinstance(tuner, ActivationSteeringModel):
        raise TypeError("`select_neurons` expects a PEFT model with an activation steering adapter.")
    if adapter_name not in tuner.peft_config:
        raise ValueError(f"Adapter '{adapter_name}' does not exist.")
    if len(tuner.peft_config) > 1:
        raise ValueError("`select_neurons` supports models with a single adapter.")
    if not math.isfinite(init_val) or init_val <= 0:
        raise ValueError("`init_val` must be a finite, positive number.")

    layers = {
        name: module for name, module in tuner.model.named_modules() if isinstance(module, ActivationSteeringLayer)
    }
    num_doses = sum(layer.activation_steering_values[adapter_name].numel() for layer in layers.values())
    if not 0 < num_neurons <= num_doses:
        raise ValueError(f"`num_neurons` must be between 1 and the number of doses ({num_doses}).")

    grads = _gradient_scores(layers, train_step, adapter_name)
    scores = torch.cat([grad.abs() for grad in grads])
    top = torch.topk(scores, num_neurons)
    if (top.values == 0).any():
        raise ValueError(
            f"Only {int((scores > 0).sum())} doses received a non-zero gradient, fewer than `num_neurons`."
        )

    keep = torch.zeros_like(scores, dtype=torch.bool)
    keep[top.indices] = True
    doses = -init_val * torch.cat(grads).sign()
    sizes = [grad.numel() for grad in grads]

    steering_pattern = {}
    for (name, layer), layer_keep, layer_doses in zip(layers.items(), keep.split(sizes), doses.split(sizes)):
        positions = layer_keep.nonzero().flatten()
        if not len(positions):
            parent, _, target_name = _get_submodules(tuner.model, name)
            setattr(parent, target_name, layer.get_base_layer())
            tuner.targeted_module_names.remove(name)
            continue

        old = layer.activation_steering_values[adapter_name]
        layer.activation_steering_values[adapter_name] = nn.Parameter(
            layer_doses[positions].to(device=old.device, dtype=old.dtype), requires_grad=old.requires_grad
        )
        for buffers in (layer.activation_steering_indices, layer.activation_steering_scales):
            buffers[adapter_name] = buffers[adapter_name][positions.to(buffers[adapter_name].device)]
        steering_pattern[name] = {
            "indices": layer.activation_steering_indices[adapter_name].tolist(),
            "init_val": layer_doses[positions].tolist(),
        }

    tuner.peft_config[adapter_name].steering_pattern = steering_pattern
