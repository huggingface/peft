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

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import nn

from peft.tuners.tuners_utils import BaseTunerLayer

from .layer import ActivationSteeringLayer, _get_residual


@torch.no_grad()
def calibrate_activation_steering(
    model: nn.Module,
    dataloader: Iterable[Mapping[str, Any]] | Mapping[str, Any],
    adapter_name: str = "default",
) -> None:
    """Calibrate the fixed residual scales of an activation steering adapter in place.

    Run baseline forwards with every adapter disabled and the model in evaluation mode. For each targeted block, pool
    outgoing residual L2 norms over calibration tokens, excluding padding and the first valid token of every sequence,
    and store their median.

    For an even number of tokens, use the average of the two middle norms, as in:
    https://github.com/lyramakesmusic/neuron-explorer/blob/5a8f8267f6bd3c8caa87d598870949ab0d25b914/sweep/neuron_stats.py

    The resulting non-trainable scales are saved with the adapter; calibration is required before inference or further
    fine-tuning of its doses.

    Args:
        model (`torch.nn.Module`):
            A PEFT model, tuner, or model with activation steering adapters injected directly.
        dataloader (`Iterable[Mapping]` or `Mapping`):
            Model input batches, or one input batch. Prepare tensors on the model's input device. An optional
            two-dimensional `attention_mask` identifies valid tokens. Each batch must be a fresh sequence, without
            `past_key_values`. For language models, passing `use_cache=False` avoids allocating unused caches.
        adapter_name (`str`, defaults to `"default"`):
            Adapter to calibrate, including an inactive adapter. Other adapters' scales and all dose parameters are
            unchanged. Repeated calls replace this adapter's scales using the new calibration data.

    All module training modes and adapter enabled states are restored, including on failure. Gradient flags and active
    adapter selection are unchanged. Scales are updated only after every targeted block has valid statistics.

    Example:
        ```python
        from peft.tuners.activation_steering import calibrate_activation_steering

        calibrate_activation_steering(model, calibration_dataloader)
        model.save_pretrained("steering-adapter")
        ```
    """
    layers = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, ActivationSteeringLayer) and adapter_name in module.activation_steering_values
    }
    if not layers:
        raise ValueError(f"No activation steering layers found for adapter '{adapter_name}'.")

    if isinstance(dataloader, Mapping):
        dataloader = [dataloader]

    norms = {name: [] for name in layers}
    attention_mask = None

    def capture(name):
        def hook(module, args, output):
            residual = _get_residual(output)
            if residual.shape[-1] != module.activation_steering_directions[adapter_name].shape[0]:
                raise ValueError("The source projection's output dimension must match the block's hidden dimension.")

            if attention_mask is None:
                eligible = torch.ones(residual.shape[:2], device=residual.device, dtype=torch.bool)
            else:
                if attention_mask.shape != residual.shape[:2]:
                    raise ValueError("The calibration attention mask must match the block output dimensions.")
                eligible = attention_mask.to(device=residual.device, dtype=torch.bool)

            eligible = eligible & (eligible.cumsum(dim=1) > 1)
            values = residual.float().norm(dim=-1)[eligible]
            if not torch.isfinite(values).all():
                raise ValueError(f"Non-finite calibration residual norms in block '{name}'.")

            norms[name].append(values.cpu())

        return hook

    training_states = [(module, module.training) for module in model.modules()]
    adapter_states = [
        (module, module.disable_adapters) for module in model.modules() if isinstance(module, BaseTunerLayer)
    ]
    handles = []
    try:
        model.eval()
        for module, _ in adapter_states:
            # Only bypass adapters; toggling requires_grad would disturb subsequent dose fine-tuning.
            module._disable_adapters = True

        for name, layer in layers.items():
            handles.append(layer.register_forward_hook(capture(name)))

        for batch in dataloader:
            if not isinstance(batch, Mapping):
                raise TypeError("Calibration batches must be mappings of model inputs.")
            if batch.get("past_key_values") is not None:
                raise ValueError("Calibration batches must not contain past_key_values.")

            attention_mask = batch.get("attention_mask")
            if attention_mask is not None and (
                not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2
            ):
                raise ValueError("Calibration requires a two-dimensional attention mask.")
            model(**batch)

        scales = {}
        for name, chunks in norms.items():
            if not chunks or sum(chunk.numel() for chunk in chunks) == 0:
                raise ValueError(f"No eligible calibration tokens in block '{name}'.")

            scale = torch.quantile(torch.cat(chunks), 0.5)
            if not torch.isfinite(scale) or scale <= 0:
                raise ValueError(f"The calibration scale for block '{name}' must be finite and positive.")
            scales[name] = scale

        for name, scale in scales.items():
            target = layers[name].activation_steering_scales[adapter_name]
            target.copy_(scale.to(device=target.device, dtype=target.dtype))

    finally:
        for handle in handles:
            handle.remove()
        for module, disabled in adapter_states:
            module._disable_adapters = disabled
        for module, training in training_states:
            module.training = training
