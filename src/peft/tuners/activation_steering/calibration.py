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

from peft.tuners.tuners_utils import BaseTunerLayer, _match_layer_index
from peft.utils.quantization_utils import get_quantization_kwargs

from .layer import ActivationSteeringLayer, get_projection_weight


def _get_hidden_states(output, batch_size):
    hidden_states = (output[0] if output else None) if isinstance(output, tuple) else output
    if isinstance(hidden_states, torch.Tensor) and hidden_states.ndim == 2 and batch_size is not None:
        # Some models, e.g. OPT, flatten the batch and sequence dimensions before the projection.
        hidden_states = hidden_states.reshape(batch_size, -1, hidden_states.shape[-1])
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
        raise ValueError("Activation steering calibration expects outputs of shape (batch, sequence, hidden).")
    return hidden_states


@torch.no_grad()
def calibrate_activation_steering(
    model: nn.Module,
    dataloader: Iterable[Mapping[str, Any]] | Mapping[str, Any],
    adapter_name: str = "default",
) -> None:
    """Calibrate the fixed scales of an activation steering adapter in place.

    Run baseline forwards with every adapter disabled and the model in evaluation mode. For each targeted projection,
    pool the L2 norms of the stream given by the adapter's `dose_unit` over calibration tokens, excluding padding and
    the first valid token of every sequence, and take their median: the output of the numbered layer containing the
    projection for `"layer_output"`, or the projection's own output for `"module_output"`. Each steered input channel
    `n` then stores the scale `median / ||W[:, n]||`, where `W` is the projection's weight, so that a dose of 1 adds a
    vector with L2 norm `median` to the projection's output.

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
    adapter selection are unchanged. Scales are updated only after every targeted projection has valid statistics.

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
    config = model.peft_config[adapter_name]

    if isinstance(dataloader, Mapping):
        dataloader = [dataloader]

    # The module whose output norms set the unit of each layer's doses; several layers can share one such stream.
    streams = {}
    for name in layers:
        if config.dose_unit == "module_output":
            streams[name] = name
        else:
            match = _match_layer_index(name, config.layers_pattern)
            if match is None:
                raise ValueError(
                    f"Could not find the numbered layer containing '{name}'. Pass `layers_pattern`, or use "
                    "`dose_unit='module_output'` to measure the unit at the projection's output."
                )
            # The match ends with the dot after the layer index.
            streams[name] = match.group(0)[:-1]

    norms = {stream: [] for stream in streams.values()}
    attention_mask = batch_size = None

    def capture(stream):
        def hook(module, args, output):
            hidden_states = _get_hidden_states(output, batch_size)
            if attention_mask is None:
                eligible = torch.ones(hidden_states.shape[:2], device=hidden_states.device, dtype=torch.bool)
            else:
                if attention_mask.shape != hidden_states.shape[:2]:
                    raise ValueError("The calibration attention mask must match the output dimensions.")
                eligible = attention_mask.to(device=hidden_states.device, dtype=torch.bool)

            eligible = eligible & (eligible.cumsum(dim=1) > 1)
            values = hidden_states.float().norm(dim=-1)[eligible]
            if not torch.isfinite(values).all():
                raise ValueError(f"Non-finite calibration norms in '{stream}'.")

            norms[stream].append(values.cpu())

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

        for stream in norms:
            handles.append(model.get_submodule(stream).register_forward_hook(capture(stream)))

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
            inputs = (
                attention_mask if attention_mask is not None else batch.get("input_ids", batch.get("inputs_embeds"))
            )
            batch_size = inputs.shape[0] if isinstance(inputs, torch.Tensor) else None
            model(**batch)

        medians = {}
        for stream, chunks in norms.items():
            if not chunks or sum(chunk.numel() for chunk in chunks) == 0:
                raise ValueError(f"No eligible calibration tokens in '{stream}'.")

            median = torch.quantile(torch.cat(chunks), 0.5)
            if not torch.isfinite(median) or median <= 0:
                raise ValueError(f"The calibration median for '{stream}' must be finite and positive.")
            medians[stream] = median

        quantization_kwargs = get_quantization_kwargs(model)
        scales = {}
        for name, layer in layers.items():
            weight = get_projection_weight(layer, quantization_kwargs)
            indices = layer.activation_steering_indices[adapter_name].to(weight.device)
            column_norms = weight.float().norm(dim=0)[indices]
            if not torch.isfinite(column_norms).all() or (column_norms == 0).any():
                raise ValueError(f"The steered weight columns of '{name}' must be finite and have non-zero L2 norms.")
            scales[name] = medians[streams[name]] / column_norms.cpu()

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
