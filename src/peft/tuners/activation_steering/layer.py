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

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

from peft.tuners._buffer_dict import BufferDict
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.quantization_utils import resolve_quantization_backend

from .config import ActivationSteeringConfig


def get_steering_directions(block, spec, quantization_kwargs):
    if isinstance(block, BaseTunerLayer):
        block = block.get_base_layer()

    try:
        source = block.get_submodule(spec["source_module"])
    except AttributeError as exc:
        raise ValueError(f"Source module '{spec['source_module']}' was not found in the targeted block.") from exc

    if isinstance(source, BaseTunerLayer):
        source = source.get_base_layer()

    backend = resolve_quantization_backend(source, **quantization_kwargs)
    if backend is not None:
        if backend.layer_type != "linear":
            raise ValueError("Activation steering requires a linear source module.")
        try:
            weight = backend.get_base_weight(source)
        except (TypeError, NotImplementedError) as exc:
            raise ValueError(f"Activation steering requires dequantization of '{spec['source_module']}'.") from exc
    elif isinstance(source, nn.Linear):
        weight = source.weight
    elif isinstance(source, Conv1D):
        weight = source.weight.T
    else:
        raise ValueError("Activation steering requires a Linear or Conv1D source module.")

    if not weight.is_floating_point():
        raise ValueError("Activation steering requires floating-point or dequantizable source weights.")
    if max(spec["indices"]) >= weight.shape[1]:
        raise ValueError(f"`indices` exceed the input dimension of '{spec['source_module']}' ({weight.shape[1]}).")
    if weight.device.type == "meta":
        raise ValueError("Activation steering requires materialized source weights.")

    directions = weight.detach()[:, spec["indices"]].float()
    norms = directions.norm(dim=0)
    if not torch.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("Selected source weight columns must be finite and have non-zero L2 norms.")

    return (directions / norms).to(weight.dtype)


def _get_residual(output):
    residual = (output[0] if output else None) if isinstance(output, tuple) else output
    if not isinstance(residual, torch.Tensor) or residual.ndim != 3:
        raise ValueError("Activation steering expects a block output of shape (batch, sequence, hidden).")
    return residual


class ActivationSteeringLayer(nn.Module, BaseTunerLayer):
    adapter_layer_names = ("activation_steering_values",)
    other_param_names = ("activation_steering_directions", "activation_steering_scales")

    def __init__(self, base_layer, adapter_name, spec, directions, config: ActivationSteeringConfig):
        super().__init__()
        self.base_layer = base_layer
        self.activation_steering_values = nn.ParameterDict()
        self.activation_steering_directions = BufferDict(persistent=False)
        self.activation_steering_scales = BufferDict(persistent=True)
        self._attention_mask: torch.Tensor | None = None
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, spec, directions, config=config)

    def update_layer(self, adapter_name, spec, directions, config: ActivationSteeringConfig):
        values = spec["init_val"]
        if not isinstance(values, list):
            values = [values] * len(spec["indices"])
        self.activation_steering_values[adapter_name] = nn.Parameter(
            torch.tensor(values, device=directions.device, dtype=directions.dtype)
        )
        self.activation_steering_directions[adapter_name] = directions
        self.activation_steering_scales[adapter_name] = torch.zeros((), device=directions.device, dtype=torch.float32)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=config.inference_mode)

    def merge(self, *args, **kwargs):
        raise ValueError("Activation steering does not support merging.")

    def unmerge(self):
        raise ValueError("Activation steering does not support unmerging.")

    def forward(self, *args, **kwargs):
        adapters = [name for name in self.active_adapters if name in self.activation_steering_values]
        if self.disable_adapters or not adapters:
            result = self.base_layer(*args, **kwargs)
        else:
            for name in adapters:
                scale = self.activation_steering_scales[name]
                if not torch.isfinite(scale) or scale <= 0:
                    raise RuntimeError(
                        f"Activation steering adapter '{name}' is not calibrated. "
                        "Call calibrate_activation_steering before using it."
                    )

            output = self.base_layer(*args, **kwargs)
            residual = _get_residual(output)
            mask = self._attention_mask
            if mask is None:
                mask = torch.ones(residual.shape[:2], device=residual.device, dtype=torch.bool)
            else:
                mask = mask[:, -residual.shape[1] :].to(device=residual.device, dtype=torch.bool)
            if mask.shape != residual.shape[:2]:
                raise ValueError("The attention mask must match the block output's batch and sequence dimensions.")

            delta = torch.zeros(residual.shape[-1], device=residual.device, dtype=torch.float32)
            for name in adapters:
                scale = self.activation_steering_scales[name]
                directions = self.activation_steering_directions[name].to(device=residual.device, dtype=torch.float32)
                if directions.shape[0] != residual.shape[-1]:
                    raise ValueError(
                        "The source projection's output dimension must match the block's hidden dimension."
                    )

                values = self.activation_steering_values[name].to(device=residual.device, dtype=torch.float32)
                delta = delta + (directions @ values) * scale.to(device=residual.device, dtype=torch.float32)
            result = residual + (delta * mask.unsqueeze(-1)).to(residual.dtype)
            result = (result, *output[1:]) if isinstance(output, tuple) else result
        return result
