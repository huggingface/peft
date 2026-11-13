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
from peft.tuners.tuners_utils import BaseTunerLayer, _get_in_out_features
from peft.utils.quantization_utils import resolve_quantization_backend

from .config import ActivationSteeringConfig


def get_projection_weight(module, quantization_kwargs):
    """Return the weight of a linear projection with shape `(out_features, in_features)`, dequantized if needed."""
    if isinstance(module, BaseTunerLayer):
        module = module.get_base_layer()

    backend = resolve_quantization_backend(module, **quantization_kwargs)
    if backend is not None:
        if backend.layer_type != "linear":
            raise ValueError("Activation steering requires a linear projection.")
        try:
            weight = backend.get_base_weight(module)
        except (TypeError, NotImplementedError) as exc:
            raise ValueError("Activation steering requires dequantization of the targeted projection.") from exc
    elif isinstance(module, nn.Linear):
        weight = module.weight
    elif isinstance(module, Conv1D):
        weight = module.weight.T
    else:
        raise ValueError("Activation steering requires a Linear or Conv1D projection.")

    if not weight.is_floating_point():
        raise ValueError("Activation steering requires floating-point or dequantizable projection weights.")
    if weight.device.type == "meta":
        raise ValueError("Activation steering requires materialized projection weights.")
    return weight.detach()


class ActivationSteeringLayer(nn.Module, BaseTunerLayer):
    adapter_layer_names = ("activation_steering_values",)
    other_param_names = ("activation_steering_indices", "activation_steering_scales")

    def __init__(self, base_layer, adapter_name, indices, init_val, config: ActivationSteeringConfig):
        super().__init__()
        self.base_layer = base_layer
        self.activation_steering_values = nn.ParameterDict()
        # The indices are restored from the config on load, while the calibrated scales are saved with the adapter.
        self.activation_steering_indices = BufferDict(persistent=False)
        self.activation_steering_scales = BufferDict(persistent=True)
        self.in_features, _ = _get_in_out_features(self.get_base_layer())
        # The attention mask and the number of input tokens of the current model call, set by a forward pre-hook.
        self._attention_mask: torch.Tensor | None = None
        self._num_tokens: int | None = None
        # Adapters whose scales were found to be calibrated, so that the check runs once instead of on every forward.
        self._calibrated_adapters: set[str] = set()
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, indices, init_val, config=config)

    def update_layer(self, adapter_name, indices, init_val, config: ActivationSteeringConfig):
        if indices is None:
            # Without a steering pattern, every input channel gets a dose of zero, leaving the model unchanged.
            indices = list(range(self.in_features))
        elif max(indices) >= self.in_features:
            raise ValueError(f"`indices` exceed the input dimension of the targeted projection ({self.in_features}).")
        values = init_val if isinstance(init_val, list) else [init_val] * len(indices)

        self.activation_steering_values[adapter_name] = nn.Parameter(torch.tensor(values, dtype=torch.float32))
        self.activation_steering_indices[adapter_name] = torch.tensor(indices, dtype=torch.long)
        self.activation_steering_scales[adapter_name] = torch.zeros(len(indices), dtype=torch.float32)
        self._calibrated_adapters.discard(adapter_name)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=config.inference_mode)

    def merge(self, *args, **kwargs):
        raise ValueError("Activation steering does not support merging.")

    def unmerge(self):
        raise ValueError("Activation steering does not support unmerging.")

    def forward(self, x, *args, **kwargs):
        adapters = [name for name in self.active_adapters if name in self.activation_steering_values]
        if self.disable_adapters or not adapters:
            return self.base_layer(x, *args, **kwargs)

        for name in adapters:
            if name not in self._calibrated_adapters:
                scales = self.activation_steering_scales[name]
                if not (torch.isfinite(scales).all() and (scales > 0).all()):
                    raise RuntimeError(
                        f"Activation steering adapter '{name}' is not calibrated. "
                        "Call calibrate_activation_steering before using it."
                    )
                self._calibrated_adapters.add(name)

        if x.ndim not in (2, 3):
            raise ValueError(
                "Activation steering expects projection inputs of shape (batch, sequence, features) or (batch * "
                "sequence, features)."
            )

        delta = torch.zeros(x.shape[-1], device=x.device, dtype=torch.float32)
        for name in adapters:
            values = self.activation_steering_values[name].to(device=x.device, dtype=torch.float32)
            scales = self.activation_steering_scales[name].to(device=x.device, dtype=torch.float32)
            delta = delta.index_add(0, self.activation_steering_indices[name].to(x.device), values * scales)

        mask = self._attention_mask
        if mask is not None:
            # The input tokens are the last positions of the mask. Some models, e.g. OPT, flatten the batch and sequence
            # dimensions before the projection.
            num_tokens = self._num_tokens or (x.shape[1] if x.ndim == 3 else None)
            shape = (mask.shape[0], num_tokens) if x.ndim == 3 else (mask.shape[0] * (num_tokens or 0),)
            if num_tokens is None or num_tokens > mask.shape[1] or x.shape[:-1] != shape:
                raise ValueError("The attention mask must match the projection input's batch and sequence dimensions.")
            mask = mask[:, -num_tokens:].to(device=x.device, dtype=torch.bool).reshape(shape)
            delta = delta * mask.unsqueeze(-1)
        return self.base_layer(x + delta.to(x.dtype), *args, **kwargs)
