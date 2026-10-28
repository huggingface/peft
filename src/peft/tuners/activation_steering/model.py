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

from torch import Tensor

from peft.tuners.tuners_utils import BaseTuner
from peft.utils.quantization_utils import get_quantization_kwargs

from .layer import ActivationSteeringLayer, get_steering_directions


def _prepare_steering_forward(model, args, kwargs):
    layers = [
        layer
        for layer in model.modules()
        if isinstance(layer, ActivationSteeringLayer)
        and not layer.disable_adapters
        and any(name in layer.activation_steering_values for name in layer.active_adapters)
    ]
    if not layers:
        return

    if len(args) > 1:
        raise ValueError(
            "When activation steering is enabled, pass attention_mask, past_key_values, and other optional model "
            "inputs by keyword; only the primary input may be positional."
        )

    mask = kwargs.get("attention_mask")
    if mask is not None and (not isinstance(mask, Tensor) or mask.ndim != 2):
        raise ValueError("Activation steering requires a two-dimensional attention mask.")

    for layer in layers:
        layer._attention_mask = mask


class ActivationSteeringModel(BaseTuner):
    """Manage additive block-output steering with independent dose parameters per selected source channel."""

    prefix = "activation_steering_"
    tuner_layer_cls = ActivationSteeringLayer

    # @staticmethod
    # def _prepare_adapter_config(peft_config, model_config):
    #     return peft_config

    @staticmethod
    def _get_spec(config, key):
        matches = [name for name in config.steering_pattern if key == name or key.endswith("." + name)]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one steering specification for '{key}', got {matches}.")

        return config.steering_pattern[matches[0]]

    def _prepare_model(self, peft_config, model):
        # Validate all targets before replacing any blocks.
        matches = {name: [] for name in peft_config.steering_pattern}
        for key, module in model.named_modules():
            for name in matches:
                if key == name or key.endswith("." + name):
                    matches[name].append((key, module))

        quantization_kwargs = get_quantization_kwargs(model)
        keys = []
        for name, targets in matches.items():
            if len(targets) != 1:
                raise ValueError(f"Steering target '{name}' must match exactly one block, found {len(targets)}.")

            key, target = targets[0]
            keys.append(key)
            get_steering_directions(target, self._get_spec(peft_config, key), quantization_kwargs)

        if any(first != second and second.startswith(first + ".") for first in keys for second in keys):
            raise ValueError("Steering targets must not contain other targeted blocks.")

    def _post_injection_hook(self, model, config, adapter_name):
        if not hasattr(model, "_activation_steering_hook"):
            model._activation_steering_hook = model.register_forward_pre_hook(
                _prepare_steering_forward, with_kwargs=True
            )

    def _replace_module(self, parent, child_name, new_module, child):
        # Blocks have no weight attribute; adapter tensors already follow the source projection's device and dtype.
        setattr(parent, child_name, new_module)

    def _create_and_replace(self, peft_config, adapter_name, target, target_name, parent, current_key):
        spec = self._get_spec(peft_config, current_key)
        directions = get_steering_directions(target, spec, get_quantization_kwargs(self.model))
        if isinstance(target, ActivationSteeringLayer):
            target.update_layer(adapter_name, spec, directions, config=peft_config)
        else:
            new_module = ActivationSteeringLayer(target, adapter_name, spec, directions, config=peft_config)
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def _check_merge_allowed(self):
        raise ValueError("Activation steering does not support merging.")

    def _unload_and_optionally_merge(self, merge=True, **kwargs):
        if merge:
            self._check_merge_allowed()
        model = super()._unload_and_optionally_merge(merge=False, **kwargs)
        model._activation_steering_hook.remove()
        del model._activation_steering_hook
        return model
