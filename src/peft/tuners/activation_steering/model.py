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

from torch import Tensor, nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, check_target_module_exists
from peft.utils.quantization_utils import get_quantization_kwargs, resolve_quantization_backend

from .layer import ActivationSteeringLayer


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

    inputs = args[0] if args else kwargs.get("input_ids")
    if inputs is None:
        inputs = kwargs.get("inputs_embeds")
    num_tokens = inputs.shape[1] if isinstance(inputs, Tensor) and inputs.ndim > 1 else None

    for layer in layers:
        layer._attention_mask = mask
        layer._num_tokens = num_tokens


def _matches(name, key):
    return key == name or key.endswith("." + name)


class ActivationSteeringModel(BaseTuner):
    """Manage additive steering of neurons, i.e. input channels of linear projections, with one dose per neuron."""

    prefix = "activation_steering_"
    tuner_layer_cls = ActivationSteeringLayer

    @staticmethod
    def _check_target_module_exists(peft_config, key):
        # With a steering pattern, only the targeted projections that have an entry are steered.
        found = check_target_module_exists(peft_config, key)
        if not found or not peft_config.steering_pattern:
            return found
        return any(_matches(name, key) for name in peft_config.steering_pattern)

    @staticmethod
    def _get_spec(config, key):
        matches = [name for name in config.steering_pattern if _matches(name, key)]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one steering specification for '{key}', got {matches}.")

        return config.steering_pattern[matches[0]]

    def _prepare_model(self, peft_config, model):
        # The attention mask of the model call belongs to the encoder, so it would not match the decoder's tokens.
        if getattr(getattr(model, "config", None), "is_encoder_decoder", False):
            raise ValueError("Activation steering does not support encoder-decoder models.")

        # Validate the steering pattern before replacing any modules: unmatched entries would be silently ignored.
        keys = [key for key, _ in model.named_modules() if self._check_target_module_exists(peft_config, key)]
        for name in peft_config.steering_pattern:
            found = [key for key in keys if _matches(name, key)]
            if len(found) != 1:
                raise ValueError(
                    f"Steering target '{name}' must match exactly one targeted module, found {len(found)}."
                )

    def _post_injection_hook(self, model, config, adapter_name):
        if not hasattr(model, "_activation_steering_hook"):
            model._activation_steering_hook = model.register_forward_pre_hook(
                _prepare_steering_forward, with_kwargs=True
            )

    def _create_and_replace(self, peft_config, adapter_name, target, target_name, parent, current_key):
        if peft_config.steering_pattern:
            spec = self._get_spec(peft_config, current_key)
            indices, init_val = spec["indices"], spec["init_val"]
        else:
            indices, init_val = None, 0.0

        if isinstance(target, ActivationSteeringLayer):
            target.update_layer(adapter_name, indices, init_val, config=peft_config)
            return

        quantization_kwargs = get_quantization_kwargs(self.model)
        if (
            not isinstance(target, (nn.Linear, Conv1D))
            and resolve_quantization_backend(target, **quantization_kwargs) is None
        ):
            raise ValueError(
                f"Target module {target} is not supported. Activation steering supports Linear and Conv1D projections."
            )
        new_module = ActivationSteeringLayer(target, adapter_name, indices, init_val, config=peft_config)
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
