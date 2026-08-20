# Copyright 2025-present the HuggingFace Inc. team.
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

import platform
import warnings

import torch

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import (
    TRANSFORMERS_MODELS_TO_SHIRA_TARGET_MODULES_MAPPING,
    get_quantization_kwargs,
    resolve_quantization_backend,
)

from .layer import Linear, ShiraLayer


def _get_tuner_layer_class(target_base_layer: torch.nn.Module) -> type[ShiraLayer] | None:
    layer_cls: type[ShiraLayer] | None = None
    if isinstance(target_base_layer, torch.nn.Linear):
        layer_cls = Linear
    elif (quant_backend := resolve_quantization_backend(target_base_layer)) is not None:
        layer_cls = {"linear": Linear}.get(quant_backend.layer_type)

    return layer_cls


class ShiraModel(BaseTuner):
    """
    Creates a Sparse High Rank Adapter (SHiRA) Model from a pretrained model.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`ShiraConfig`]): The configuration of the SHiRA model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The SHiRA model.

    Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import ShiraConfig, get_peft_model

        >>> base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        >>> config = ShiraConfig(r=32)
        >>> model = get_peft_model(base_model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`ShiraConfig`]): The configuration of the SHiRA model.
    """

    prefix: str = "shira_"
    tuner_layer_cls = ShiraLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_SHIRA_TARGET_MODULES_MAPPING

    def _create_and_replace(
        self,
        shira_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = get_quantization_kwargs(self)
        kwargs["bias"] = bias
        if shira_config.mask_type == "random":
            kwargs["random_seed"] = shira_config.random_seed

        for k, v in optional_kwargs.items():
            kwargs[k] = v

        if isinstance(target, Linear):
            mask = (
                shira_config.mask_fn(target.base_layer, shira_config.r, **kwargs)
                if shira_config.mask_fn is not None
                else None
            )
            target.update_layer(
                adapter_name,
                mask,
                shira_config.r,
                config=shira_config,
                **kwargs,
            )
        else:
            new_module = self._create_new_module(shira_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(shira_config, adapter_name, target, **kwargs):
        _ = kwargs.pop("bias", False)

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        layer_cls = _get_tuner_layer_class(target_base_layer)

        if layer_cls is Linear:
            if shira_config.fan_in_fan_out:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                shira_config.fan_in_fan_out = False
        elif layer_cls is None:
            raise TypeError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`."
            )

        mask = (
            shira_config.mask_fn(target_base_layer, shira_config.r, **kwargs)
            if shira_config.mask_fn is not None
            else None
        )

        new_module = layer_cls(
            target,
            mask,
            adapter_name,
            config=shira_config,
            r=shira_config.r,
            **kwargs,
        )

        return new_module

    @classmethod
    def _get_adapter_state_dict(cls, model, config, adapter_name, state_dict, unwanted_adapter_names):
        from peft.utils.save_and_load import _filter_state_dict_for_adapter_name

        to_return = super()._get_adapter_state_dict(model, config, adapter_name, state_dict, unwanted_adapter_names)
        if platform.system() == "Windows":
            warnings.warn(
                "Windows has issues saving integers into safetensors. Hence, we convert shira_indices to float32 "
                "before saving on Windows OS. The shira_indices will always be converted to integers when loading."
            )
        for name, module in model.named_modules():
            if hasattr(module, "shira_indices"):
                for k, v in module.shira_indices.items():
                    # Windows has some issues with saving integers into safetensors. Tests fail with some kind of
                    # PermissionError. This results in failed tests, so we are converting indices to float32 before
                    # saving and then converting them back to int when loading. This is happening only for Windows,
                    # not for Linux and Mac-OS.
                    to_return[f"{name}.shira_indices.{k}"] = (
                        v.to(torch.float32) if platform.system() == "Windows" else v
                    )
                    # the above may contain other adapter names, so filter again
                    to_return = _filter_state_dict_for_adapter_name(to_return, unwanted_adapter_names)
        return to_return

    @classmethod
    def _remap_adapter_state_dict_for_load(cls, model, config, adapter_name, state_dict):
        peft_model_state_dict = super()._remap_adapter_state_dict_for_load(model, config, adapter_name, state_dict)
        if platform.system() == "Windows":
            warnings.warn(
                "Windows has issues saving integers into safetensors. Hence, we had converted shira_indices "
                "to float32 before saving on Windows OS. The shira_indices will always be converted to integers "
                "when loading."
            )
        for name, module in model.named_modules():
            if hasattr(module, "shira_indices"):
                # The shira_indices are stored in a plain dict, not as registered buffers, so their checkpoint keys
                # have no counterpart in the model state_dict and are thus not remapped, i.e. they carry no adapter
                # name.
                if f"{name}.shira_indices" in peft_model_state_dict:
                    shira_indices_values = peft_model_state_dict.pop(f"{name}.shira_indices")
                    # Convert shira_indices to int in case they were saved on a Windows OS and are being loaded
                    # on a Linux or a Mac-OS system. If they were saved in Linux or Mac-OS, they are already
                    # integers and the following will not affect anything.
                    module.shira_indices[adapter_name] = shira_indices_values.to(torch.int)
        return peft_model_state_dict
