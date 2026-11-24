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

from __future__ import annotations

import warnings

import torch
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import (
    TRANSFORMERS_MODELS_TO_DUMMY_TARGET_MODULES_MAPPING,
    get_quantization_kwargs,
    resolve_quantization_backend,
)

from .layer import Conv1d, Conv2d, Conv3d, DummyLayer, Embedding, Linear


def _get_tuner_layer_class(target_base_layer: torch.nn.Module) -> type[DummyLayer] | None:
    layer_cls: type[DummyLayer] | None = None
    if isinstance(target_base_layer, (torch.nn.Linear, Conv1D)):
        layer_cls = Linear
    elif isinstance(target_base_layer, torch.nn.Conv1d):
        layer_cls = Conv1d
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        layer_cls = Conv2d
    elif isinstance(target_base_layer, torch.nn.Conv3d):
        layer_cls = Conv3d
    elif isinstance(target_base_layer, torch.nn.Embedding):
        layer_cls = Embedding
    elif (quant_backend := resolve_quantization_backend(target_base_layer)) is not None:
        layer_cls = {"linear": Linear}.get(quant_backend.layer_type)

    return layer_cls


class DummyModel(BaseTuner):
    """
    Creates a Dummy adapter model from a pretrained model.

    The Dummy method is not a research method. It is a minimal, deliberately trivial tuner that serves as a template
    for contributors who want to add a new PEFT method and as a test vehicle for PEFT features that any tuner should
    support (saving, loading, merging, quantization, mixed adapters, ...).

    Its implementation is essentially a plain LoRA without any of the optimizations or extras that LoRA offers, see
    [`DummyConfig`] for details.

    Args:
        model (`torch.nn.Module`):
            The model to which the adapter tuner layers will be attached.
        config ([`DummyConfig`]): The configuration of the Dummy model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    Returns:
        `torch.nn.Module`: The Dummy model.

    Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import DummyConfig, get_peft_model

        >>> base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        >>> config = DummyConfig(target_modules=["q_proj", "v_proj"])
        >>> model = get_peft_model(base_model, config)
        ```

    **Attributes**:
        - **model** ([`~torch.nn.Module`]) -- The model to be adapted.
        - **peft_config** ([`DummyConfig`]): The configuration of the Dummy model.
    """

    prefix: str = "dummy_"
    tuner_layer_cls = DummyLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_DUMMY_TARGET_MODULES_MAPPING

    def _create_and_replace(
        self,
        dummy_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "bias": bias,
        }
        kwargs.update(get_quantization_kwargs(self))

        # If it is not a DummyLayer, create a new module, else update it with new adapters
        if isinstance(target, DummyLayer):
            target.update_layer(adapter_name, config=dummy_config, **kwargs)
        else:
            new_module = self._create_new_module(dummy_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(dummy_config, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        layer_cls = _get_tuner_layer_class(target_base_layer)
        if layer_cls is None:
            raise TypeError(
                f"Target module {target} is not supported. Currently, `torch.nn.Linear`, "
                "`transformers.pytorch_utils.Conv1D`, `torch.nn.Conv1d/2d/3d` (optionally quantized), and"
                " `torch.nn.Embedding` are supported."
            )

        if isinstance(target_base_layer, Conv1D):
            if not dummy_config.fan_in_fan_out:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True."
                )
                dummy_config.fan_in_fan_out = True
        elif isinstance(
            target_base_layer, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.Embedding)
        ):
            if dummy_config.fan_in_fan_out:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is a torch.nn layer. "
                    "Setting fan_in_fan_out to False."
                )
                dummy_config.fan_in_fan_out = False

        new_module = layer_cls(target, adapter_name, config=dummy_config, **kwargs)
        return new_module
