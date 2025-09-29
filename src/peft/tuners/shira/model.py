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

import warnings

import torch

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import (
    TRANSFORMERS_MODELS_TO_SHIRA_TARGET_MODULES_MAPPING,
)

from .layer import Linear, ShiraLayer


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
        kwargs = {}
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
                init_weights=shira_config.init_weights,
            )
        else:
            new_module = self._create_new_module(shira_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(shira_config, adapter_name, target, **kwargs):
        fan_in_fan_out = shira_config.fan_in_fan_out

        _ = kwargs.pop("bias", False)

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            if fan_in_fan_out:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                fan_in_fan_out = shira_config.fan_in_fan_out = False
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`."
            )

        mask = (
            shira_config.mask_fn(target_base_layer, shira_config.r, **kwargs)
            if shira_config.mask_fn is not None
            else None
        )

        new_module = Linear(
            target,
            mask,
            adapter_name,
            shira_config.r,
            fan_in_fan_out,
            init_weights=shira_config.init_weights,
            **kwargs,
        )

        return new_module
