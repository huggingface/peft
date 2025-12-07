# Copyright 2024-present the HuggingFace Inc. team.
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

import re
import warnings
from itertools import chain

import torch
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import (
    TRANSFORMERS_MODELS_TO_FOURIERFT_TARGET_MODULES_MAPPING,
)

from .layer import FourierFTLayer, FourierFTLinear


class FourierFTModel(BaseTuner):
    """
    Creates FourierFT model from a pretrained transformers model.

    The method is described in detail in https://huggingface.co/papers/2405.03003.

    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`FourierFTConfig`]): The configuration of the FourierFT model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    Returns:
        `torch.nn.Module`: The FourierFT model.

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`FourierFTConfig`]): The configuration of the Fourier model.
    """

    prefix: str = "fourierft_"
    tuner_layer_cls = FourierFTLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_FOURIERFT_TARGET_MODULES_MAPPING

    def _create_and_replace(
        self,
        fourierft_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")
        # Regexp matching - Find key which matches current target_name in patterns provided
        pattern_keys = list(chain(fourierft_config.n_frequency_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys), current_key)

        n_frequency = fourierft_config.n_frequency_pattern.get(target_name_key, fourierft_config.n_frequency)
        scaling = fourierft_config.scaling
        random_loc_seed = fourierft_config.random_loc_seed
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "n_frequency": n_frequency,
            "scaling": scaling,
            "fan_in_fan_out": fourierft_config.fan_in_fan_out,
            "init_weights": fourierft_config.init_weights,
            "random_loc_seed": fourierft_config.random_loc_seed,
        }
        kwargs["bias"] = bias
        if isinstance(target, FourierFTLayer):
            target.update_layer(
                adapter_name,
                n_frequency,
                scaling,
                fourierft_config.init_weights,
                random_loc_seed,
            )
        else:
            new_module = self._create_new_module(fourierft_config, adapter_name, target, **kwargs)
            if adapter_name != self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(fourierft_config, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = fourierft_config.fan_in_fan_out = False
        elif isinstance(target_base_layer, Conv1D):
            kwargs["is_target_conv_1d_layer"] = True
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = fourierft_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`."
            )

        new_module = FourierFTLinear(target, adapter_name, **kwargs)

        return new_module
