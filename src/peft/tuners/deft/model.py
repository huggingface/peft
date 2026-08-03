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


import warnings

import torch
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_DEFT_TARGET_MODULES_MAPPING

from .layer import DeftLayer, DeftLinear


class DeftModel(BaseTuner):
    """
    Creates a DEFT (Decompositional Efficient Fine-Tuning) model from a pretrained model.

    DEFT freezes the base weights and learns, per target module, a low-rank projection direction `P` and an injection
    matrix `R`. The effective weight becomes `(I - P_proj) @ W + Q_P @ R`, replacing a sub-space of `W` with newly
    injected content (see [`DeftConfig`] for the available `decomposition_method` variants).

    Args:
        model (`torch.nn.Module`): The model to which the adapter tuner layers will be attached.
        config ([`DeftConfig`]): The configuration of the DEFT model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    Returns:
        `torch.nn.Module`: The DEFT model.

    **Attributes**:
        - **model** ([`~torch.nn.Module`]) -- The model to be adapted.
        - **peft_config** ([`DeftConfig`]): The configuration of the DEFT model.
    """

    prefix: str = "deft_"
    tuner_layer_cls = DeftLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_DEFT_TARGET_MODULES_MAPPING

    def _create_and_replace(
        self,
        deft_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # If it is not a DeftLayer, create a new module, else update it with new adapters
        if not isinstance(target, DeftLayer):
            new_module = self._create_new_module(deft_config, adapter_name, target, r=deft_config.r)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
        else:
            target.update_layer(
                adapter_name,
                r=deft_config.r,
                config=deft_config,
            )

    @staticmethod
    def _create_new_module(deft_config, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            if deft_config.fan_in_fan_out:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                deft_config.fan_in_fan_out = False
            new_module = DeftLinear(target, adapter_name, config=deft_config, **kwargs)
        elif isinstance(target_base_layer, Conv1D):
            if not deft_config.fan_in_fan_out:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True."
                )
                deft_config.fan_in_fan_out = True
            new_module = DeftLinear(target, adapter_name, config=deft_config, **kwargs)
        else:
            raise TypeError(
                f"Target module {target} is not supported. Currently, only `torch.nn.Linear` and "
                "`transformers.pytorch_utils.Conv1D` are supported."
            )

        return new_module
