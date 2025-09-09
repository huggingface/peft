# Copyright 2023-present the HuggingFace Inc. team.
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
from dataclasses import replace

import torch
from transformers.pytorch_utils import Conv1D

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import (
    TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_submodules,
)

from .layer import Conv2d, Conv3d, IA3Layer, Linear


class IA3Model(BaseTuner):
    """
    Creates a Infused Adapter by Inhibiting and Amplifying Inner Activations ((IA)^3) model from a pretrained
    transformers model. The method is described in detail in https://huggingface.co/papers/2205.05638

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`IA3Config`]): The configuration of the (IA)^3 model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    Returns:
        `torch.nn.Module`: The (IA)^3 model.

    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM, ia3Config
        >>> from peft import IA3Model, IA3Config

        >>> config = IA3Config(
        ...     peft_type="IA3",
        ...     task_type="SEQ_2_SEQ_LM",
        ...     target_modules=["k", "v", "w0"],
        ...     feedforward_modules=["w0"],
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> ia3_model = IA3Model(config, model)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`ia3Config`]): The configuration of the (IA)^3 model.
    """

    prefix: str = "ia3_"
    base_layer_cls = IA3Layer

    @staticmethod
    def _create_new_module(ia3_config, adapter_name, target, **kwargs):
        # avoid eager bnb import
        if is_bnb_available():
            import bitsandbytes as bnb

            from .bnb import Linear8bitLt

        if is_bnb_4bit_available():
            from .bnb import Linear4bit

        loaded_in_8bit = kwargs.pop("loaded_in_8bit", False)
        loaded_in_4bit = kwargs.pop("loaded_in_4bit", False)
        is_feedforward = kwargs.pop("is_feedforward", False)

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if loaded_in_8bit and isinstance(target_base_layer, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target_base_layer.state.has_fp16_weights,
                    "threshold": target_base_layer.state.threshold,
                    "index": target_base_layer.index,
                }
            )
            new_module = Linear8bitLt(target, adapter_name, is_feedforward=is_feedforward, **eightbit_kwargs)
        elif loaded_in_4bit and isinstance(target_base_layer, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target_base_layer.compute_dtype,
                    "compress_statistics": target_base_layer.weight.compress_statistics,
                    "quant_type": target_base_layer.weight.quant_type,
                }
            )
            new_module = Linear4bit(target, adapter_name, is_feedforward=is_feedforward, **fourbit_kwargs)
        elif isinstance(target, torch.nn.Conv2d):
            new_module = Conv2d(target, adapter_name, is_feedforward=is_feedforward, **kwargs)
        elif isinstance(target, torch.nn.Conv3d):
            new_module = Conv3d(target, adapter_name, is_feedforward=is_feedforward, **kwargs)
        elif isinstance(target_base_layer, torch.nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = ia3_config.fan_in_fan_out = False
            new_module = Linear(target, adapter_name, is_feedforward=is_feedforward, **kwargs)
        elif isinstance(target_base_layer, Conv1D):
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = ia3_config.fan_in_fan_out = True
            new_module = Linear(
                target, adapter_name, is_feedforward=is_feedforward, is_target_conv_1d_layer=True, **kwargs
            )
        else:
            raise ValueError(
                f"Target module {target} is not supported. "
                f"Currently, only `torch.nn.Linear`, `torch.nn.Conv2d`, and `Conv1D` are supported."
            )
        return new_module

    def _create_and_replace(
        self,
        ia3_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        # check if target module is in feedforward_modules
        is_feedforward = self._check_target_module_feedforward(ia3_config, current_key)

        kwargs = {
            "fan_in_fan_out": ia3_config.fan_in_fan_out,
            "init_ia3_weights": ia3_config.init_ia3_weights,
            "is_feedforward": is_feedforward,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }

        if isinstance(target, IA3Layer):
            target.update_layer(
                adapter_name,
                ia3_config.init_ia3_weights,
            )
        else:
            new_module = self._create_new_module(ia3_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _check_target_module_feedforward(ia3_config, key) -> bool:
        """
        A helper private method that checks if the target module `key` matches with a feedforward module specified in
        `ia3_config`
        """
        if isinstance(ia3_config.feedforward_modules, str):
            is_feedforward = bool(re.fullmatch(ia3_config.feedforward_modules, key))
        else:
            is_feedforward = any(key.endswith(target_key) for target_key in ia3_config.feedforward_modules)
        return is_feedforward

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        if peft_config.feedforward_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING:
                raise ValueError("Please specify `feedforward_modules` in `peft_config`")
            peft_config.feedforward_modules = set(
                TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def _check_add_weighted_adapter(self, adapters: list[str]) -> tuple[str, str]:
        """
        Helper function to check if the arguments to add_weighted_adapter are valid and compatible with the underlying
        model.
        """
        # Validate existence of adapters
        for adapter in adapters:
            if adapter not in self.peft_config:
                raise ValueError(f"Adapter {adapter} does not exist")

        # Check for conflicting modules_to_save
        modules_to_save_wrappers = [module for module in self.modules() if isinstance(module, ModulesToSaveWrapper)]
        if any(
            sum(adapter in wrapper.modules_to_save for adapter in adapters) > 1 for wrapper in modules_to_save_wrappers
        ):
            raise ValueError("Cannot add weighted adapters targeting the same module with modules_to_save.")

        # Ensure all adapters have compatible target and feedforward module types
        target_module_types = {type(self.peft_config[adapter].target_modules) for adapter in adapters}
        feedforward_module_types = {type(self.peft_config[adapter].feedforward_modules) for adapter in adapters}
        if len(target_module_types) > 1 or len(feedforward_module_types) > 1:
            raise ValueError("All adapter configs should have the same type for target and feedforward modules.")

        # Combine target and feedforward modules
        if str in target_module_types:
            new_target_modules = "|".join(f"({self.peft_config[adapter].target_modules})" for adapter in adapters)
        else:
            new_target_modules = set.union(*(self.peft_config[adapter].target_modules for adapter in adapters))

        if str in feedforward_module_types:
            new_feedforward_modules = "|".join(
                f"({self.peft_config[adapter].feedforward_modules})" for adapter in adapters
            )
        else:
            new_feedforward_modules = set.union(
                *(self.peft_config[adapter].feedforward_modules for adapter in adapters)
            )

        return new_target_modules, new_feedforward_modules

    def add_weighted_adapter(
        self,
        adapters: list[str],
        weights: list[float],
        adapter_name: str,
    ) -> None:
        """
        This method adds a new adapter by merging the given adapters with the given weights.

        Args:
            adapters (`list`):
                List of adapter names to be merged.
            weights (`list`):
                List of weights for each adapter.
            adapter_name (`str`):
                Name of the new adapter.
        """
        if adapter_name in list(self.peft_config.keys()):
            return

        new_target_modules, new_feedforward_modules = self._check_add_weighted_adapter(
            adapters=adapters,
        )

        self.peft_config[adapter_name] = replace(
            self.peft_config[adapters[0]],
            target_modules=new_target_modules,
            feedforward_modules=new_feedforward_modules,
        )
        self.inject_adapter(self.model, adapter_name)

        # Do we really need that?
        _freeze_adapter(self.model, adapter_name)

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, IA3Layer):
                if adapter_name in target.ia3_l:
                    target_ia3_l = target.ia3_l[adapter_name]
                else:
                    continue

                target_ia3_l.data = target_ia3_l.data.zero_()
                for adapter, weight in zip(adapters, weights):
                    if adapter in target.ia3_l:
                        current_adapter_ia3_l = target.ia3_l[adapter]
                    else:
                        continue
                    target_ia3_l.data += current_adapter_ia3_l.data * weight
