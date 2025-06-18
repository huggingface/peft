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

import warnings
from typing import Optional

from torch import nn
from torch.nn.modules import Module
from tqdm import tqdm

from peft.config import PeftConfig
from peft.tuners.tuners_utils import BaseTuner, _get_submodules, check_target_module_exists
from peft.utils import TRANSFORMERS_MODELS_TO_LNTUNING_TARGET_MODULES_MAPPING, ModulesToSaveWrapper

from .layer import LNTuningLayer


class LNTuningModel(BaseTuner):
    """
    Creates LayerNorm tuning from a pretrained transformer model.

    The method is described in detail in https://huggingface.co/papers/2312.11420.

    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`LNTuningConfig`]): The configuration of the Lora model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            This option has no effect on LN tuning but exists for consistency with other PEFT methods.

    Returns:
        'torch.nn.Module': The adapted model with LayerNorm tuned on.

    Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import get_peft_model, TaskType, LNTuningConfig

        >>> peft_config = LNTuningConfig(
        ...     task_type=TaskType.CAUSAL_LM,
        ... )

        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> model = get_peft_model(model, peft_config)
        >>> model.print_trainable_parameters()
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LNTuningConfig`]): The configuration of the Lora model.
    """

    prefix: str = "ln_tuning_"

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        # self.adapter_name = adapter_name
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "model":  # see #1892: prevent infinite recursion if class is not initialized
                raise
            return getattr(self.model, name)

    # TODO: here need to handle the modules_to_save rather than the target_modules
    @staticmethod
    def _prepare_adapter_config(peft_config: PeftConfig, model_config: dict) -> PeftConfig:
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LNTUNING_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_LNTUNING_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def _create_and_replace(
        self,
        peft_config: PeftConfig,
        adapter_name: str,
        target: Module,
        target_name: str,
        parent: Module,
        current_key: str,
    ) -> None:
        # replace the original module with a same new module
        new_module = self._create_new_module(peft_config, target, adapter_name)
        if adapter_name != self.active_adapter:
            new_module.requires_grad_(False)
        self._replace_module(parent, target_name, new_module, target)

    def _create_new_module(
        self,
        peft_config: PeftConfig,
        target: Module,
        adapter_name: str,
    ) -> Module:
        if not isinstance(target, LNTuningLayer):
            new_module = LNTuningLayer(target, adapter_name)
        else:
            new_module = target
            new_module.update_layer(target.base_layer, adapter_name)
        return new_module

    def _replace_module(self, parent: Module, child_name: str, new_module: Module, child: Module) -> None:
        setattr(parent, child_name, new_module)

        if hasattr(child, "base_layer"):
            child = child.base_layer

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        for name, module in new_module.named_modules():
            weight = child.qweight if hasattr(child, "qweight") else child.weight
            module.to(weight.device)

    def _mark_only_adapters_as_trainable(self, model: Module):
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

    def _check_target_module_exists(self, peft_config: PeftConfig, key: str) -> bool:
        return check_target_module_exists(peft_config, key)

    def _set_adapter_layers(self, enabled: bool) -> None:
        for module in self.model.modules():
            if isinstance(module, (LNTuningLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self) -> None:
        """Enable all adapters.

        Call this if you have previously disabled all adapters and want to re-enable them.
        """
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self) -> None:
        """Disable all adapters.

        When disabling all adapters, the model output corresponds to the output of the base model.
        """
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name: str) -> None:
        for module in self.model.modules():
            if isinstance(module, LNTuningLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        self._unloading_checks(adapter_names)
        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        desc = "Unloading adapters " + ("and merging " if merge else "") + "model"

        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue

            if hasattr(target, "base_layer"):
                if merge:
                    target.merge(adapter_names)
                self._replace_module(parent, target_name, target.get_base_layer(), target)

        return self.model

    def unload(self):
        return self._unload_and_optionally_merge(merge=False)

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ) -> nn.Module:
        return self._unload_and_optionally_merge(merge=True)

    def _cast_adapter_dtype(self, adapter_name: str, autocast_adapter_dtype: bool = True) -> None:
        # Note: LN Tuning does not add adapter layers, instead it creates copies of the original layer. For this reason,
        # we need to skip adapter autocasting, otherwise we would change the dtype of copies of the original layer,
        # resulting in dtype errors down the line.
        pass
