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

from typing import Optional

from torch.nn.modules import Module
from tqdm import tqdm

from peft.config import PeftConfig
from peft.tuners.tuners_utils import BaseTuner, _get_submodules
from peft.utils import TRANSFORMERS_MODELS_TO_LNTUNING_TARGET_MODULES_MAPPING

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
    tuner_layer_cls = LNTuningLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LNTUNING_TARGET_MODULES_MAPPING

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

    def _unloading_checks(self, adapter_names: Optional[list[str]]):
        adapters_to_consider = adapter_names or self.active_adapters
        is_modules_to_save_available = any(
            self.peft_config[adapter].modules_to_save for adapter in adapters_to_consider
        )
        if is_modules_to_save_available and len(adapters_to_consider) > 1:
            raise ValueError("Cannot unload multiple adapters that specify `modules_to_save`.")

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

    def _cast_adapter_dtype(self, adapter_name: str, autocast_adapter_dtype: bool = True) -> None:
        # Note: LN Tuning does not add adapter layers, instead it creates copies of the original layer. For this reason,
        # we need to skip adapter autocasting, otherwise we would change the dtype of copies of the original layer,
        # resulting in dtype errors down the line.
        pass
