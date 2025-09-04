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

import warnings
from typing import Any, Union

from torch import nn

from peft.tuners import adalora, loha, lokr, lora, oft, shira
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    PeftType,
    _get_submodules,
    get_auto_gptq_quant_linear,
)


# Collection of constants used for all tuners
COMPATIBLE_TUNER_TYPES = (PeftType.LORA, PeftType.LOHA, PeftType.LOKR, PeftType.ADALORA, PeftType.OFT, PeftType.SHIRA)
PREFIXES = [
    lora.LoraModel.prefix,
    lokr.LoKrModel.prefix,
    loha.LoHaModel.prefix,
    oft.OFTModel.prefix,
    shira.ShiraModel.prefix,
]
Configs = Union[
    lora.LoraConfig, loha.LoHaConfig, lokr.LoKrConfig, adalora.AdaLoraConfig, oft.OFTConfig, shira.ShiraConfig
]
Layers = (
    lora.layer.LoraLayer,
    loha.layer.LoHaLayer,
    lokr.layer.LoKrLayer,
    adalora.layer.AdaLoraLayer,
    oft.OFTLayer,
    shira.ShiraLayer,
)


class MixedModel(BaseTuner):
    """
    A class that allows to mix different types of adapters in a single model.

    Note: This class should usually not be initialized directly. Instead, use `get_peft_model` with the argument
    `mixed=True`.

    Args:
        model (:obj:`nn.Module`):
            The model to be tuned.
        config (:obj:`PeftConfig`):
            The config of the model to be tuned. The adapter type must be compatible.
        adapter_name (:obj:`str`):
            The name of the first adapter.
    """

    def __init__(self, model: nn.Module, config: Configs, adapter_name: str) -> None:
        super().__init__(model, config, adapter_name)

    def _check_new_adapter_config(self, config: Configs) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        if not isinstance(config, Configs.__args__):
            raise ValueError(
                f"{self.__class__.__name__} only supports {COMPATIBLE_TUNER_TYPES} configs, but got {type(config)}."
            )

        biases = (getattr(config, "bias", None) for config in self.peft_config)
        biases = [bias for bias in biases if bias not in (None, "none")]
        if len(biases) > 1:
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )

    @staticmethod
    def _check_target_module_exists(config: Configs, key: str):
        return check_target_module_exists(config, key)

    def _create_and_replace(
        self,
        config: Configs,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if isinstance(config, adalora.AdaLoraConfig):
            adalora.AdaLoraModel._create_and_replace(self, config, *args, **kwargs)
        elif isinstance(config, lora.LoraConfig):
            lora.LoraModel._create_and_replace(self, config, *args, **kwargs)
        elif isinstance(config, loha.LoHaConfig):
            loha.LoHaModel._create_and_replace(self, config, *args, **kwargs)
        elif isinstance(config, lokr.LoKrConfig):
            lokr.LoKrModel._create_and_replace(self, config, *args, **kwargs)
        elif isinstance(config, oft.OFTConfig):
            oft.OFTModel._create_and_replace(self, config, *args, **kwargs)
        elif isinstance(config, shira.ShiraConfig):
            shira.ShiraModel._create_and_replace(self, config, *args, **kwargs)
        else:
            raise ValueError(f"Unsupported config type {type(config)}, should be one of {COMPATIBLE_TUNER_TYPES}.")

    def _replace_module(self, parent, child_name, new_module, child) -> None:
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.get_base_layer()
        elif hasattr(child, "quant_linear_module"):
            # TODO maybe not necessary to have special treatment?
            child = child.quant_linear_module

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if any(prefix in name for prefix in PREFIXES):
                module.to(child.weight.device)
            if "ranknum" in name:
                module.to(child.weight.device)

    @staticmethod
    def _create_new_module(config, adapter_name, target, **kwargs):
        gptq_quantization_config = kwargs.get("gptq_quantization_config", None)
        AutoGPTQQuantLinear = get_auto_gptq_quant_linear(gptq_quantization_config)
        if (gptq_quantization_config is not None) or (AutoGPTQQuantLinear is not None):
            raise ValueError(f"GPTQ quantization not supported for {config.peft_type.value} (yet).")

        loaded_in_8bit = kwargs.pop("loaded_in_8bit", False)
        loaded_in_4bit = kwargs.pop("loaded_in_4bit", False)
        if loaded_in_8bit or loaded_in_4bit:
            raise ValueError(f"8bit and 4bit quantization not supported for {config.peft_type.value} (yet).")

        if isinstance(config, adalora.AdaLoraConfig):
            new_module = adalora.AdaLoraModel._create_new_module(config, adapter_name, target, **kwargs)
        elif isinstance(config, lora.LoraConfig):
            new_module = lora.LoraModel._create_new_module(config, adapter_name, target, **kwargs)
        elif isinstance(config, loha.LoHaConfig):
            new_module = loha.LoHaModel._create_new_module(config, adapter_name, target, **kwargs)
        elif isinstance(config, lokr.LoKrConfig):
            new_module = lokr.LoKrModel._create_new_module(config, adapter_name, target, **kwargs)
        elif isinstance(config, oft.OFTConfig):
            new_module = oft.OFTModel._create_new_module(config, adapter_name, target, **kwargs)
        elif isinstance(config, shira.ShiraConfig):
            new_module = shira.ShiraModel._create_new_module(config, adapter_name, target, **kwargs)
        else:
            raise ValueError(f"Unknown config type {type(config)}, should be one of {COMPATIBLE_TUNER_TYPES}.")
        return new_module

    def set_adapter(self, adapter_name: Union[str, list[str]]) -> None:
        for module in self.model.modules():
            if isinstance(module, Layers):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")

            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def add_weighted_adapter(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(f"Weighted adapters are not supported for {self.__class__.__name__} (yet).")

    def delete_adapter(self, adapter_name: Union[str, list[str]]) -> None:
        """
        Deletes an existing adapter.

        Args:
            adapter_name (Union[str, list[str]]): Name of the adapter(s) to delete.
        """
        if isinstance(adapter_name, str):
            adapter_names = [adapter_name]
        else:
            adapter_names = adapter_name

        mismatched = set(adapter_names) - set(self.peft_config.keys())
        if mismatched:
            raise ValueError(
                f"Adapter(s) {sorted(mismatched)} not found, available adapters: {sorted(self.peft_config.keys())}"
            )

        for adapter_name in adapter_names:
            del self.peft_config[adapter_name]

            key_list = [key for key, _ in self.model.named_modules() if not any(prefix in key for prefix in PREFIXES)]
            new_adapter = None
            for key in key_list:
                _, target, _ = _get_submodules(self.model, key)
                if isinstance(target, BaseTunerLayer):
                    target.delete_adapter(adapter_name)
                    if new_adapter is None:
                        new_adapter = target.active_adapters[:]

        self.active_adapter = new_adapter or []
        self._delete_auxiliary_adapter(adapter_name, new_active_adapters=new_adapter)

    def generate(self, *args: Any, **kwargs: Any):
        return self.model.generate(*args, **kwargs)
