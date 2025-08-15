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
from typing import Any, Optional, Union

from torch import nn
from tqdm import tqdm

from peft.tuners import adalora, loha, lokr, lora, oft
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    PeftType,
    _get_submodules,
    get_auto_gptq_quant_linear,
)


# Collection of constants used for all tuners
COMPATIBLE_TUNER_TYPES = (PeftType.LORA, PeftType.LOHA, PeftType.LOKR, PeftType.ADALORA, PeftType.OFT)
PREFIXES = [lora.LoraModel.prefix, lokr.LoKrModel.prefix, loha.LoHaModel.prefix, oft.OFTModel.prefix]
Configs = Union[lora.LoraConfig, loha.LoHaConfig, lokr.LoKrConfig, adalora.AdaLoraConfig, oft.OFTConfig]
Layers = (lora.layer.LoraLayer, loha.layer.LoHaLayer, lokr.layer.LoKrLayer, adalora.layer.AdaLoraLayer, oft.OFTLayer)


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

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if not any(prefix in n for prefix in PREFIXES):
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = getattr(self.peft_config[active_adapter], "bias", "none")
            if bias == "none":
                continue

            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "lora_only":
                # TODO: check if this is needed for other supported types
                for m in model.modules():
                    if isinstance(m, Layers) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise ValueError(f"Requested bias: {bias}, is not implemented.")

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
        else:
            raise ValueError(f"Unknown config type {type(config)}, should be one of {COMPATIBLE_TUNER_TYPES}.")
        return new_module

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "model":  # see #1892: prevent infinite recursion if class is not initialized
                raise
            return getattr(self.model, name)

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        for active_adapter in self.active_adapters:
            val = getattr(self.peft_config[active_adapter], "bias", "none")
            if val != "none":
                msg = (
                    f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                    "output as the base model would without adaption."
                )
                warnings.warn(msg)
        self._set_adapter_layers(enabled=False)

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

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        if merge:
            if getattr(self.model, "quantization_method", None) == "gptq":
                raise ValueError("Cannot merge layers when the model is gptq quantized")

        def merge_recursively(module):
            # helper function to recursively merge the base_layer of the target
            path = []
            layer = module
            while hasattr(layer, "base_layer"):
                path.append(layer)
                layer = layer.base_layer
            for layer_before, layer_after in zip(path[:-1], path[1:]):
                layer_after.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                layer_before.base_layer = layer_after.base_layer
            module.merge(safe_merge=safe_merge, adapter_names=adapter_names)

        key_list = [key for key, _ in self.model.named_modules() if not any(prefix in key for prefix in PREFIXES)]
        desc = "Unloading " + ("and merging " if merge else "") + "model"

        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue

            if hasattr(target, "base_layer"):
                if merge:
                    merge_recursively(target)
                self._replace_module(parent, target_name, target.get_base_layer(), target)
            elif isinstance(target, ModulesToSaveWrapper):
                # save any additional trainable modules part of `modules_to_save`
                new_module = target.modules_to_save[target.active_adapter]
                if hasattr(new_module, "base_layer"):
                    # check if the module is itself a tuner layer
                    if merge:
                        new_module.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                    new_module = new_module.get_base_layer()
                setattr(parent, target_name, new_module)

        return self.model

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

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ) -> nn.Module:
        r"""
        This method merges the layers into the base model. This is needed if someone wants to use the base model as a
        standalone model.

        Args:
            progressbar (`bool`):
                whether to show a progressbar indicating the unload and merge process
            safe_merge (`bool`):
                whether to activate the safe merging check to check if there is any potential Nan in the adapter
                weights
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )

    def unload(self) -> nn.Module:
        """
        Gets back the base model by removing all the lora modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)

    def generate(self, *args: Any, **kwargs: Any):
        return self.model.generate(*args, **kwargs)
