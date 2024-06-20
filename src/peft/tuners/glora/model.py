import re
import warnings
from dataclasses import asdict
from enum import Enum
from itertools import chain
from typing import Dict, Type, Union

import torch
import torch.nn as nn
from tqdm import tqdm

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_submodules,
)

from .config import GLoraConfig
from .layer import GLoraLayer, Linear


class GLoraModel(BaseTuner):
    """
    Creates Generalized Low Rank Adapter (GLora) model from a pretrained transformers model.
    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`GLoraConfig`]): The configuration of the Lora model.
    Returns:
        `torch.nn.Module`: The Lora model.
    Example:
        ```py
        >>> from transformers import AutoModelForSeq2SeqLM, GLoraConfig
        >>> from peft import GLoraModel, GLoraConfig
        >>> config = GLoraConfig(
        ...     peft_type="GLORA",
        ...     task_type="SEQ_2_SEQ_LM",
        ...     r=8,
        ...     target_modules=["q", "v"],
        ... )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> glora_model = GLoraModel(config, model)
        ```
        ```py
        >>> import transformers
        >>> from peft import GLoraConfig, PeftModel, get_peft_model, prepare_model_for_int8_training
        >>> target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
        >>> config = GLoraConfig(
        ...     r=4, target_modules=target_modules, task_type="CAUSAL_LM"
        ... )
        >>> model = transformers.GPTJForCausalLM.from_pretrained(
        ...     "kakaobrain/kogpt",
        ...     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
        ...     pad_token_id=tokenizer.eos_token_id,
        ...     use_cache=False,
        ...     device_map={"": rank},
        ...     torch_dtype=torch.float16,
        ...     load_in_8bit=True,
        ... )
        >>> model = prepare_model_for_int8_training(model)
        >>> glora_model = get_peft_model(model, config)
        ```
    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`GLoraConfig`]): The configuration of the Lora model.
    """

    prefix: str = "glora"
    layers_mapping: Dict[Type[torch.nn.Module], Type[GLoraLayer]] = {
        torch.nn.Linear: Linear,
    }

    def __init__(self, model, config, adapter_name) -> None:
        super().__init__(model, config, adapter_name)


    def add_adapter(self, adapter_name, config=None):
        print("add_adapter")
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_glora_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "LoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )
        mark_only_glora_as_trainable(self.model)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _check_target_module_exists(self, glora_config, key):
        if isinstance(glora_config.target_modules, str):
            target_module_found = re.fullmatch(glora_config.target_modules, key)
        else:
            target_module_found = any(key.endswith(target_key) for target_key in glora_config.target_modules)
        return target_module_found


    def _find_and_replace(self, adapter_name):
        glora_config = self.peft_config[adapter_name]
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if not self._check_target_module_exists(glora_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)
            new_module = self._create_new_module(glora_config, adapter_name, target)
            self._replace_module(parent, target_name, new_module, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {glora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if hasattr(old_module, "bias"):
            if old_module.bias is not None:
                new_module.bias = old_module.bias

        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "glora_" in name:
                module.to(old_module.weight.device)
            if "ranknum" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    @staticmethod
    def _prepare_glora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
        return peft_config

    @staticmethod
    def _replace_module(parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        new_module.weight = child.weight
        if hasattr(child, "bias"):
            if child.bias is not None:
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            new_module.state = child.state
            new_module.to(child.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "glora_" in name:
                module.to(child.weight.device)
            if "ranknum" in name:
                module.to(child.weight.device)

    def merge_and_unload(self, progressbar: bool = False):
        r"""
        This method merges the LoRa layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.
        Args:
            progressbar (bool): whether to show a progressbar indicating the unload and merge process
        Example:
        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModel
        >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
        >>> peft_model_id = "smangrul/falcon-40B-int4-peft-glora-sfttrainer-sample"
        >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
        >>> merged_model = model.merge_and_unload()
        ```
        """
        return self._unload_and_optionally_merge(progressbar=progressbar)

    def _unload_and_optionally_merge(self, merge=True, progressbar: bool = False):
        key_list = [key for key, _ in self.model.named_modules() if "glora" not in key]
        for key in tqdm(key_list, disable=not progressbar):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, GLoraLayer):
                bias = True
                new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                if merge:
                    target.merge()
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    def _create_and_replace(
        self,
        glora_config: GLoraConfig,
        adapter_name: str,
        target: Union[GLoraLayer, nn.Module],
        target_name: str,
        parent: nn.Module,
        current_key: str,
    ) -> None:
        """
        A private method to create and replace the target module with the adapter module.
        """

        kwargs = glora_config.to_dict()

        if isinstance(target, GLoraLayer):
            target.update_layer(adapter_name, **kwargs)
        else:
            new_module = self._create_new_module(glora_config, adapter_name, target, **kwargs)
            self._replace_module(parent, target_name, new_module, target)


    @classmethod
    def _create_new_module(cls, config: GLoraConfig, adapter_name: str, target: nn.Module, **kwargs) -> GLoraLayer:
        # Find corresponding subtype of provided target module
        new_module_cls = None
        for subtype, target_cls in cls.layers_mapping.items():
            if (
                hasattr(target, "base_layer")
                and isinstance(target.get_base_layer(), subtype)
                and isinstance(target, BaseTunerLayer)
            ):
                # nested tuner layers are allowed
                new_module_cls = target_cls
                break
            elif isinstance(target, subtype):
                new_module_cls = target_cls
                break

        # We didn't find corresponding type, so adapter for this layer is not supported
        if new_module_cls is None:
            supported_modules = ", ".join(layer.__name__ for layer in cls.layers_mapping.keys())
            raise ValueError(
                f"Target module of type {type(target)} not supported, "
                f"currently only adapters for {supported_modules} are supported"
            )

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target
        kwargs.pop('r', None)
        in_features = target_base_layer.in_features
        out_features = target_base_layer.out_features
        if isinstance(target_base_layer, torch.nn.Linear):
                        new_module = new_module_cls(target=target,
                                        adapter_name=adapter_name,
                                        in_features= in_features,
                                        out_features= out_features,
                                        r=config.r,
                                        **kwargs)
        else:
            supported_modules = ", ".join(layer.__name__ for layer in cls.layers_mapping.keys())
            raise ValueError(
                f"Target module of type {type(target)} not supported, "
                f"currently only adapters for {supported_modules} are supported"
            )

        return new_module


    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            raise ValueError("Please specify `target_modules` in `peft_config`")
        return peft_config

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


@staticmethod
def mark_only_glora_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if "glora_" not in n:
            p.requires_grad = False
