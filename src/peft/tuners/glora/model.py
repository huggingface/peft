import random
import re
import warnings
from dataclasses import asdict
from enum import Enum
from typing import Any, Optional

import torch
from torch import nn
from tqdm import tqdm

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.glora.layer import GLoraLayer, mark_only_glora_as_trainable
from peft.tuners.lora.aqlm import dispatch_aqlm
from peft.tuners.lora.awq import dispatch_awq
from peft.tuners.lora.eetq import dispatch_eetq
from peft.tuners.lora.gptq import dispatch_gptq
from peft.tuners.lora.hqq import dispatch_hqq
from peft.tuners.lora.layer import Conv2d
from peft.tuners.lora.tp_layer import dispatch_megatron
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_submodules,
)

from .config import GLoraConfig
from .layer import Linear


random.seed(56)


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

    prefix: str = "glora_"

    def __init__(self, model, config, adapter_name) -> None:
        super().__init__(model, config, adapter_name)

    @staticmethod
    def _check_target_module_exists(poly_config, key):
        return check_target_module_exists(poly_config, key)

    def _check_new_adapter_config(self, config: GLoraConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.
        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.
        """
        # TODO: there should be a check if any of the existing adapters actually has bias != "none", or else the check
        # does not fully correspond to the error message.
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
        )

    def add_adapter(self, adapter_name, config=None):
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

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            if hasattr(new_module, "W_q"):  # HQQ
                new_module.W_q = child.W_q
            else:
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
            if (self.prefix in name) or ("ranknum" in name):
                weight = (
                    child.qweight
                    if hasattr(child, "qweight")
                    else child.W_q
                    if hasattr(child, "W_q")
                    else child.weight
                )
                module.to(weight.device)

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
                new_module = Linear(target.in_features, target.out_features, bias=bias)
                if merge:
                    target.merge()
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    # def _create_and_replace(
    #     self,
    #     glora_config,
    #     adapter_name,
    #     target,
    #     target_name,
    #     parent,
    #     current_key,
    # ):
    #     pattern_keys = list(chain(glora_config.rank_pattern.keys(), glora_config.alpha_pattern.keys()))
    #     target_name_key = next(filter(lambda key: re.match(rf".*\.{key}$", current_key), pattern_keys), current_key)
    #     r = glora_config.rank_pattern.get(target_name_key, glora_config.r)
    #     target_modules = glora_config.alpha_pattern.get(target_name_key, glora_config.target_modules)

    #     kwargs = {
    #         "r": r,
    #         "target_modules" : target_modules
    #     }
    #     new_module = self._create_new_module(glora_config, adapter_name, target, **kwargs)
    #     if adapter_name not in self.active_adapters:
    #         # adding an additional adapter: it is not automatically trainable
    #         new_module.requires_grad_(False)
    #         self._replace_module(parent, target_name, new_module, target)

    def _create_and_replace(
        self,
        glora_config: GLoraConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        **optional_kwargs: Any,
    ):
        if isinstance(target, GLoraLayer):
            target.update_layer(adapter_name, glora_config)
        else:
            new_module = self._create_new_module(
                glora_config,
                adapter_name,
                target,
            )
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue

            if bias == "all":
                for n, p in model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "glora_only":
                for m in model.modules():
                    if isinstance(m, GLoraLayer) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def disable_adapter_layers(self) -> None:
        """Disable all adapters.

        When disabling all adapters, the model output corresponds to the output of the base model.
        """
        for active_adapter in self.active_adapters:
            val = self.peft_config[active_adapter].bias
            if val != "none":
                msg = (
                    f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                    "output as the the base model would without adaption."
                )
                warnings.warn(msg)
        self._set_adapter_layers(enabled=False)

    def enable_adapter_layers(self) -> None:
        """Enable all adapters.

        Call this if you have previously disabled all adapters and want to re-enable them.
        """
        self._set_adapter_layers(enabled=True)

    # @staticmethod
    # def _create_new_module(glora_config, adapter_name, target, **kwargs):
    #     if isinstance(target, BaseTunerLayer):
    #         target_base_layer = target.get_base_layer()
    #     else:
    #         target_base_layer = target

    #     if isinstance(target_base_layer, Linear):
    #         return Linear(target, adapter_name, glora_config, **kwargs)
    #     else:
    #         raise ValueError(
    #             f"Target module {target} is not supported. Currently, only the following modules are supported: "
    #             "`torch.nn.Linear`."
    #         )

    @staticmethod
    def _create_new_module(glora_config, adapter_name, target, **kwargs):
        # Collect dispatcher functions to decide what backend to use for the replaced LoRA layer. The order matters,
        # because the first match is always used. Therefore, the default layers should be checked last.
        dispatchers = []
        kwargs["fan_in_fan_out"] = False
        # avoid eager bnb import
        if is_bnb_available():
            from peft.tuners.lora.bnb import dispatch_bnb_8bit

            dispatchers.append(dispatch_bnb_8bit)

        if is_bnb_4bit_available():
            from peft.tuners.lora.bnb import dispatch_bnb_4bit

            dispatchers.append(dispatch_bnb_4bit)

        dispatchers.extend(
            [
                dispatch_eetq,
                dispatch_aqlm,
                dispatch_awq,
                dispatch_gptq,
                dispatch_hqq,
                dispatch_default_glora,
            ]
        )

        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(target, adapter_name, glora_config=glora_config, **kwargs)
            if new_module is not None:  # first match wins
                break

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, Linear):
            return Linear(target, adapter_name, glora_config, **kwargs)
        if new_module is None:
            # no module could be matched
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`, `transformers.pytorch_utils.Conv1D`."
            )

        return new_module


@staticmethod
def dispatch_default_glora(
    target: torch.nn.Module,
    adapter_name: str,
    glora_config: GLoraConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None
    #kwargs["fan_in_fan_out"] = False
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Conv2d):
        kwargs.update(glora_config.loftq_config)
        new_module = Conv2d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = glora_config.fan_in_fan_out = False
        kwargs.update(glora_config.loftq_config)
        new_module = Linear(target, adapter_name, target.in_features, target.out_features, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Conv1d):
        if not kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to False but the target module is `Conv1D`. " "Setting fan_in_fan_out to True."
            )
            kwargs["fan_in_fan_out"] = glora_config.fan_in_fan_out = True
        kwargs.update(glora_config.loftq_config)
        new_module = Linear(target, adapter_name, target.in_features, target.out_features, **kwargs)

    return new_module
