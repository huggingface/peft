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

import math
import warnings
from dataclasses import asdict
from enum import Enum
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn.init import _calculate_correct_fan
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import (
    TRANSFORMERS_MODELS_TO_VERA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _get_submodules,
)

from .._buffer_dict import BufferDict
from ..tuners_utils import _maybe_include_all_linear_layers
from .config import VeraConfig
from .layer import Linear, VeraLayer


def _kaiming_init(
    tensor_or_shape: Union[torch.Tensor, tuple[int, ...]],
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Kaiming Uniform Initialisation adapted to accept a `torch.Generator` object for PRNG.

    Args:
        tensor_or_shape (`Union[torch.Tensor, tuple[int, ...]]`):
            Tensor to initialise, or shape of new tensor to create and then initialise.
        generator: (`torch.Generator`):
            Generator object that manages the state of the PRNG algorithm in use.

    Returns:
        `torch.Tensor`: The initialised tensor.
    """
    if isinstance(tensor_or_shape, tuple):
        tensor = torch.empty(tensor_or_shape)
    else:
        tensor = tensor_or_shape
    fan = _calculate_correct_fan(tensor, "fan_in")
    gain = math.sqrt(2)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std

    with torch.no_grad():
        return tensor.uniform_(-bound, bound, generator=generator)


class VeraModel(BaseTuner):
    """
    Creates Vector-based Random Matrix Adaptation (Vera) model from a pretrained transformers model.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`VeraConfig`]): The configuration of the Vera model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    Returns:
        `torch.nn.Module`: The Vera model.

    Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import VeraConfig, get_peft_model

        >>> base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        >>> config = VeraConfig(r=128)
        >>> model = get_peft_model(base_model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`VeraConfig`]): The configuration of the Vera model.
    """

    prefix: str = "vera_lambda_"

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

    def _find_dim(self, config) -> tuple[int, int]:
        """
        Finds the largest input and output dimensions across linear layers that have been wrapped with VeRA.

        This will be used for determining the size of the shared vera_A and vera_B matrices.
        """
        model_config = self.get_model_config(self.model)

        peft_config = self._prepare_adapter_config(config, model_config)
        peft_config = _maybe_include_all_linear_layers(peft_config, self.model)

        largest_shape = None
        for key, module in self.model.named_modules():
            if not self._check_target_module_exists(peft_config, key):
                continue

            if isinstance(module, nn.Linear):
                module_shape = module.out_features, module.in_features
            elif isinstance(module, Conv1D):
                module_shape = module.weight.ds_shape if hasattr(module.weight, "ds_shape") else module.weight.shape
                module_shape = module_shape[::-1]
            else:
                continue

            if largest_shape is None:
                largest_shape = module_shape
                continue

            if module_shape != largest_shape:
                largest_shape = tuple(max(a, b) for a, b in zip(largest_shape, module_shape))

        if largest_shape is None:
            msg = "No layers types compatible with VeRA were found. Please check `peft_config.target_modules`."
            raise ValueError(msg)

        return largest_shape

    def _init_vera_A_vera_B(self, config: VeraConfig, adapter_name: str) -> None:
        linear_out_dim, linear_in_dim = self._find_dim(config)

        # use of persistent to exclude vera_A and vera_B from the state dict if we choose not to save them.
        self.vera_A = BufferDict({}, persistent=config.save_projection)
        self.vera_B = BufferDict({}, persistent=config.save_projection)

        # deterministic init of vera_A and vera_B if we know the key
        generator = torch.Generator(device="cpu").manual_seed(config.projection_prng_key)
        vera_A = _kaiming_init((config.r, linear_in_dim), generator=generator)
        vera_B = _kaiming_init((linear_out_dim, config.r), generator=generator)

        self.vera_A[adapter_name] = vera_A
        self.vera_B[adapter_name] = vera_B

    def _pre_injection_hook(self, model: nn.Module, config: VeraConfig, adapter_name: str) -> None:
        self._init_vera_A_vera_B(config, adapter_name)

    def _check_new_adapter_config(self, config: VeraConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        # the below todo is copied from LoRA
        # TODO: there should be a check if any of the existing adapters actually has bias != "none", or else the check
        # does not fully correspond to the error message.
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )

        for existing_config in self.peft_config.values():
            if existing_config is config:
                # skip the current config
                continue

            if existing_config.projection_prng_key != config.projection_prng_key:
                raise ValueError(
                    f"Vera PRNG initialisation key must be the same for all adapters. Got {config.projection_prng_key=} but "
                    f"previous config had {existing_config.projection_prng_key}."
                )

        save_project_unique_values = sorted({config.save_projection for config in self.peft_config.values()})
        if len(save_project_unique_values) > 1:
            raise ValueError(
                "VeRA projection weights must be saved for all adapters or none, but got multiple different values: "
                f"{save_project_unique_values}"
            )

    @staticmethod
    def _check_target_module_exists(vera_config, key):
        return check_target_module_exists(vera_config, key)

    def _create_and_replace(
        self,
        vera_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        r = vera_config.r
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": r,
            "vera_dropout": vera_config.vera_dropout,
            "fan_in_fan_out": vera_config.fan_in_fan_out,
            "init_weights": vera_config.init_weights,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }
        kwargs["bias"] = bias

        if isinstance(target, Linear):
            target.update_layer(
                adapter_name,
                self.vera_A,
                self.vera_B,
                r,
                vera_config.vera_dropout,
                vera_config.init_weights,
                d_initial=vera_config.d_initial,
            )
        else:
            new_module = self._create_new_module(vera_config, self.vera_A, self.vera_B, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _replace_module(parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

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

        meta = torch.device("meta")
        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "vera_" in name:
                if not any(p.device == meta for p in module.parameters()):
                    module.to(child.weight.device)

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
            elif bias == "vera_only":
                for m in model.modules():
                    if isinstance(m, VeraLayer) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    @staticmethod
    def _create_new_module(vera_config, vera_A, vera_B, adapter_name, target, **kwargs):
        # avoid eager bnb import
        if is_bnb_available():
            import bitsandbytes as bnb

            from .bnb import Linear8bitLt

        if is_bnb_4bit_available():
            from .bnb import Linear4bit

        bias = kwargs.pop("bias", False)
        loaded_in_8bit = kwargs.get("loaded_in_8bit", False)
        loaded_in_4bit = kwargs.get("loaded_in_4bit", False)

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
            return Linear8bitLt(target, adapter_name, vera_A, vera_B, **eightbit_kwargs)
        elif loaded_in_4bit and isinstance(target_base_layer, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target_base_layer.compute_dtype,
                    "compress_statistics": target_base_layer.weight.compress_statistics,
                    "quant_type": target_base_layer.weight.quant_type,
                }
            )
            return Linear4bit(target, adapter_name, vera_A, vera_B, **fourbit_kwargs)
        elif isinstance(target_base_layer, torch.nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = vera_config.fan_in_fan_out = False
        elif isinstance(target_base_layer, Conv1D):
            kwargs["is_target_conv_1d_layer"] = True
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = vera_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `transformers.pytorch_utils.Conv1D`."
            )
        new_module = Linear(
            target,
            vera_A,
            vera_B,
            adapter_name,
            bias=bias,
            d_initial=vera_config.d_initial,
            **kwargs,
        )

        return new_module

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "model":  # see #1892: prevent infinite recursion if class is not initialized
                raise
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        for active_adapter in self.active_adapters:
            val = self.peft_config[active_adapter].bias
            if val != "none":
                msg = (
                    f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                    "output as the base model would without adaption."
                )
                warnings.warn(msg)
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, VeraLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_VERA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_VERA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        # we cannot use self.prefix as we want to include non-trainable vera parameters
        key_list = [key for key, _ in self.model.named_modules() if "vera" not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue

            if hasattr(target, "base_layer"):
                if merge:
                    target.merge(safe_merge=safe_merge, adapter_names=adapter_names)

                self._replace_module(parent, target_name, target.get_base_layer(), target)
            elif isinstance(target, ModulesToSaveWrapper):
                # save any additional trainable modules part of `modules_to_save`
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    def delete_adapter(self, adapter_name: str):
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]

        # we cannot use self.prefix as we want to include non-trainable vera parameters
        key_list = [key for key, _ in self.model.named_modules() if "vera" not in key]
        new_adapter = None
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, VeraLayer):
                target.delete_adapter(adapter_name)
                if new_adapter is None:
                    new_adapter = target.active_adapter[:]

        self.active_adapter = new_adapter or []
        self._delete_auxiliary_adapter(adapter_name, new_active_adapters=new_adapter)

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ):
        r"""
        This method merges the Vera layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.

        Args:
            progressbar (`bool`):
                whether to show a progressbar indicating the unload and merge process
            safe_merge (`bool`):
                whether to activate the safe merging check to check if there is any potential Nan in the adapter
                weights
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.

        Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModel

        >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
        >>> peft_model_id = "smangrul/falcon-40B-int4-peft-lora-sfttrainer-sample"
        >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
        >>> merged_model = model.merge_and_unload()
        ```
        """
        return self._unload_and_optionally_merge(
            progressbar=progressbar, safe_merge=safe_merge, adapter_names=adapter_names
        )

    def unload(self):
        """
        Gets back the base model by removing all the Vera modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)
