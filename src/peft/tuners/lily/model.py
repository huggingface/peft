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

from __future__ import annotations

import math
import operator
import re
import warnings
from contextlib import contextmanager
from dataclasses import asdict, replace
from enum import Enum
from functools import partial, reduce
from itertools import chain
from typing import Literal, Optional, Dict

import torch
from torch import nn
from tqdm import tqdm

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    check_target_module_exists,
    onload_layer,
    replicate_layers,
)
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_submodules,
    get_peft_model_state_dict,
    get_quantization_config,
)
from peft.utils.constants import TRANSFORMERS_MODELS_TO_LILY_TARGET_MODULES_MAPPING
from .config import LilyConfig
from .layer import LilyLayer, Linear

def _adapter_names_pre_forward_hook(target, args, kwargs, adapter_names):
    # pre-forward hook to inject the adapter_names argument when using mixed adapter batches inference
    kwargs["adapter_names"] = adapter_names
    return args, kwargs

class LilyModel(BaseTuner):
    """
    Creates Low-Rank Interconnectd Adaptation Across Layers (Lily) model from a pretrained transformers model.

    The method is described in detail in https://arxiv.org/abs/2407.09946.

    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`LilyConfig`]): The configuration of the Lily model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The Lily PEFT model.

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LilyConfig`]): The configuration of the Lily model.
    """

    prefix: str = "lily_"

    def __init__(self, model, config, adapter_name) -> None:
        super().__init__(model, config, adapter_name)

    def _check_new_adapter_config(self, config: LilyConfig) -> None:
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

    @staticmethod
    def _check_target_module_exists(lily_config, key):
        return check_target_module_exists(lily_config, key)

    def _create_and_replace(
        self,
        lily_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        lily_A,
        lily_B,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        r = lily_config.r # if there's any special rank setting
        lily_scaling = lily_config.lily_scaling
        num_A = lily_config.num_A
        num_B = lily_config.num_B
        out_features, in_features = target.weight.shape
        kwargs = {
            "in_features" : in_features,
            "out_features" : out_features,
        }

        if isinstance(target, LilyLayer):
            target.update_layer(
                adapter_name,
                r,
                lily_scaling=lily_scaling,
                lily_dropout=lily_config.lily_dropout,
                lily_A=lily_A,
                lily_B=lily_B,
                num_A=num_A,
                num_B=num_B,
            )
        else:
            new_module = self._create_new_module(lily_config, adapter_name, target, lily_A, lily_B, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        meta = torch.device("meta")
        # dispatch to correct device
        for name, module in new_module.named_modules():
            if (self.prefix in name) or ("ranknum" in name):
                if hasattr(child, "qweight"):
                    weight = child.qweight
                elif hasattr(child, "W_q"):
                    weight = child.W_q
                elif hasattr(child, "weight"):
                    weight = child.weight
                elif getattr(child, "in_proj_weight", None) is not None:  # MHA
                    weight = child.in_proj_weight
                else:
                    weight = next(child.parameters())
                if not any(p.device == meta for p in module.parameters()):
                    module.to(weight.device)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

    @staticmethod
    def _create_new_module(lily_config, adapter_name, target, lily_A, lily_B, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        new_module = None
        if isinstance(target_base_layer, torch.nn.Linear):
            new_module = Linear(target, adapter_name, r=lily_config.r, lily_scaling=lily_config.lily_scaling, lily_dropout=lily_config.lily_dropout, num_A=lily_config.num_A, num_B=lily_config.num_B, lily_A=lily_A, lily_B=lily_B, **kwargs)

        return new_module

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "base_model":
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

    def enable_adapter_layers(self) -> None:
        """Enable all adapters.

        Call this if you have previously disabled all adapters and want to re-enable them.
        """
        self._set_adapter_layers(enabled=True)

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

    def set_adapter(self, adapter_name: str | list[str]) -> None:
        """Set the active adapter(s).

        Args:
            adapter_name (`str` or `list[str]`): Name of the adapter(s) to be activated.
        """
        for module in self.model.modules():
            if isinstance(module, LilyLayer):
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            # we directly use lora's mapping for now
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def delete_adapter(self, adapter_name: str) -> None:
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]

        key_list = [key for key, _ in self.model.named_modules() if 'lily' not in key]
        new_adapter = None
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, LilyLayer):
                target.delete_adapter(adapter_name)
                if new_adapter is None:
                    new_adapter = target.active_adapters[:]

        self.active_adapter = new_adapter or []

    def num_of_layers(self, model, adapter_name):
        peft_config = self.peft_config[adapter_name]
        one_target_key = None
        one_target_keys = list(peft_config.target_modules)
        counter = {}

        key_list = [key for key, _ in model.named_modules()]

        for one_target_key in one_target_keys:
            # target moduls found
            counter[one_target_key] = {}

            for key in key_list:
                if key.endswith(one_target_key):
                    # print(f"find {key} matching {one_target_key}")
                    _, target, _ = _get_submodules(model, key)
                    if target.weight.shape not in counter[one_target_key]:
                        counter[one_target_key][target.weight.shape] = 0
                    counter[one_target_key][target.weight.shape] += 1

        return counter
        
    def inject_adapter(self, model: nn.Module, adapter_name: str,
        autocast_adapter_dtype: bool = True,
        low_cpu_mem_usage: bool = False,
        state_dict: Optional[dict[str, torch.Tensor]] = None) -> None:
        """
        Override BaseTuner to allow custom deployment of adapters for Lily.
        """
        peft_config = self.peft_config[adapter_name]
        self._check_new_adapter_config(peft_config)

        # _check_for_modules_to_save = getattr(peft_config, "modules_to_save", None) is not None
        # _has_modules_to_save = False

        model_config = getattr(model, "config", {"model_type": "custom"})
        if hasattr(model_config, "to_dict"):
            model_config = model_config.to_dict()
        peft_config = self._prepare_adapter_config(peft_config, model_config)
        self._prepare_model(peft_config, model)
        is_target_modules_in_base_model = False

        key_list = [key for key, _ in model.named_modules()]

        num_of_target: int = len(peft_config.target_modules)
        counter: int = 0

        lily_As: Dict[str, Dict] = {}
        lily_Bs: Dict[str, Dict] = {}

        for key in peft_config.target_modules:
            lily_As[key] = {}
            lily_Bs[key] = {} 

        num_layers = self.num_of_layers(model, adapter_name)

        stride = {}

        for target in num_layers.keys():
            stride[target] = {}
            for shape in num_layers[target].keys():
                assert num_layers[target][shape] % peft_config.num_A == 0, f"num of layers {num_layers[target][shape]} for target {target} with shape {shape} is not divisible by num_A {peft_config.num_A}"
                stride[target][shape] = num_layers[target][shape] // peft_config.num_A

        import logging

        # some logging for information
        logging.basicConfig(level=logging.INFO)
        logging.info("=" * 50)
        logging.info("Lily adapter injecting, configuration as follows:")
        logging.info(f"num of As: {peft_config.num_A}")
        logging.info(f"num of Bs: {peft_config.num_B}")
        logging.info(f"stride for each A is {stride}")
        logging.info(f"num of targets: {num_of_target}")
        # print num_layers for each target and each shape
        for target in num_layers.keys():
            for shape in num_layers[target].keys():
                logging.info(f"num of layers for target {target} with shape {shape}: {num_layers[target][shape]}")
        logging.info("=" * 50)

        counter = {}
        for key in peft_config.target_modules:
            counter[key] = {}

        for key in key_list:
            # find target modules
            if isinstance(peft_config.target_modules, str):
                target_module_found = re.fullmatch(peft_config.target_modules, key)
                matched_target = peft_config.target_modules if target_module_found else None
            else:
                for target_key in peft_config.target_modules:
                    if key.endswith(target_key):
                        target_module_found = True
                        matched_target = target_key
                        break
                else:
                    target_module_found = False
                    matched_target = None

            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = _get_submodules(model, key)

                self.targeted_module_names.append(key)
                if isinstance(target, torch.nn.Linear):
                    out_features, in_features = target.weight.shape
                    shape = target.weight.shape

                    if shape not in lily_Bs[matched_target]:
                        lily_Bs[matched_target][shape] = nn.Linear(out_features, peft_config.num_B * peft_config.r, bias=False)

                    if shape not in counter[matched_target]:
                        counter[matched_target][shape] = 0

                    if counter[matched_target][shape] % stride[matched_target][shape] == 0:
                        # setting new As across 'stride' layers.
                        logging.info(f"Creating new Lily A for target {matched_target} with shape {shape} at layer count {counter[matched_target][shape]}")

                        lily_As[matched_target][shape] = nn.Linear(in_features=in_features, out_features=peft_config.r, bias=False)
                        
                    self._create_and_replace(peft_config, adapter_name, target, target_name, parent, current_key=key, lily_A=lily_As[matched_target][shape], lily_B=lily_Bs[matched_target][shape])

                    counter[matched_target][shape] += 1

        self.set_adapter(self.active_adapters)
        self._mark_only_adapters_as_trainable(model)