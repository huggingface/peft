# coding=utf-8
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

import re
import warnings
from itertools import chain
from typing import Union

import torch
from torch import nn
from tqdm import tqdm

from peft.tuners.tuners_utils import BaseTuner
from peft.utils import (
    COMMON_LAYERS_PATTERN,
    ModulesToSaveWrapper,
    _get_submodules,
)

from .layer import Conv2d, Linear, LoHaLayer


class LoHaModel(BaseTuner):
    """
    Creates Low-Rank Hadamard Product model from a pretrained model. The method is partially described in
    https://arxiv.org/abs/2108.06098 Current implementation heavily borrows from
    https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/modules/loha.py

    Args:
        model (`torch.nn.Module`): The model to which the adapter tuner layers will be attached.
        config ([`LoHaConfig`]): The configuration of the LoHa model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The LoHa model.

    Example:
        #TODO: Add example

    **Attributes**:
        - **model** ([`~torch.nn.Module`]) -- The model to be adapted.
        - **peft_config** ([`LoHaConfig`]): The configuration of the LoHa model.
    """

    def __init__(self, model, config, adapter_name):
        super().__init__(model, config, adapter_name)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            raise ValueError("Please specify `target_modules` in `peft_config`")
        return peft_config

    @staticmethod
    def _check_target_module_exists(loha_config, key):
        """
        A helper private method to check if the passed module's key name matches any of the target modules in the
        adatper_config.
        """
        if isinstance(loha_config.target_modules, str):
            target_module_found = re.fullmatch(loha_config.target_modules, key)
        else:
            target_module_found = any(
                re.match(f".*\.{target_key}$", key) for target_key in loha_config.target_modules
            ) or any(target_key == key for target_key in loha_config.target_modules)
            is_using_layer_indexes = getattr(loha_config, "layers_to_transform", None) is not None
            layer_indexing_pattern = getattr(loha_config, "layers_pattern", None)

            if is_using_layer_indexes and target_module_found:
                layers_pattern = COMMON_LAYERS_PATTERN if layer_indexing_pattern is None else layer_indexing_pattern
                layers_pattern = [layers_pattern] if isinstance(layers_pattern, str) else layers_pattern

                for pattern in layers_pattern:
                    layer_index = re.match(f".*.{pattern}\.(\d+)\.*", key)
                    if layer_index is not None:
                        layer_index = int(layer_index.group(1))
                        if isinstance(loha_config.layers_to_transform, int):
                            target_module_found = layer_index == loha_config.layers_to_transform
                        else:
                            target_module_found = layer_index in loha_config.layers_to_transform

                        break
                    else:
                        target_module_found = False
        return target_module_found

    def _create_and_replace(
        self,
        loha_config,
        adapter_name: str,
        target: Union[LoHaLayer, nn.Module],
        target_name,
        parent,
        **optional_kwargs,
    ):
        """
        A private method to create and replace the target module with the adapter module.
        """

        # Regexp matching - Find key which matches current target_name in patterns provided
        current_key = optional_kwargs["current_key"]
        pattern_keys = list(chain(loha_config.rank_pattern.keys(), loha_config.alpha_pattern.keys()))
        target_name_key = next(filter(lambda key: re.match(f"(.*\.)?{key}$", current_key), pattern_keys), target_name)

        r = loha_config.rank_pattern.get(target_name_key, loha_config.r)
        alpha = loha_config.alpha_pattern.get(target_name_key, loha_config.alpha)

        kwargs = {
            "r": r,
            "alpha": alpha,
            "rank_dropout": loha_config.rank_dropout,
            "module_dropout": loha_config.module_dropout,
            "use_effective_conv2d": loha_config.use_effective_conv2d,
            "init_weights": loha_config.init_weights,
        }

        if isinstance(target, LoHaLayer):
            target.update_layer(adapter_name, **kwargs)
        else:
            new_module = self._create_new_module(loha_config, adapter_name, target, **kwargs)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(loha_config, adapter_name, target, **kwargs) -> LoHaLayer:
        if isinstance(target, torch.nn.Conv2d):
            new_module = Conv2d(
                target.in_channels,
                target.out_channels,
                target.weight.size()[2:],
                stride=target.stride,
                padding=target.padding,
                dilation=target.dilation,
                groups=target.groups,
                bias=target.bias is not None,
                padding_mode=target.padding_mode,
                device=target.weight.device,
                dtype=target.weight.dtype,
                adapter_name=adapter_name,
                **kwargs,
            )
        elif isinstance(target, torch.nn.Linear):
            new_module = Linear(
                target.in_features,
                target.out_features,
                bias=target.bias is not None,
                device=target.weight.device,
                dtype=target.weight.dtype,
                adapter_name=adapter_name,
                **kwargs,
            )
        return new_module

    @staticmethod
    def _replace_module(parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable
        new_module.weight = child.weight
        if hasattr(child, "bias"):
            new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            new_module.state = child.state
            new_module.to(child.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "hada_" in name:
                module.to(child.weight.device)

    def _mark_only_adapters_as_trainable(self) -> None:
        for n, p in self.model.named_parameters():
            if "hada_" not in n:
                p.requires_grad = False

    def merge_and_unload(self, progressbar: bool = False):
        return self._unload_and_optionally_merge(progressbar=progressbar)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, LoHaLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)

    def _unload_and_optionally_merge(self, merge=True, progressbar: bool = False):
        if merge:
            if getattr(self.model, "quantization_method", None) == "gptq":
                raise ValueError("Cannot merge LOHA layers when the model is gptq quantized")

        key_list = [key for key, _ in self.model.named_modules() if "hada" not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, LoHaLayer):
                if isinstance(target, nn.Conv2d):
                    new_module = torch.nn.Conv2d(
                        target.in_channels,
                        target.out_channels,
                        kernel_size=target.kernel_size,
                        stride=target.stride,
                        padding=target.padding,
                        dilation=target.dilation,
                    )
                elif isinstance(target, nn.Linear):
                    bias = target.bias is not None
                    new_module = torch.nn.Linear(
                        target.in_features,
                        target.out_features,
                        bias=bias,
                        device=target.weight.device,
                    )
                if merge:
                    target.merge()
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model
