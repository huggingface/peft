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

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import (
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
        ```py
        >>> from diffusers import StableDiffusionPipeline
        >>> from peft import LoHaModel, LoHaConfig

        >>> config_te = LoHaConfig(
        ...     r=8,
        ...     lora_alpha=32,
        ...     target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
        ...     rank_dropout=0.0,
        ...     module_dropout=0.0,
        ...     init_weights=True,
        ... )
        >>> config_unet = LoHaConfig(
        ...     r=8,
        ...     lora_alpha=32,
        ...     target_modules=[
        ...         "proj_in",
        ...         "proj_out",
        ...         "to_k",
        ...         "to_q",
        ...         "to_v",
        ...         "to_out.0",
        ...         "ff.net.0.proj",
        ...         "ff.net.2",
        ...     ],
        ...     rank_dropout=0.0,
        ...     module_dropout=0.0,
        ...     init_weights=True,
        ...     use_effective_conv2d=True,
        ... )

        >>> model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> model.text_encoder = LoHaModel(model.text_encoder, config_te, "default")
        >>> model.unet = LoHaModel(model.unet, config_unet, "default")
        ```

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

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, LoHaLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            raise ValueError("Please specify `target_modules` in `peft_config`")
        return peft_config

    @staticmethod
    def _check_target_module_exists(loha_config, key):
        return check_target_module_exists(loha_config, key)

    def _create_and_replace(
        self,
        loha_config,
        adapter_name: str,
        target: Union[LoHaLayer, nn.Module],
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        """
        A private method to create and replace the target module with the adapter module.
        """

        # Regexp matching - Find key which matches current target_name in patterns provided
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
        else:
            raise ValueError(
                "Target module not found, currently only adapters for nn.Linear and nn.Conv2d are supported"
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
                else:
                    raise ValueError(
                        "Cannot convert current module to torch module, currently only adapters for nn.Linear and nn.Conv2d are supported"
                    )
                if merge:
                    target.merge()
                self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model
