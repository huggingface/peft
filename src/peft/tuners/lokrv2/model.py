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
from dataclasses import asdict
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    check_target_module_exists,
    onload_layer,
)
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _get_submodules,
)

from .config import LoKrV2Config
from .layer import Conv2d, Linear, LoKrLayerv2


class LoKrModelv2(BaseTuner):
    """
    Creates Low-Rank Kronecker Product model from a pretrained model. The original method is partially described in
    https://arxiv.org/abs/2108.06098 and in https://arxiv.org/abs/2309.14859 Current implementation heavily borrows
    from
    https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/modules/lokr.py

    Args:
        model (`torch.nn.Module`): The model to which the adapter tuner layers will be attached.
        config ([`LoKrConfig`]): The configuration of the LoKr model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    Returns:
        `torch.nn.Module`: The LoKr model.

    Example:
        ```py
        >>> from diffusers import StableDiffusionPipeline
        >>> from peft import LoKrModel, LoKrConfig

        >>> config_te = LoKrConfig(
        ...     r=8,
        ...     lora_alpha=32,
        ...     target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
        ...     rank_dropout=0.0,
        ...     module_dropout=0.0,
        ...     init_weights=True,
        ... )
        >>> config_unet = LoKrConfig(
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
        >>> model.text_encoder = LoKrModel(model.text_encoder, config_te, "default")
        >>> model.unet = LoKrModel(model.unet, config_unet, "default")
        ```

    **Attributes**:
        - **model** ([`~torch.nn.Module`]) -- The model to be adapted.
        - **peft_config** ([`LoKrConfig`]): The configuration of the LoKr model.
    """

    prefix: str = "lokr_"

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

    def _check_new_adapter_config(self, config: LoKrV2Config) -> None:
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
    def _check_target_module_exists(LoKrV2Config, key):
        return check_target_module_exists(LoKrV2Config, key)

    def _create_and_replace(
        self,
        lokr_config: LoKrV2Config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        r = lokr_config.r
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": r,
            "rank_dropout": lokr_config.rank_dropout,
            "fan_in_fan_out": lokr_config.fan_in_fan_out,
            "init_weights": lokr_config.init_weights,
            "rank_dropout_scale": lokr_config.rank_dropout_scale,
            "use_effective_conv2d": lokr_config.use_effective_conv2d,
            "decompose_both": lokr_config.decompose_both,
            "decompose_factor": lokr_config.decompose_factor,
            "use_scalar": lokr_config.use_scalar,
        }
        kwargs["bias"] = bias

        if isinstance(target, LoKrLayerv2):
            target.update_layer(
                adapter_name,
                r=lokr_config.r,
                alpha=lokr_config.alpha,
                rank_dropout=lokr_config.rank_dropout,
                module_dropout=lokr_config.module_dropout,
                init_weights=lokr_config.init_weights,
                use_effective_conv2d=lokr_config.use_effective_conv2d,
                decompose_both=lokr_config.decompose_both,
                decompose_factor=lokr_config.decompose_factor,
                full_matrix=lokr_config.use_full_matrix,
                use_upstream=lokr_config.use_upstream,
            )
        else:
            new_module = self._create_new_module(lokr_config, adapter_name, target, **kwargs)
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
            elif bias == "lokr_only":
                for name, m in model.named_modules():
                    if isinstance(m, LoKrLayerv2) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

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

    @staticmethod
    def _create_new_module(lokr_config, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = lokr_config.fan_in_fan_out = False
            new_module = Linear(target, adapter_name, **kwargs)
        elif isinstance(target_base_layer, torch.nn.Conv2d):
            new_module = Conv2d(target, adapter_name, **kwargs)
        else:
            raise ValueError(
                f"Target module {target} is not supported. "
                "Currently, only `torch.nn.Linear` and `torch.nn.Conv2d` are supported."
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
                    "output as the the base model would without adaption."
                )
                warnings.warn(msg)
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, LoKrLayerv2):
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
            self._check_merge_allowed()

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            with onload_layer(target):
                if hasattr(target, "base_layer"):
                    if merge:
                        target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
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

    def delete_adapter(self, adapter_name: str) -> None:
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        new_adapter = None
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, LoKrLayerv2):
                target.delete_adapter(adapter_name)
                if new_adapter is None:
                    new_adapter = target.active_adapters[:]

        self.active_adapter = new_adapter or []

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ) -> torch.nn.Module:
        r"""
        This method merges the LoKr layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.

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

    def unload(self) -> torch.nn.Module:
        """
        Gets back the base model by removing all the LoKr modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)
