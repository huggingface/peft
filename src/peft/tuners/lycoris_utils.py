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
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from tqdm import tqdm

from peft.config import PeftConfig
from peft.utils import (
    ModulesToSaveWrapper,
    _get_submodules,
)

from .tuners_utils import BaseTuner, BaseTunerLayer, check_adapters_to_merge, check_target_module_exists


@dataclass
class LycorisConfig(PeftConfig):
    r"""
    A base config for LyCORIS like adapters
    """

    rank_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to ranks which are different from the default rank specified by `r`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 8`}"
            )
        },
    )
    alpha_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to alphas which are different from the default alpha specified by `alpha`. "
                "For example, `{model.decoder.layers.0.encoder_attn.k_proj: 32`}"
            )
        },
    )


class LycorisLayer(BaseTunerLayer):
    r"""
    A base layer for LyCORIS like adapters
    """

    # adapter_layer_names needs to be defined on the child class
    other_param_names = ("r", "alpha", "scaling", "rank_dropout", "module_dropout")

    def __init__(self, base_layer: nn.Module) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.alpha = {}
        self.scaling = {}
        self.rank_dropout = {}
        self.module_dropout = {}

        # Tuner info
        self._disable_adapters = False
        self.merged_adapters = []

    @property
    @abstractmethod
    def _available_adapters(self) -> set[str]: ...

    def _init_empty_weights(self, cls, *args, **kwargs) -> None:
        # A helper method that allows to initialize the layer of the given class without spending time to initialize the
        # model weights. The implementation is inspired by
        # https://pytorch.org/docs/stable/generated/torch.nn.utils.skip_init.html but this function cannot be used
        # directly.
        # Instead of this approach, it would be possible to bypass the __init__ of the class but that runs the risk of
        # omitting important logic inside that __init__.
        kwargs = kwargs.copy()
        final_device = kwargs.pop("device", "cpu")
        cls.__init__(self, *args, device="meta", **kwargs)
        self.to_empty(device=final_device)

    @abstractmethod
    def create_adapter_parameters(self, adapter_name: str, r: int, **kwargs): ...

    # TODO: refactor LoRA to use the same approach
    @abstractmethod
    def _get_delta_activations(self, adapter_name: str, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Activations added on top of the base layer output (i.e. after the base layer forward pass)"""

    @abstractmethod
    def get_delta_weight(self, adapter_name: str) -> torch.Tensor: ...

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If `True`, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If `None`, all active adapters will be merged.
                Defaults to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self._available_adapters:
                base_layer = self.get_base_layer()
                if safe_merge:
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    @abstractmethod
    def reset_adapter_parameters(self, adapter_name: str): ...

    def set_scale(self, adapter, scale):
        if adapter not in self._available_adapters:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale * self.alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self._available_adapters:
                continue

            self.scaling[active_adapter] *= scale

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self._available_adapters:
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self._available_adapters:
                continue

            if scale is None:
                self.scaling[active_adapter] = self.alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale

    @abstractmethod
    def update_layer(self, adapter_name: str, r: int, alpha: float, **kwargs): ...


class LycorisTuner(BaseTuner):
    r"""
    A base tuner for LyCORIS like adapters

    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    """

    prefix: str
    layers_mapping: dict[type[torch.nn.Module], type[LycorisLayer]]

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False):
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "model":  # see #1892: prevent infinite recursion if class is not initialized
                raise
            return getattr(self.model, name)

    @staticmethod
    def _check_target_module_exists(config, key):
        return check_target_module_exists(config, key)

    @abstractmethod
    def _create_and_replace(
        self,
        config: LycorisConfig,
        adapter_name: str,
        target: Union[LycorisLayer, nn.Module],
        target_name,
        parent,
        current_key,
    ): ...

    @classmethod
    def _create_new_module(cls, config: LycorisConfig, adapter_name: str, target: nn.Module, **kwargs) -> LycorisLayer:
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

        if isinstance(target_base_layer, torch.nn.Conv2d):
            new_module = new_module_cls(target, adapter_name=adapter_name, **kwargs)
        elif isinstance(target_base_layer, torch.nn.Linear):
            new_module = new_module_cls(target, adapter_name=adapter_name, **kwargs)
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

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

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
            if self.prefix in name:
                if not any(p.device == meta for p in module.parameters()):
                    module.to(child.weight.device)

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def _unload_and_optionally_merge(
        self,
        merge: bool = True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        if merge:
            if getattr(self.model, "quantization_method", None) == "gptq":
                raise ValueError("Cannot merge LOHA layers when the model is gptq quantized")

        self._unloading_checks(adapter_names)
        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
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
                new_module = target.modules_to_save[target.active_adapter]
                if hasattr(new_module, "base_layer"):
                    # check if the module is itself a tuner layer
                    if merge:
                        new_module.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                    new_module = new_module.get_base_layer()
                setattr(parent, target_name, new_module)

        return self.model

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

    def merge_and_unload(
        self, progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None
    ) -> torch.nn.Module:
        r"""
        This method merges the adapter layers into the base model. This is needed if someone wants to use the base
        model as a standalone model.

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
        Gets back the base model by removing all the lora modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)

    def set_adapter(self, adapter_name: str | list[str]) -> None:
        """Set the active adapter(s).

        Additionally, this function will set the specified adapters to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (`str` or `list[str]`): Name of the adapter(s) to be activated.
        """
        for module in self.model.modules():
            if isinstance(module, LycorisLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    def delete_adapter(self, adapter_name: str) -> None:
        """
        Deletes an existing adapter.

        Args:
            adapter_name (`str`): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        new_adapter = None
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, LycorisLayer):
                target.delete_adapter(adapter_name)
                if new_adapter is None:
                    new_adapter = target.active_adapters[:]

        self.active_adapter = new_adapter or []
