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

from peft.config import PeftConfig

from .tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    _get_in_out_features,
    check_adapters_to_merge,
)


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
                "For example, `{'^model.decoder.layers.0.encoder_attn.k_proj': 16}`."
            )
        },
    )
    alpha_pattern: Optional[dict] = field(
        default_factory=dict,
        metadata={
            "help": (
                "The mapping from layer names or regexp expression to alphas which are different from the default alpha specified by `alpha`. "
                "For example, `{'^model.decoder.layers.0.encoder_attn.k_proj': 16}`."
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
        self.rank_dropout_scale = {}
        self.module_dropout = {}

        # Tuner info
        self._disable_adapters = False
        self.merged_adapters = []
        # flag to enable/disable casting of input to weight dtype during forward call
        self.cast_input_dtype_enabled = True

        in_features, out_features = _get_in_out_features(self.get_base_layer())
        self.in_features = in_features
        self.out_features = out_features

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
    tuner_layer_cls = LycorisLayer
    layers_mapping: dict[type[torch.nn.Module], type[LycorisLayer]]

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

        if isinstance(target_base_layer, (torch.nn.Conv2d, torch.nn.Conv1d)):
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
