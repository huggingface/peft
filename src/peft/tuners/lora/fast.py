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

import math
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer

from .config import LoraConfig


class FastLoraLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features

    def get_base_layer(self) -> nn.Module:
        # For fast layers, nested base layers are not possible, so we can simplify this
        return self.base_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        raise TypeError(f"{self.__class__.__name__} does not support merging.")

    def unmerge(self) -> None:
        raise TypeError(f"{self.__class__.__name__} does not support unmerging.")

    @property
    def merged(self) -> bool:
        return False

    @property
    def disable_adapters(self) -> bool:
        raise TypeError(f"{self.__class__.__name__} does not support disabling adapters.")

    def update_layer(self, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora):
        # This method is only for the basics, implement the actual learnable parameters in the subclasses
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r = r
        self.lora_alpha = lora_alpha
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
        self.scaling = lora_alpha / math.sqrt(r) if use_rslora else lora_alpha / r

    def reset_lora_parameters(self, init_lora_weights):
        # possibly override this for non-Linear layers
        if init_lora_weights is False:
            return

        if init_lora_weights is True:
            # initialize A the same way as the default for nn.Linear and B to zero
            # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        elif init_lora_weights.lower() == "gaussian":
            nn.init.normal_(self.lora_A.weight, std=1 / self.r)
        else:
            raise ValueError(f"Unknown initialization {init_lora_weights=}")
        nn.init.zeros_(self.lora_B.weight)

    def loftq_init(self):
        from peft.utils.loftq_utils import loftq_init

        weight = self.get_base_layer().weight
        kwargs = {
            "num_bits": self.kwargs.get("loftq_bits", 4),
            "reduced_rank": self.r,
            "num_iter": self.kwargs.get("loftq_iter", 1),
        }

        qweight, lora_A, lora_B = loftq_init(weight, **kwargs)
        # initialize A the same way as the default for nn.Linear and B to zero
        self.lora_A.weight.data = lora_A
        self.lora_B.weight.data = lora_B
        self.get_base_layer().weight.data = qweight

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling = scale * self.lora_alpha / self.r

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        self.scaling *= scale

    def unscale_layer(self, scale=None) -> None:
        if scale is None:
            self.scaling = self.lora_alpha / self.r
        else:
            self.scaling /= scale

    def enable_adapters(self, enabled: bool) -> None:
        pass

    def set_adapter(self, adapter_names: str | list[str]) -> None:
        if not isinstance(adapter_names, str):
            adapter_name = list(adapter_names)[0]
        else:
            adapter_name = adapter_names

        if adapter_name != self._active_adapter:
            raise TypeError(f"{self.__class__.__name__} does not support changing the active adapter.")

    def delete_adapter(self, adapter_name: str) -> None:
        raise TypeError(f"{self.__class__.__name__} does not support deleting adapters.")


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class Linear(nn.Module, FastLoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        FastLoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(r, lora_alpha, lora_dropout, init_lora_weights, use_rslora)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def update_layer(self, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora):
        super().update_layer(r, lora_alpha, lora_dropout, init_lora_weights, use_rslora)

        # Actual trainable parameters
        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)

        if init_lora_weights == "loftq":
            self.loftq_init()
        elif init_lora_weights:
            self.reset_lora_parameters(init_lora_weights)

        # check weight and qweight (for GPTQ)
        for weight_name in ("weight", "qweight"):
            weight = getattr(self.get_base_layer(), weight_name, None)
            if weight is not None:
                # the layer is already completely initialized, this is an update
                if weight.dtype.is_floating_point or weight.dtype.is_complex:
                    self.to(weight.device, dtype=weight.dtype)
                else:
                    self.to(weight.device)
                break

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype
        result = self.base_layer(x, *args, **kwargs)
        x = x.to(self.lora_A.weight.dtype)
        result += self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora.fast." + rep


class Embedding(nn.Module, FastLoraLayer):
    # LoRA implemented in a Embedding layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        FastLoraLayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer(r, lora_alpha, lora_dropout, init_lora_weights, use_rslora)

    def update_layer(self, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora):
        super().update_layer(r, lora_alpha, lora_dropout, init_lora_weights, use_rslora)

        # Actual trainable parameters
        weight_A = torch.randn((r, self.in_features))
        weight_B = torch.randn((self.out_features, r))
        self.lora_embedding_A = nn.Parameter(weight_A)
        self.lora_embedding_B = nn.Parameter(weight_B)

        if init_lora_weights == "loftq":
            self.loftq_init()
        elif init_lora_weights:
            self.reset_lora_parameters(init_lora_weights)

        base_layer = self.get_base_layer()
        weight = getattr(base_layer, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            self.to(base_layer.weight.device, dtype=weight.dtype)
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, init_lora_weights):
        if init_lora_weights is False:
            return

        nn.init.zeros_(self.lora_B.weight)
        # initialize a the same way as the default for nn.linear and b to zero
        nn.init.zeros_(self.lora_embedding_A)
        nn.init.normal_(self.lora_embedding_B)

    def loftq_init(self):
        from peft.utils.loftq_utils import loftq_init

        weight = self.get_base_layer().weight
        kwargs = {
            "num_bits": self.kwargs.get("loftq_bits", 4),
            "reduced_rank": self.r,
            "num_iter": self.kwargs.get("loftq_iter", 1),
        }

        qweight, lora_A, lora_B = loftq_init(weight, **kwargs)
        # initialize a the same way as the default for nn.linear and b to zero
        self.lora_embedding_A.weight.data = lora_A
        self.lora_embedding_B.weight.data = lora_B
        self.get_base_layer().weight.data = qweight

    def _embed(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        base_layer = self.get_base_layer()
        return F.embedding(
            input,
            weight,
            padding_idx=base_layer.padding_idx,
            max_norm=base_layer.max_norm,
            norm_type=base_layer.norm_type,
            scale_grad_by_freq=base_layer.scale_grad_by_freq,
            sparse=base_layer.sparse,
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        result = self.base_layer(x, *args, **kwargs)
        after_A = self._embed(x, self.lora_embedding_A)
        result += (after_A @ self.lora_embedding_B) * self.scaling
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora.fast." + rep


class Conv2d(nn.Module, FastLoraLayer):
    # Lora implemented in a conv2d layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        FastLoraLayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self.update_layer(r, lora_alpha, lora_dropout, init_lora_weights, use_rslora)

    def update_layer(self, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora):
        super().update_layer(r, lora_alpha, lora_dropout, init_lora_weights, use_rslora)

        # Actual trainable parameters
        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)

        if init_lora_weights == "loftq":
            self.loftq_init()
        elif init_lora_weights:
            self.reset_lora_parameters(init_lora_weights)

        # check weight and qweight (for GPTQ)
        for weight_name in ("weight", "qweight"):
            weight = getattr(self.get_base_layer(), weight_name, None)
            if weight is not None:
                # the layer is already completely initialized, this is an update
                if weight.dtype.is_floating_point or weight.dtype.is_complex:
                    self.to(weight.device, dtype=weight.dtype)
                else:
                    self.to(weight.device)
                break

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        result = self.base_layer(x, *args, **kwargs)
        x = x.to(self.lora_A.weight.dtype)
        result += self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora.fast." + rep


def dispatch_fast(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: LoraConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None
    if getattr(lora_config, "fast_train_mode", False) is False:
        return new_module

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Embedding):
        embedding_kwargs = kwargs.copy()
        embedding_kwargs.pop("fan_in_fan_out", None)
        embedding_kwargs.update(lora_config.loftq_config)
        new_module = Embedding(target, adapter_name, **embedding_kwargs)
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        kwargs.update(lora_config.loftq_config)
        new_module = Conv2d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        new_module = Linear(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        if not kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to False but the target module is `Conv1D`. " "Setting fan_in_fan_out to True."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
        kwargs.update(lora_config.loftq_config)
        new_module = Linear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)

    return new_module
