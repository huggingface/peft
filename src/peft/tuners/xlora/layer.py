# Copyright 2024-present the HuggingFace Inc. team.
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

from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor

from peft.tuners import lora

from .config import XLoraConfig


class XLoraLayer:
    """
    A XLoraLayer wraps any LoraLayer and performs the XLora operation on the LoRA adaptors specified. Its primary API
    is the forward method, which uses the scalings to execute the XLora algorithm.
    """

    def __init__(
        self,
        model: nn.Module,  # XLoraModel
        target: lora.LoraLayer,
        target_forward: Callable[..., Any],
        layer_number: int,
        config: XLoraConfig,
    ) -> None:
        self.model = model
        self.target_forward = target_forward
        self.target = target
        self.layer_number = layer_number
        self.config = config

    """
    Apply the scalings for the adapter.
    """

    @staticmethod
    def apply_scalings_to_x(x: torch.Tensor, scalings_layer: torch.Tensor, adapter: int) -> torch.Tensor:
        # scalings_layer = [batch_size, seq_len, n_classes]
        scalings = scalings_layer[:, :, adapter].unsqueeze(-1)
        # scalings_layer = [batch_size, seq_len, 1]
        return x * scalings

    """
    Get the scalings for this layer, potentially applying topk and topk+softmax. This is called before
    `apply_scalings_to_x`
    """

    def get_maybe_topk_scalings(self, scalings) -> torch.Tensor:
        # xlora_scalings = [batch_size, seq_len, n_classes]
        xlora_scalings: Tensor = scalings[:, :, self.layer_number, :]  # type: ignore

        if self.config.top_k_lora is not None:
            _, topk_indices = torch.topk(xlora_scalings, k=self.config.top_k_lora, dim=-1)

            # Mask the topk to True, the rest to False
            mask = torch.zeros_like(xlora_scalings, dtype=torch.bool)
            mask.scatter_(-1, topk_indices, True)

            xlora_scalings = xlora_scalings * mask.to(xlora_scalings.dtype)

        if self.config.enable_softmax_topk:
            nonzero_mask = xlora_scalings != 0
            softmax_res_nonzero = torch.softmax(xlora_scalings[nonzero_mask], dim=-1)
            xlora_scalings[nonzero_mask] = softmax_res_nonzero

        return xlora_scalings


class XLoraLinearLayer(XLoraLayer):
    def __init__(
        self,
        model: nn.Module,
        target: lora.Linear,
        target_forward: Callable[..., Any],
        layer_number: int,
        config: XLoraConfig,
    ) -> None:
        super().__init__(model, target, target_forward, layer_number, config)

    def forward(self, x: Tensor, *args: Any, scalings: Optional[Tensor] = None, **kwargs: Any) -> Tensor:
        """
        This method is designed to be a drop-in-replacement for the LoRA layers' .forward method. To use it, a bound
        method must be created (bound to an instance of the XLoraLayer class).
        """

        previous_dtype = x.dtype
        if scalings is not None:
            xlora_scalings = self.get_maybe_topk_scalings(scalings)

        result = self.target.base_layer(x, *args, **kwargs)

        # Ignore if disabled. We want to make sure this is always run.
        if not self.target.merged:
            for adapter_n, active_adapter in enumerate(self.target.active_adapters):
                # TODO: implement X-LoRA with Lora+Dora layers
                if self.target.use_dora[active_adapter]:
                    raise ValueError("X-LoRA currently does not support LoRA layers with DoRA")
                if active_adapter not in self.target.lora_A.keys():
                    continue
                lora_A = self.target.lora_A[active_adapter]
                lora_B = self.target.lora_B[active_adapter]
                dropout = self.target.lora_dropout[active_adapter]
                scaling = self.target.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)  # type: ignore
                if scalings is not None:
                    x_mod = self.apply_scalings_to_x(x, xlora_scalings, adapter_n)
                    scaling_weight = self.config.global_scaling_weight
                else:
                    x_mod = x
                    scaling_weight = 1
                result += lora_B(lora_A(dropout(x_mod))) * scaling * scaling_weight

        result = result.to(previous_dtype)
        return result


class XLoraEmbeddingLayer(XLoraLayer):
    def __init__(
        self,
        model: nn.Module,
        target: lora.Embedding,
        target_forward: Callable[..., Any],
        layer_number: int,
        config: XLoraConfig,
    ) -> None:
        super().__init__(model, target, target_forward, layer_number, config)

    def forward(self, x: Tensor, *args: Any, scalings: Optional[Tensor] = None, **kwargs: Any) -> Tensor:
        """
        This method is designed to be a drop-in-replacement for the LoRA layers' .forward method. To use it, a bound
        method must be created (bound to an instance of the XLoraLayer class).
        """

        if scalings is not None:
            xlora_scalings = self.get_maybe_topk_scalings(scalings)

        result = self.target.base_layer(x, *args, **kwargs)

        # Ignore if disabled. We want to make sure this is always run.
        if not self.target.merged:
            for adapter_n, active_adapter in enumerate(self.target.active_adapters):
                # TODO: implement X-LoRA with Lora+Dora layers
                if self.target.use_dora.get(active_adapter, False):
                    raise ValueError("X-LoRA currently does not support LoRA layers with DoRA")
                if active_adapter not in self.target.lora_embedding_A:
                    continue
                embedding_A = self.target.lora_embedding_A[active_adapter].T
                embedding_B = self.target.lora_embedding_B[active_adapter].T
                scaling = self.target.scaling[active_adapter]
                after_A = self.target._embed(x, embedding_A)  # type: ignore
                if scalings is not None:
                    after_A_mod = self.apply_scalings_to_x(after_A, xlora_scalings, adapter_n)
                    scaling_weight = self.config.global_scaling_weight
                else:
                    after_A_mod = after_A
                    scaling_weight = 1
                result += (after_A_mod @ embedding_B) * scaling * scaling_weight

        return result


class XLoraConv2dLayer(XLoraLayer):
    def __init__(
        self,
        model: nn.Module,
        target: lora.Conv2d,
        target_forward: Callable[..., Any],
        layer_number: int,
        config: XLoraConfig,
    ) -> None:
        super().__init__(model, target, target_forward, layer_number, config)

    def forward(self, x: Tensor, *args: Any, scalings: Optional[Tensor] = None, **kwargs: Any) -> Tensor:
        """
        This method is designed to be a drop-in-replacement for the LoRA layers' .forward method. To use it, a bound
        method must be created (bound to an instance of the XLoraLayer class).
        """

        previous_dtype = x.dtype

        if scalings is not None:
            xlora_scalings = self.get_maybe_topk_scalings(scalings)

        result = self.target.base_layer(x, *args, **kwargs)

        # Ignore if disabled. We want to make sure this is always run.
        if not self.target.merged:
            for adapter_n, active_adapter in enumerate(self.target.active_adapters):
                # TODO: implement X-LoRA with Lora+Dora layers
                if self.target.use_dora[active_adapter]:
                    raise ValueError("X-LoRA currently does not support LoRA layers with DoRA")
                if active_adapter not in self.target.lora_A.keys():
                    continue
                lora_A = self.target.lora_A[active_adapter]
                lora_B = self.target.lora_B[active_adapter]
                dropout = self.target.lora_dropout[active_adapter]
                scaling = self.target.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)  # type: ignore
                if scalings is not None:
                    x_mod = self.apply_scalings_to_x(x, xlora_scalings, adapter_n)
                    scaling_weight = self.config.global_scaling_weight
                else:
                    x_mod = x
                    scaling_weight = 1
                result += lora_B(lora_A(dropout(x_mod))) * scaling * scaling_weight

        result = result.to(previous_dtype)
        return result
