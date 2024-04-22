from typing import Any, Callable

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

    @staticmethod
    def apply_scalings_to_x(x: torch.Tensor, scalings_layer: torch.Tensor, adapter: int) -> torch.Tensor:
        # scalings_layer = [batch_size, seq_len, n_classes]
        scalings = scalings_layer[:, :, adapter].unsqueeze(-1)
        # scalings_layer = [batch_size, seq_len, 1]
        return x * scalings

    def get_maybe_topk_scalings(self) -> torch.Tensor:
        # XLora_scalings = [batch_size, seq_len, n_classes]
        XLora_scalings: Tensor = self.scalings[:, :, self.layer_number, :]  # type: ignore

        if self.config.top_k_lora is not None:
            _, topk_indices = torch.topk(XLora_scalings, k=self.config.top_k_lora, dim=-1)

            # Mask the topk to True, the rest to False
            mask = torch.zeros_like(XLora_scalings, dtype=torch.bool)
            mask.scatter_(-1, topk_indices, True)

            XLora_scalings = XLora_scalings * mask.to(XLora_scalings.dtype)

        if self.config.enable_softmax_topk:
            nonzero_mask = XLora_scalings != 0
            softmax_res_nonzero = torch.softmax(XLora_scalings[nonzero_mask], dim=-1)
            XLora_scalings[nonzero_mask] = softmax_res_nonzero

        return XLora_scalings


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

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """
        This method is designed to be a drop-in-replacement for the LoRA layers' .forward method. To use it, a bound
        method must be created (bound to an instance of the XLoraLayer class).
        """

        previous_dtype = x.dtype
        XLora_scalings = self.get_maybe_topk_scalings()

        # Ignore if disabled. We want to make sure this is always run.
        if self.target.merged:
            result = self.target.base_layer(x, *args, **kwargs)
        else:
            result = self.target.base_layer(x, *args, **kwargs)

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
                x_mod = self.apply_scalings_to_x(x, XLora_scalings, adapter_n)
                result += lora_B(lora_A(dropout(x_mod))) * scaling * self.config.global_scaling_weight

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

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """
        This method is designed to be a drop-in-replacement for the LoRA layers' .forward method. To use it, a bound
        method must be created (bound to an instance of the XLoraLayer class).
        """

        XLora_scalings = self.get_maybe_topk_scalings()

        # Ignore if disabled. We want to make sure this is always run.
        if self.target.merged:
            result = self.target.base_layer(x, *args, **kwargs)
        else:
            result = self.target.base_layer(x, *args, **kwargs)
            for adapter_n, active_adapter in enumerate(self.target.active_adapters):
                # TODO: implement X-LoRA with Lora+Dora layers
                if self.target.use_dora[active_adapter]:
                    raise ValueError("X-LoRA currently does not support LoRA layers with DoRA")
                if active_adapter not in self.target.lora_embedding_A:
                    continue
                embedding_A = self.target.lora_embedding_A[active_adapter].T
                embedding_B = self.target.lora_embedding_B[active_adapter].T
                scaling = self.target.scaling[active_adapter]
                x_mod = self.apply_scalings_to_x(x, XLora_scalings, adapter_n)
                after_A = self.target._embed(x_mod, embedding_A)  # type: ignore
                result += (after_A @ embedding_B) * scaling * self.config.global_scaling_weight

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

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        """
        This method is designed to be a drop-in-replacement for the LoRA layers' .forward method. To use it, a bound
        method must be created (bound to an instance of the XLoraLayer class).
        """

        previous_dtype = x.dtype
        XLora_scalings = self.get_maybe_topk_scalings()

        # Ignore if disabled. We want to make sure this is always run.
        if self.target.merged:
            result = self.target.base_layer(x, *args, **kwargs)
        else:
            result = self.target.base_layer(x, *args, **kwargs)
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
                x_mod = self.apply_scalings_to_x(x, XLora_scalings, adapter_n)
                result += lora_B(lora_A(dropout(x_mod))) * scaling * self.config.global_scaling_weight

        result = result.to(previous_dtype)
        return result
