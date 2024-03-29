from typing import Any, Callable

import torch
import torch.nn as nn
from peft.tuners import lora
from torch import Tensor

from xlora.xlora_config import XLoraConfig


class XLoRALayer:
    """
    A XLoRALayer wraps any LoraLayer and performs the XLoRA operation on the LoRA adaptors specified.
    Its primary API is the forward method, which uses the scalings to execute the
    XLoRA algorithm.
    """
    
    def __init__(
        self,
        model: nn.Module, # XLoraModel
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
        # xlora_scalings = [batch_size, seq_len, n_classes]
        xlora_scalings: Tensor = self.model.internal_xlora_scalings[:, :, self.layer_number, :]  # type: ignore

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


class XLoRALinearLayer(XLoRALayer):
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
        This method is designed to be a drop-in-replacement for the LoRA layers' .forward method.
        To use it, a bound method must be created (bound to an instance of the XLoRALayer class).
        """

        previous_dtype = x.dtype
        xlora_scalings = self.get_maybe_topk_scalings()

        if self.target.disable_adapters:
            if self.target.merged:
                self.target.unmerge()
            result = self.target.base_layer(x, *args, **kwargs)
        elif self.target.merged:
            result = self.target.base_layer(x, *args, **kwargs)
        else:
            result = self.target.base_layer(x, *args, **kwargs)

            for adapter_n, active_adapter in enumerate(self.target.active_adapters):
                if active_adapter not in self.target.lora_A.keys():
                    continue
                lora_A = self.target.lora_A[active_adapter]
                lora_B = self.target.lora_B[active_adapter]
                dropout = self.target.lora_dropout[active_adapter]
                scaling = self.target.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)  # type: ignore
                x_mod = self.apply_scalings_to_x(x, xlora_scalings, adapter_n)
                result += lora_B(lora_A(dropout(x_mod))) * scaling * self.config.global_scaling_weight

        result = result.to(previous_dtype)
        return result


class XLoRAEmbeddingLayer(XLoRALayer):
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
        This method is designed to be a drop-in-replacement for the LoRA layers' .forward method.
        To use it, a bound method must be created (bound to an instance of the XLoRALayer class).
        """

        xlora_scalings = self.get_maybe_topk_scalings()

        # TODO: no dtype conversion here, unlike in Linear, is that correct?
        if self.target.disable_adapters:
            if self.target.merged:
                self.target.unmerge()
            result = self.target.base_layer(x, *args, **kwargs)
        elif self.target.merged:
            result = self.target.base_layer(x, *args, **kwargs)
        else:
            result = self.target.base_layer(x, *args, **kwargs)
            for adapter_n, active_adapter in enumerate(self.target.active_adapters):
                if active_adapter not in self.target.lora_embedding_A:
                    continue
                embedding_A = self.target.lora_embedding_A[active_adapter].T
                embedding_B = self.target.lora_embedding_B[active_adapter].T
                scaling = self.target.scaling[active_adapter]
                x_mod = self.apply_scalings_to_x(x, xlora_scalings, adapter_n)
                after_A = self.target._embed(x_mod, embedding_A)  # type: ignore
                result += (after_A @ embedding_B) * scaling * self.config.global_scaling_weight

        return result


class XLoRAConv2dLayer(XLoRALayer):
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
        This method is designed to be a drop-in-replacement for the LoRA layers' .forward method.
        To use it, a bound method must be created (bound to an instance of the XLoRALayer class).
        """

        previous_dtype = x.dtype
        xlora_scalings = self.get_maybe_topk_scalings()

        if self.target.disable_adapters:
            if self.target.merged:
                self.target.unmerge()
            result = self.target.base_layer(x, *args, **kwargs)
        elif self.target.merged:
            result = self.target.base_layer(x, *args, **kwargs)
        else:
            result = self.target.base_layer(x, *args, **kwargs)
            for adapter_n, active_adapter in enumerate(self.target.active_adapters):
                if active_adapter not in self.target.lora_A.keys():
                    continue
                lora_A = self.target.lora_A[active_adapter]
                lora_B = self.target.lora_B[active_adapter]
                dropout = self.target.lora_dropout[active_adapter]
                scaling = self.target.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)  # type: ignore
                x_mod = self.apply_scalings_to_x(x, xlora_scalings, adapter_n)
                result += lora_B(lora_A(dropout(x_mod))) * scaling * self.config.global_scaling_weight

        result = result.to(previous_dtype)
        return result
