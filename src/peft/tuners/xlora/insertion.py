from typing import Any, Callable

import torch
from torch import Tensor, nn

from peft.tuners import lora

from .classifier import XLoraClassifier
from .config import XLoraConfig


class XLoraLayer:
    """
    A XLoraLayer wraps any LoraLayer and performs the XLora operation on the LoRA adaptors specified.
    Its primary API is the forward method, which uses the scalings to execute the
    XLora algorithm.
    """

    def __init__(
        self,
        model: nn.Module,  # PeftModel
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
            _, topk_indices = torch.topk(xlora_scalings, k=self.config.top_k_lora, dim=1)

            # Mask the topk to True, the rest to False
            mask = torch.zeros_like(xlora_scalings, dtype=torch.bool)
            mask.scatter_(1, topk_indices, True)

            xlora_scalings = xlora_scalings * mask.to(xlora_scalings.dtype)

        classifier: XLoraClassifier = self.model.base_model.internal_xlora_classifier  # type: ignore
        if classifier.config.enable_softmax_topk:
            nonzero_mask = xlora_scalings != 0
            softmax_res_nonzero = torch.softmax(xlora_scalings[nonzero_mask], dim=-1)
            xlora_scalings[nonzero_mask] = softmax_res_nonzero

        return xlora_scalings

    def forward(self, x: Tensor, *args: Any, **kwargs: Any) -> Tensor:
        return self.target.forward(x, *args, **kwargs)
