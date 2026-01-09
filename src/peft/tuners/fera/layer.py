# layer.py
import math
from typing import Any

import torch
import torch.nn as nn

from peft.tuners.tuners_utils import BaseTunerLayer


class LoRAExpert(nn.Module):
    """Simple LoRA Expert module used internally by FeRA."""

    def __init__(self, in_features, out_features, rank, alpha, dropout, init_weights=True):
        super().__init__()
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = alpha / rank

        if init_weights:
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        down = self.lora_down(self.dropout(x))
        up = self.lora_up(down)
        return up * self.scale


class FeRALayer(BaseTunerLayer):
    """Base layer for FeRA."""

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.experts = nn.ModuleDict({})  # Change to ModuleDict to handle multiple adapters
        self.rank = {}
        self.alpha = {}
        self.dropout = {}
        self.num_experts = {}
        self.kwargs = kwargs

        self.current_routing_weights = None

    def update_layer(self, adapter_name, rank, alpha, dropout, num_experts, init_weights):
        self.rank[adapter_name] = rank
        self.alpha[adapter_name] = alpha
        self.dropout[adapter_name] = dropout
        self.num_experts[adapter_name] = num_experts

        if isinstance(self.base_layer, nn.Linear):
            in_features, out_features = self.base_layer.in_features, self.base_layer.out_features
        else:
            raise ValueError(f"Unsupported layer type {type(self.base_layer)}")

        expert_list = nn.ModuleList(
            [LoRAExpert(in_features, out_features, rank, alpha, dropout, init_weights) for _ in range(num_experts)]
        )

        self.experts[adapter_name] = expert_list

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def set_routing_weights(self, weights: torch.Tensor):
        """Called by the Tuner/Model to broadcast global routing weights."""
        self.current_routing_weights = weights


class FeRALinear(nn.Module, FeRALayer):
    """FeRA implementation for Linear layers."""

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        rank: int = 4,
        lora_alpha: float = 8.0,
        dropout: float = 0.0,
        num_experts: int = 3,
        init_weights: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        FeRALayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, rank, lora_alpha, dropout, num_experts, init_weights)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        # 1. Base Output
        result = self.base_layer(x, *args, **kwargs)

        if self.disable_adapters:
            return result

        # 2. Iterate over active adapters
        for active_adapter in self.active_adapters:
            if active_adapter not in self.experts:
                continue

            if self.current_routing_weights is None:
                continue

            expert_list = self.experts[active_adapter]

            # 3. Compute Experts Output
            # x: (B, Seq, Dim) -> expert_outs: (B, Num_Experts, Seq, Dim)
            expert_outputs = [expert(x.to(result.dtype)) for expert in expert_list]
            expert_outputs = torch.stack(expert_outputs, dim=1)

            # 4. Weighted Sum
            # 这里的 weights 是全局传入的 (B, Num_Experts)
            weights = self.current_routing_weights

            view_shape = (weights.shape[0], weights.shape[1]) + (1,) * (expert_outputs.ndim - 2)
            w = weights.view(view_shape)

            adapter_out = torch.sum(w * expert_outputs, dim=1)
            result = result + adapter_out.to(previous_dtype)

        return result

    def __repr__(self) -> str:
        return f"FeRALinear(in_features={self.base_layer.in_features}, out_features={self.base_layer.out_features}, num_experts={len(self.experts[self.active_adapter])})"
