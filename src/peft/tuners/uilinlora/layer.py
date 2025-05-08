from __future__ import annotations
from typing import Dict, List, Optional
import warnings

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

__all__ = ["UILinLoRALayer", "Linear"]


# ---------------------------------------------------------------------------
# Mixin with row-trainable logic
# ---------------------------------------------------------------------------
class UILinLoRALayer(BaseTunerLayer):
    adapter_layer_names: tuple[str, ...] = ("uilinlora_adapter", "uilinlora_bias")
    other_param_names: tuple[str, ...] = ("uilinlora_alpha", "uilinlora_dropout")

    def __init__(self, base_layer: nn.Module, *, fan_in_fan_out: bool = False):
        self.base_layer = base_layer
        self.fan_in_fan_out = fan_in_fan_out

        # ── adapter containers ────────────────────────────────────────────
        self.uilinlora_adapter = nn.ParameterDict()
        self.uilinlora_bias = nn.ParameterDict()
        self.uilinlora_dropout = nn.ModuleDict()
        self.uilinlora_alpha: Dict[str, float] = {}

        # ── runtime state ─────────────────────────────────────────────────
        self._disable_adapters: bool = False
        self._merged_adapters: List[str] = []
        self._active_adapters: List[str] = []

        base = self.get_base_layer()

        if isinstance(base, nn.Linear):
            self.in_features, self.out_features = base.in_features, base.out_features

        elif hasattr(base, "weight") and hasattr(base.weight, "ds_shape"):
            # ds_shape = (out_features, in_features)
            self.out_features, self.in_features = base.weight.ds_shape

        elif isinstance(base, nn.Conv1d):
            self.in_features, self.out_features = base.in_channels, base.out_channels

        else:
            raise ValueError(f"Unsupported base layer type {type(base)}")

    def update_layer(
        self,
        adapter_name: str,
        uilinlora_alpha: float = 1.0,
        uilinlora_dropout: float = 0.0,
        init_uilinlora_weights: bool = True,
        bias: str = "none",
    ) -> None:
        if adapter_name not in self.uilinlora_adapter:
            base_layer = self.get_base_layer()
            base_w = base_layer.weight
            device = base_w.device
            adapter_shape = (self.out_features, self.in_features)
            
            full = torch.empty(adapter_shape, dtype=torch.float32, device=device)
            nn.init.kaiming_uniform_(full, a=math.sqrt(5))
            self.uilinlora_adapter[adapter_name] = nn.Parameter(full, requires_grad=True)


            # optional bias
            if bias != "none" and adapter_name not in self.uilinlora_bias:
                self.uilinlora_bias[adapter_name] = nn.Parameter(
                    torch.zeros(self.out_features, dtype=torch.float32, device=device)
                )

            self.uilinlora_dropout[adapter_name] = nn.Dropout(p=uilinlora_dropout) if uilinlora_dropout > 0.0 else nn.Identity()

        # Get device from base layer for uilinlora_alpha
        base_layer = self.get_base_layer()
        device = base_layer.weight.device
        
        # Ensure uilinlora_alpha is float32
        self.uilinlora_alpha[adapter_name] = torch.tensor(uilinlora_alpha, dtype=torch.float32, device=device)
        if adapter_name not in self.active_adapters:
            self.active_adapters.append(adapter_name)

    def get_base_layer(self) -> nn.Module:
        return (
            self.base_layer
            if not hasattr(self.base_layer, "get_base_layer")
            else self.base_layer.get_base_layer()
        )


class Linear(nn.Linear, UILinLoRALayer):
    def __init__(
        self,
        base_layer: nn.Linear,
        adapter_name: str,
        *,
        uilinlora_alpha: float = 1.0,
        uilinlora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        init_uilinlora_weights: bool = True,
        bias: str = "none",
        **kwargs,
    ) -> None:
        # Avoid creating new parameters with possibly int8 dtype
        super(nn.Linear, self).__init__()
        UILinLoRALayer.__init__(self, base_layer, fan_in_fan_out=fan_in_fan_out)
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            uilinlora_alpha=uilinlora_alpha,
            uilinlora_dropout=uilinlora_dropout,
            init_uilinlora_weights=init_uilinlora_weights,
            bias=bias,
        )

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        if adapter not in self.uilinlora_adapter:
            return torch.zeros_like(self.get_base_layer().weight, dtype=torch.float32)

        w = self.uilinlora_adapter[adapter] * self.uilinlora_alpha[adapter]
        return w.T if self.fan_in_fan_out else w


    def get_delta_bias(self, adapter: str) -> Optional[torch.Tensor]:
        return self.uilinlora_bias.get(adapter)

    def merge(self, *, safe_merge: bool = False, adapter_names: Optional[List[str]] = None):
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        base = self.get_base_layer()
        for name in adapter_names:
            if name not in self.uilinlora_adapter:
                continue
            if safe_merge:
                new_w = base.weight.data + self.get_delta_weight(name)
                if not torch.isfinite(new_w).all():
                    raise ValueError(f"NaNs detected while merging adapter {name}")
                base.weight.data = new_w
            else:
                base.weight.data += self.get_delta_weight(name)


            db = self.get_delta_bias(name)
            if db is not None:
                if base.bias is None:
                    base.bias = nn.Parameter(db.clone())
                else:
                    base.bias.data += db
            self.merged_adapters.append(name)

    def unmerge(self):
        if not self.merged:
            warnings.warn("Already unmerged.")
            return
        base = self.get_base_layer()
        while self.merged_adapters:
            name = self.merged_adapters.pop()
            if name not in self.uilinlora_adapter:
                continue
            base.weight.data -= self.get_delta_weight(name)
            db = self.get_delta_bias(name)
            if db is not None and base.bias is not None:
                base.bias.data -= db

    def forward(self, x: torch.Tensor, *args, **kwargs):
        prev_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)

        if self.merged:
            return self.base_layer(x, *args, **kwargs)

        # ── base path ───────────────────────────────────────────────
        result = self.base_layer(x, *args, **kwargs)   # keep x as-is

        # ── adapter path(s) ─────────────────────────────────────────
        for name in self.active_adapters:
            if name not in self.uilinlora_adapter:
                continue

            w = self.get_delta_weight(name)            # FP32, (out,in)

            x_drop = self.uilinlora_dropout[name](x)
            x_fp32 = x_drop.to(w.dtype)                # cast only this copy

            a = F.linear(x_fp32, w)                    # FP32 matmul
            if name in self.uilinlora_bias:
                a = a + self.uilinlora_bias[name]            # FP32

            result = result + a.to(result.dtype)       # back to base dtype

        return result.to(prev_dtype)                   # restore caller’s dtype


    def get_base_layer(self):
        return self.base_layer
    
    def __repr__(self):
        return f"UILinLoRALayer({self.get_base_layer().__repr__()})"

    @property
    def disable_adapters(self):
        return self._disable_adapters

    @disable_adapters.setter
    def disable_adapters(self, value: bool):
        self._disable_adapters = value

    @property
    def active_adapters(self):
        return self._active_adapters

    @active_adapters.setter
    def active_adapters(self, value: List[str]):
        self._active_adapters = value

    @property
    def merged_adapters(self):
        return self._merged_adapters

    @merged_adapters.setter
    def merged_adapters(self, value: List[str]):
        self._merged_adapters = value

