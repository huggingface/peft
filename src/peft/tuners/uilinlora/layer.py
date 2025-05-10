from __future__ import annotations
from typing import Dict, List, Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import dequantize_module_weight

__all__ = ["UILinLoRALayer", "Linear"]

class UILinLoRALayer(BaseTunerLayer):
    adapter_layer_names = ("uilinlora_sigma", "uilinlora_D", "uilinlora_E")
    other_param_names = ("uilinlora_alpha", "uilinlora_dropout", "rank", "scaling_factor", "enforce_sv_positive")

    def __init__(self, base_layer: nn.Module, *, fan_in_fan_out: bool = False, layer_idx: int = 0):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.fan_in_fan_out = fan_in_fan_out
        self.layer_idx = layer_idx

        self.uilinlora_sigma = nn.ParameterDict()
        self.uilinlora_D = nn.ParameterDict()
        self.uilinlora_E = nn.ParameterDict()
        self.uilinlora_dropout = nn.ModuleDict()
        self.uilinlora_bias = nn.ParameterDict()
        self._meta: Dict[str, dict] = {}

        self._active_adapters: List[str] = []
        self._merged_adapters: List[str] = []
        self._disable_adapters: bool = False

    def update_layer(
        self,
        adapter_name: str,
        *,
        rank: int | list[int] = 4,
        scaling_factor: float = 1.0,
        enforce_sv_positive: bool = False,
        uilinlora_dropout: float = 0.0,
        init_uilinlora_weights: bool = True,
        bias: str = "none",
        **kwargs,
    ):
        if adapter_name in self.uilinlora_sigma:
            return

        base_w = self.get_base_layer().weight.detach()
        
        # Try to dequantize if it's a quantized weight
        try:
            base_w = dequantize_module_weight(self.get_base_layer())
        except (TypeError, AttributeError):
            # If dequantization fails, use the original weight
            pass
        
        # Ensure base_w is a 2D matrix
        if base_w.dim() > 2:
            base_w = base_w.view(base_w.size(0), -1)
        elif base_w.dim() == 1:
            base_w = base_w.view(1, -1)

        U, S, Vh = torch.linalg.svd(base_w.float(), full_matrices=False)
        ids = torch.argsort(S)[:rank]
        print(f"ids shape: {ids.shape}")
        U_r, V_r = U[:, ids], Vh[ids, :]

        self.register_buffer(f"{adapter_name}_U", U_r, persistent=True)
        self.register_buffer(f"{adapter_name}_V", V_r, persistent=True)

        diag = torch.full((rank,), 1e-7)
        self.uilinlora_sigma[adapter_name] = nn.Parameter(diag)

        d_in = torch.ones(self.in_features)
        d_out = torch.ones(self.out_features)
        self.uilinlora_D[adapter_name] = nn.Parameter(d_in)
        self.uilinlora_E[adapter_name] = nn.Parameter(d_out)


        self._meta[adapter_name] = dict(sf=scaling_factor, pos=enforce_sv_positive)
        self.active_adapters.append(adapter_name)

        self.uilinlora_dropout[adapter_name] = nn.Dropout(uilinlora_dropout)

        if bias != "none":
            b = torch.zeros(self.out_features)
            self.uilinlora_bias[adapter_name] = nn.Parameter(b)

    def get_base_layer(self) -> nn.Module:
        return self.base_layer if not hasattr(self.base_layer, "get_base_layer") else self.base_layer.get_base_layer()

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
        # Initialize nn.Linear with the base layer's parameters
        nn.Linear.__init__(
            self,
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            bias=base_layer.bias is not None
        )
        # Initialize UILinLoRALayer
        UILinLoRALayer.__init__(self, base_layer, fan_in_fan_out=fan_in_fan_out, layer_idx=kwargs.pop("layer_idx", 0))
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            rank=kwargs.pop("rank"),
            scaling_factor=kwargs.pop("scaling_factor"),
            enforce_sv_positive=kwargs.pop("enforce_sv_positive"),
            uilinlora_alpha=uilinlora_alpha,
            uilinlora_dropout=uilinlora_dropout,
            init_uilinlora_weights=init_uilinlora_weights,
            bias=bias,
        )

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        if adapter not in self.uilinlora_sigma:
            return torch.zeros_like(self.get_base_layer().weight, dtype=torch.float32)

        diag = self.uilinlora_sigma[adapter]
        if self._meta[adapter]["pos"]:
            diag = torch.relu(diag)

        U = getattr(self, f"{adapter}_U")  # shape: (out_features, rank)
        V = getattr(self, f"{adapter}_V")  # shape: (rank, in_features)
        Dv = self.uilinlora_D[adapter]
        Ev = self.uilinlora_E[adapter]
        Σ = torch.diag(diag)  # shape: (rank, rank)

        # 1. low-rank product
        core = U @ Σ @ V  # shape: (out_features, in_features)

        # 2. per-column scale
        core = core * Dv  # broadcast on columns

        # 3. per-row scale
        core = Ev.unsqueeze(1) * core  # broadcast on rows

        return self._meta[adapter]["sf"] * core.to(self.get_base_layer().weight.dtype)

    def get_delta_bias(self, adapter: str) -> Optional[torch.Tensor]:
        return self.uilinlora_bias.get(adapter)

    def merge(self, *, safe_merge: bool = False, adapter_names: Optional[List[str]] = None):
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        base = self.get_base_layer()
        for name in adapter_names:
            if name not in self.uilinlora_sigma:
                continue
            if safe_merge:
                new_w = base.weight.data + self.get_delta_weight(name).detach()
                if not torch.isfinite(new_w).all():
                    raise ValueError(f"NaNs detected while merging adapter {name}")
                base.weight.data = new_w
            else:
                base.weight.data += self.get_delta_weight(name).detach()

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
            if name not in self.uilinlora_sigma:
                continue
            base.weight.data -= self.get_delta_weight(name).detach()
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

        result = self.base_layer(x, *args, **kwargs)

        for name in self.active_adapters:
            if name not in self.uilinlora_sigma:
                continue

            w = self.get_delta_weight(name).detach()
            x_drop = self.uilinlora_dropout[name](x)
            x_fp32 = x_drop.to(w.dtype)

            a = F.linear(x_fp32, w)
            if name in self.uilinlora_bias:
                a = a + self.uilinlora_bias[name]
            a = a.to(result.dtype)

            result = result + a.to(result.dtype)

        return result.to(prev_dtype)

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
