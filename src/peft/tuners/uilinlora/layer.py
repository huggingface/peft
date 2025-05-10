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

    def __init__(self, base_layer: nn.Module, *, fan_in_fan_out: bool = False):
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.fan_in_fan_out = fan_in_fan_out

        self.uilinlora_sigma = nn.ParameterDict()
        self.uilinlora_D = nn.ParameterDict()
        self.uilinlora_E = nn.ParameterDict()
        self.uilinlora_dropout = nn.ModuleDict()
        self._meta: Dict[str, dict] = {}

        self._active_adapters: List[str] = []
        self._merged_adapters: List[str] = []
        self._disable_adapters: bool = False

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name: str,
        *,
        rank: int,
        scaling_factor: float = 1.0,
        enforce_sv_positive: bool = False,
        uilinlora_dropout: float = 0.0,
        init_uilinlora_weights: bool = True,
        d_initial: float = 1e-7,
        **kwargs,
    ):
        print("update_layer adapter_name", adapter_name)
        print("update_layer self.uilinlora_sigma.keys()", self.uilinlora_sigma.keys())
        if adapter_name in self.uilinlora_sigma.keys():
            return

        base_w = self.get_base_layer().weight

        try:
            base_w = dequantize_module_weight(self.get_base_layer())
        except (TypeError, AttributeError):
            pass

        if base_w.dim() > 2:
            base_w = base_w.view(base_w.size(0), -1)
        elif base_w.dim() == 1:
            base_w = base_w.view(1, -1)

        # Compute SVD and slice the smallest singular vectors
        U, S, Vh = torch.linalg.svd(base_w.float(), full_matrices=False)
        ids = torch.argsort(S)[:rank]
        U_r, V_r = U[:, ids], Vh[ids, :]

        self.register_buffer(f"{adapter_name}_U", U_r, persistent=True)
        self.register_buffer(f"{adapter_name}_V", V_r, persistent=True)

        # Initialize parameters
        self.uilinlora_sigma[adapter_name] = nn.Parameter(torch.full((rank,), d_initial))
        self.uilinlora_D[adapter_name] = nn.Parameter(torch.ones(self.in_features))
        self.uilinlora_E[adapter_name] = nn.Parameter(torch.ones(self.out_features))

        self._meta[adapter_name] = dict(sf=scaling_factor, pos=enforce_sv_positive)

        # Add dropout
        self.uilinlora_dropout.update(nn.ModuleDict({
            adapter_name: nn.Dropout(uilinlora_dropout) if uilinlora_dropout > 0.0 else nn.Identity()
        }))

        # Move newly added parameters to the base layer's device
        self._move_adapter_to_device_of_base_layer(adapter_name)

        # Activate the adapter
        self.set_adapter(self.active_adapters)


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
        **kwargs,
    ) -> None:
        # Initialize nn.Linear with the base layer's parameters
        super(nn.Linear, self).__init__()
        # Initialize UILinLoRALayer
        UILinLoRALayer.__init__(self, base_layer, fan_in_fan_out=fan_in_fan_out)
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            rank=kwargs.pop("rank"),
            scaling_factor=kwargs.pop("scaling_factor"),
            enforce_sv_positive=kwargs.pop("enforce_sv_positive"),
            uilinlora_alpha=uilinlora_alpha,
            uilinlora_dropout=uilinlora_dropout,
            init_uilinlora_weights=init_uilinlora_weights)

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        """
        Return the effective weight matrix of the adapter: ΔW = E * U * Σ * V * D

        This is a proper, differentiable version safe for inspection or merging.
        """
        diag = self.uilinlora_sigma[adapter]
        if self._meta[adapter]["pos"]:
            diag = torch.relu(diag)

        U = getattr(self, f"{adapter}_U")         # (out, r)
        V = getattr(self, f"{adapter}_V")         # (r, in)
        D = self.uilinlora_D[adapter]             # (in,)
        E = self.uilinlora_E[adapter]             # (out,)

        # Broadcast D and E
        VD = V * D.unsqueeze(0)                   # (r, in)
        UE = U * E.unsqueeze(1)                   # (out, r)

        Σ = torch.diag(diag)                      # (r, r)

        core = UE @ Σ @ VD                        # (out, in)
        return self._meta[adapter]["sf"] * core.to(self.get_base_layer().weight.dtype)



    def merge(self, *, safe_merge: bool = False, adapter_names: Optional[List[str]] = None):
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.uilinlora_sigma.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
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

    def unmerge(self):
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.uilinlora_sigma.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        print("forward !!!")
        print("self.active_adapters", self.active_adapters)
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)

        if self.merged:
            return self.base_layer(x, *args, **kwargs)

        result = self.base_layer(x, *args, **kwargs)

        for name in self.active_adapters:
            if name not in self.uilinlora_sigma.keys():
                print("name not in self.uilinlora_sigma.keys()", name)
                continue
            print("name in self.uilinlora_sigma.keys()", name)

            diag = self.uilinlora_sigma[name]
            if self._meta[name]["pos"]:
                diag = torch.relu(diag)

            U = getattr(self, f"{name}_U")               # (out, r)
            V = getattr(self, f"{name}_V")               # (r, in)
            D = self.uilinlora_D[name]                   # (in,)
            E = self.uilinlora_E[name]                   # (out,)

            x_casted = x.to(diag.dtype)
            
            x_proj = F.linear(self.uilinlora_dropout[name](x_casted), V * D.unsqueeze(0))  # (B, r)
            x_proj = x_proj * diag                                                  # (B, r)
            delta = F.linear(x_proj, U * E.unsqueeze(1))                             # (B, out)

            result = result + self._meta[name]["sf"] * delta
        print("result shape", result.shape)

        return result


    def get_base_layer(self):
        return self.base_layer

    def __repr__(self):
        return f"UILinLoRALayer({self.get_base_layer().__repr__()})"





