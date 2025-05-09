from __future__ import annotations

import warnings
from typing import Optional

import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils.integrations import dequantize_bnb_weight

from peft.tuners.uilinlora.layer import UILinLoRALayer


if is_bnb_available():

    class Linear8bitLt(torch.nn.Module, UILinLoRALayer):
        # Row-trainable adapter implemented in a dense layer with 8-bit quantization
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            uilinlora_alpha: float = 1.0,
            uilinlora_dropout: float = 0.0,
            fan_in_fan_out: bool = False,
            init_uilinlora_weights: bool = True,
            bias: str = "none",
            **kwargs,
        ) -> None:
            super().__init__()
            UILinLoRALayer.__init__(self, base_layer)
            self.fan_in_fan_out = fan_in_fan_out

            self._active_adapter = adapter_name
            self.update_layer(
                adapter_name,
                uilinlora_alpha,
                uilinlora_dropout,
                init_uilinlora_weights,
                bias,
            )

        def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
            if self.merged:
                warnings.warn(
                    f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                    f"You are now additionally merging {','.join(self.active_adapters)}."
                )

            adapter_names = check_adapters_to_merge(self, adapter_names)
            if not adapter_names:
                return

            for active_adapter in adapter_names:
                if active_adapter not in self.uilinlora_adapter.keys():
                    continue

                warnings.warn(
                    "Merge row-trainable module to 8-bit linear may get different generations due to rounding errors."
                )
                row_data = self.get_delta_weight(active_adapter)

                weight = self.get_base_layer().weight
                state = self.get_base_layer().state
                if state.SCB is None:
                    state.SCB = weight.SCB

                output = dequantize_bnb_weight(weight, state)
                w_data = output.to(row_data.dtype).to(row_data.device) + row_data

                if safe_merge and not torch.isfinite(w_data).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )

                self.get_base_layer().weight = bnb.nn.Int8Params(
                    w_data.to("cpu"), requires_grad=False, has_fp16_weights=weight.has_fp16_weights
                ).to(weight.device)
                state.reset_grads()
                self.merged_adapters.append(active_adapter)

        def unmerge(self) -> None:
            if not self.merged:
                warnings.warn("Already unmerged. Nothing to do")
                return

            while len(self.merged_adapters) > 0:
                active_adapter = self.merged_adapters.pop()
                if active_adapter not in self.uilinlora_adapter.keys():
                    continue
                warnings.warn(
                    "Unmerge row-trainable module to 8-bit linear may get different generations due to rounding errors."
                )
                row_data = self.get_delta_weight(active_adapter)

                weight = self.get_base_layer().weight
                state = self.get_base_layer().state
                if state.SCB is None:
                    state.SCB = weight.SCB
                output = dequantize_bnb_weight(weight, state=state)

                w_data = output.to(row_data.dtype).to(row_data.device) - row_data

                self.get_base_layer().weight = bnb.nn.Int8Params(
                    w_data.to("cpu"), requires_grad=False, has_fp16_weights=weight.has_fp16_weights
                ).to(weight.device)
                state.reset_grads()

        def get_delta_weight(self, adapter: str) -> torch.Tensor:
            if adapter not in self.uilinlora_adapter:
                return torch.zeros_like(self.get_base_layer().weight, dtype=torch.float32)

            diag = self.uilinlora_adapter[adapter]
            if self._meta[adapter]["pos"]:
                diag = torch.relu(diag)

            U = getattr(self, f"{adapter}_U")  # shape: (out_features, rank)
            V = getattr(self, f"{adapter}_V")  # shape: (rank, in_features)
            D = getattr(self, f"{adapter}_D")  # shape: (in_features,)
            E = getattr(self, f"{adapter}_E")  # shape: (out_features,)
            Σ = torch.diag(diag)  # shape: (rank, rank)

            # Create diagonal matrices for D and E
            D = torch.diag_embed(D) if D.ndim == 1 else D  # shape: (in_features, in_features)
            E = torch.diag_embed(E) if E.ndim == 1 else E  # shape: (out_features, out_features)

            # Compute U @ Σ @ V first
            UΣV = U @ Σ @ V  # shape: (out_features, in_features)
            
            # Apply D and E using matrix multiplication
            ΔW = E @ UΣV @ D  # shape: (out_features, in_features)
            return self._meta[adapter]["sf"] * ΔW

        def forward(self, x: torch.Tensor, *args, **kwargs):
            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                return self.base_layer(x, *args, **kwargs)

            if self.merged:
                return self.base_layer(x, *args, **kwargs)

            result = self.base_layer(x, *args, **kwargs)
            for name in self.active_adapters:
                if name not in self.uilinlora_adapter:
                    continue
                w = self.get_delta_weight(name)
                x_fp32 = x.to(w.dtype)
                result = result + F.linear(x_fp32, w)

            return result


        def __repr__(self) -> str:
            rep = super().__repr__()
            return "row." + rep


if is_bnb_4bit_available():

    class Linear4bit(torch.nn.Module, UILinLoRALayer):
        # Row-trainable adapter implemented in a dense layer with 4-bit quantization
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            uilinlora_alpha: float = 1.0,
            uilinlora_dropout: float = 0.0,
            fan_in_fan_out: bool = False,
            init_uilinlora_weights: bool = True,
            bias: str = "none",
            **kwargs,
        ) -> None:
            super().__init__()
            UILinLoRALayer.__init__(self, base_layer)
            self.fan_in_fan_out = fan_in_fan_out

            self._active_adapter = adapter_name
            self.update_layer(
                adapter_name=adapter_name,
                rank=kwargs.pop("rank", 4),
                scaling_factor=kwargs.pop("scaling_factor", 1.0),
                enforce_sv_positive=kwargs.pop("enforce_sv_positive", True),
                uilinlora_alpha=uilinlora_alpha,
                uilinlora_dropout=uilinlora_dropout,
                init_uilinlora_weights=init_uilinlora_weights,
                bias=bias,
                **kwargs,
            )

        def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
            if self.merged:
                warnings.warn(
                    f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                    f"You are now additionally merging {','.join(self.active_adapters)}."
                )

            adapter_names = check_adapters_to_merge(self, adapter_names)
            if not adapter_names:
                return

            for active_adapter in adapter_names:
                if active_adapter not in self.uilinlora_adapter.keys():
                    continue

                warnings.warn(
                    "Merge row-trainable module to 4-bit linear may get different generations due to rounding errors."
                )
                row_data = self.get_delta_weight(active_adapter)

                weight = self.get_base_layer().weight
                kwargs = weight.__dict__
                # torch.compile can introduce attributes preceded by '_', remove them
                kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
                w_data = bnb.functional.dequantize_4bit(weight.data, weight.quant_state) + row_data

                if safe_merge and not torch.isfinite(w_data).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )

                self.get_base_layer().weight = bnb.nn.Params4bit(w_data.to("cpu"), requires_grad=False, **kwargs).to(
                    weight.device
                )
                self.merged_adapters.append(active_adapter)

        def unmerge(self) -> None:
            if not self.merged:
                warnings.warn("Already unmerged. Nothing to do")
                return

            while len(self.merged_adapters) > 0:
                active_adapter = self.merged_adapters.pop()
                if active_adapter not in self.uilinlora_adapter.keys():
                    continue
                warnings.warn(
                    "Unmerge row-trainable module to 4-bit linear may get different generations due to rounding errors."
                )
                row_data = self.get_delta_weight(active_adapter)

                weight = self.get_base_layer().weight
                kwargs = weight.__dict__
                w_data = bnb.functional.dequantize_4bit(weight.data, weight.quant_state) - row_data

                self.get_base_layer().weight = bnb.nn.Params4bit(w_data.to("cpu"), requires_grad=False, **kwargs).to(
                    weight.device
                )

        def get_delta_weight(self, adapter: str) -> torch.Tensor:
            if adapter not in self.uilinlora_adapter:
                return torch.zeros_like(self.get_base_layer().weight, dtype=torch.float32)

            diag = self.uilinlora_adapter[adapter]
            if self._meta[adapter]["pos"]:
                diag = torch.relu(diag)                       # (r,)

            # buffers
            U  = getattr(self, f"{adapter}_U")                # (out, r)
            V  = getattr(self, f"{adapter}_V")                # (r,  in)
            Dv = getattr(self, f"{adapter}_D")                # (in,)
            Ev = getattr(self, f"{adapter}_E")                # (out,)
            Σ  = torch.diag(diag)                             # (r, r)

            # 1. low-rank product
            core = U @ Σ @ V                                  # (out, in)

            # 2. per-column scale
            core = core * Dv                                  # broadcast on columns

            # 3. per-row scale
            core = Ev.unsqueeze(1) * core                     # broadcast on rows

            # 4. respect fan_in_fan_out wrappers
            core = transpose(core, self.fan_in_fan_out)

            return self._meta[adapter]["sf"] * core.to(torch.float32)

        def forward(self, x: torch.Tensor, *args, **kwargs):
            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                return self.base_layer(x, *args, **kwargs)

            if self.merged:
                return self.base_layer(x, *args, **kwargs)

            result = self.base_layer(x, *args, **kwargs)

            for name in self.active_adapters:
                if name not in self.uilinlora_adapter:
                    continue
                w = self.get_delta_weight(name)
                x_fp32 = x.to(w.dtype)
                result = result + F.linear(x_fp32, w)

            return result


        def __repr__(self) -> str:
            rep = super().__repr__()
            return "row." + rep 