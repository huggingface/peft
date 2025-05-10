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
from peft.utils.other import transpose

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
                if active_adapter not in self.uilinlora_sigma.keys():
                    continue

                warnings.warn(
                    "Merge row-trainable module to 8-bit linear may get different generations due to rounding errors."
                )
                row_data = self.get_delta_weight(active_adapter).detach()

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
                if active_adapter not in self.uilinlora_sigma.keys():
                    continue
                warnings.warn(
                    "Unmerge row-trainable module to 8-bit linear may get different generations due to rounding errors."
                )
                row_data = self.get_delta_weight(active_adapter).detach()

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
            if adapter not in self.uilinlora_sigma:
                return torch.zeros_like(self.get_base_layer().weight, dtype=torch.float32)

            diag = self.uilinlora_sigma[adapter]
            if self._meta[adapter]["pos"]:
                diag = torch.relu(diag)                       # (r,)

            # buffers
            U  = getattr(self, f"{adapter}_U")       # (out, r)
            V  = getattr(self, f"{adapter}_V")       # (r,  in)
            Dv = self.uilinlora_D[adapter]                   # (in,)
            Ev = self.uilinlora_E[adapter]                   # (out,)
            Σ  = torch.diag(diag)                             # (r, r)

            # 1. low-rank product
            core = U @ Σ @ V                                  # (out, in)

            # 2. per-column scale
            core = core * Dv                                  # broadcast on columns

            # 3. per-row scale
            core = Ev.unsqueeze(1) * core                     # broadcast on rows

            # 4. respect fan_in_fan_out wrappers
            core = transpose(core, self.fan_in_fan_out)

            return self._meta[adapter]["sf"] * core.to(self.get_base_layer().weight.dtype)

        def forward(self, x: torch.Tensor, *args, **kwargs):
            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                return self.base_layer(x, *args, **kwargs)

            if self.merged:
                return self.base_layer(x, *args, **kwargs)

            # Base layer forward first (already fp16 / bf16 on a 4-bit module)
            result = self.base_layer(x, *args, **kwargs)
            compute_dtype = x.dtype            # fp16 or bf16 — whatever the model runs in

            for name in self.active_adapters:
                if name not in self.uilinlora_sigma:
                    continue

                # --- diagonal scaling (σ) --------------------------------
                diag = self.uilinlora_sigma[name].to(compute_dtype)
                if self._meta[name]["pos"]:
                    diag = torch.relu(diag)

                # --- frozen buffers, cast once ---------------------------
                U = getattr(self, f"{name}_U").to(compute_dtype)              # (out, r)
                V = getattr(self, f"{name}_V").to(compute_dtype)              # (r,  in)
                D = self.uilinlora_D[name].to(compute_dtype)                  # (in,)
                E = self.uilinlora_E[name].to(compute_dtype)                  # (out,)

                # fuse per-column & per-row scales
                V_scaled = V * D.unsqueeze(0)                                 # (r, in)
                U_scaled = U * E.unsqueeze(1)                                 # (out, r)

                # --- adapter computation ---------------------------------
                x_drop = self.uilinlora_dropout[name](x)                      # stay in fp16/bf16
                x_proj = F.linear(x_drop, V_scaled) * diag                    # (B, r)
                delta  = F.linear(x_proj, U_scaled)                           # (B, out)

                result = result + self._meta[name]["sf"] * delta

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
                if active_adapter not in self.uilinlora_sigma.keys():
                    continue

                warnings.warn(
                    "Merge row-trainable module to 4-bit linear may get different generations due to rounding errors."
                )
                row_data = self.get_delta_weight(active_adapter).detach()

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
                if active_adapter not in self.uilinlora_sigma.keys():
                    continue
                warnings.warn(
                    "Unmerge row-trainable module to 4-bit linear may get different generations due to rounding errors."
                )
                row_data = self.get_delta_weight(active_adapter).detach()

                weight = self.get_base_layer().weight
                kwargs = weight.__dict__
                w_data = bnb.functional.dequantize_4bit(weight.data, weight.quant_state) - row_data

                self.get_base_layer().weight = bnb.nn.Params4bit(w_data.to("cpu"), requires_grad=False, **kwargs).to(
                    weight.device
                )

        def get_delta_weight(self, adapter: str) -> torch.Tensor:
            diag = self.uilinlora_sigma[adapter]
            if self._meta[adapter]["pos"]:
                diag = torch.relu(diag.clone())

            U = getattr(self, f"{adapter}_U")         # (out, r)
            V = getattr(self, f"{adapter}_V")         # (r, in)
            D = self.uilinlora_D[adapter]             # (in,)
            E = self.uilinlora_E[adapter]             # (out,)

            # Broadcast D and E
            VD = V * D.unsqueeze(0).clone()                   # (r, in)
            UE = U * E.unsqueeze(1).clone()                   # (out, r)

            Σ = torch.diag(diag)                      # (r, r)

            core = UE @ Σ @ VD                        # (out, in)
            return self._meta[adapter]["sf"] * core.to(self.get_base_layer().weight.dtype)

        def forward(self, x: torch.Tensor, *args, **kwargs):
            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                return self.base_layer(x, *args, **kwargs)

            if self.merged:
                return self.base_layer(x, *args, **kwargs)

            # Base layer forward first (already fp16 / bf16 on a 4-bit module)
            result = self.base_layer(x, *args, **kwargs)
            compute_dtype = x.dtype            # fp16 or bf16 — whatever the model runs in

            for name in self.active_adapters:
                if name not in self.uilinlora_sigma:
                    continue

                # --- diagonal scaling (σ) --------------------------------
                diag = self.uilinlora_sigma[name].to(compute_dtype)
                if self._meta[name]["pos"]:
                    diag = torch.relu(diag)

                # --- frozen buffers, cast once ---------------------------
                U = getattr(self, f"{name}_U").to(compute_dtype)              # (out, r)
                V = getattr(self, f"{name}_V").to(compute_dtype)              # (r,  in)
                D = self.uilinlora_D[name].to(compute_dtype)                  # (in,)
                E = self.uilinlora_E[name].to(compute_dtype)                  # (out,)

                # fuse per-column & per-row scales
                V_scaled = V * D.unsqueeze(0)                                 # (r, in)
                U_scaled = U * E.unsqueeze(1)                                 # (out, r)

                # --- adapter computation ---------------------------------
                x_drop = self.uilinlora_dropout[name](x)                      # stay in fp16/bf16
                x_proj = F.linear(x_drop, V_scaled) * diag                    # (B, r)
                delta  = F.linear(x_proj, U_scaled)                           # (B, out)

                result = result + self._meta[name]["sf"] * delta

            return result



        def __repr__(self) -> str:
            rep = super().__repr__()
            return "row." + rep 