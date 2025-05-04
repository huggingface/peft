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

from peft.tuners.diag.layer import DiagLayer


if is_bnb_available():

    class Linear8bitLt(torch.nn.Module, DiagLayer):
        # Row-trainable adapter implemented in a dense layer with 8-bit quantization
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            diag_alpha: float = 1.0,
            diag_dropout: float = 0.0,
            fan_in_fan_out: bool = False,
            init_diag_weights: bool = True,
            bias: str = "none",
            **kwargs,
        ) -> None:
            super().__init__()
            DiagLayer.__init__(self, base_layer)
            self.fan_in_fan_out = fan_in_fan_out

            self._active_adapter = adapter_name
            self.update_layer(
                adapter_name,
                diag_alpha,
                diag_dropout,
                init_diag_weights,
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
                if active_adapter not in self.row_weight.keys():
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
                if active_adapter not in self.row_weight.keys():
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

        def get_delta_weight(self, adapter) -> torch.Tensor:
            if adapter not in self.row_weight.keys():
                return torch.zeros_like(self.get_base_layer().weight)

            row_weight = self.row_weight[adapter]
            if self.fan_in_fan_out:
                row_weight = row_weight.T

            return row_weight

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                result = self.base_layer(x, *args, **kwargs)
            elif self.merged:
                result = self.base_layer(x, *args, **kwargs)
            else:
                result = self.base_layer(x, *args, **kwargs)
                if self.active_adapter in self.row_weight.keys():
                    row_weight = self.row_weight[self.active_adapter]
                    if self.fan_in_fan_out:
                        row_weight = row_weight.T
                    result = result + F.linear(x, row_weight)

            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "row." + rep


if is_bnb_4bit_available():

    class Linear4bit(torch.nn.Module, DiagLayer):
        # Row-trainable adapter implemented in a dense layer with 4-bit quantization
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            diag_alpha: float = 1.0,
            diag_dropout: float = 0.0,
            fan_in_fan_out: bool = False,
            init_diag_weights: bool = True,
            bias: str = "none",
            **kwargs,
        ) -> None:
            super().__init__()
            DiagLayer.__init__(self, base_layer)
            self.fan_in_fan_out = fan_in_fan_out

            self._active_adapter = adapter_name
            self.update_layer(
                adapter_name,
                diag_alpha,
                diag_dropout,
                init_diag_weights,
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
                if active_adapter not in self.row_weight.keys():
                    continue

                warnings.warn(
                    "Merge row-trainable module to 4-bit linear may get different generations due to rounding errors."
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
                if active_adapter not in self.row_weight.keys():
                    continue
                warnings.warn(
                    "Unmerge row-trainable module to 4-bit linear may get different generations due to rounding errors."
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

        def get_delta_weight(self, adapter) -> torch.Tensor:
            if adapter not in self.row_weight.keys():
                return torch.zeros_like(self.get_base_layer().weight)

            row_weight = self.row_weight[adapter]
            if self.fan_in_fan_out:
                row_weight = row_weight.T

            return row_weight

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                result = self.base_layer(x, *args, **kwargs)
            elif self.merged:
                result = self.base_layer(x, *args, **kwargs)
            else:
                result = self.base_layer(x, *args, **kwargs)
                if self.active_adapter in self.row_weight.keys():
                    row_weight = self.row_weight[self.active_adapter]
                    if self.fan_in_fan_out:
                        row_weight = row_weight.T
                    result = result + F.linear(x, row_weight)

            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "row." + rep 