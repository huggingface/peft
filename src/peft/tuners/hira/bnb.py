# Copyright 2025-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import warnings
from typing import Any, Optional

import bitsandbytes as bnb
import torch
import torch.nn.functional as F

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import dequantize_bnb_weight
from peft.utils.other import transpose

from .layer import HiraLayer


if is_bnb_available():

    class Linear8bitLt(torch.nn.Module, HiraLayer):
        # HiRA implemented in a dense layer
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            r: int = 0,
            hira_dropout: float = 0,
            init_hira_weights: bool = True,
            **kwargs,
        ) -> None:
            super().__init__()
            HiraLayer.__init__(self, base_layer)
            self.fan_in_fan_out = False

            self._active_adapter = adapter_name
            self.update_layer(
                adapter_name,
                r,
                hira_dropout=hira_dropout,
                init_hira_weights=init_hira_weights,
            )

        def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
            """
            Merge the active adapter weights into the base weights

            Args:
                safe_merge (`bool`, *optional*):
                    If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                    before merging the weights. This is useful if you want to check if the merge operation will produce
                    NaNs. Defaults to `False`.
                adapter_names (`list[str]`, *optional*):
                    The list of adapter names that should be merged. If None, all active adapters will be merged.
                    Defaults to `None`.
            """
            adapter_names = check_adapters_to_merge(self, adapter_names)
            if not adapter_names:
                # no adapter to merge
                return

            for active_adapter in adapter_names:
                if active_adapter not in self.hira_A.keys():
                    continue

                warnings.warn(
                    "Merge hira module to 8-bit linear may get different generations due to rounding errors."
                )

                weight = self.get_base_layer().weight
                state = self.get_base_layer().state
                if state.SCB is None:
                    state.SCB = weight.SCB

                # Dequantize the result of identity matrix and int8 weight because bitsandbytes does not support int8
                # dequantization directly
                output = dequantize_bnb_weight(weight, state=state)
                hira_data = self.get_delta_weight(active_adapter)
                hira_data = output.to(hira_data.dtype).to(hira_data.device) * hira_data
                w_data = output.to(hira_data.dtype).to(hira_data.device) + hira_data

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
            """
            This method unmerges all merged adapter layers from the base weights.
            """
            if not self.merged:
                warnings.warn("Already unmerged. Nothing to do.")
                return

            while len(self.merged_adapters) > 0:
                active_adapter = self.merged_adapters.pop()
                if active_adapter not in self.hira_A.keys():
                    continue
                warnings.warn(
                    "Unmerge hira module to 8-bit linear may get different generations due to rounding errors."
                )

                weight = self.get_base_layer().weight
                state = self.get_base_layer().state
                if state.SCB is None:
                    state.SCB = weight.SCB
                output = dequantize_bnb_weight(weight, state=state)

                hira_data = self.get_delta_weight(active_adapter)
                w_data = output.to(hira_data.dtype).to(hira_data.device) / (1 + hira_data)
                self.get_base_layer().weight = bnb.nn.Int8Params(
                    w_data.to("cpu"), requires_grad=False, has_fp16_weights=weight.has_fp16_weights
                ).to(weight.device)

                state.reset_grads()

        def get_delta_weight(self, adapter):
            return transpose(
                self.hira_B[adapter] @ self.hira_A[adapter],
                False,
            )

        def _mixed_batch_forward(
            self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
        ) -> torch.Tensor:
            # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
            # extra argument that allows mixing different adapters in the same batch at inference time.
            result = self.base_layer(x, *args, **kwargs)

            unique_adapters = set(adapter_names)
            sub_batch_indices_list = []
            for adapter in unique_adapters:
                sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

            # dequantitize
            weight = self.get_base_layer().weight
            state = self.get_base_layer().state
            if state.SCB is None:
                state.SCB = weight.SCB
            # Dequantize the result of identity matrix and int8 weight because bitsandbytes does not support int8
            # dequantization directly
            dequant_w = dequantize_bnb_weight(weight, state=state)
            for i, active_adapter in enumerate(unique_adapters):
                if active_adapter == "__base__":
                    continue
                if active_adapter not in self.hira_A.keys():
                    continue

                hira_A = self.hira_A[active_adapter]
                hira_B = self.hira_B[active_adapter]
                dropout = self.hira_dropout[active_adapter]

                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = result.dtype
                    x = self._cast_input_dtype(x, hira_A.dtype)

                # getting the sub-batch, passing it to HiRA layers and updating the corresponding indices of the linear
                # layer output
                sub_batch = x[sub_batch_indices_list[i]]
                sub_batch = dropout(sub_batch)
                prod_AB = torch.mm(hira_A.T, hira_B.T)

                eff_weight = transpose(dequant_w.to(prod_AB.dtype).to(prod_AB.device), self.fan_in_fan_out) * prod_AB.T
                hira_out = F.linear(sub_batch, eff_weight)
                if requires_conversion:
                    hira_out = hira_out.to(expected_dtype)
                result[sub_batch_indices_list[i]] += hira_out

            return result

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            self._check_forward_args(x, *args, **kwargs)
            adapter_names = kwargs.pop("adapter_names", None)

            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                result = self.base_layer(x, *args, **kwargs)
            elif adapter_names is not None:
                result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
            elif self.merged:
                result = self.base_layer(x, *args, **kwargs)
            else:
                # 2) now compute the HiRA “delta” in float32 and add it:
                # dequantize once per forward
                raw_weight = self.get_base_layer().weight
                state = self.get_base_layer().state
                dequant_w = dequantize_bnb_weight(raw_weight, state=state)  # float32 tensor [out, in]
                result = self.base_layer(x, *args, **kwargs)
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.hira_A.keys():
                        continue
                    hira_A = self.hira_A[active_adapter]
                    hira_B = self.hira_B[active_adapter]
                    dropout = self.hira_dropout[active_adapter]

                    requires_conversion = not torch.is_autocast_enabled()
                    if requires_conversion:
                        expected_dtype = result.dtype
                        x = self._cast_input_dtype(x, hira_A.dtype)
                    dropout_sub = dropout(x)
                    # compute Δ = B @ A  → shape (out, in)
                    prod_AB = hira_B @ hira_A
                    # effective weight = W₀ ⊙ Δ
                    eff_w = dequant_w * prod_AB
                    hira_out = F.linear(dropout_sub, eff_w)
                    if requires_conversion:
                        hira_out = hira_out.to(expected_dtype)
                    result = result + hira_out
                    if requires_conversion:
                        result = result.to(expected_dtype)

            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "hira." + rep

    def dispatch_bnb_8bit(target: torch.nn.Module, adapter_name: str, **kwargs):
        new_module = None

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        loaded_in_8bit = kwargs.get("loaded_in_8bit", False)
        if loaded_in_8bit and isinstance(target_base_layer, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = Linear8bitLt(target, adapter_name, **eightbit_kwargs)

        return new_module


if is_bnb_4bit_available():

    class Linear4bit(torch.nn.Module, HiraLayer):
        # HiRA implemented in a dense layer
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            r: int = 0,
            hira_dropout: float = 0.0,
            init_hira_weights: bool = True,
            **kwargs,
        ) -> None:
            super().__init__()
            HiraLayer.__init__(self, base_layer)
            self.fan_in_fan_out = False

            self._active_adapter = adapter_name
            self.update_layer(
                adapter_name,
                r,
                hira_dropout=hira_dropout,
                init_hira_weights=init_hira_weights,
            )

        def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
            """
            Merge the active adapter weights into the base weights

            Args:
                safe_merge (`bool`, *optional*):
                    If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                    before merging the weights. This is useful if you want to check if the merge operation will produce
                    NaNs. Defaults to `False`.
                adapter_names (`list[str]`, *optional*):
                    The list of adapter names that should be merged. If None, all active adapters will be merged.
                    Defaults to `None`.
            """
            adapter_names = check_adapters_to_merge(self, adapter_names)
            if not adapter_names:
                # no adapter to merge
                return

            for active_adapter in adapter_names:
                if active_adapter not in self.hira_A.keys():
                    continue

                warnings.warn(
                    "Merge hira module to 4-bit linear may get different generations due to rounding errors."
                )
                # Refer to https://gist.github.com/ChrisHayduk/1a53463331f52dca205e55982baf9930
                weight = self.get_base_layer().weight
                kwargs = weight.__dict__

                output = dequantize_bnb_weight(weight, state=weight.quant_state)
                hira_data = self.get_delta_weight(active_adapter)
                w_data = output * (1 + hira_data)

                if safe_merge and not torch.isfinite(w_data).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )

                if "bnb_quantized" in kwargs:
                    kwargs["bnb_quantized"] = False
                kwargs["requires_grad"] = False
                kwargs.pop("data", None)
                # torch.compile can introduce attributes preceded by '_', remove them
                kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
                self.get_base_layer().weight = bnb.nn.Params4bit(w_data.to("cpu"), **kwargs).to(weight.device)

                self.merged_adapters.append(active_adapter)

        def unmerge(self) -> None:
            """
            This method unmerges all merged adapter layers from the base weights.
            """
            if not self.merged:
                warnings.warn("Already unmerged. Nothing to do.")
                return

            while len(self.merged_adapters) > 0:
                active_adapter = self.merged_adapters.pop()
                if active_adapter not in self.hira_A.keys():
                    continue
                warnings.warn(
                    "Unmerge hira module to 4-bit linear may get different generations due to rounding errors."
                )

                weight = self.get_base_layer().weight
                kwargs = weight.__dict__
                output = dequantize_bnb_weight(weight, state=weight.quant_state)

                hira_data = self.get_delta_weight(active_adapter)
                w_data = output / (1 + hira_data)

                if "bnb_quantized" in kwargs:
                    kwargs["bnb_quantized"] = False
                kwargs["requires_grad"] = False
                kwargs.pop("data", None)
                self.get_base_layer().weight = bnb.nn.Params4bit(w_data.to("cpu"), **kwargs).to(weight.device)

        def get_delta_weight(self, adapter):
            return transpose(
                self.hira_B[adapter] @ self.hira_A[adapter],
                False,
            )

        def _mixed_batch_forward(
            self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
        ) -> torch.Tensor:
            # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
            # extra argument that allows mixing different adapters in the same batch at inference time.
            result = self.base_layer(x, *args, **kwargs)

            unique_adapters = set(adapter_names)
            sub_batch_indices_list = []
            for adapter in unique_adapters:
                sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])
            weight = self.get_base_layer().weight
            dequant_w = dequantize_bnb_weight(weight, state=weight.quant_state)
            for i, active_adapter in enumerate(unique_adapters):
                if active_adapter == "__base__":
                    continue
                if active_adapter not in self.hira_A.keys():
                    continue

                hira_A = self.hira_A[active_adapter]
                hira_B = self.hira_B[active_adapter]
                dropout = self.hira_dropout[active_adapter]

                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = result.dtype
                    x = self._cast_input_dtype(x, hira_A.dtype)

                # getting the sub-batch, passing it to HiRA layers and updating the corresponding indices of the linear
                # layer output
                sub_batch = x[sub_batch_indices_list[i]]
                prod_AB = torch.mm(hira_A.T, hira_B.T)
                sub_batch = dropout(sub_batch)
                eff_weight = transpose(dequant_w.to(prod_AB.dtype).to(prod_AB.device), self.fan_in_fan_out) * prod_AB.T
                hira_out = F.linear(sub_batch, eff_weight)
                if requires_conversion:
                    hira_out = hira_out.to(expected_dtype)
                result[sub_batch_indices_list[i]] += hira_out

            return result

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            self._check_forward_args(x, *args, **kwargs)
            adapter_names = kwargs.pop("adapter_names", None)

            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                result = self.base_layer(x, *args, **kwargs)
            elif adapter_names is not None:
                result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
            elif self.merged:
                result = self.base_layer(x, *args, **kwargs)
            else:
                result = self.base_layer(x, *args, **kwargs)
                # As per Tim Dettmers, for 4bit, we need to defensively clone here.
                # The reason is that in some cases, an error can occur that backprop
                # does not work on a manipulated view. This issue may be solved with
                # newer PyTorch versions but this would need extensive testing to be
                # sure.
                result = result.clone()
                # dequanted w
                weight = self.get_base_layer().weight
                dequant_w = dequantize_bnb_weight(weight, state=weight.quant_state)
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.hira_A.keys():
                        continue
                    hira_A = self.hira_A[active_adapter]
                    hira_B = self.hira_B[active_adapter]
                    dropout = self.hira_dropout[active_adapter]

                    requires_conversion = not torch.is_autocast_enabled()
                    if requires_conversion:
                        expected_dtype = result.dtype
                        x = self._cast_input_dtype(x, hira_A.dtype)

                    dropout_sub = dropout(x)
                    # compute Δ = B @ A  → shape (out, in)
                    prod_AB = hira_B @ hira_A
                    # effective weight = W₀ ⊙ Δ
                    eff_w = dequant_w * prod_AB
                    hira_out = F.linear(dropout_sub, eff_w)
                    if requires_conversion:
                        hira_out = hira_out.to(expected_dtype)
                    result = result + hira_out
                    if requires_conversion:
                        result = result.to(expected_dtype)

            return result

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "hira." + rep

    def dispatch_bnb_4bit(target: torch.nn.Module, adapter_name: str, **kwargs):
        new_module = None

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        loaded_in_4bit = kwargs.get("loaded_in_4bit", False)
        if loaded_in_4bit and is_bnb_4bit_available() and isinstance(target_base_layer, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target_base_layer.compute_dtype,
                    "compress_statistics": target_base_layer.weight.compress_statistics,
                    "quant_type": target_base_layer.weight.quant_type,
                }
            )
            new_module = Linear4bit(target, adapter_name, **fourbit_kwargs)

        return new_module
