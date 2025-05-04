# Copyright 2024-present the HuggingFace Inc. team.
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
from typing import Optional

import bitsandbytes as bnb
import torch

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils.integrations import dequantize_bnb_weight
from peft.utils.other import transpose

from .layer import VeraLayer


if is_bnb_available():

    class Linear8bitLt(torch.nn.Module, VeraLayer):
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            vera_A,
            vera_B,
            r: int = 0,
            vera_dropout: float = 0.0,
            fan_in_fan_out: bool = False,
            init_weights: bool = True,
            d_initial: float = 0.1,
            **kwargs,
        ) -> None:
            super().__init__()
            VeraLayer.__init__(self, base_layer)
            self.fan_in_fan_out = fan_in_fan_out

            self._active_adapter = adapter_name
            self.update_layer(
                adapter_name,
                vera_A,
                vera_B,
                r,
                vera_dropout=vera_dropout,
                init_weights=init_weights,
                d_initial=d_initial,
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
                if active_adapter not in self.vera_lambda_d.keys():
                    continue

                warnings.warn(
                    "Merge vera module to 8-bit linear may get different generations due to rounding errors."
                )
                vera_data = self.get_delta_weight(active_adapter)

                weight = self.get_base_layer().weight
                state = self.get_base_layer().state
                if state.SCB is None:
                    state.SCB = weight.SCB

                output = dequantize_bnb_weight(weight, state)
                w_data = output.to(vera_data.dtype).to(vera_data.device) + vera_data

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
                if active_adapter not in self.vera_lambda_d.keys():
                    continue
                warnings.warn(
                    "Unmerge vera module to 8-bit linear may get different generations due to rounding errors."
                )
                vera_data = self.get_delta_weight(active_adapter)

                weight = self.get_base_layer().weight
                state = self.get_base_layer().state
                if state.SCB is None:
                    state.SCB = weight.SCB
                output = dequantize_bnb_weight(weight, state=state)

                w_data = output.to(vera_data.dtype).to(vera_data.device) - vera_data

                self.get_base_layer().weight = bnb.nn.Int8Params(
                    w_data.to("cpu"), requires_grad=False, has_fp16_weights=weight.has_fp16_weights
                ).to(weight.device)
                state.reset_grads()

        def get_delta_weight(self, adapter) -> torch.Tensor:
            """
            Compute the delta weight for the given adapter.

            Args:
                adapter (str): The name of the adapter for which the delta weight should be computed.

            Returns:
                torch.Tensor: The computed delta weight for the VeRA adapter.

            Note:
                This method implements the VeRA-specific weight update. Unlike LoRA, VeRA uses shared projection
                matrices (vera_A and vera_B) across all layers, along with per-layer trainable parameters (lambda_d and
                lambda_b).
            """
            # Retrieve shared projection matrices
            vera_A = self.vera_A[adapter]
            vera_B = self.vera_B[adapter]

            # Retrieve per-layer trainable parameters
            device = vera_B.device
            dtype = vera_B.dtype

            # In case users wants to merge the adapter weights that are in
            # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
            # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
            cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

            lambda_d = self.vera_lambda_d[adapter]
            lambda_b = self.vera_lambda_b[adapter]

            if cast_to_fp32:
                vera_A = vera_A.float()
                vera_B = vera_B.float()
                lambda_d = lambda_d.float()
                lambda_b = lambda_b.float()

            sliced_A = vera_A[:, : self.in_features].to(lambda_d.device)
            sliced_B = vera_B[: self.out_features, :].to(lambda_d.device)
            lambda_b = lambda_b.unsqueeze(-1)
            lambda_d = lambda_d.unsqueeze(-1)

            # VeRA-specific computation:
            # 1. Apply lambda_d to the input projection (vera_A)
            # 2. Apply lambda_b to the output projection (vera_B)
            # 3. Compute the outer product of the scaled projections
            output_tensor = transpose((lambda_b * sliced_B) @ (lambda_d * sliced_A), self.fan_in_fan_out)

            if cast_to_fp32:
                output_tensor = output_tensor.to(dtype=dtype)

            return output_tensor

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            """
            Perform the forward pass using the VeRA adapter.

            Args:
                x (torch.Tensor): Input tensor.

            Returns:
                torch.Tensor: Output tensor after applying the VeRA adaptation.

            Note:
                This method implements the VeRA-specific forward pass. It applies the shared projections (vera_A and
                vera_B) along with the per-layer trainable parameters (lambda_d and lambda_b) to compute the adapter
                output.
            """
            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                result = self.base_layer(x, *args, **kwargs)
            elif self.merged:
                result = self.base_layer(x, *args, **kwargs)
            else:
                result = self.base_layer(x, *args, **kwargs)
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.vera_lambda_d.keys():
                        continue

                    lambda_d = self.vera_lambda_d[active_adapter]
                    lambda_b = self.vera_lambda_b[active_adapter]

                    vera_A = self.vera_A[active_adapter]
                    vera_B = self.vera_B[active_adapter]

                    dropout = self.vera_dropout[active_adapter]

                    requires_conversion = not torch.is_autocast_enabled()
                    if requires_conversion:
                        expected_dtype = result.dtype
                        compute_dtype = lambda_d.dtype
                        if x.dtype != compute_dtype:
                            x = x.to(compute_dtype)

                    sliced_A = vera_A[:, : self.in_features].to(x.device)
                    sliced_B = vera_B[: self.out_features, :].to(x.device)

                    x_temp = dropout(x.to(lambda_d.dtype))

                    adapter_output = lambda_b * torch.nn.functional.linear(
                        lambda_d * torch.nn.functional.linear(x_temp, sliced_A), sliced_B
                    )

                    if requires_conversion:
                        adapter_output = adapter_output.to(expected_dtype)

                    result = result + adapter_output

            # Ensure the output tensor has the same dtype as the input tensor
            return result.to(x.dtype)

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "vera." + rep


if is_bnb_4bit_available():

    class Linear4bit(torch.nn.Module, VeraLayer):
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            vera_A,
            vera_B,
            r: int = 0,
            vera_dropout: float = 0.0,
            fan_in_fan_out: bool = False,
            init_weights: bool = True,
            d_initial: float = 0.1,
            **kwargs,
        ) -> None:
            super().__init__()
            VeraLayer.__init__(self, base_layer)
            self.fan_in_fan_out = fan_in_fan_out

            self._active_adapter = adapter_name
            self.update_layer(
                adapter_name,
                vera_A,
                vera_B,
                r,
                vera_dropout=vera_dropout,
                init_weights=init_weights,
                d_initial=d_initial,
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
                if active_adapter not in self.vera_lambda_d.keys():
                    continue

                warnings.warn(
                    "Merge vera module to 4-bit linear may get different generations due to rounding errors."
                )
                vera_data = self.get_delta_weight(active_adapter)

                weight = self.get_base_layer().weight
                kwargs = weight.__dict__
                # torch.compile can introduce attributes preceded by '_', remove them
                kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
                w_data = bnb.functional.dequantize_4bit(weight.data, weight.quant_state) + vera_data

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
                if active_adapter not in self.vera_lambda_d.keys():
                    continue
                warnings.warn(
                    "Unmerge vera module to 4-bit linear may get different generations due to rounding errors."
                )
                vera_data = self.get_delta_weight(active_adapter)

                weight = self.get_base_layer().weight
                kwargs = weight.__dict__
                w_data = bnb.functional.dequantize_4bit(weight.data, weight.quant_state) - vera_data

                self.get_base_layer().weight = bnb.nn.Params4bit(w_data.to("cpu"), requires_grad=False, **kwargs).to(
                    weight.device
                )

        def get_delta_weight(self, adapter) -> torch.Tensor:
            vera_A = self.vera_A[adapter]
            vera_B = self.vera_B[adapter]

            device = vera_B.device
            dtype = vera_B.dtype

            cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

            lambda_d = self.vera_lambda_d[adapter]
            lambda_b = self.vera_lambda_b[adapter]

            if cast_to_fp32:
                vera_A = vera_A.float()
                vera_B = vera_B.float()
                lambda_d = lambda_d.float()
                lambda_b = lambda_b.float()

            sliced_A = vera_A[:, : self.in_features].to(lambda_d.device)
            sliced_B = vera_B[: self.out_features, :].to(lambda_d.device)
            lambda_b = lambda_b.unsqueeze(-1)
            lambda_d = lambda_d.unsqueeze(-1)

            output_tensor = transpose((lambda_b * sliced_B) @ (lambda_d * sliced_A), self.fan_in_fan_out)

            if cast_to_fp32:
                output_tensor = output_tensor.to(dtype=dtype)

            return output_tensor

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                result = self.base_layer(x, *args, **kwargs)
            elif self.merged:
                result = self.base_layer(x, *args, **kwargs)
            else:
                result = self.base_layer(x, *args, **kwargs)
                result = result.clone()
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.vera_lambda_d.keys():
                        continue

                    lambda_d = self.vera_lambda_d[active_adapter]
                    lambda_b = self.vera_lambda_b[active_adapter]

                    vera_A = self.vera_A[active_adapter]
                    vera_B = self.vera_B[active_adapter]

                    dropout = self.vera_dropout[active_adapter]

                    requires_conversion = not torch.is_autocast_enabled()
                    if requires_conversion:
                        expected_dtype = result.dtype
                        compute_dtype = lambda_d.dtype
                        if x.dtype != compute_dtype:
                            x = x.to(compute_dtype)

                    sliced_A = vera_A[:, : self.in_features].to(x.device)
                    sliced_B = vera_B[: self.out_features, :].to(x.device)

                    x_temp = dropout(x.to(lambda_d.dtype))

                    adapter_output = lambda_b * torch.nn.functional.linear(
                        lambda_d * torch.nn.functional.linear(x_temp, sliced_A), sliced_B
                    )

                    if requires_conversion:
                        adapter_output = adapter_output.to(expected_dtype)

                    result = result + adapter_output

            # Ensure the output tensor has the same dtype as the input tensor
            return result.to(x.dtype)

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "vera." + rep
