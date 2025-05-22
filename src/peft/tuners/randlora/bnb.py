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
from typing import Optional

import bitsandbytes as bnb
import torch

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils.integrations import dequantize_bnb_weight
from peft.utils.other import transpose

from .layer import RandLoraLayer, UniqueBaseGrad


if is_bnb_available():

    class Linear8bitLt(torch.nn.Module, RandLoraLayer):
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            randlora_A,
            randlora_B,
            r: int = 0,
            randlora_alpha: int = 0,
            randlora_dropout: float = 0.0,
            fan_in_fan_out: bool = False,
            init_weights: bool = True,
            **kwargs,
        ) -> None:
            super().__init__()
            RandLoraLayer.__init__(self, base_layer)
            self.fan_in_fan_out = fan_in_fan_out

            self._active_adapter = adapter_name
            self.update_layer(
                adapter_name,
                randlora_A,
                randlora_B,
                r,
                randlora_alpha=randlora_alpha,
                randlora_dropout=randlora_dropout,
                init_weights=init_weights,
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
                return

            for active_adapter in adapter_names:
                if active_adapter not in self.randlora_lambda.keys():
                    continue

                warnings.warn(
                    "Merge RandLora module to 8-bit linear may get different generations due to rounding errors."
                )
                randlora_data = self.get_delta_weight(active_adapter)

                weight = self.get_base_layer().weight
                state = self.get_base_layer().state
                if state.SCB is None:
                    state.SCB = weight.SCB

                output = dequantize_bnb_weight(weight, state)
                w_data = output.to(randlora_data.dtype).to(randlora_data.device) + randlora_data

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
                warnings.warn("Already unmerged. Nothing to do")
                return

            while len(self.merged_adapters) > 0:
                active_adapter = self.merged_adapters.pop()
                if active_adapter not in self.randlora_lambda.keys():
                    continue
                warnings.warn(
                    "Unmerge randlora module to 8-bit linear may get different generations due to rounding errors."
                )
                randlora_data = self.get_delta_weight(active_adapter)

                weight = self.get_base_layer().weight
                state = self.get_base_layer().state
                if state.SCB is None:
                    state.SCB = weight.SCB
                output = dequantize_bnb_weight(weight, state=state)

                w_data = output.to(randlora_data.dtype).to(randlora_data.device) - randlora_data

                self.get_base_layer().weight = bnb.nn.Int8Params(
                    w_data.to("cpu"), requires_grad=False, has_fp16_weights=weight.has_fp16_weights
                ).to(weight.device)
                state.reset_grads()

        def get_scaled_bases(self, adapter, device=None) -> list[torch.Tensor, torch.Tensor]:
            """
            Performs scaling on the smallest random base (randlora_A) and returns randlora_A and randlora_B in the
            correct order to fit the target layers' dimensions

            Args:
                adapter (str):
                    The name of the adapter for which the delta weight should be computed.
            """

            randlora_A = self.randlora_A[adapter]
            randlora_B = self.randlora_B[adapter]

            if device is None:
                device = randlora_B.device
            dtype = randlora_B.dtype

            # In case users wants to merge the adapter weights that are in
            # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
            # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
            cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

            randlora_lambda = self.randlora_lambda[adapter].to(device)
            randlora_gamma = self.randlora_gamma[adapter].to(device)

            if cast_to_fp32:
                randlora_A = randlora_A.float()
                randlora_B = randlora_B.float()
                randlora_lambda = randlora_lambda.float()
                randlora_gamma = randlora_gamma.float()

            # The trainable parameters are always applied to randlora_A, the smallest basis.
            min_dim, max_dim = min(self.out_features, self.in_features), max(self.out_features, self.in_features)

            # As adapted layers may have different shapes and RandLora contains a single shared pair of A and B matrices,
            # we initialize these matrices with the largest required size for each dimension.
            # During the forward pass, required submatrices are sliced out from the shared randlora_A and randlora_B.
            sliced_A = randlora_A[:, : self.num_bases, :min_dim].to(device)
            sliced_B = randlora_B[:max_dim, : self.num_bases, :].to(device)

            # Flattening the matrices over the rank and number of bases dimensions is more memory efficient
            update_B = sliced_B.flatten(start_dim=1)
            update_A = UniqueBaseGrad.apply(sliced_A, randlora_lambda, randlora_gamma).flatten(end_dim=1)
            if min_dim == self.in_features:
                return update_A, update_B

            return update_B.T, update_A.T

        def get_delta_weight(self, adapter) -> torch.Tensor:
            """
            Compute the delta weight for the given adapter.

            Args:
                adapter (str):
                    The name of the adapter for which the delta weight should be computed.
            """

            update_B, update_A = self.get_scaled_bases(adapter)

            update = update_B @ update_A
            output_tensor = transpose(update, self.fan_in_fan_out)

            scaling = self.scaling[adapter]

            return output_tensor * scaling

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            """
            Perform the forward pass using the RandLora adapter.

            Args:
                x (torch.Tensor): Input tensor.

            Returns:
                torch.Tensor: Output tensor after applying the RandLora adaptation.

            Note:
                This method implements the RandLora-specific forward pass. It applies the shared projections
                (randlora_A and randlora_B) along with the per-layer trainable parameters (lambda and gamma) to compute
                the adapter output.
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
                    if active_adapter not in self.randlora_lambda.keys():
                        continue

                    update_B, update_A = self.get_scaled_bases(active_adapter, device=x.device)

                    requires_conversion = not torch.is_autocast_enabled()
                    if requires_conversion:
                        expected_dtype = result.dtype
                        compute_dtype = update_A.dtype
                        if x.dtype != compute_dtype:
                            x = x.to(compute_dtype)

                    dropout = self.randlora_dropout[active_adapter]
                    x_temp = dropout(x.to(update_A.dtype))

                    adapter_output = torch.nn.functional.linear(torch.nn.functional.linear(x_temp, update_B), update_A)

                    if requires_conversion:
                        adapter_output = adapter_output.to(expected_dtype)

                    scaling = self.scaling[active_adapter]
                    result = result + adapter_output * scaling

            # Ensure the output tensor has the same dtype as the input tensor
            return result.to(x.dtype)

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "randlora." + rep


if is_bnb_4bit_available():

    class Linear4bit(torch.nn.Module, RandLoraLayer):
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            randlora_A,
            randlora_B,
            r: int = 0,
            randlora_alpha: int = 0,
            randlora_dropout: float = 0.0,
            fan_in_fan_out: bool = False,
            init_weights: bool = True,
            **kwargs,
        ) -> None:
            super().__init__()
            RandLoraLayer.__init__(self, base_layer)
            self.fan_in_fan_out = fan_in_fan_out
            self._active_adapter = adapter_name
            self.update_layer(
                adapter_name,
                randlora_A,
                randlora_B,
                r,
                randlora_alpha=randlora_alpha,
                randlora_dropout=randlora_dropout,
                init_weights=init_weights,
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
                return

            for active_adapter in adapter_names:
                if active_adapter not in self.randlora_lambda.keys():
                    continue

                warnings.warn(
                    "Merge RandLora module to 4-bit linear may get different generations due to rounding errors."
                )
                randlora_data = self.get_delta_weight(active_adapter)

                weight = self.get_base_layer().weight
                kwargs = weight.__dict__
                w_data = bnb.functional.dequantize_4bit(weight.data, weight.quant_state) + randlora_data

                if safe_merge and not torch.isfinite(w_data).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )

                self.get_base_layer().weight = bnb.nn.Params4bit(w_data.to("cpu"), requires_grad=False, **kwargs).to(
                    weight.device
                )
                self.merged_adapters.append(active_adapter)

        def unmerge(self) -> None:
            """
            This method unmerges all merged adapter layers from the base weights.
            """
            if not self.merged:
                warnings.warn("Already unmerged. Nothing to do")
                return

            while len(self.merged_adapters) > 0:
                active_adapter = self.merged_adapters.pop()
                if active_adapter not in self.randlora_lambda.keys():
                    continue
                warnings.warn(
                    "Unmerge RandLora module to 4-bit linear may get different generations due to rounding errors."
                )
                randlora_data = self.get_delta_weight(active_adapter)

                weight = self.get_base_layer().weight
                kwargs = weight.__dict__
                w_data = bnb.functional.dequantize_4bit(weight.data, weight.quant_state) - randlora_data

                self.get_base_layer().weight = bnb.nn.Params4bit(w_data.to("cpu"), requires_grad=False, **kwargs).to(
                    weight.device
                )

        def get_scaled_bases(self, adapter, device=None) -> list[torch.Tensor, torch.Tensor]:
            """
            Performs scaling on the smallest random base (randlora_A) and returns randlora_A and randlora_B in the
            correct order to fit the target layers' dimensions

            Args:
                adapter (str):
                    The name of the adapter for which the delta weight should be computed.
            """

            randlora_A = self.randlora_A[adapter]
            randlora_B = self.randlora_B[adapter]
            if device is None:
                device = randlora_B.device
            dtype = randlora_B.dtype

            # In case users wants to merge the adapter weights that are in
            # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
            # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
            cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

            randlora_lambda = self.randlora_lambda[adapter].to(device)
            randlora_gamma = self.randlora_gamma[adapter].to(device)

            if cast_to_fp32:
                randlora_A = randlora_A.float()
                randlora_B = randlora_B.float()
                randlora_lambda = randlora_lambda.float()
                randlora_gamma = randlora_gamma.float()

            # The trainable parameters are always applied to randlora_A, the smallest basis.
            min_dim, max_dim = min(self.out_features, self.in_features), max(self.out_features, self.in_features)

            # As adapted layers may have different shapes and RandLora contains a single shared pair of A and B matrices,
            # we initialize these matrices with the largest required size for each dimension.
            # During the forward pass, required submatrices are sliced out from the shared randlora_A and randlora_B.
            sliced_A = randlora_A[:, : self.num_bases, :min_dim].to(device)
            sliced_B = randlora_B[:max_dim, : self.num_bases, :].to(device)
            # Flattening the matrices over the rank and number of bases dimensions is more memory efficient
            update_B = sliced_B.flatten(start_dim=1)
            update_A = UniqueBaseGrad.apply(sliced_A, randlora_lambda, randlora_gamma).flatten(end_dim=1)
            if min_dim == self.in_features:
                return update_A, update_B

            return update_B.T, update_A.T

        def get_delta_weight(self, adapter) -> torch.Tensor:
            """
            Compute the delta weight for the given adapter.

            Args:
                adapter (str):
                    The name of the adapter for which the delta weight should be computed.
            """
            update_B, update_A = self.get_scaled_bases(adapter)

            update = update_B @ update_A
            output_tensor = transpose(update, self.fan_in_fan_out)

            scaling = self.scaling[adapter]

            return output_tensor * scaling

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
                    if active_adapter not in self.randlora_lambda.keys():
                        continue

                    update_B, update_A = self.get_scaled_bases(active_adapter, device=x.device)

                    requires_conversion = not torch.is_autocast_enabled()
                    if requires_conversion:
                        expected_dtype = result.dtype
                        compute_dtype = update_A.dtype
                        if x.dtype != compute_dtype:
                            x = x.to(compute_dtype)

                    dropout = self.randlora_dropout[active_adapter]
                    x_temp = dropout(x.to(update_A.dtype))

                    adapter_output = torch.nn.functional.linear(torch.nn.functional.linear(x_temp, update_B), update_A)

                    if requires_conversion:
                        adapter_output = adapter_output.to(expected_dtype)

                    scaling = self.scaling[active_adapter]
                    result = result + adapter_output * scaling

            # Ensure the output tensor has the same dtype as the input tensor
            return result.to(x.dtype)

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "randlora." + rep
