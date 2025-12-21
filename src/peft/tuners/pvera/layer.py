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

import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .._buffer_dict import BufferDict


class PVeRALayer(BaseTunerLayer):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = ("pvera_lambda_b", "pvera_lambda_d")
    other_param_names = ("pvera_A", "pvera_B")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.pvera_dropout = nn.ModuleDict({})

        # For storing vector scale
        self.pvera_lambda_b = nn.ParameterDict({})
        self.pvera_lambda_d = nn.ParameterDict({})

        # Stores a reference to the pvera_A/B BufferDict.
        # Set to `None` otherwise to avoid computation with random weights
        self.pvera_A: Optional[BufferDict] = None
        self.pvera_B: Optional[BufferDict] = None

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name,
        pvera_A: BufferDict,
        pvera_B: BufferDict,
        r,
        pvera_dropout,
        init_weights,
        d_initial: float = 0.1,
        inference_mode: bool = False,
        **kwargs,
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        if pvera_dropout > 0.0:
            pvera_dropout_layer = nn.Dropout(p=pvera_dropout)
        else:
            pvera_dropout_layer = nn.Identity()

        self.pvera_dropout.update(nn.ModuleDict({adapter_name: pvera_dropout_layer}))
        # Actual trainable parameters
        self.pvera_lambda_b[adapter_name] = nn.Parameter(torch.ones(self.out_features), requires_grad=True)
        self.pvera_lambda_d[adapter_name] = nn.Parameter(torch.randn(r * 2), requires_grad=True)

        # non trainable references to pvera_A/B buffers
        self.pvera_A = pvera_A
        self.pvera_B = pvera_B
        if adapter_name not in pvera_A:
            # This means that this is not the first PVeRA adapter. We have to add an entry in the dict for this adapter.
            if len(self.pvera_A) < 1:
                raise ValueError(
                    "The `pvera_A` and `pvera_B` buffers are empty. This should not happen. Please report this issue."
                )
            # we can take any of the existing adapter's parameters, as they should all be identical
            pvera_A_param = list(self.pvera_A.values())[0]
            pvera_B_param = list(self.pvera_B.values())[0]

            error_tmpl = (
                "{} has a size of {} but {} or greater is required; this probably happened because an additional PVeRA "
                "adapter was added after the first one with incompatible shapes."
            )
            # check input size
            if pvera_A_param.shape[1] < self.in_features:
                raise ValueError(error_tmpl.format("pvera_A", pvera_A_param.shape[1], self.in_features))
            # check output size
            if pvera_B_param.shape[0] < self.out_features:
                raise ValueError(error_tmpl.format("pvera_B", pvera_B_param.shape[0], self.out_features))
            # check r
            error_tmpl = (
                "{} has a size of {} but {} or greater is required; this probably happened because an additional PVeRA "
                "adapter with a lower rank was added after the first one; loading the adapters "
                "in reverse order may solve this."
            )
            if pvera_A_param.shape[0] < self.r[adapter_name]:
                raise ValueError(error_tmpl.format("pvera_A", pvera_A_param.shape[0], self.r[adapter_name]))
            if pvera_B_param.shape[1] < self.r[adapter_name]:
                raise ValueError(error_tmpl.format("pvera_B", pvera_B_param.shape[1], self.r[adapter_name]))

            self.pvera_A[adapter_name] = pvera_A_param
            self.pvera_B[adapter_name] = pvera_B_param

        if init_weights:
            self.reset_pvera_parameters(adapter_name, d_initial=d_initial)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=inference_mode)

    def reset_pvera_parameters(self, adapter_name, d_initial: float = 0.1):
        if adapter_name in self.pvera_lambda_d.keys():
            with torch.no_grad():
                nn.init.zeros_(self.pvera_lambda_d[adapter_name]).fill_(d_initial)
                nn.init.zeros_(self.pvera_lambda_b[adapter_name])


class Linear(nn.Linear, PVeRALayer):
    # PVeRA implemented in a dense layer
    def __init__(
        self,
        base_layer,
        pvera_A: BufferDict,
        pvera_B: BufferDict,
        adapter_name: str,
        r: int = 0,
        pvera_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_weights: bool = True,
        d_initial: float = 0.1,
        sample_at_inference: bool = False,
        **kwargs,
    ) -> None:
        # this gets the init from nn.Linear's super perspective, i.e. nn.Module.__init__, which should always be called
        super(nn.Linear, self).__init__()
        PVeRALayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self.sample_at_inference = sample_at_inference

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, pvera_A, pvera_B, r, pvera_dropout, init_weights, d_initial=d_initial)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
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
            if active_adapter in self.pvera_lambda_d.keys():
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

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.pvera_lambda_d.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        pvera_A = self.pvera_A[adapter]
        pvera_B = self.pvera_B[adapter]

        device = pvera_B.device
        dtype = pvera_B.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        lambda_d = self.pvera_lambda_d[adapter]
        lambda_b = self.pvera_lambda_b[adapter]

        if cast_to_fp32:
            pvera_A = pvera_A.float()
            pvera_B = pvera_B.float()
            lambda_d = lambda_d.float()
            lambda_b = lambda_b.float()

        sliced_A = pvera_A[:, : self.in_features].to(lambda_d.device)
        sliced_B = pvera_B[: self.out_features, :].to(lambda_d.device)
        lambda_b = lambda_b.unsqueeze(-1)
        lambda_d = lambda_d.unsqueeze(-1)
        output_tensor = transpose((lambda_b * sliced_B) @ (lambda_d * sliced_A), self.fan_in_fan_out)

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

        return output_tensor

    def _reparametrize(self, mu, logvar, sample_at_inference):
        if self.training or (not self.training and sample_at_inference):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        return z

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.pvera_lambda_d.keys():
                    continue

                lambda_d = self.pvera_lambda_d[active_adapter]
                lambda_b = self.pvera_lambda_b[active_adapter]

                pvera_A = self.pvera_A[active_adapter]
                pvera_B = self.pvera_B[active_adapter]

                # As adapted layers may have different shapes and PVeRA contains a single shared pair of A and B matrices,
                # we initialize these matrices with the largest required size for each dimension.
                # During the forward pass, required submatrices are sliced out from the shared pvera_A and pvera_B.
                sliced_A = pvera_A[:, : self.in_features].to(x.device)
                sliced_B = pvera_B[: self.out_features, :].to(x.device)

                dropout = self.pvera_dropout[active_adapter]
                x = x.to(lambda_d.dtype)
                mu, logvar = (lambda_d * F.linear(dropout(x), sliced_A)).chunk(2, dim=-1)
                result = result + lambda_b * F.linear(
                    self._reparametrize(mu, logvar, self.sample_at_inference), sliced_B
                )

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "pvera." + rep
