# Copyright 2023-present the HuggingFace Inc. team.
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


class VeraLayer(BaseTunerLayer):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = ("vera_lambda_b", "vera_lambda_d")
    other_param_names = ("vera_A", "vera_B")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.vera_dropout = nn.ModuleDict({})

        # For storing vector scale
        self.vera_lambda_b = nn.ParameterDict({})
        self.vera_lambda_d = nn.ParameterDict({})

        # Stores a reference to the vera_A/B BufferDict.
        # Set to `None` otherwise to avoid computation with random weights
        self.vera_A: Optional[BufferDict] = None
        self.vera_B: Optional[BufferDict] = None

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
        vera_A: BufferDict,
        vera_B: BufferDict,
        r,
        vera_dropout,
        init_weights,
        d_initial: float = 0.1,
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        if vera_dropout > 0.0:
            vera_dropout_layer = nn.Dropout(p=vera_dropout)
        else:
            vera_dropout_layer = nn.Identity()

        self.vera_dropout.update(nn.ModuleDict({adapter_name: vera_dropout_layer}))
        # Actual trainable parameters
        self.vera_lambda_b[adapter_name] = nn.Parameter(torch.ones(self.out_features), requires_grad=True)
        self.vera_lambda_d[adapter_name] = nn.Parameter(torch.randn(r), requires_grad=True)

        # non trainable references to vera_A/B buffers
        self.vera_A = vera_A
        self.vera_B = vera_B
        if adapter_name not in vera_A:
            # This means that this is not the first VeRA adapter. We have to add an entry in the dict for this adapter.
            if len(self.vera_A) < 1:
                raise ValueError(
                    "The `vera_A` and `vera_B` buffers are empty. This should not happen. Please report this issue."
                )
            # we can take any of the existing adapter's parameters, as they should all be identical
            vera_A_param = list(self.vera_A.values())[0]
            vera_B_param = list(self.vera_B.values())[0]

            error_tmpl = (
                "{} has a size of {} but {} or greater is required; this probably happened because an additional VeRA "
                "adapter was added after the first one with incompatible shapes."
            )
            # check input size
            if vera_A_param.shape[1] < self.in_features:
                raise ValueError(error_tmpl.format("vera_A", vera_A_param.shape[1], self.in_features))
            # check output size
            if vera_B_param.shape[0] < self.out_features:
                raise ValueError(error_tmpl.format("vera_B", vera_B_param.shape[0], self.out_features))
            # check r
            error_tmpl = (
                "{} has a size of {} but {} or greater is required; this probably happened because an additional VeRA "
                "adapter with a lower rank was added after the first one; loading the adapters "
                "in reverse order may solve this."
            )
            if vera_A_param.shape[0] < self.r[adapter_name]:
                raise ValueError(error_tmpl.format("vera_A", vera_A_param.shape[0], self.r[adapter_name]))
            if vera_B_param.shape[1] < self.r[adapter_name]:
                raise ValueError(error_tmpl.format("vera_B", vera_B_param.shape[1], self.r[adapter_name]))

            self.vera_A[adapter_name] = vera_A_param
            self.vera_B[adapter_name] = vera_B_param

        if init_weights:
            self.reset_vera_parameters(adapter_name, d_initial=d_initial)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_vera_parameters(self, adapter_name, d_initial: float = 0.1):
        if adapter_name in self.vera_lambda_d.keys():
            with torch.no_grad():
                nn.init.zeros_(self.vera_lambda_d[adapter_name]).fill_(d_initial)
                nn.init.zeros_(self.vera_lambda_b[adapter_name])


class Linear(nn.Linear, VeraLayer):
    # Vera implemented in a dense layer
    def __init__(
        self,
        base_layer,
        vera_A: BufferDict,
        vera_B: BufferDict,
        adapter_name: str,
        r: int = 0,
        vera_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_weights: bool = True,
        d_initial: float = 0.1,
        **kwargs,
    ) -> None:
        # this gets the init from nn.Linear's super perspective, i.e. nn.Module.__init__, which should always be called
        super(nn.Linear, self).__init__()
        VeraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, vera_A, vera_B, r, vera_dropout, init_weights, d_initial=d_initial)
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
            if active_adapter in self.vera_lambda_d.keys():
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
            if active_adapter in self.vera_lambda_d.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        vera_A = self.vera_A[adapter]
        vera_B = self.vera_B[adapter]

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
        output_tensor = transpose((lambda_b * sliced_B) @ (lambda_d * sliced_A), self.fan_in_fan_out)

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

        return output_tensor

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
                if active_adapter not in self.vera_lambda_d.keys():
                    continue

                lambda_d = self.vera_lambda_d[active_adapter]
                lambda_b = self.vera_lambda_b[active_adapter]

                vera_A = self.vera_A[active_adapter]
                vera_B = self.vera_B[active_adapter]

                # As adapted layers may have different shapes and VeRA contains a single shared pair of A and B matrices,
                # we initialize these matrices with the largest required size for each dimension.
                # During the forward pass, required submatrices are sliced out from the shared vera_A and vera_B.
                sliced_A = vera_A[:, : self.in_features].to(x.device)
                sliced_B = vera_B[: self.out_features, :].to(x.device)

                dropout = self.vera_dropout[active_adapter]
                x = x.to(lambda_d.dtype)
                result = result + lambda_b * F.linear(lambda_d * F.linear(dropout(x), sliced_A), sliced_B)

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "vera." + rep
