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

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.other import transpose


class GraloraLayer(BaseTunerLayer):
    # List all names of layers that may contain adapter weight
    adapter_layer_names = ("gralora_A", "gralora_B", "gralora_A_general", "gralora_B_general")
    other_param_names = ("r", "hybrid_r", "gralora_alpha", "scaling", "gralora_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.gralora_alpha = {}
        self.gralora_k = {}
        self.hybrid_r = {}
        self.scaling = {}
        self.gralora_dropout = nn.ModuleDict({})

        # Set to `None` otherwise to avoid computation with random weight
        self.gralora_A = nn.ParameterDict({})
        self.gralora_B = nn.ParameterDict({})
        self.gralora_A_general = nn.ModuleDict({})
        self.gralora_B_general = nn.ModuleDict({})

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

    def _move_adapter_to_device_of_base_layer(self, adapter_name: str, device: Optional[torch.device] = None) -> None:
        """
        Move the adapter of the given name to the device of the base layer.
        """
        from peft.tuners._buffer_dict import BufferDict

        if device is None:
            # check weight and qweight (for GPTQ)
            for weight_name in ("weight", "qweight"):
                weight = getattr(self.get_base_layer(), weight_name, None)
                if weight is not None:
                    device = weight.device
                    dtype = weight.dtype
                    break
            else:
                # no break encountered: could not determine the device
                return

        # loop through all potential adapter layers and move them to the device of the base layer; be careful to only
        # move this specific adapter to the device, as the other adapters could be on different devices
        # see #1639
        for adapter_layer_name in self.adapter_layer_names + self.other_param_names:
            adapter_layer = getattr(self, adapter_layer_name, None)
            if not isinstance(adapter_layer, (nn.ModuleDict, nn.ParameterDict, BufferDict)):
                continue
            if adapter_name not in adapter_layer:
                continue
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                adapter_layer[adapter_name] = adapter_layer[adapter_name].to(device, dtype=dtype)
            else:
                adapter_layer[adapter_name] = adapter_layer[adapter_name].to(device)

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    @property
    def bias(self) -> torch.Tensor:
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            return base_layer.bias
        elif isinstance(base_layer, Conv1D):
            return base_layer.bias
        else:
            return None

    def update_layer(
        self,
        adapter_name,
        module_name,
        r,
        gralora_alpha,
        gralora_dropout,
        gralora_k: int = 2,
        hybrid_r: int = 0,
        init_weights: bool = True,
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.gralora_alpha[adapter_name] = gralora_alpha
        self.gralora_k[adapter_name] = gralora_k
        self.hybrid_r[adapter_name] = hybrid_r

        if gralora_dropout > 0.0:
            gralora_dropout_layer = nn.Dropout(p=gralora_dropout)
        else:
            gralora_dropout_layer = nn.Identity()

        self.gralora_dropout.update(nn.ModuleDict({adapter_name: gralora_dropout_layer}))

        # Actual trainable parameters
        subblock_in_features = self.in_features // gralora_k
        subblock_out_features = self.out_features // gralora_k

        gralora_r = r - hybrid_r  # gralora_r is the rank allocated to gralora method
        assert gralora_r % gralora_k == 0, f"r should be divisible by gralora_k, but got {r} and {gralora_k}"

        gralora_A = nn.ParameterList()
        gralora_B = nn.ParameterList()
        for _ in range(gralora_k):
            new_A = nn.Parameter(torch.zeros(gralora_r, subblock_in_features))
            new_B = nn.Parameter(torch.zeros(subblock_out_features, gralora_r))
            if init_weights:
                # Initialize to identity: A is random, B is zero
                nn.init.kaiming_uniform_(new_A, a=math.sqrt(5))
                # new_B is already initialized to zeros
            else:
                # Initialize to random: both A and B are random (for testing)
                nn.init.kaiming_uniform_(new_A, a=math.sqrt(5))
                nn.init.kaiming_uniform_(new_B, a=math.sqrt(5))
            gralora_A.append(new_A)
            gralora_B.append(new_B)
        # stack A and B and transpose to get the final shape
        gralora_A = torch.stack(tuple(gralora_A), dim=0)  # [N, gralora_r, in_features//N]
        gralora_A = gralora_A.transpose(1, 2).contiguous()  # [N, in_features//N, gralora_r]

        gralora_B = torch.stack(tuple(gralora_B), dim=0)  # [N, out_features//N, gralora_r]
        gralora_B = gralora_B.transpose(1, 2).contiguous()  # [N, gralora_r, out_features//N]

        if hybrid_r > 0:
            general_gralora_A = nn.Linear(self.in_features, hybrid_r, bias=False)
            general_gralora_B = nn.Linear(hybrid_r, self.out_features, bias=False)
            if init_weights:
                # Initialize to identity: A is random, B is zero
                nn.init.kaiming_uniform_(general_gralora_A.weight, a=math.sqrt(5))
                nn.init.zeros_(general_gralora_B.weight)
            else:
                # Initialize to random: both A and B are random (for testing)
                nn.init.kaiming_uniform_(general_gralora_A.weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(general_gralora_B.weight, a=math.sqrt(5))
        else:
            general_gralora_A = nn.Identity()
            general_gralora_B = nn.Identity()

        self.gralora_A[adapter_name] = gralora_A
        self.gralora_B[adapter_name] = gralora_B
        self.gralora_A_general[adapter_name] = general_gralora_A
        self.gralora_B_general[adapter_name] = general_gralora_B

        self.module_name = module_name

        self.scaling[adapter_name] = gralora_alpha / r
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)


class Linear(nn.Linear, GraloraLayer):
    # Gralora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        module_name,
        r: int = 0,
        gralora_alpha: int = 1,
        gralora_dropout: float = 0.0,
        gralora_k: int = 2,
        hybrid_r: int = 0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        init_weights: bool = True,
        **kwargs,
    ) -> None:
        # this gets the init from nn.Linear's super perspective, i.e. nn.Module.__init__, which should always be called
        super(nn.Linear, self).__init__()
        GraloraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name, module_name, r, gralora_alpha, gralora_dropout, gralora_k, hybrid_r, init_weights
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
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        from peft.tuners.tuners_utils import check_adapters_to_merge

        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.gralora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    orig_weights += delta_weight

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    base_layer.weight.data += delta_weight

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
            if active_adapter in self.gralora_A.keys():
                delta_weight = self.get_delta_weight(active_adapter)
                self.get_base_layer().weight.data -= delta_weight

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for GraLoRA adapter.

        GraLoRA applies block-wise low-rank adaptation with information exchange. This method computes the equivalent
        weight matrix that would be added to the base weight during merge.

        Args:
            adapter (str): The name of the adapter

        Returns:
            torch.Tensor: The delta weight matrix with shape [out_features, in_features]
        """
        gralora_A = self.gralora_A[adapter]  # [N, in_features//N, rank]
        gralora_B = self.gralora_B[adapter]  # [N, rank, out_features//N]
        gralora_A_general = self.gralora_A_general[adapter]
        gralora_B_general = self.gralora_B_general[adapter]

        device = gralora_A.device
        dtype = gralora_A.dtype

        gralora_k = self.gralora_k[adapter]
        hybrid_r = self.hybrid_r[adapter]
        r = self.r[adapter]

        # Handle CPU fp16/bf16 casting
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        if cast_to_fp32:
            gralora_A = gralora_A.float()
            gralora_B = gralora_B.float()

        # Get dimensions
        in_features = self.in_features
        out_features = self.out_features
        subblock_in = in_features // gralora_k
        subblock_out = out_features // gralora_k
        gralora_rank = r - hybrid_r
        subblock_gralora_rank = gralora_rank // gralora_k

        # Simulate the forward pass computation to get equivalent weight matrix
        # We need to compute: W_delta such that W_delta @ x = gralora_forward(x) - base_forward(x)

        # Create an identity matrix for each input dimension and compute output
        # This gives us the columns of the weight matrix
        delta_weight = torch.zeros(out_features, in_features, device=device, dtype=gralora_A.dtype)

        # Process in batches to avoid memory issues
        batch_size = min(256, in_features)
        for start_idx in range(0, in_features, batch_size):
            end_idx = min(start_idx + batch_size, in_features)
            batch_len = end_idx - start_idx

            # Create identity input: [batch_len, in_features]
            x = torch.zeros(batch_len, in_features, device=device, dtype=gralora_A.dtype)
            for i in range(batch_len):
                x[i, start_idx + i] = 1.0

            # Apply GraLoRA transformation (following forward logic)
            # x shape: [batch_len, in_features]
            N = gralora_k

            # Reshape x: [batch_len, N, in_features//N]
            x_reshaped = x.view(batch_len, N, in_features // N)

            # Apply gralora_A: [batch_len, N, in_features//N] @ [N, in_features//N, rank]
            # Result: [batch_len, N, rank]
            temp = torch.einsum("bni, nir -> bnr", x_reshaped, gralora_A)

            # Reshape and permute for information exchange
            # [batch_len, N, rank] -> [batch_len, N, N, subblock_rank]
            temp = temp.view(batch_len, N, N, subblock_gralora_rank)
            # Permute: [batch_len, N, N, subblock_rank] -> [batch_len, N, N, subblock_rank]
            temp = temp.permute(0, 2, 1, 3)
            # Reshape: [batch_len, N, N * subblock_rank]
            temp = temp.reshape(batch_len, N, N * subblock_gralora_rank)

            # Apply gralora_B: [batch_len, N, N*subblock_rank] @ [N, rank, out_features//N]
            # Note: rank here is actually gralora_rank = N * subblock_gralora_rank
            # Result: [batch_len, N, out_features//N]
            output = torch.einsum("bnr, nro -> bno", temp, gralora_B)

            # Reshape to [batch_len, out_features]
            output = output.reshape(batch_len, out_features)

            # Store in delta_weight (transpose because weight is [out, in])
            delta_weight[:, start_idx:end_idx] = output.T

        # Add hybrid LoRA component if present
        if hybrid_r > 0:
            # general_A: [in_features, hybrid_r], general_B: [hybrid_r, out_features]
            weight_A_general = gralora_A_general.weight  # [hybrid_r, in_features]
            weight_B_general = gralora_B_general.weight  # [out_features, hybrid_r]

            if cast_to_fp32:
                weight_A_general = weight_A_general.float()
                weight_B_general = weight_B_general.float()

            # Compute delta for hybrid part: [out_features, hybrid_r] @ [hybrid_r, in_features]
            delta_weight += weight_B_general @ weight_A_general

        # Apply scaling and transpose if needed
        delta_weight = transpose(delta_weight, self.fan_in_fan_out) * self.scaling[adapter]

        # Cast back if needed
        if cast_to_fp32:
            delta_weight = delta_weight.to(dtype=dtype)

        return delta_weight

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
            torch_result_dtype = result.dtype

            # Handle 2D input: [batch, features] -> [batch, 1, features]
            # This is common for MLPs and other non-sequence models
            x_is_2d = x.ndim == 2
            if x_is_2d:
                x = x.unsqueeze(1)  # [B, F] -> [B, 1, F]

            for active_adapter in self.active_adapters:
                if active_adapter not in self.gralora_A.keys():
                    continue
                gralora_A = self.gralora_A[active_adapter]
                gralora_B = self.gralora_B[active_adapter]

                gralora_A_general = self.gralora_A_general[active_adapter]
                gralora_B_general = self.gralora_B_general[active_adapter]

                r = self.r[active_adapter]
                gralora_k = self.gralora_k[active_adapter]
                hybrid_r = self.hybrid_r[active_adapter]

                assert len(gralora_A) == len(gralora_B)

                dropout = self.gralora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                gralora_dtype = gralora_A.dtype
                gralora_rank = r - hybrid_r

                B, L, in_features = x.shape
                N = gralora_k
                subblock_gralora_rank = gralora_rank // N

                output = torch.einsum(
                    "bljr, jro -> bljo",
                    torch.einsum(
                        "blni, nir -> blnr",
                        dropout(x.to(gralora_dtype)).view(B, L, N, in_features // N),
                        gralora_A,
                    )
                    .view(B, L, N, N, subblock_gralora_rank)
                    .permute(0, 1, 3, 2, 4)
                    .reshape(B, L, N, N * subblock_gralora_rank),
                    gralora_B,
                ).reshape(B, L, -1)

                # Squeeze back to 2D if input was 2D
                if x_is_2d:
                    output = output.squeeze(1)  # [B, 1, F] -> [B, F]

                result += scaling * output.to(torch_result_dtype)
                if hybrid_r > 0:
                    hybrid_output = gralora_B_general(gralora_A_general(dropout(x.to(gralora_dtype))))
                    if x_is_2d:
                        hybrid_output = hybrid_output.squeeze(1)
                    result += scaling * hybrid_output.to(torch_result_dtype)

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "gralora." + rep
