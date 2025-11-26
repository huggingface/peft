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

import math
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .config import HiRAConfig


class HiRALayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names: tuple[str, ...] = ("hira_A", "hira_B", "hira_embedding_A", "hira_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names: tuple[str, ...] = ("r", "hira_dropout")

    def __init__(self, base_layer: nn.Module, ephemeral_gpu_offload: bool = False, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.hira_dropout = nn.ModuleDict({})
        self.hira_A = nn.ParameterDict({})
        self.hira_B = nn.ParameterDict({})
        # For Embedding layer
        self.hira_embedding_A = nn.ParameterDict({})
        self.hira_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self._caches: dict[str, Any] = {}
        self.ephemeral_gpu_offload: bool = ephemeral_gpu_offload
        # flag to enable/disable casting of input to weight dtype during forward call
        self.cast_input_dtype_enabled: bool = True
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv1d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Conv3d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        else:
            # possibly support user provided custom layer types using dynamic dispatch
            if hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
                in_features, out_features = base_layer.in_features, base_layer.out_features
            else:
                in_features, out_features = None, None
            warnings.warn(
                f"Unsupported layer type '{type(base_layer)}' encountered, proceed at your own risk.", UserWarning
            )

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self,
        adapter_name,
        r,
        hira_dropout,
        init_hira_weights,
    ):
        # collect the kwargs
        kwargs = locals().copy()
        del kwargs["self"]

        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        if hira_dropout > 0.0:
            hira_dropout_layer = nn.Dropout(p=hira_dropout)
        else:
            hira_dropout_layer = nn.Identity()

        self.hira_dropout.update(nn.ModuleDict({adapter_name: hira_dropout_layer}))
        # Actual trainable parameters
        self.hira_A.update(nn.ParameterDict({adapter_name: nn.Parameter(torch.randn(r, self.in_features))}))
        self.hira_B.update(nn.ParameterDict({adapter_name: nn.Parameter(torch.randn(self.out_features, r))}))

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        if init_hira_weights:
            self.reset_hira_parameters(adapter_name, init_hira_weights)
        self._move_adapter_to_device_of_base_layer(adapter_name)

        self.set_adapter(self.active_adapters)

    def reset_hira_parameters(self, adapter_name, init_hira_weights):
        if init_hira_weights is False:
            return

        if adapter_name in self.hira_A.keys():
            if init_hira_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(self.hira_A[adapter_name], a=math.sqrt(5))
            elif init_hira_weights.lower() == "gaussian":
                nn.init.normal_(self.hira_A[adapter_name], std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_hira_weights=}")
            nn.init.zeros_(self.hira_B[adapter_name])
        if adapter_name in self.hira_embedding_A.keys():
            # Initialize A to zeros and B the same way as the default for nn.Embedding, see:
            nn.init.zeros_(self.hira_embedding_A[adapter_name])
            nn.init.normal_(self.hira_embedding_B[adapter_name])

    def _cache_store(self, key: str, value: Any) -> None:
        self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key)
        return value

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)

        if self.merged:
            # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
            # adapters. Therefore, it is better to raise an error in this case.
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)

    def _mixed_batch_forward(
        self,
        x: torch.Tensor,
        *args: Any,
        adapter_names: list[str],
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward pass that allows *different* adapters to be used for different examples in the same batch
        (``adapter_names`` must have length == len(x)).

        The base projection is computed once; the HiRA update is then added separately for each sub-batch that shares
        the same adapter.
        """
        # 0. run the expensive base layer once for the whole batch
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        # 1. collect indices for each adapter in the batch
        adapter_to_indices: dict[str, list[int]] = {}
        for idx, name in enumerate(adapter_names):
            adapter_to_indices.setdefault(name, []).append(idx)

        # 2. apply the HiRA update for each (adapter, sub-batch)
        for active_adapter, idxs in adapter_to_indices.items():
            if active_adapter == "__base__":
                continue  # base only → nothing to add
            if active_adapter not in self.hira_A:
                continue  # adapter not initialised

            # --- grab HiRA params & dropout ---
            hira_A = self.hira_A[active_adapter]  # (r, in_dim)
            hira_B = self.hira_B[active_adapter]  # (out_dim, r)
            dropout = self.hira_dropout[active_adapter]

            # --- slice out sub-batch for this adapter ---
            sub_batch = x[idxs]
            sub_batch = self._cast_input_dtype(sub_batch, hira_A.dtype)
            sub_batch = dropout(sub_batch)

            # --- compute element-wise modulated weight: W0 ⊙ (B @ A) ---
            prod_AB = torch.mm(hira_A.T, hira_B.T)  # (in_dim, out_dim)
            assert prod_AB.T.shape == self.get_base_layer().weight.shape
            eff_weight = transpose(self.get_base_layer().weight, self.fan_in_fan_out) * prod_AB.T

            hira_out = F.linear(sub_batch, eff_weight)
            result[idxs] += hira_out.to(torch_result_dtype)

        return result


class Linear(nn.Module, HiRALayer):
    # HiRA implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        hira_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_hira_weights: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        HiRALayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            hira_dropout=hira_dropout,
            init_hira_weights=init_hira_weights,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

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
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.hira_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weight = base_layer.weight.data.clone()
                    orig_dtype = orig_weight.dtype
                    delta_weight = self.get_delta_weight(active_adapter)
                    orig_weight *= 1 + delta_weight.to(orig_dtype)

                    if not torch.isfinite(orig_weight).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weight
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    base_layer.weight.data *= 1 + delta_weight
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
            if active_adapter in self.hira_A.keys():
                weight = self.get_base_layer().weight
                orig_dtype = weight.dtype
                delta_weight = self.get_delta_weight(active_adapter)
                weight.data /= 1 + delta_weight.to(orig_dtype)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.hira_B[adapter].device
        dtype = self.hira_B[adapter].dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.hira_A[adapter]
        weight_B = self.hira_B[adapter]

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()
        output_tensor = transpose((weight_B @ weight_A), self.fan_in_fan_out)
        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.hira_A[adapter].data = weight_A.to(dtype)
            self.hira_B[adapter].data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
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
            torch_result_dtype = result.dtype

            hira_A_keys = self.hira_A.keys()
            for active_adapter in self.active_adapters:
                if active_adapter not in hira_A_keys:
                    continue
                hira_A = self.hira_A[active_adapter]
                hira_B = self.hira_B[active_adapter]
                _prod_AB = torch.mm(hira_A.T, hira_B.T)
                x = self._cast_input_dtype(x, hira_A.dtype)
                dropout = self.hira_dropout[active_adapter]
                dropout_sub = dropout(x)
                hira_result = F.linear(
                    dropout_sub, transpose(self.get_base_layer().weight, self.fan_in_fan_out) * _prod_AB.T
                )
                result = result + hira_result

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "hira." + rep


class Embedding(nn.Module, HiRALayer):
    # HiRA implemented in a Embedding layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        hira_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        init_hira_weights: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        HiRALayer.__init__(self, base_layer)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            hira_dropout=hira_dropout,
            init_hira_weights=init_hira_weights,
        )

    def update_layer(self, adapter_name, r, hira_dropout, init_hira_weights):
        # collect the kwargs
        kwargs = locals().copy()
        del kwargs["self"]

        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        if hira_dropout > 0.0:
            hira_dropout_layer = nn.Dropout(p=hira_dropout)
        else:
            hira_dropout_layer = nn.Identity()

        self.hira_dropout[adapter_name] = hira_dropout_layer
        # Actual trainable parameters
        weight_A = torch.randn((r, self.in_features))
        weight_B = torch.randn((self.out_features, r))
        self.hira_embedding_A[adapter_name] = nn.Parameter(weight_A)
        self.hira_embedding_B[adapter_name] = nn.Parameter(weight_B)
        self.reset_hira_parameters(adapter_name, init_hira_weights)

        # call this before init of the hira variants
        self._move_adapter_to_device_of_base_layer(adapter_name)

        self.set_adapter(self.active_adapters)

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
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.hira_embedding_A.keys():
                base_layer = self.get_base_layer()
                orig_dtype = base_layer.weight.dtype
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weight = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter).to(orig_dtype)
                    orig_weight *= 1 + delta_weight
                    if not torch.isfinite(orig_weight).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    base_layer.weight.data = orig_weight
                else:
                    delta_weight = self.get_delta_weight(active_adapter).to(orig_dtype)
                    base_layer.weight.data *= 1 + delta_weight
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
            orig_dtype = self.get_base_layer().weight.dtype
            if active_adapter in self.hira_embedding_A.keys():
                weight = self.get_base_layer().weight
                weight.data /= 1 + self.get_delta_weight(active_adapter).to(orig_dtype)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.hira_embedding_B[adapter].device
        dtype = self.hira_embedding_A[adapter].dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.hira_embedding_A[adapter]
        weight_B = self.hira_embedding_B[adapter]

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()
        output_tensor = transpose(weight_B @ weight_A, True)

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.hira_embedding_A[adapter] = weight_A.to(dtype)
            self.hira_embedding_B[adapter] = weight_B.to(dtype)

        return output_tensor

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # Compute base embedding once for efficiency
        result = self.base_layer(x, *args, **kwargs)

        # Group batch indices by adapter
        adapter_to_indices = {}
        for idx, adapter_name in enumerate(adapter_names):
            adapter_to_indices.setdefault(adapter_name, []).append(idx)

        # Apply HiRA embedding updates
        for adapter_name, indices in adapter_to_indices.items():
            if adapter_name == "__base__":
                continue
            if adapter_name not in self.hira_embedding_A:
                continue

            embedding_A = self.hira_embedding_A[adapter_name]  # shape: (r, num_embeddings)
            embedding_B = self.hira_embedding_B[adapter_name]  # shape: (embedding_dim, r)

            sub_batch = x[indices]  # shape: (sub_batch_size, sequence_length)

            # Compute the low-rank update: (B @ A)[:, sub_batch].T
            low_rank_update = F.embedding(sub_batch, (embedding_B @ embedding_A).T)

            # Element-wise modulation with base embedding
            base_sub_embedding = result[indices]
            hira_update = base_sub_embedding * low_rank_update

            # Update the result tensor
            result[indices] += hira_update

        return result

    def _embed(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        base_layer = self.get_base_layer()
        return F.embedding(
            input,
            weight,
            padding_idx=base_layer.padding_idx,
            max_norm=base_layer.max_norm,
            norm_type=base_layer.norm_type,
            scale_grad_by_freq=base_layer.scale_grad_by_freq,
            sparse=base_layer.sparse,
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        HiRA forward for Embedding layer. Supports mixed adapters per batch or single adapter.
        """
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        # Adapter disabled or after merge: use base embedding
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)
        if adapter_names is not None:
            return self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        if self.merged:
            return self.base_layer(x, *args, **kwargs)

        # Single adapter active: compute base embedding + HiRA residual
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype
        base_weight = self.get_base_layer().weight  # (num_embeddings, embedding_dim)

        for adapter in self.active_adapters:
            if adapter not in self.hira_embedding_A:
                continue
            # HiRA factors
            hira_A = self.hira_embedding_A[adapter]  # (r, num_embeddings)
            hira_B = self.hira_embedding_B[adapter]  # (embedding_dim, r)

            # Compute modulation matrix: B @ A -> (embedding_dim, num_embeddings)
            mod_matrix = torch.mm(hira_B, hira_A)
            # Element-wise modulated embedding weight
            eff_weight = base_weight * mod_matrix.T  # (num_embeddings, embedding_dim)

            # Compute HiRA residual via embedding lookup
            hira_out = F.embedding(x, eff_weight)
            result = result + hira_out

        return result.to(torch_result_dtype)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "hira." + rep


class _ConvNd(nn.Module, HiRALayer):
    # HiRA implemented in a conv(2,3)d layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        hira_dropout: float = 0.0,
        init_hira_weights: Union[bool, str] = True,
        **kwargs,
    ) -> None:
        super().__init__()
        HiRALayer.__init__(self, base_layer)

        if base_layer.groups > 1:
            warnings.warn("HiRA adapter added to ConvNd layer with groups > 1. Merging is not supported.")

        self._active_adapter = adapter_name
        self._kernel_dim = base_layer.weight.dim()

        self.update_layer(
            adapter_name,
            r,
            hira_dropout=hira_dropout,
            init_hira_weights=init_hira_weights,
        )

    def update_layer(self, adapter_name, r, hira_dropout, init_hira_weights):
        # collect the kwargs
        kwargs = locals().copy()
        del kwargs["self"]

        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        if hira_dropout > 0.0:
            hira_dropout_layer = nn.Dropout(p=hira_dropout)
        else:
            hira_dropout_layer = nn.Identity()

        self.hira_dropout[adapter_name] = hira_dropout_layer
        # Determine base conv parameters
        base = self.get_base_layer()
        conv_cls = type(base)
        in_channels = base.in_channels
        out_channels = base.out_channels
        kernel_size = base.kernel_size
        stride = base.stride
        padding = base.padding
        dilation = getattr(base, "dilation", (1,) * (base.weight.dim() - 2))
        groups = getattr(base, "groups", 1)
        # Spatial dims for B: 1 in each spatial dimension
        spatial_ones = (1,) * (base.weight.dim() - 2)

        # HiRA factor A: conv from in_channels to r
        weight_shape_A = (r, in_channels, *kernel_size)
        weight_shape_B = (out_channels, r, *spatial_ones)

        self.hira_A[adapter_name] = nn.Parameter(torch.randn(weight_shape_A))
        self.hira_B[adapter_name] = nn.Parameter(torch.randn(weight_shape_B))

        # Initialize HiRA parameters (A with Kaiming, B zeros)
        if init_hira_weights:
            # A: same init as base conv weight
            nn.init.kaiming_uniform_(self.hira_A[adapter_name], a=math.sqrt(5))
            # B: initialize to zero
            nn.init.zeros_(self.hira_B[adapter_name])

        # Place adapters on correct device and register
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights inside the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.hira_A.keys():
                base_layer = self.get_base_layer()
                orig_dtype = base_layer.weight.dtype

                if base_layer.groups > 1:
                    # https://github.com/huggingface/peft/pull/2403
                    raise NotImplementedError("Merging is not supported for _ConvNd layers with groups > 1!")

                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weight = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    orig_weight *= 1 + delta_weight.to(orig_dtype)

                    if not torch.isfinite(orig_weight).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weight

                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    base_layer.weight.data *= 1 + delta_weight.to(orig_dtype)

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
            if active_adapter in self.hira_A.keys():
                weight = self.get_base_layer().weight
                orig_dtype = weight.dtype
                delta_weight = self.get_delta_weight(active_adapter)
                weight.data /= 1 + delta_weight.to(orig_dtype)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.hira_B[adapter].device
        dtype = self.hira_A[adapter].dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.hira_A[adapter]
        weight_B = self.hira_B[adapter]

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        if self.get_base_layer().weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            output_tensor = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            output_tensor = self.conv_fn(weight_A.transpose(0, 1), weight_B)

            if self.get_base_layer().groups > 1:
                output_tensor = output_tensor
            else:
                output_tensor = output_tensor.transpose(0, 1)

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.hira_A[adapter].weight.data = weight_A.to(dtype)
            self.hira_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

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
            torch_result_dtype = result.dtype
            base = self.get_base_layer()
            # Determine which convolution function to use
            if isinstance(base, nn.Conv1d):
                conv_fn = F.conv1d
            elif isinstance(base, nn.Conv2d):
                conv_fn = F.conv2d
            elif isinstance(base, nn.Conv3d):
                conv_fn = F.conv3d
            else:
                raise TypeError(f"Unsupported conv layer {type(base)} for HiRA.")
            for active_adapter in self.active_adapters:
                if active_adapter not in self.hira_A.keys():
                    continue
                hira_A = self.hira_A[active_adapter]
                dropout = self.hira_dropout[active_adapter]
                x = self._cast_input_dtype(x, hira_A.dtype)
                x_in = self._cast_input_dtype(x, hira_A.dtype)
                x_drop = dropout(x_in)
                # low-rank factor B@A
                bia = self.get_delta_weight(active_adapter)  # now returns (B@A) tensor
                # element-wise modulate base weight: W0 ⊙ (B@A)
                base_weight = base.weight
                eff_weight = base_weight * bia
                hira_out = conv_fn(
                    x_drop,
                    eff_weight,
                    bias=None,
                    stride=base.stride,
                    padding=base.padding,
                    dilation=getattr(base, "dilation", 1),
                    groups=base.groups,
                )
                result = result + hira_out

            result = result.to(torch_result_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "hira." + rep


class Conv2d(_ConvNd):
    # HiRA implemented in a conv2d layer
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._kernel_dim == 4:
            raise ValueError(f"Conv2d layer kernel must have 4 dimensions, not {self._kernel_dim}")
        self.conv_fn = F.conv2d


class Conv1d(_ConvNd):
    # HiRA implemented in a conv1d layer
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._kernel_dim == 3:
            raise ValueError(f"Conv1d layer kernel must have 3 dimensions, not {self._kernel_dim}")
        self.conv_fn = F.conv1d


class Conv3d(_ConvNd):
    # HiRA implemented in a conv3d layer
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._kernel_dim == 5:
            raise ValueError(f"Conv3d layer kernel must have 5 dimensions, not {self._kernel_dim}")
        self.conv_fn = F.conv3d


def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    hira_config: HiRAConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    """
    Dispatch function for HiRA adapters that avoids LoFT-Q related config.
    """
    # Determine the base module (unwrap BaseTunerLayer)
    if isinstance(target, BaseTunerLayer):
        base = target.get_base_layer()
    else:
        base = target

    # Common HiRA init kwargs
    module_kwargs = {
        "r": hira_config.r,
        "hira_dropout": hira_config.hira_dropout,
        "init_hira_weights": hira_config.init_hira_weights,
    }
    # If fan_in_fan_out present in config, include it
    if hasattr(hira_config, "fan_in_fan_out"):
        module_kwargs["fan_in_fan_out"] = hira_config.fan_in_fan_out

    new_module = None
    # Embedding
    if isinstance(base, nn.Embedding):
        new_module = Embedding(target, adapter_name, **module_kwargs)
    # Conv layers
    elif isinstance(base, nn.Conv2d):
        new_module = Conv2d(target, adapter_name, **module_kwargs)
    elif isinstance(base, nn.Conv3d):
        new_module = Conv3d(target, adapter_name, **module_kwargs)
    elif isinstance(base, nn.Conv1d):
        new_module = Conv1d(target, adapter_name, **module_kwargs)
    # Linear layers
    elif isinstance(base, nn.Linear):
        # Linear always uses fan_in_fan_out=False
        if module_kwargs.get("fan_in_fan_out", False):
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            module_kwargs["fan_in_fan_out"] = False
        new_module = Linear(target, adapter_name, **module_kwargs)
    # HuggingFace Conv1D
    elif isinstance(base, Conv1D):
        # Conv1D expects fan_in_fan_out=True
        if not module_kwargs.get("fan_in_fan_out", False):
            warnings.warn(
                "fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True."
            )
            module_kwargs["fan_in_fan_out"] = True
        new_module = Linear(target, adapter_name, is_target_conv_1d_layer=True, **module_kwargs)

    return new_module
