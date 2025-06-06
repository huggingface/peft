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
from __future__ import annotations

from typing import Any

import torch
from accelerate.utils.imports import is_xpu_available
from torch import nn

from peft.utils.other import transpose

from .dora import DoraConv1dLayer, DoraConv2dLayer, DoraConv3dLayer, DoraEmbeddingLayer, DoraLinearLayer
from .layer import Conv1d, Conv2d, Conv3d, Embedding, Linear, LoraVariant, _ConvNd


class DoraLinearVariant(LoraVariant):
    @staticmethod
    def init(module: Linear, adapter_name: str, **kwargs: Any) -> None:
        if not module.lora_magnitude_vector:
            # first dora layer being added, add lora_magnitude_vector to the list of learnable parameters
            module.adapter_layer_names = module.adapter_layer_names[:] + ("lora_magnitude_vector",)

        dora_layer = DoraLinearLayer(fan_in_fan_out=getattr(module, "fan_in_fan_out", False))
        lora_A = module.lora_A[adapter_name].weight
        lora_B = module.lora_B[adapter_name].weight
        place_on_cpu = module.ephemeral_gpu_offload and (lora_A.device.type == "cpu" or lora_B.device.type == "cpu")
        if module.ephemeral_gpu_offload:
            if lora_A.device.type in ["cuda", "xpu"]:
                lora_B = lora_B.to(lora_A.device)
            else:
                if lora_B.device.type not in ["cuda", "xpu"]:
                    if is_xpu_available():
                        lora_B = lora_B.to("xpu")
                    else:
                        lora_B = lora_B.to("cuda")
                lora_A = lora_A.to(lora_B.device)
        scaling = module.scaling[adapter_name]
        dora_layer.update_layer(
            base_layer=module.get_base_layer(),
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=scaling,
            place_on_cpu=place_on_cpu,
        )
        module.lora_magnitude_vector[adapter_name] = dora_layer

    @staticmethod
    def merge_safe(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        orig_dtype = orig_weight.dtype
        delta_weight = module.get_delta_weight(active_adapter)

        # since delta_weight already includes scaling, set it to 1 here
        weight_norm = (
            module.lora_magnitude_vector[active_adapter]
            .get_weight_norm(orig_weight, transpose(delta_weight, module.fan_in_fan_out), scaling=1)
            .detach()
        )
        # We need to cache weight_norm because it has to be based on the original weights. We
        # cannot calculate it on the fly based on the merged weights when unmerging because its a
        # different value
        module._cache_store(f"{active_adapter}-weight_norm", weight_norm)
        dora_factor = module.lora_magnitude_vector[active_adapter].weight / weight_norm
        dora_factor = transpose(dora_factor.view(-1, 1), module.fan_in_fan_out)
        new_weight = dora_factor * (orig_weight + delta_weight)
        new_weight = new_weight.to(orig_dtype)
        return new_weight

    @staticmethod
    def merge_unsafe(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> None:
        orig_dtype = orig_weight.dtype
        delta_weight = module.get_delta_weight(active_adapter)
        weight_norm = (
            module.lora_magnitude_vector[active_adapter]
            .get_weight_norm(orig_weight, transpose(delta_weight, module.fan_in_fan_out), scaling=1)
            .detach()
        )
        # We need to cache weight_norm because it has to be based on the original weights. We
        # cannot calculate it on the fly based on the merged weights when unmerging because its a
        # different value
        module._cache_store(f"{active_adapter}-weight_norm", weight_norm)
        dora_factor = module.lora_magnitude_vector[active_adapter].weight / weight_norm
        dora_factor = transpose(dora_factor.view(-1, 1), module.fan_in_fan_out)
        new_weight = dora_factor * (orig_weight.data + delta_weight)
        new_weight = new_weight.to(orig_dtype)
        orig_weight.data = new_weight

    @staticmethod
    def unmerge(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        orig_dtype = orig_weight.dtype
        delta_weight = module.get_delta_weight(active_adapter)
        weight_norm = module._cache_pop(f"{active_adapter}-weight_norm")
        dora_factor = module.lora_magnitude_vector[active_adapter].weight / weight_norm
        new_weight = orig_weight.data / dora_factor.view(-1, 1) - delta_weight
        new_weight = new_weight.to(orig_dtype)
        return new_weight

    @staticmethod
    def forward(module: Linear, active_adapter: str, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        lora_A = module.lora_A[active_adapter]
        lora_B = module.lora_B[active_adapter]
        dropout = module.lora_dropout[active_adapter]
        scaling = module.scaling[active_adapter]

        if isinstance(dropout, nn.Identity) or not module.training:
            base_result = result
        else:
            x = dropout(x)
            base_result = None

        result = result + module.lora_magnitude_vector[active_adapter](
            x,
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=scaling,
            base_layer=module.get_base_layer(),
            base_result=base_result,
        )
        return result


class QALoraLinearVariant(LoraVariant):
    @staticmethod
    def init(module: Linear, adapter_name: str, **kwargs: Any) -> None:
        """
        Initializes QALoRA specific parameters for a given adapter.

        Args:
            module (Linear): The linear module to be adapted.
            adapter_name (str): The name of the adapter.
            **kwargs: Additional keyword arguments.
                qalora_group_size (int, optional): The size of groups for pooling. Defaults to 16.
        """
        # Store the qalora_group_size as a module attribute
        # It's expected that qalora_group_size is passed in kwargs from LoraConfig
        qalora_group_size = kwargs.get("qalora_group_size", 16)  # Default to 16 if not specified in kwargs
        if not hasattr(module, "qalora_group_size"):
            module.qalora_group_size = {}
        module.qalora_group_size[adapter_name] = qalora_group_size

        # Store original dimensions for use in get_delta_weight
        if not hasattr(module, "orig_in_features"):
            module.orig_in_features = {}
        module.orig_in_features[adapter_name] = module.in_features

        # Create and store pooling factor for scaling
        if not hasattr(module, "qalora_scaling_factor"):
            module.qalora_scaling_factor = {}

        if module.in_features % qalora_group_size == 0:
            # Paper's pseudocode for QALoRA forward pass scales the pooled input x_pooled by (D_in / L_pseudo),
            # where L_pseudo is the group_size.
            # The get_delta_weight calculation also uses a paper_scaling_factor.
            # For consistency with the paper's forward pass logic,
            # this factor could represent (D_in / group_size) or simply group_size
            # depending on interpretation.
            # Let's assume paper_scaling_factor is related to how x_pooled is scaled before LoRA.
            # The current implementation calculates it as module.in_features / qalora_group_size,
            # which is the number of groups (L).
            # If the intention is to scale by group_size, this should be `qalora_group_size`.
            # For now, keeping the existing logic for qalora_scaling_factor:
            module.qalora_scaling_factor[adapter_name] = module.in_features / qalora_group_size
        else:
            # No special scaling if dimensions don't align
            module.qalora_scaling_factor[adapter_name] = 1.0

    @staticmethod
    def get_delta_weight(module: Linear, active_adapter: str) -> torch.Tensor:
        """
        Computes the QALoRA delta weight.
        The formula (B_std @ (A_std @ P.T)) @ P is used for delta_w_core.

        Args:
            module (Linear): The adapted linear module.
            active_adapter (str): The currently active adapter.

        Returns:
            torch.Tensor: The calculated delta weight.
        """
        if (
            not hasattr(module, "qalora_group_size")
            or active_adapter not in module.qalora_group_size
            or not hasattr(module, "qalora_scaling_factor")
            or active_adapter not in module.qalora_scaling_factor
        ):
            # Fall back to standard LoRA delta weight
            lora_A_weight = module.lora_A[active_adapter].weight
            lora_B_weight = module.lora_B[active_adapter].weight
            return (lora_B_weight @ lora_A_weight) * module.scaling[active_adapter]

        group_size = module.qalora_group_size[active_adapter]
        in_features = module.orig_in_features.get(active_adapter, module.in_features)

        # This is L (number of groups) as calculated in init.
        # Pseudocode suggests scaling by group_size.
        paper_scaling_factor = module.qalora_scaling_factor.get(active_adapter, 1.0)

        if in_features % group_size == 0:
            pooled_dim = in_features // group_size

            lora_A_weight = module.lora_A[active_adapter].weight  # Shape: (rank, in_features)
            lora_B_weight = module.lora_B[active_adapter].weight  # Shape: (out_features, rank)
            device = lora_A_weight.device
            dtype = lora_A_weight.dtype

            # --- Vectorized creation of pooling_matrix ---
            # P has shape (pooled_dim, in_features)
            # Each row i of P has 1.0/group_size for columns i*group_size to (i+1)*group_size - 1
            eye_matrix = torch.eye(pooled_dim, device=device, dtype=dtype)
            pooling_matrix = (1.0 / group_size) * eye_matrix.repeat_interleave(group_size, dim=1)
            # --- End Vectorized pooling_matrix ---

            # A_std shape: (rank, in_features)
            # P.T shape: (in_features, pooled_dim)
            # A_pseudo = A_std @ P.T  shape: (rank, pooled_dim)
            A_pseudo = lora_A_weight @ pooling_matrix.T

            # B_std shape: (out_features, rank)
            # B_std @ A_pseudo shape: (out_features, pooled_dim)
            B_A_pseudo = lora_B_weight @ A_pseudo

            # (B_std @ A_pseudo) @ P shape: (out_features, in_features)
            delta_w_core = B_A_pseudo @ pooling_matrix

            # Apply LoRA scaling 's' and the QALoRA specific scaling
            return delta_w_core * module.scaling[active_adapter] * paper_scaling_factor
        else:
            # Fall back to standard LoRA
            lora_A_weight = module.lora_A[active_adapter].weight
            lora_B_weight = module.lora_B[active_adapter].weight
            return (lora_B_weight @ lora_A_weight) * module.scaling[active_adapter]

    @staticmethod
    def merge_safe(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        orig_dtype = orig_weight.dtype
        delta_weight = QALoraLinearVariant.get_delta_weight(module, active_adapter)
        new_weight = orig_weight + delta_weight
        new_weight = new_weight.to(orig_dtype)
        return new_weight

    @staticmethod
    def merge_unsafe(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> None:
        orig_dtype = orig_weight.dtype
        delta_weight = QALoraLinearVariant.get_delta_weight(module, active_adapter)
        orig_weight.data = (orig_weight.data + delta_weight).to(orig_dtype)

    @staticmethod
    def unmerge(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        orig_dtype = orig_weight.dtype
        delta_weight = QALoraLinearVariant.get_delta_weight(module, active_adapter)
        new_weight = orig_weight - delta_weight
        new_weight = new_weight.to(orig_dtype)
        return new_weight

    @staticmethod
    def forward(module: Linear, active_adapter: str, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        lora_A_layer = module.lora_A[active_adapter]
        lora_B_layer = module.lora_B[active_adapter]

        lora_A_weight = lora_A_layer.weight
        lora_B_weight = lora_B_layer.weight

        dropout = module.lora_dropout[active_adapter]
        lora_scaling_coefficient = module.scaling[active_adapter]

        group_size = module.qalora_group_size.get(active_adapter, 16)
        paper_scaling_factor = module.qalora_scaling_factor.get(active_adapter, 1.0)

        x_dropped = dropout(x) if module.training and not isinstance(dropout, nn.Identity) else x

        orig_shape = x_dropped.shape
        in_features = orig_shape[-1]

        x_2d = x_dropped.reshape(-1, in_features) if len(orig_shape) > 2 else x_dropped

        if in_features % group_size == 0:
            pooled_dim = in_features // group_size

            # 1. Group and pool the input x_2d
            x_grouped = x_2d.reshape(x_2d.shape[0], pooled_dim, group_size)
            x_pooled = x_grouped.mean(dim=-1)

            # 2. Apply QALoRA specific scaling
            x_pooled_scaled = x_pooled * paper_scaling_factor

            # --- Vectorized creation of lora_A_pooled_weight ---
            # lora_A_weight shape: (rank, in_features) = (rank, pooled_dim * group_size)
            # Reshape to (rank, pooled_dim, group_size) then mean over group_size dimension.
            # lora_A_pooled_weight shape: (rank, pooled_dim)
            lora_A_weight_reshaped = lora_A_weight.reshape(lora_A_weight.shape[0], pooled_dim, group_size)
            lora_A_pooled_weight = lora_A_weight_reshaped.mean(dim=2)
            # --- End Vectorized lora_A_pooled_weight ---

            intermediate = x_pooled_scaled @ lora_A_pooled_weight.t()
            delta = intermediate @ lora_B_weight.t()
            delta = delta * lora_scaling_coefficient
        else:
            if isinstance(lora_A_layer, nn.Linear) and isinstance(lora_B_layer, nn.Linear):
                intermediate = lora_A_layer(x_2d)
                delta = lora_B_layer(intermediate) * lora_scaling_coefficient
            else:
                intermediate = x_2d @ lora_A_weight.t()
                delta = intermediate @ lora_B_weight.t() * lora_scaling_coefficient

        if len(orig_shape) > 2:
            delta = delta.reshape(orig_shape[:-1] + (delta.shape[-1],))

        return result + delta


class DoraEmbeddingVariant(DoraLinearVariant):
    @staticmethod
    def init(module: Embedding, adapter_name: str, **kwargs: Any) -> None:
        if module.lora_magnitude_vector is None:
            # first dora layer being added, add lora_magnitude_vector to the list of learnable parameters
            module.adapter_layer_names = module.adapter_layer_names[:] + ("lora_magnitude_vector",)

        dora_layer = DoraEmbeddingLayer(fan_in_fan_out=True)
        lora_embedding_A = module.lora_embedding_A[adapter_name]
        lora_embedding_B = module.lora_embedding_B[adapter_name]
        scaling = module.scaling[adapter_name]
        dora_layer.update_layer(
            base_layer=module.get_base_layer(), lora_A=lora_embedding_A, lora_B=lora_embedding_B, scaling=scaling
        )
        module.lora_magnitude_vector[adapter_name] = dora_layer

    @staticmethod
    def merge_safe(module: Embedding, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        orig_dtype = orig_weight.dtype
        delta_weight = module.get_delta_weight(active_adapter)

        # since delta_weight already includes scaling, set it to 1 here
        weight_norm = (
            module.lora_magnitude_vector[active_adapter]
            .get_weight_norm(orig_weight, delta_weight.T, scaling=1)
            .detach()
        )
        # We need to cache weight_norm because it has to be based on the original weights. We
        # cannot calculate it on the fly based on the merged weights when unmerging because its a
        # different value
        module._cache_store(f"{active_adapter}-weight_norm", weight_norm)
        dora_factor = module.lora_magnitude_vector[active_adapter].weight / weight_norm
        dora_factor = dora_factor.view(1, -1)
        new_weight = dora_factor * (orig_weight + delta_weight)
        new_weight = new_weight.to(orig_dtype)
        return new_weight

    @staticmethod
    def merge_unsafe(module: Embedding, active_adapter: str, orig_weight: torch.Tensor) -> None:
        orig_dtype = orig_weight.dtype
        delta_weight = module.get_delta_weight(active_adapter)
        weight_norm = (
            module.lora_magnitude_vector[active_adapter]
            .get_weight_norm(orig_weight, delta_weight.T, scaling=1)
            .detach()
        )
        # We need to cache weight_norm because it has to be based on the original weights. We
        # cannot calculate it on the fly based on the merged weights when unmerging because its a
        # different value
        module._cache_store(f"{active_adapter}-weight_norm", weight_norm)
        dora_factor = module.lora_magnitude_vector[active_adapter].weight / weight_norm
        dora_factor = dora_factor.view(1, -1)
        new_weight = dora_factor * (orig_weight.data + delta_weight)
        new_weight = new_weight.to(orig_dtype)
        orig_weight.data = new_weight

    @staticmethod
    def unmerge(module: Embedding, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        orig_dtype = orig_weight.dtype
        delta_weight = module.get_delta_weight(active_adapter)
        weight_norm = module._cache_pop(f"{active_adapter}-weight_norm")
        dora_factor = module.lora_magnitude_vector[active_adapter].weight / weight_norm
        new_weight = orig_weight.data / dora_factor.view(1, -1) - delta_weight
        new_weight = new_weight.to(orig_dtype)
        return new_weight

    @staticmethod
    def forward(module: Embedding, active_adapter: str, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        embedding_A = module.lora_embedding_A[active_adapter].T
        embedding_B = module.lora_embedding_B[active_adapter].T
        scaling = module.scaling[active_adapter]

        mag_norm_scale, dora_result = module.lora_magnitude_vector[active_adapter](
            x,
            lora_A=embedding_A,
            lora_B=embedding_B,
            scaling=scaling,
            base_layer=module.get_base_layer(),
            embed_fn=module._embed,
        )
        result = mag_norm_scale * result + dora_result
        return result


class _DoraConvNdVariant(LoraVariant):
    @staticmethod
    def init_convd_variant(module: _ConvNd, adapter_name: str, dora_layer: nn.Module) -> None:
        if module.lora_magnitude_vector is None:
            # first dora layer being added, add lora_magnitude_vector to the list of learnable parameters
            module.adapter_layer_names = module.adapter_layer_names[:] + ("lora_magnitude_vector",)

        lora_A = module.lora_A[adapter_name].weight
        lora_B = module.lora_B[adapter_name].weight
        scaling = module.scaling[adapter_name]
        dora_layer.update_layer(base_layer=module.get_base_layer(), lora_A=lora_A, lora_B=lora_B, scaling=scaling)
        module.lora_magnitude_vector[adapter_name] = dora_layer

    @staticmethod
    def merge_safe(module: _ConvNd, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        orig_dtype = orig_weight.dtype
        delta_weight = module.get_delta_weight(active_adapter)

        # since delta_weight already includes scaling, set it to 1 here
        weight_norm = (
            module.lora_magnitude_vector[active_adapter].get_weight_norm(orig_weight, delta_weight, scaling=1).detach()
        )
        # We need to cache weight_norm because it has to be based on the original weights. We
        # cannot calculate it on the fly based on the merged weights when unmerging because its a
        # different value
        module._cache_store(f"{active_adapter}-weight_norm", weight_norm)
        dora_factor = module.lora_magnitude_vector[active_adapter].weight / weight_norm
        new_weight = dora_factor.view(*module._get_dora_factor_view()) * (orig_weight + delta_weight)
        new_weight = new_weight.to(orig_dtype)
        return new_weight

    @staticmethod
    def merge_unsafe(module: _ConvNd, active_adapter: str, orig_weight: torch.Tensor) -> None:
        orig_dtype = orig_weight.dtype
        delta_weight = module.get_delta_weight(active_adapter)
        # since delta_weight already includes scaling, set it to 1 here
        weight_norm = (
            module.lora_magnitude_vector[active_adapter].get_weight_norm(orig_weight, delta_weight, scaling=1).detach()
        )
        # We need to cache weight_norm because it has to be based on the original weights. We
        # cannot calculate it on the fly based on the merged weights when unmerging because its a
        # different value
        module._cache_store(f"{active_adapter}-weight_norm", weight_norm)
        dora_factor = module.lora_magnitude_vector[active_adapter].weight / weight_norm
        new_weight = dora_factor.view(*module._get_dora_factor_view()) * (orig_weight.data + delta_weight)
        new_weight = new_weight.to(orig_dtype)
        orig_weight.data = new_weight

    @staticmethod
    def unmerge(module: _ConvNd, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        orig_dtype = orig_weight.dtype
        delta_weight = module.get_delta_weight(active_adapter)
        weight_norm = module._cache_pop(f"{active_adapter}-weight_norm")
        dora_factor = module.lora_magnitude_vector[active_adapter].weight / weight_norm
        new_weight = orig_weight.data / dora_factor.view(*module._get_dora_factor_view()) - delta_weight
        new_weight = new_weight.to(orig_dtype)
        return new_weight

    @staticmethod
    def forward(module: _ConvNd, active_adapter: str, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        lora_A = module.lora_A[active_adapter]
        lora_B = module.lora_B[active_adapter]
        dropout = module.lora_dropout[active_adapter]
        scaling = module.scaling[active_adapter]

        if isinstance(dropout, nn.Identity) or not module.training:
            base_result = result
        else:
            x = dropout(x)
            base_result = None

        result = result + module.lora_magnitude_vector[active_adapter](
            x,
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=scaling,
            base_layer=module.get_base_layer(),
            base_result=base_result,
        )
        return result


class DoraConv1dVariant(_DoraConvNdVariant):
    @staticmethod
    def init(module: Conv1d, adapter_name: str, **kwargs: Any) -> None:
        dora_layer = DoraConv1dLayer(fan_in_fan_out=False)
        _DoraConvNdVariant.init_convd_variant(module, adapter_name, dora_layer=dora_layer)


class DoraConv2dVariant(_DoraConvNdVariant):
    @staticmethod
    def init(module: Conv2d, adapter_name: str, **kwargs: Any) -> None:
        dora_layer = DoraConv2dLayer(fan_in_fan_out=False)
        _DoraConvNdVariant.init_convd_variant(module, adapter_name, dora_layer=dora_layer)


class DoraConv3dVariant(_DoraConvNdVariant):
    @staticmethod
    def init(module: Conv3d, adapter_name: str, **kwargs: Any) -> None:
        dora_layer = DoraConv3dLayer(fan_in_fan_out=False)
        _DoraConvNdVariant.init_convd_variant(module, adapter_name, dora_layer=dora_layer)
