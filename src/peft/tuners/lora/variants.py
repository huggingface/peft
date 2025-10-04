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

import collections
import warnings
from typing import Any, Optional

import torch
from accelerate.utils.imports import is_xpu_available
from torch import nn

from peft.utils.other import transpose

from .arrow import ArrowLoraLinearLayer
from .config import PeftConfig
from .dora import DoraConv1dLayer, DoraConv2dLayer, DoraConv3dLayer, DoraEmbeddingLayer, DoraLinearLayer
from .layer import Conv1d, Conv2d, Conv3d, Embedding, Linear, LoraVariant, _ConvNd


class ArrowLinearVariant(LoraVariant):
    @staticmethod
    def init(module: Linear, adapter_name: str, **kwargs):
        """
        Initialise the ArrowLoraLinearLayer() inside lora_arrow. lora_arrow is nn.ModuleDict(), serving as a container
        for ArrowLoraLinearLayer(). A layer of the base model with LoRA adapter loaded on it will be like:
        ----------------------------------------------------
             (qkv_proj): lora.Linear4bit or lora.Linear(
                (base_layer): Linear4bit or Linear (lora_dropout): ModuleDict( ... ) (lora_A): ModuleDict( ... )
                (lora_B): ModuleDict( ... ) (lora_embedding_A): ParameterDict( ... ) (lora_embedding_B): ParameterDict(
                ... ) (lora_magnitude_vector): ModuleDict( ... ) (lora_arrow): ModuleDict(
                    (arrow_router): ArrowLoraLinearLayer() )
            )
        ----------------------------------------------------

        Args:
            module (Linear): LoRA Layer of the model, containing base_layer, lora_A, lora_B, etc.
            adapter_name (str): name of the adapter that will be put in lora_arrow.
            The adapter_name is "arrow_router" by default, set in create_arrow_model() in ./arrow.py
        """
        # Checking for arrow necessary config
        arrow_config = kwargs.get("arrow_config")
        if arrow_config is None:
            raise ValueError("ArrowLinearVariant.init() did not receive an arrow_config")

        # 1-a) build the ArrowLoRALayer
        arrow_layer = ArrowLoraLinearLayer(
            in_features=module.in_features,
            arrow_config=arrow_config,
        ).to(module.weight.device)

        # 1-b) register a container if it doesn’t exist yet
        if not hasattr(module, "lora_arrow"):
            module.lora_arrow = nn.ModuleDict()

        module.lora_arrow[adapter_name] = arrow_layer

    @staticmethod
    def forward(
        module: Linear,
        *,
        active_adapter: str,
        x: torch.Tensor,
        result: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Parameters mirror those in PEFT’s `LoraVariant.forward`. Called every time the host Linear does a fwd pass.

        build_prototypes() and gen_know_sub() should run only once before routing. Both are implemented in
        ArrowLoraLinearLayer (see ./arrow.py). They are lazily invoked in the forward pass below. Attributes of
        ArrowLoraLinearLayer() class ensure they execute only a single time.

        Args:
            module (Linear): LoRA Layer of the model
            active_adapter (str): name of the arrow route, which should be active to perform arrow.
            x (torch.Tensor): input to the layer
            result (torch.Tensor): output of the base layer.

        Return value:
            output of the base model + delta weight computed by arrow layer.
        """
        arrow = module.lora_arrow[active_adapter]  # ArrowLoraLinearLayer
        # Apply GenKnowSub the 1st time if applcable. By calling arrow/on_adapter_change(),
        # gen_know_sub() is redone for newly added adapters after arrow.create_arrow_model().
        arrow.gen_know_sub(module.lora_A, module.lora_B)
        # lazily build prototypes the 1st time after GenKnowSub. By calling arrow/on_adapter_change(),
        # build_prototypes() is redone for newly added adapters after arrow.create_arrow_model().
        arrow.build_prototypes(module.lora_A, module.lora_B)

        # A forward path of ArrowLoraLinearLayer is called so routing performs.
        # Accept and ignore extra variant kwargs (e.g., 'alora_offsets') for compatibility
        delta = arrow(
            x,
            lora_A=module.lora_A,
            lora_B=module.lora_B,
            dropout=module.lora_dropout[active_adapter],
            scaling=module.scaling,
        )
        return result + delta

    """
    Since Arrow is a Mixture-of-Experts (MoE) approach, merging adapters is not meaningful or even possible: for each
    token, the top-k LoRA experts are dynamically selected and routed. Because of this per-token routing, there is no
    single set of weights that can represent a merged adapter.
    """

    @staticmethod
    def merge_safe(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("Cannot merge an active Arrow router adapter. Remove it first.")

    @staticmethod
    def merge_unsafe(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> None:
        raise RuntimeError("Cannot merge an active Arrow router adapter. Remove it first.")

    @staticmethod
    def unmerge(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("Cannot unmerge an active Arrow router adapter. Remove it first.")


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
    def forward(
        module: Linear,
        active_adapter: str,
        x: torch.Tensor,
        result: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
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
    def forward(
        module: Embedding,
        active_adapter: str,
        x: torch.Tensor,
        result: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
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
    def forward(
        module: _ConvNd,
        active_adapter: str,
        x: torch.Tensor,
        result: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
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


class KasaLinearVariant(LoraVariant):
    @staticmethod
    def init(module: Linear, adapter_name: str, **kwargs: Any) -> None:
        if not module.lora_diag:
            module.adapter_layer_names = module.adapter_layer_names[:] + ("lora_diag",)

        # Initialize lora_diag
        module.lora_diag[adapter_name] = nn.Parameter(torch.randn(module.r[adapter_name]), requires_grad=True)

        # see https://github.com/juyongjiang/KaSA/blob/f85e88c22d0fa4cb8ab2923d7c2bf1bbec152da3/peft/src/peft/tuners/lora/layer.py#L132

        # SVD
        weight = module.get_base_layer().weight
        dtype = weight.dtype
        svd_rank = module.in_features - module.r[adapter_name]
        weight = weight.to(torch.float32)
        U, S, Vh = torch.linalg.svd(weight.data, full_matrices=False)
        U_principle, S_principle, Vh_principle = U[:, :svd_rank], S[:svd_rank], Vh[:svd_rank, :]
        module.get_base_layer().weight.data = (U_principle @ torch.diag(S_principle) @ Vh_principle).to(dtype)

    @staticmethod
    def _get_delta_weight(weight_A, weight_B, lora_diag, scaling, fan_in_fan_out):
        diag = torch.diag(lora_diag)
        delta = weight_B @ diag @ weight_A
        if fan_in_fan_out:
            delta = delta.transpose(0, 1)
        delta = delta * scaling
        return delta
            
    @staticmethod
    def merge_safe(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        delta_weight = module.get_delta_weight(active_adapter)
        return orig_weight + delta_weight

    @staticmethod
    def merge_unsafe(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> None:
        delta_weight = module.get_delta_weight(active_adapter)
        orig_weight.data += delta_weight

    @staticmethod
    def unmerge(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        delta_weight = module.get_delta_weight(active_adapter)
        return orig_weight - delta_weight

    @staticmethod
    def forward(module: Linear, active_adapter: str, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        lora_A = module.lora_A[active_adapter]
        lora_B = module.lora_B[active_adapter]
        dropout = module.lora_dropout[active_adapter]
        scaling = module.scaling[active_adapter]
        diag = torch.diag(module.lora_diag[active_adapter])

        # KASA calculation
        # see https://github.com/juyongjiang/KaSA/blob/f85e88c22d0fa4cb8ab2923d7c2bf1bbec152da3/peft/src/peft/tuners/lora/layer.py#L602C21-L602C110
        lora_output = lora_B(torch.einsum("ijk,kl->ijl", lora_A(x), diag)) * scaling
        return result + lora_output


class QALoraLinearVariant(LoraVariant):
    @staticmethod
    def init(module: Linear, adapter_name: str, **kwargs: Any) -> None:
        """
        Initializes QALoRA specific parameters for a given adapter.

        Args:
            module (Linear): The linear module to be adapted.
            adapter_name (str): The name of the adapter.
            **kwargs: Additional keyword arguments.
                qalora_group_size (int): The size of groups for pooling. This is expected to be passed.
        """
        if "qalora_group_size" not in kwargs:
            raise ValueError(
                "`use_qalora=True` requires 'qalora_group_size' to be provided in kwargs."
                " Please ensure it is passed from the LoraConfig."
            )

        if module.in_features is not None and module.in_features % kwargs["qalora_group_size"] != 0:
            raise ValueError(
                f"`use_qalora=True` requires `module.in_features` ({module.in_features}) to be"
                f"divisible by 'qalora_group_size' ({kwargs['qalora_group_size']})"
            )
        qalora_group_size = kwargs["qalora_group_size"]

        if "qalora_group_size" not in module.other_param_names:
            module.other_param_names = module.other_param_names + ("qalora_group_size",)

        if not hasattr(module, "qalora_group_size"):
            module.qalora_group_size = {}
        module.qalora_group_size[adapter_name] = qalora_group_size

        old_lora_A_layer = module.lora_A[adapter_name]
        r = old_lora_A_layer.out_features
        device = old_lora_A_layer.weight.device
        dtype = old_lora_A_layer.weight.dtype

        new_lora_A_layer = nn.Linear(
            old_lora_A_layer.in_features // module.qalora_group_size[adapter_name],
            r,
            bias=False,
            device=device,
            dtype=dtype,
        )
        module.lora_A[adapter_name] = new_lora_A_layer

    @staticmethod
    def get_delta_weight(module: Linear, active_adapter: str) -> torch.Tensor:
        raise NotImplementedError("QALoRA for GPTQ layers does not support 'get_delta_weight'.")

    @staticmethod
    def merge_safe(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("QALoRA for GPTQ layers does not support 'safe_merge'.")

    @staticmethod
    def merge_unsafe(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> None:
        raise NotImplementedError("QALoRA for GPTQ layers does not support 'merge_unsafe'.")

    @staticmethod
    def unmerge(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("QALoRA for GPTQ layers does not support 'unmerge'.")

    @staticmethod
    def forward(
        module: Linear,
        active_adapter: str,
        x: torch.Tensor,
        result: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        lora_A_weight = module.lora_A[active_adapter].weight
        lora_B_weight = module.lora_B[active_adapter].weight
        dropout = module.lora_dropout[active_adapter]
        scaling = module.scaling[active_adapter]
        group_size = module.qalora_group_size[active_adapter]

        x_dropped = dropout(x) if module.training and not isinstance(dropout, nn.Identity) else x
        orig_shape = x_dropped.shape

        # Reshape to 2D
        if len(orig_shape) > 2:
            x_flat = x_dropped.view(-1, module.in_features)
        else:
            x_flat = x_dropped

        batch_size, in_features = x_flat.shape
        pooled_features = in_features // group_size

        x_pooled = x_flat.view(batch_size, pooled_features, group_size).mean(dim=2)

        x_pooled_scaled = x_pooled * pooled_features

        # LoRA computation
        delta = x_pooled_scaled @ lora_A_weight.t() @ lora_B_weight.t() * scaling

        # Reshape back
        if len(orig_shape) > 2:
            delta = delta.view(orig_shape[:-1] + (delta.size(-1),))

        return result + delta


class ALoraLinearVariant(LoraVariant):
    @staticmethod
    def init(module: Linear, adapter_name: str, **kwargs: Any) -> None:
        pass

    @staticmethod
    def merge_safe(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("aLoRA does not support safe merging.")

    @staticmethod
    def merge_unsafe(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> None:
        raise NotImplementedError("aLoRA does not support merging.")

    @staticmethod
    def unmerge(module: Linear, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("aLoRA does not support unmerging.")

    @staticmethod
    def forward(
        module: Linear,
        active_adapter: str,
        x: torch.Tensor,
        result: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        alora_offsets = kwargs.get("alora_offsets", None)
        lora_A = module.lora_A[active_adapter]
        lora_B = module.lora_B[active_adapter]
        dropout = module.lora_dropout[active_adapter]
        scaling = module.scaling[active_adapter]
        x = x.to(lora_A.weight.dtype)
        result_shape = result.shape
        B = result_shape[0]  # batch
        if len(result_shape) == 3:
            T = result_shape[1]  # tokens
        else:
            T = 1
        D = result_shape[-1]  # dimensions
        Dx = x.shape[-1]
        device = result.device
        if alora_offsets is None:  # use base model only, but ensure 0 gradient
            mask = torch.zeros((B, T), dtype=torch.bool)
        else:
            # If alora_offsets[i] is None, this means that the invocation sequence was not found in the
            # input. As a result, the weights should not be activated anywhere (equivalent to base model).
            # Convert None -> 0 and clip to T
            offsets = torch.tensor(
                [0 if o is None else min(int(o), T) for o in alora_offsets],
                device=device,
                dtype=torch.long,
            )
            # Mask True on the last `offsets[i]` positions for each row i
            pos = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
            mask = pos >= (T - offsets).unsqueeze(1)

        # Flatten for vectorization
        x_flat = x.view(-1, Dx)
        res_flat = result.view(-1, D)
        mask_flat = mask.view(-1)

        # Compute adapter on the selected tokens only
        res_flat[mask_flat] += lora_B(lora_A(dropout(x_flat[mask_flat]))) * scaling
        return result


def calculate_alora_offsets(
    peft_config: PeftConfig, active_adapter: str, input_ids: torch.Tensor, adapter_names: Optional[list[str]] = None
) -> list[int]:
    """
    This is a helper function for Activated LoRA (aLoRA) that searches each input token sequence for the last occurence
    of the appropriate "alora_invocation_tokens" invocation sequence. The calculated alora_offset is the location of
    the *start* of the invocation tokens, counting backward from the end (will therefore always be >=
    len(alora_invocation_tokens). If adapter_names is passed, then each input uses the appropriate invocation sequence
    for the specified adapter for that row. Logic is provided to handle mixed collections of adapters for which not all
    are aLoRAs (e.g. some base model, some LoRA).
    """
    if input_ids is None:
        return []

    batch_size = input_ids.shape[0]
    alora_offsets = [None] * batch_size

    cached_invocation_tensors = {}
    adapters_to_process_indices = collections.defaultdict(list)

    for i in range(batch_size):
        current_adapter_name = adapter_names[i] if adapter_names and i < len(adapter_names) else active_adapter

        if current_adapter_name == "__base__":
            alora_offsets[i] = None
            continue

        if current_adapter_name not in peft_config:
            warnings.warn(f"Adapter '{current_adapter_name}' not found in peft_config. Using base model for row {i}.")
            alora_offsets[i] = None
            continue

        current_peft_config = peft_config[current_adapter_name]

        invocation_tokens = getattr(current_peft_config, "alora_invocation_tokens", None)
        if invocation_tokens is None:
            alora_offsets[i] = None  # Not an aLoRA adapter or wrong type
            continue

        if current_adapter_name not in cached_invocation_tensors:
            cached_invocation_tensors[current_adapter_name] = torch.tensor(
                invocation_tokens, dtype=torch.long, device=input_ids.device
            )

        adapters_to_process_indices[current_adapter_name].append(i)

    for adapter_name_to_process, indices in adapters_to_process_indices.items():
        current_invocation_ids_tensor = cached_invocation_tensors[adapter_name_to_process]
        invocation_len = len(current_invocation_ids_tensor)

        for i in indices:
            sequence = input_ids[i]
            seq_len = len(sequence)
            best_match_start_idx = -1

            possible_starts = (sequence == current_invocation_ids_tensor[0]).nonzero(as_tuple=True)[0]

            for start_idx_tensor in possible_starts:
                idx = start_idx_tensor.item()
                if idx + invocation_len <= seq_len:
                    if torch.equal(sequence[idx : idx + invocation_len], current_invocation_ids_tensor):
                        if idx > best_match_start_idx:
                            best_match_start_idx = idx

            if best_match_start_idx != -1:
                offset_val = seq_len - best_match_start_idx
                alora_offsets[i] = offset_val if offset_val > 0 else None
            else:  # Invocation sequence not found in input
                alora_offsets[i] = None
    return alora_offsets


def is_alora_relevant_in_batch(model: nn.Module, adapter_names: Optional[list[str]] = None):
    """
    Helper function to determine if the current batch has any aLoRA adapters.
    """
    is_alora_relevant = False
    if getattr(model.active_peft_config, "alora_invocation_tokens", None):
        is_alora_relevant = True
    elif adapter_names:
        for name in adapter_names:
            if name == "__base__":
                continue
            config_ = model.peft_config.get(name)
            if config_ and getattr(config_, "alora_invocation_tokens", None):
                is_alora_relevant = True
                break

    return is_alora_relevant


def get_alora_offsets_for_forward(
    model: nn.Module, input_ids: torch.Tensor = None, inputs_embeds: torch.Tensor = None, **kwargs
):
    """
    Wrapper around calculate_alora_offsets, for the .forward of the model. It only calculates alora_offsets if the
    batch contains aLoRA adapters.
    """
    adapter_names_for_offset_calc = kwargs.get("adapter_names", None)
    if not is_alora_relevant_in_batch(model, adapter_names_for_offset_calc):
        # Nothing to compute
        return kwargs
    alora_offsets = kwargs.get("alora_offsets")
    if alora_offsets is None:
        if input_ids is None and inputs_embeds is not None:
            warnings.warn(
                "Cannot calculate aLoRA offsets when only inputs_embeds are provided. Disabling aLoRA for this forward pass."
            )
            kwargs["alora_offsets"] = None
        elif input_ids is not None:
            kwargs["alora_offsets"] = calculate_alora_offsets(
                model.peft_config,
                model.active_adapter,
                input_ids,
                adapter_names=adapter_names_for_offset_calc,
            )
        else:
            kwargs["alora_offsets"] = None
    return kwargs


def get_alora_offsets_for_generate(model: nn.module, *args, **kwargs):
    """
    Wrapper around calculate_alora_offsets, for the .generate of the model. It only calculates alora_offsets if the
    batch contains aLoRA adapters.
    """
    adapter_names_for_offset_calc = kwargs.get("adapter_names")
    if not is_alora_relevant_in_batch(model, adapter_names_for_offset_calc):
        # Nothing to compute
        return kwargs
    alora_offsets_from_kwargs = kwargs.get("alora_offsets")
    if alora_offsets_from_kwargs is None:
        current_input_ids = kwargs.get("input_ids")
        if current_input_ids is None:  # args[0] is usually input_ids
            if args and isinstance(args[0], torch.Tensor):
                current_input_ids = args[0]
            else:
                current_input_ids = None

        if current_input_ids is not None:
            if current_input_ids.ndim == 1:
                current_input_ids = current_input_ids.unsqueeze(0)
            calculated_offsets = calculate_alora_offsets(
                model.peft_config,
                model.active_adapter,
                current_input_ids,
                adapter_names=adapter_names_for_offset_calc,
            )
            kwargs["alora_offsets"] = calculated_offsets

        else:
            warnings.warn(
                "Cannot calculate aLoRA offsets during generate as input_ids are not available. Disabling aLoRA."
            )

            kwargs["alora_offsets"] = None
    return kwargs
