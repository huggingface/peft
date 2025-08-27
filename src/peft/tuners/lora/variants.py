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

import os
from copy import deepcopy
from typing import Any

import torch
from accelerate.utils.imports import is_xpu_available
from torch import nn
from transformers import PreTrainedModel

from peft.utils.other import transpose

from .arrow import GKS_ADAPTER_PREFIX, TASK_ADAPTER_PREFIX, ArrowLoraLinearLayer
from .config import ArrowConfig
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
            The adapter_name is "arrow_router" by default, set in create_arrow_model() in ./variants.py
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
        # gen_know_sub() is done again for new added adapters after variants/create_arrow_model().
        arrow.gen_know_sub(module.lora_A, module.lora_B)
        # lazily build prototypes the 1st time after GenKnowSub. By calling arrow/on_adapter_change(),
        # build_prototypes() is done again for new added adapters after variants/create_arrow_model().
        arrow.build_prototypes(module.lora_A, module.lora_B)

        # A forward path of ArrowLoraLinearLayer is called so routing performs.
        delta = arrow(
            x,
            lora_A=module.lora_A,
            lora_B=module.lora_B,
            dropout=module.lora_dropout[active_adapter],
            scaling=module.scaling,
        )
        return result + delta


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
    def forward(module: Linear, active_adapter: str, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
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


def check_loaded_lora_compatibility_arrow(model, adapter_names: list[str]):
    """
    After loading all adapters into `model`, check they share:
      - the same LoRA rank (r)
      - identical weight shapes
      - identical sets of target_modules
    Returns the sorted list of target module names (e.g. ["qkv_proj","o_proj"]).
    """
    reference = None  # will hold {'r':…, 'shapes':(Ashape,Bshape), 'modules':set([...])}

    for name in adapter_names:
        curr_modules = set()
        curr_r = None
        curr_shapes = None

        # Use named_modules to get both module and its name in the model hierarchy
        for full_name, module in model.named_modules():
            if hasattr(module, "lora_A") and name in module.lora_A:
                A = module.lora_A[name].weight
                B = module.lora_B[name].weight

                # extract the *attribute* name (last component of the path)
                mod_name = full_name.split(".")[-1]

                curr_modules.add(mod_name)
                curr_r = A.shape[1]
                curr_shapes = (A.shape, B.shape)

        if reference is None:
            reference = {"r": curr_r, "shapes": curr_shapes, "modules": curr_modules}
        else:
            if curr_r != reference["r"]:
                raise ValueError(f"[{name}] rank mismatch: {curr_r} != {reference['r']}")
            if curr_shapes != reference["shapes"]:
                raise ValueError(f"[{name}] shape mismatch: {curr_shapes} != {reference['shapes']}")
            if curr_modules != reference["modules"]:
                raise ValueError(
                    f"[{name}] target_modules mismatch:\n"
                    f"  this adapter -> {sorted(curr_modules)}\n"
                    f"  reference   -> {sorted(reference['modules'])}"
                )

    agreed_modules = sorted(reference["modules"])
    return agreed_modules


def ensure_adapters_target_linear_layers_only(model, adapter_names: list[str]):
    """
    Validate that every module holding LoRA weights for any of `adapter_names` is Linear-like: nn.Linear,
    bitsandbytes.nn.Linear4bit, nn.Conv1d, or transformers.models.gpt2.modeling_gpt2.Conv1D. If not, raise.
    """
    import torch.nn as nn

    # Optional bnb Linear4bit
    Linear4bit = None
    try:
        import bitsandbytes as bnb  # type: ignore

        Linear4bit = bnb.nn.Linear4bit
    except Exception:
        pass

    # HF GPT-2 Conv1D (linear wrapper)
    HFConv1D = None
    try:
        from transformers.models.gpt2.modeling_gpt2 import Conv1D as HFConv1D  # type: ignore
    except Exception:
        pass

    allowed_types = (nn.Linear, nn.Conv1d)
    if Linear4bit is not None:
        allowed_types = allowed_types + (Linear4bit,)
    if HFConv1D is not None:
        allowed_types = allowed_types + (HFConv1D,)

    offenders = []

    for full_name, module in model.named_modules():
        if hasattr(module, "lora_A"):
            for name in adapter_names:
                if name in getattr(module, "lora_A", {}):
                    base = getattr(module, "base_layer", None) or getattr(module, "original_module", None)
                    layer_to_check = base if base is not None else module

                    if not isinstance(layer_to_check, allowed_types):
                        offenders.append((name, full_name, type(layer_to_check).__name__))

    if offenders:
        lines = [
            "LoRA adapters must only target Linear-like layers "
            "(nn.Linear, nn.Conv1d, HF Conv1D, or bitsandbytes.nn.Linear4bit). Found:"
        ]
        for name, full_name, tname in offenders:
            lines.append(f"  - adapter '{name}' on module '{full_name}' of type {tname}")
        raise TypeError("\n".join(lines))


def get_adapter_hyperparams(model, adapter_name: str):
    """
    Return (r, lora_alpha, scaling) for `adapter_name`. Prefers the config dict when available; falls back to the first
    LoRA layer.
    """
    # 1) Try from the config dict
    r = alpha = scaling = None
    try:
        cfg = model.peft_config[adapter_name]
        r = getattr(cfg, "r", None)
        alpha = getattr(cfg, "lora_alpha", None)
        # scaling is usually alpha / r inside layers; keep None for now
    except Exception:
        pass

    # 2) Fall back to inspecting a LoRA-wrapped layer
    for _, module in model.named_modules():
        if hasattr(module, "lora_A") and adapter_name in getattr(module, "lora_A", {}):
            # r from A’s shape: (r, in_features)
            if r is None:
                r = int(module.lora_A[adapter_name].weight.shape[0])
            # alpha from layer dict if present; otherwise leave as-is
            if alpha is None and hasattr(module, "lora_alpha"):
                alpha = module.lora_alpha.get(adapter_name, None)
            # scaling stored per-layer in PEFT
            if hasattr(module, "scaling"):
                scaling = module.scaling.get(adapter_name, None)
            break

    if r is None or alpha is None or scaling is None:
        raise ValueError(
            f"Could not infer hyperparams for adapter '{adapter_name}' (r={r}, alpha={alpha}, scaling={scaling})."
        )
    return r, alpha, scaling


def check_lora_rank_consistency(model, adapter_names: list[str]) -> int:
    """
    Ensure all listed adapters share identical LoRA rank `r` across all LoRA layers. Returns the agreed rank `r`.
    """
    ref_r = None
    found = False

    for name in adapter_names:
        for _, module in model.named_modules():
            if hasattr(module, "lora_A") and name in module.lora_A:
                found = True
                r_layer = int(module.lora_A[name].weight.shape[0])
                if ref_r is None:
                    ref_r = r_layer
                elif r_layer != ref_r:
                    raise ValueError(f"LoRA rank mismatch: adapter '{name}' has r={r_layer}, expected r={ref_r}")

    if not found:
        raise ValueError("No LoRA layers found for the provided adapters.")

    return ref_r


def _resolve_adapter_source(path: str, base_kwargs: dict) -> tuple[str, dict]:
    """
    Resolve a user-provided adapter `path` into (model_id, adapter_kwargs).

    Supports:
      - Local path to a folder that contains `adapter_config.json`
      - Hub path with subfolder, e.g. "user/repo/ts_expert_0[/more/...]", which becomes:
            model_id="user/repo", subfolder="ts_expert_0[/more/...]"
      - Plain Hub repo id "user/repo" (no subfolder)

    `base_kwargs` are copied and `subfolder` is set if needed (not overwritten if already provided).
    """
    kw = deepcopy(base_kwargs)

    # Local directory case
    if os.path.isdir(path):
        # Expect adapter_config.json directly inside this folder
        if not os.path.isfile(os.path.join(path, "adapter_config.json")):
            raise ValueError(f"Local adapter path '{path}' does not contain 'adapter_config.json'.")
        return path, kw

    # Hub-style path: "namespace/repo[/optional/sub/dir]"
    parts = path.strip("/").split("/")
    if len(parts) >= 2:
        model_id = "/".join(parts[:2])
        # if there's a subfolder portion beyond "namespace/repo"
        if len(parts) > 2:
            subfolder = "/".join(parts[2:])
            kw.setdefault("subfolder", subfolder)
        return model_id, kw

    # Fallback (unlikely): treat as model_id as-is
    return path, kw


def create_arrow_model(
    base_model: PreTrainedModel,
    task_specific_adapter_paths: list[str],  # path of task-specific LoRAs
    arrow_config: ArrowConfig,
    general_adapter_paths: list[str] | None = None,  # path to the trained general-knowledge LoRAs
    **adapter_kwargs: Any,
):
    if task_specific_adapter_paths is None or len(task_specific_adapter_paths) == 0:
        raise ValueError("`task_specific_adapter_paths` should contain at least one adapter path")

    from peft import LoraConfig, PeftModel

    # Load the first TS expert to get a PeftModel
    model_id0, kw0 = _resolve_adapter_source(task_specific_adapter_paths[0], adapter_kwargs)
    initial_ts_expert_name = f"{TASK_ADAPTER_PREFIX}0"
    model = PeftModel.from_pretrained(
        base_model,
        model_id=model_id0,
        adapter_name=initial_ts_expert_name,
        **kw0,
    )

    # Load other task-specific adapters
    for i in range(1, len(task_specific_adapter_paths)):
        ts_expert_name = f"{TASK_ADAPTER_PREFIX}{i}"
        mid, kw = _resolve_adapter_source(task_specific_adapter_paths[i], adapter_kwargs)
        model.load_adapter(
            model_id=mid,
            adapter_name=ts_expert_name,
            **kw,
        )
    arrow_config.task_adapter_names = [f"{TASK_ADAPTER_PREFIX}{i}" for i in range(len(task_specific_adapter_paths))]

    # Load general adapters (for GKS) if requested
    if arrow_config.use_gks:
        if general_adapter_paths is None or len(general_adapter_paths) == 0:
            raise ValueError("You should provide general LoRA paths if you want to use GenKnowSub.")
        for i in range(len(general_adapter_paths)):
            gen_expert_name = f"{GKS_ADAPTER_PREFIX}{i}"
            mid, kw = _resolve_adapter_source(general_adapter_paths[i], adapter_kwargs)
            model.load_adapter(
                model_id=mid,
                adapter_name=gen_expert_name,
                **kw,
            )
        arrow_config.gks_adapter_names = [f"{GKS_ADAPTER_PREFIX}{i}" for i in range(len(general_adapter_paths))]
    else:
        arrow_config.gks_adapter_names = []

    # Check LoRA compatibility
    target_modules = check_loaded_lora_compatibility_arrow(
        model, adapter_names=arrow_config.task_adapter_names + arrow_config.gks_adapter_names
    )

    # Enforce adapters are ONLY on Linear/Linear4bit
    ensure_adapters_target_linear_layers_only(
        model, adapter_names=arrow_config.task_adapter_names + arrow_config.gks_adapter_names
    )

    # Check for the consistency of adapters' rank
    r = check_lora_rank_consistency(
        model, adapter_names=arrow_config.task_adapter_names + arrow_config.gks_adapter_names
    )

    # --- Add the Arrow router (dummy LoRA) and activate it
    router_cfg = LoraConfig(
        arrow_config=arrow_config,
        target_modules=target_modules,
        r=r,
    )
    model.add_adapter(adapter_name="arrow_router", peft_config=router_cfg)
    model.set_adapter("arrow_router")

    return model
