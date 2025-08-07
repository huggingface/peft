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
from transformers import PreTrainedModel

from peft.utils.other import transpose

from .arrow import ArrowLoraLinearLayer
from .config import ArrowConfig, LoraConfig
from .dora import DoraConv1dLayer, DoraConv2dLayer, DoraConv3dLayer, DoraEmbeddingLayer, DoraLinearLayer
from .layer import Conv1d, Conv2d, Conv3d, Embedding, Linear, LoraVariant, _ConvNd


class ArrowLinearVariant(LoraVariant):
    """
    Variant wrapper for ArrowLoRA on `nn.Linear` and `bnb.nn.Linear4bit`. One instance (actually: one *class*) serves
    *all* adapters; per-adapter state lives in `module.lora_arrow[adapter_name]`, an ArrowLoRALayer.
    """

    # ------------------------------------------------------------------
    # 1) INIT  – runs **once per adapter** during `model.load_adapter(...)`
    # ------------------------------------------------------------------
    @staticmethod
    def init(module: Linear, adapter_name: str, **kwargs):
        """
        `module` : the patched Linear (already owns lora_A/B/etc.) `adapter_name` : "cluster7", "router", ... `kwargs`
        : every extra field you added to LoraConfig
                         (arrow_top_k, arrow_expert_num, etc.)
        """
        # print(kwargs)
        # print(adapter_name)
        # Checking for arrow necessary config

        arrow_config = kwargs["kwargs"].get("arrow_config")
        if arrow_config is None:
            raise ValueError("ArrowLinearVariant.init() did not receive a arrow_config")

        # 1-a) build the ArrowLoRALayer
        arrow_layer = ArrowLoraLinearLayer(
            in_features=module.in_features,  # **check**
            arrow_config=arrow_config,
        ).to(module.weight.device)

        # 1-b) register a container if it doesn’t exist yet
        if not hasattr(module, "lora_arrow"):
            module.lora_arrow = nn.ModuleDict()
            # ensure PEFT knows to save / load these params
            module.other_param_names = module.other_param_names + ("lora_arrow",)

        module.lora_arrow[adapter_name] = arrow_layer
        # *** DO NOT compute prototypes here *** –
        # we may not have loaded *all* experts yet.  ArrowLoraLinearLayer.forward()
        # will lazily call `build_prototypes()` the first time it runs.

    # ------------------------------------------------------------------
    # 2) FORWARD – called every time the host Linear does a fwd pass
    # ------------------------------------------------------------------
    @staticmethod
    def forward(
        module: Linear,
        *,
        active_adapter: str,
        x: torch.Tensor,
        result: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters mirror those in PEFT’s `LoraVariant.forward`.

        * `result` = base linear output (`x @ W₀ᵀ + b`)
        * return = result + Δ (Δ from Arrow routing)
        """
        arrow = module.lora_arrow[active_adapter]  # ArrowLoraLinearLayer
        # Apply GenKnowSub the 1st time if applcable.
        arrow.gen_know_sub(module.lora_A, module.lora_B)
        # lazily build prototypes the 1st time after GenKnowSub
        arrow.build_prototypes(module.lora_A, module.lora_B)

        delta = arrow(
            x,
            lora_A=module.lora_A,  # dict of nn.Linear
            lora_B=module.lora_B,
            dropout=module.lora_dropout[active_adapter],
            scaling=module.scaling[active_adapter],
        )
        return result + delta

    # ------------------------------------------------------------------
    # 3) MERGE / UNMERGE  (optional)
    # ------------------------------------------------------------------
    # If you plan to support `model.merge_and_unload()`, add:
    #
    # @staticmethod
    # def merge_weights(module: Linear, adapter_name: str) -> None: ...
    #
    # For most research prototypes you can skip this block; PEFT will simply
    # fall back to keeping Arrow in “un-merged” mode.


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
    print(f"✅ All adapters target: {agreed_modules}")
    return agreed_modules


def create_arrow_model(
    base_model: PreTrainedModel,
    task_specific_adapter_paths: list[str],  # path of task-specific LoRAs
    arrow_config: ArrowConfig,
    general_adapter_paths: list[str] | None = None,  # path to the trained general-knowledge LoRAs
    **adapter_kwargs: Any,
):
    def split_repo_and_subfolder(path: str):
        """
        Splits a Hugging Face path into repo_id and subfolder. For example:
        "TahaBa/phi3-mini-clustered-flan/ts_expert_0"
        """
        parts = path.strip("/").split("/")
        if len(parts) <= 2:
            return path, None  # It's a repo without subfolder
        repo_id = "/".join(parts[:2])
        subfolder = "/".join(parts[2:])
        return repo_id, subfolder

    if task_specific_adapter_paths is None:
        raise ValueError("`task_specific_adapter_paths` should contain at least one adapter path")

    from peft import PeftModel

    # Loading the first expert on the model so we have a PeftModel
    repo_id, sub_folder = split_repo_and_subfolder(task_specific_adapter_paths[0])
    initial_ts_expert_name = "ts_expert_0"
    model = PeftModel.from_pretrained(
        base_model,
        model_id=repo_id,
        adapter_name=initial_ts_expert_name,
        subfolder=sub_folder,
        **adapter_kwargs,
    )

    # Loading other task-specific adapters
    for i in range(1, len(task_specific_adapter_paths)):
        ts_expert_name = f"ts_expert_{i}"
        repo_id, sub_folder = split_repo_and_subfolder(task_specific_adapter_paths[i])
        model.load_adapter(
            model_id=repo_id,
            adapter_name=ts_expert_name,
            subfolder=sub_folder,
            **adapter_kwargs,
        )
    # Adding task-specific adapter names to the arrow_config
    arrow_config.ts_names = [f"ts_expert_{i}" for i in range(len(task_specific_adapter_paths))]

    # Loading general LoRAs if available (to be used in GenKnowSub)
    if arrow_config.use_gks:
        if general_adapter_paths is None:
            raise ValueError("You should provide general LoRA paths if want to use GenKnowSub.")
        for i in range(0, len(general_adapter_paths)):
            gen_expert_name = f"gen_expert_{i}"
            repo_id, sub_folder = split_repo_and_subfolder(general_adapter_paths[i])
            model.load_adapter(
                model_id=repo_id,
                adapter_name=gen_expert_name,
                subfolder=sub_folder,
                **adapter_kwargs,
            )
        # Adding general adapter names to the arrow_config
        arrow_config.gen_names = [f"gen_expert_{i}" for i in range(len(general_adapter_paths))]
    else:
        arrow_config.gen_names = []

    # Now checking the compatibility of all adapters
    target_modules = check_loaded_lora_compatibility_arrow(
        model, adapter_names=arrow_config.ts_names + arrow_config.gen_names
    )

    # Add a dummy LoRA as the arrow router
    # 1) Define the router’s config:
    router_cfg = LoraConfig(
        r=2,  # rank of router adapter (dummy)
        use_arrow=True,  # turns on Arrow routing instead of vanilla LoRA
        arrow_config=arrow_config,  # all the necessary configs for arrow
        target_modules=target_modules,  # target modules which adapters are applied on
    )

    # 2) Create the adapter “router” (weights are randomly init’d LoRA mats,
    #    but will never actually be applied because use_arrow=True):
    model.add_adapter(
        adapter_name="router",
        peft_config=router_cfg,
    )

    # 3) Tell the model “hey, from now on use the ‘router’ adapter”:
    model.set_adapter("router")

    return model
