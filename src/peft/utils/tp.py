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
"""New-path (DTensor-based) helpers for PEFT's LoRA + Transformers Tensor Parallel integration.

Transformers PR #47579 replaced the hook-based TP API (`add_tensor_parallel_hooks_to_module`,
per-module `_hf_tp_plan`/`_hf_device_mesh` attributes, `EmbeddingParallel`) with a DTensor-native
one: the plan/mesh live on the top-level model (`model.tp_plan`, `model._device_mesh`), and
`TensorParallelLayer` subclasses (`ColwiseParallel`, `RowwiseParallel`, ...) expose
`shard_param`/`install_forward` instead.

The old-path code (still needed for transformers < the DTensor migration) is left untouched at
its existing call sites in `lora/model.py`, `lora/layer.py`, and `save_and_load.py`, gated by
`peft.import_utils.is_transformers_dtensor_tp`. This module only holds the new-path
implementations, so the two code paths stay easy to tell apart and the old one stays unmodified.
"""

from __future__ import annotations

import re

import torch
from torch import nn


def _replace_layer_number_by_wildcard(name: str) -> str:
    return re.sub(r"\.\d+(\.|$)", lambda m: ".*" + m.group(1), name)


def get_tp_plan_and_mesh(model, current_key: str):
    tp_plan = getattr(model, "tp_plan", None)
    device_mesh = getattr(model, "_device_mesh", None)
    if not tp_plan or device_mesh is None:
        return None, None
    plan_name = tp_plan.get(_replace_layer_number_by_wildcard(current_key))
    if plan_name is None:
        return None, None
    return plan_name, device_mesh


def stamp_tp_shim_attrs(base_layer, model, current_key: str) -> None:
    if getattr(base_layer, "_hf_tp_plan", None) is not None:
        return  # already set, e.g. when adding a 2nd adapter to an already-processed layer
    tp_plan, device_mesh = get_tp_plan_and_mesh(model, current_key)
    if tp_plan is None:
        return
    base_layer._hf_tp_plan = tp_plan
    base_layer._hf_device_mesh = device_mesh


def add_lora_tp_hooks_dtensor(tp_module: nn.Module, tp_plan_name: str, device_mesh, *, parameter_name: str) -> None:
    from transformers.distributed.tensor_parallel import ALL_PARALLEL_STYLES

    style = ALL_PARALLEL_STYLES[tp_plan_name]
    style.validate_param(tp_module, "weight", device_mesh, parameter_name=parameter_name)
    style.shard_param(tp_module, "weight", device_mesh)
    style.install_forward(tp_module, device_mesh)


def shard_lora_tensor_for_load(full_tensor: torch.Tensor, tp_plan_name: str, device_mesh, *, is_embedding_weight: bool = False):
    from torch.distributed.tensor import Shard, distribute_tensor

    if tp_plan_name == "colwise":
        placement = Shard(0)  # lora_B.weight: (out_features, r)
    elif tp_plan_name == "rowwise":
        placement = Shard(-1)  # lora_A.weight: (r, in_features)
    elif tp_plan_name == "embedding_rowwise":
        # base embedding weight: (vocab_size, embedding_dim) -> shard the vocab dim (dim 0).
        # lora_embedding_A: (r, vocab_size) -> same vocab dim, but it sits at dim 1 here.
        placement = Shard(0) if is_embedding_weight else Shard(1)
    else:
        raise ValueError(f"Unsupported tensor parallel plan {tp_plan_name!r} for LoRA state dict sharding.")

    return distribute_tensor(full_tensor, device_mesh, [placement], src_data_rank=None)


class LoraEmbeddingATPHolder(nn.Embedding):

    def __init__(self, lora_embedding_A_weight: nn.Parameter):
        nn.Module.__init__(self)
        num_embeddings, embedding_dim = lora_embedding_A_weight.T.shape
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.0
        self.scale_grad_by_freq = False
        self.sparse = False
        self._parameters["weight"] = nn.Parameter(
            lora_embedding_A_weight.T.contiguous(), requires_grad=lora_embedding_A_weight.requires_grad
        )


def make_embedding_lora_tp_fns(style, holder: LoraEmbeddingATPHolder, device_mesh):
    """New-path replacement for `EmbeddingParallel._prepare_input_fn`/`_prepare_output_fn`.
    `holder` is the same `LoraEmbeddingATPHolder` instance the old path also builds in
    `Embedding.update_layer`, so construction stays symmetric between the two branches.
    """
    style.shard_param(holder, "weight", device_mesh)
    sharded_weight = nn.Parameter(holder.weight.T, requires_grad=holder.weight.requires_grad)

    def input_fn(inputs):
        (x,), _ = style.transform_inputs_pre_forward(holder, inputs, {}, device_mesh)
        return x

    def output_fn(output):
        return style.transform_output_post_forward(holder, output, device_mesh)

    return sharded_weight, input_fn, output_fn
