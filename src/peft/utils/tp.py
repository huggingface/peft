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


def add_lora_tp_hooks_dtensor(tp_module: nn.Module, tp_plan_name: str, device_mesh, *, module_name: str) -> None:
    from transformers.distributed.tensor_parallel import ALL_PARALLEL_STYLES

    style = ALL_PARALLEL_STYLES[tp_plan_name]
    # Shard every parameter of the module (weight, and bias when lora_bias=True), matching
    # transformers' own `apply_tensor_parallelism`, which shards all of a module's parameters
    # before installing the forward transform.
    for p_name, _ in list(tp_module.named_parameters(recurse=False)):
        style.validate_param(tp_module, p_name, device_mesh, parameter_name=f"{module_name}.{p_name}")
        style.shard_param(tp_module, p_name, device_mesh)
    style.install_forward(tp_module, device_mesh)


class LoraEmbeddingATPHolder(nn.Embedding):
    """ 
    In LoRA, the embedding A weight is a learnable parameter that is added to the original embedding weight, but the TP
    API acts on modules rather than individual parameters. This class wraps the LoRA embedding A weight in an 
    `nn.Embedding` module, allowing it to be treated as a module by the TP API.
    """
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

