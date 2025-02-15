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

import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge


class TrainableTokensLayer(nn.Module, BaseTunerLayer):
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        token_indices: list[int],
        **kwargs,
    ) -> None:
        super().__init__()

        self.base_layer = base_layer
        self._active_adapter = adapter_name
        self.token_indices = token_indices
        self.kwargs = kwargs

        # we store the delta weight on particular tokens
        self.trainable_tokens_delta_tokens = nn.ParameterDict({})

        # Mark the weight as unmerged
        self.merged_adapters = []

        # we set parameters on layer sizing
        self.num_trainable_embeddings = len(token_indices)  # this is similar to `num_embeddings`
        self.embedding_dim = base_layer.embedding_dim  # token from the embedding
        self.num_total_embeddings = base_layer.num_embeddings  # total number of tokens in the vocabulary

        # we set the number of trainable tokens
        self.update_layer(adapter_name)

    def update_layer(self, adapter_name):
        # we initialize the delta embedding weights and store them in a trainable parameter
        #
        # TODO use existing embedding layer values for initialization? there is the problem
        # that some tests assume that after init should be the same as disabled adapters.
        # so either we let this break the assumptions or we initialize the deltas to zero.
        # since we're adding on top of the existing embeddings, I think that zero is sensible.
        values = torch.zeros(self.num_trainable_embeddings * self.base_layer.weight.shape[-1])

        # cause safetensors doesn't support sparse tensors, we need to store the values in a dense tensor and then convert it to a sparse tensor when called
        self.trainable_tokens_delta_tokens[adapter_name] = nn.Parameter(values, requires_grad=True)

    def get_sparse_delta_tokens(self, adapter_name):
        # we created the indices of the sparse tensor
        r = torch.Tensor(self.token_indices).long()
        c = torch.arange(self.embedding_dim)
        indices = torch.stack(
            [r.repeat(self.embedding_dim), c.repeat(self.num_trainable_embeddings, 1).t().reshape(-1)], dim=1
        ).T

        # we create the sparse tensor from our `delta_tokens` and `indices
        return torch.sparse_coo_tensor(
            indices=indices,
            values=self.trainable_tokens_delta_tokens[adapter_name],
            size=(self.num_total_embeddings, self.embedding_dim),
        )

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)

        if not adapter_names:
            # no adapter to merge
            return

        for adapter_name in adapter_names:
            orig_weights = self.base_layer.weight.data
            merged = orig_weights + self.get_sparse_delta_tokens(adapter_name)

            if safe_merge and not torch.isfinite(merged).all():
                raise ValueError(
                    f"NaNs detected in the merged weights. The adapter {adapter_name} seems to be broken"
                )
            else:
                self.base_layer.weight.data = merged
                self.merged_adapters.append(adapter_name)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            adapter_name = self.merged_adapters.pop()
            self.base_layer.weight.data -= self.get_sparse_delta_tokens(adapter_name)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.disable_adapters or not self.active_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            deltas = None
            for active_adapter in self.active_adapters:
                delta = self.get_sparse_delta_tokens(active_adapter)
                if deltas is None:
                    deltas = delta
                else:
                    deltas += delta

            W = self.base_layer.weight

            result = F.embedding(
                input=x,
                weight=W + deltas,
                padding_idx=self.base_layer.padding_idx,
                max_norm=self.base_layer.max_norm,
                norm_type=self.base_layer.norm_type,
                scale_grad_by_freq=self.base_layer.scale_grad_by_freq,
                sparse=self.base_layer.sparse,
            )

        return result
