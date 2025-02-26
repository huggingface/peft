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

from peft.tuners._buffer_dict import BufferDict
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge


class TrainableTokensLayer(nn.Module, BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("trainable_tokens_delta",)

    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("token_indices", "trainable_tokens_original")

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
        self.token_indices = {}
        self.kwargs = kwargs

        # we store the updated weights of particular tokens and their originals. we assume
        # that the count of new tokens is far smaller than the number of total tokens.
        self.trainable_tokens_delta = nn.ParameterDict({})
        self.trainable_tokens_original = BufferDict({})

        # Mark the weight as unmerged
        self.merged_adapters = []

    def update_layer(self, adapter_name, **kwargs):
        self.token_indices[adapter_name] = kwargs["token_indices"]
        init_weights = kwargs.get("init_weights", True)

        # we initialize the delta embedding weights from the base embedding matrix and replace values instead of
        # adding/subtracting deltas. we do it this way and use `embedding.weight.index_copy()` to write the updated
        # values during `forward()` to avoid that the user resizing the embedding matrix, effectively filling the new
        # token space with random values, training the model with TrainableTokensLayer, initializing the model anew -
        # thus re-initializing the new embeddings again with new random variables. If we would add/subtract deltas
        # onto the new values, we would get undefined behavior. By replacing the specific token values we always
        # get defined behavior.
        #
        if init_weights:
            values = self.get_base_layer().weight[self.token_indices[adapter_name]]
        else:
            values = torch.rand_like(self.get_base_layer().weight[self.token_indices[adapter_name]])

        self.trainable_tokens_delta[adapter_name] = nn.Parameter(values.clone(), requires_grad=True)
        self.trainable_tokens_original[adapter_name] = values.clone()

        self._move_adapter_to_device_of_base_layer(adapter_name)

    def _check_overlapping_tokens(self, adapter_names):
        """Raises an error if the token indices of the given adapter names are overlapping.
        This is currently not supported and can lead to undefined behavior of the model if no specific merging between
        the overlapping indices' values is applied.
        """
        if len(adapter_names) <= 1:
            return

        indices = set()

        # we take already merged adapters into account as well since they can be overriden by new adapters as well.
        for adapter_name in set(adapter_names + self.merged_adapters):
            index_set = set(self.token_indices[adapter_name])
            if len(indices.intersection(index_set)):
                raise ValueError(
                    f"Token indices of adapter {adapter_name} are already defined and would result in "
                    "undefined merging behavior. Only disjunct token indices are currently supported."
                )
            indices.update(index_set)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)

        if not adapter_names:
            # no adapter to merge
            return

        self._check_overlapping_tokens(adapter_names)

        merged = self.base_layer.weight.data

        for adapter_name in adapter_names:
            index = torch.tensor(self.token_indices[adapter_name]).to(merged.device)
            deltas = self.trainable_tokens_delta[adapter_name].to(merged)
            merged = merged.index_copy(dim=0, index=index, source=deltas)

            if safe_merge and not torch.isfinite(merged).all():
                raise ValueError(f"NaNs detected in the merged weights. The adapter {adapter_name} seems to be broken")

        self.base_layer.weight.data = merged
        self.merged_adapters.extend(adapter_names)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            adapter_name = self.merged_adapters.pop()

            index = torch.tensor(self.token_indices[adapter_name]).to(self.base_layer.weight.device)
            originals = self.trainable_tokens_original[adapter_name].to(self.base_layer.weight)
            self.base_layer.weight.data.index_copy_(dim=0, index=index, source=originals)

    def forward_adapters(self, x: torch.Tensor, active_adapters, *args, **kwargs) -> torch.Tensor:
        if self.disable_adapters or not active_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            self._check_overlapping_tokens(active_adapters)

            W = self.base_layer.weight

            for adapter_name in active_adapters:
                index = torch.tensor(self.token_indices[adapter_name]).to(W.device)
                deltas = self.trainable_tokens_delta[adapter_name].to(W)
                W = W.index_copy(dim=0, index=index, source=deltas)

            result = F.embedding(
                input=x,
                weight=W,
                padding_idx=self.base_layer.padding_idx,
                max_norm=self.base_layer.max_norm,
                norm_type=self.base_layer.norm_type,
                scale_grad_by_freq=self.base_layer.scale_grad_by_freq,
                sparse=self.base_layer.sparse,
            )

        return result

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.forward_adapters(x, self.active_adapters, *args, **kwargs)
