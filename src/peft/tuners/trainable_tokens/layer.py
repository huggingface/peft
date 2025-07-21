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
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners._buffer_dict import BufferDict
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import check_deepspeed_zero3_enabled, gather_params_ctx


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
        tied_adapter: Optional[TrainableTokensLayer] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.base_layer = base_layer
        self._active_adapter = adapter_name
        self.kwargs = kwargs

        # wrap the tied adapter in a list so that it is excluded from .(named_)modules() and, therefore,
        # not included in the state dict since it would be a copy of the tied adapter anyway.
        self._tied_adapter = [tied_adapter] if tied_adapter else []

        # we store the updated weights of particular tokens and their originals. we assume
        # that the count of new tokens is far smaller than the number of total tokens.
        #
        # In case we have weight tying with another token adapter, we'll have no actual
        # references on our own but use everything from the tied adapter.
        if not self.tied_adapter:
            self.trainable_tokens_delta = nn.ParameterDict({})
            self.trainable_tokens_original = BufferDict({})
            self.token_indices = {}
        else:
            self.trainable_tokens_delta = self.tied_adapter.trainable_tokens_delta
            self.trainable_tokens_original = self.tied_adapter.trainable_tokens_original
            self.token_indices = self.tied_adapter.token_indices

        # Mark the weight as unmerged
        self.merged_adapters = []

    @property
    def tied_adapter(self):
        if self._tied_adapter:
            return self._tied_adapter[0]
        return None

    def _collect_token_weights(self, weight: torch.Tensor, rows: torch.Tensor, embed_dim: int) -> torch.Tensor:
        """DeepSpeed zero3 specific code to initialize trainable tokens.

        Ensures that only the necessary weights are collected to a single rank, initialized, and then shared with all
        ranks.
        """
        src_rank = 0
        # right now, only CUDA is implemented
        device = torch.device("cuda", torch.cuda.current_device())

        with gather_params_ctx([weight], modifier_rank=None):
            if dist.get_rank() == src_rank:
                token_weights = weight[rows].clone()
            else:
                # build an empty tensor with correct shape/type/device
                token_weights = torch.empty(
                    (len(rows), embed_dim),
                    dtype=weight.dtype,
                    device=device,
                )

        # share the weights with all ranks
        dist.broadcast(token_weights, src=src_rank)
        return token_weights

    def update_layer(self, adapter_name, **kwargs):
        if kwargs.get("tied_adapter", None):
            # as a tied adapter, we're just following whatever the adpater we're tied to does, we don't update anything.
            return

        self.token_indices[adapter_name] = kwargs["token_indices"]
        init_weights = kwargs.get("init_weights", True)

        # we initialize the delta embedding weights from the base embedding matrix and replace values instead of
        # adding/subtracting deltas. we do it this way and use `embedding.weight.index_copy()` to write the updated
        # values during `forward()` to avoid that the user resizing the embedding matrix, effectively filling the new
        # token space with random values, training the model with TrainableTokensLayer, initializing the model anew -
        # thus re-initializing the new embeddings again with new random variables. If we would add/subtract deltas
        # onto the new values, we would get undefined behavior. By replacing the specific token values we always
        # get defined behavior.
        weight = self.get_base_layer().weight
        embed_dim = self.get_base_layer().embedding_dim

        if init_weights:
            if check_deepspeed_zero3_enabled():
                values = self._collect_token_weights(weight, self.token_indices[adapter_name], embed_dim)
            else:
                values = self.weight[self.token_indices[adapter_name]]
        else:
            # random init with matching dtype/device
            values = torch.randn(
                (len(self.token_indices[adapter_name]), embed_dim),
                dtype=weight.dtype,
                device=weight.device,
            )

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

        # we take already merged adapters into account as well since they can be overridden by new adapters as well.
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

    def get_merged_weights(self, active_adapters):
        W = self.base_layer.weight

        for adapter_name in active_adapters:
            index = torch.tensor(self.token_indices[adapter_name]).to(W.device)
            deltas = self.trainable_tokens_delta[adapter_name].to(W)
            W = W.index_copy(dim=0, index=index, source=deltas)

        return W

    def forward_adapters(self, x: torch.Tensor, active_adapters, *args, **kwargs) -> torch.Tensor:
        if self.disable_adapters or not active_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            self._check_overlapping_tokens(active_adapters)

            W = self.get_merged_weights(active_adapters)

            # Normally it should be very clear that we're wrapping Embedding layers but there are cases, such as
            # tying weights with an LM head where the layer we wrap is a Linear layer. Therefore we must choose
            # accordingly.
            #
            # TODO: the isinstance checks, especially the one for nn.Linear, may not hold for quantized layers;
            # TODO: we may need to find a better way to detect quantized layers.
            if isinstance(self.base_layer, torch.nn.Embedding):
                result = F.embedding(
                    input=x,
                    weight=W,
                    padding_idx=self.base_layer.padding_idx,
                    max_norm=self.base_layer.max_norm,
                    norm_type=self.base_layer.norm_type,
                    scale_grad_by_freq=self.base_layer.scale_grad_by_freq,
                    sparse=self.base_layer.sparse,
                )
            elif isinstance(self.base_layer, torch.nn.Linear):
                # Probably a tied adapter that wraps an LM head.
                result = F.linear(
                    input=x,
                    weight=W,
                )
            else:
                raise ValueError(
                    "TrainableTokensLayer wraps an unknown layer type, maybe you are targeting the wrong layer?"
                )

        return result

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.forward_adapters(x, self.active_adapters, *args, **kwargs)
