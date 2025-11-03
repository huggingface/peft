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

import math

import torch
import torch.nn as nn

from ..lora.layer import Linear


class BlockDiagonalLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, nblocks: int, bias: bool = True):
        """Implementation of a block-diagonal linear layer. The weight matrix is divded into nblocks with size out_features //
        nblocks, in_features// nblocks."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nblocks = nblocks

        self.weight = nn.Parameter(torch.empty(out_features, in_features // nblocks))
        # BD-LoRA initialization should overwrite this, so the initialization does not matter for our implementation. I would guess that we could
        # also leave the tensor empty to save a bit of compute - although I cannot 100% verify that init_lora_weights is always set to True with
        # all possible LoRA parameters
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle non-2D inputs by flattening first dimension, as these are usually
        # treated as batched over in nn.Linear Layers. We simply call this flattened first dimension
        # the batch size from now on
        if not len(x.shape) == 2:
            x = x.reshape(-1, x.shape[-1])

        B = x.shape[0]
        nb = self.nblocks
        m = x.shape[-1] // nb
        n = self.out_features // nb

        x = x.reshape(B, nb, m)
        assert self.weight.shape == (nb * n, m)
        w = self.weight.view(nb, n, m)

        # x: (B, nb, m)
        # weight: (nb, n, m)
        # output should be (B, nb, n) (and then we reshape to stack the blocks)
        out = torch.einsum("bim,inm->bin", x, w)
        return out.reshape(B, -1)

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, nblocks={self.nblocks})"


class LoraABlockdiagonalLinear(Linear):
    """LoRA adapter for RowParallelLinear where A is block-diagonal"""

    def __init__(self, *args, **kwargs):
        self.nblocks = kwargs.pop("nblocks", 1)
        super().__init__(*args, **kwargs)

    def update_layer(self, adapter_name, r, *args, **kwargs):
        super().update_layer(adapter_name, r, *args, **kwargs)
        # A is block-diagonal (for row-parallel)
        layer = BlockDiagonalLinear(self.in_features, r, nblocks=self.nblocks, bias=False)  # type: ignore
        self.lora_A.update(nn.ModuleDict({adapter_name: layer}))  # type: ignore


class LoraBBlockdiagonalLinear(Linear):
    """LoRA adapter for ColumnParallelLinear where B is block-diagonal"""

    def __init__(self, *args, **kwargs):
        self.nblocks = kwargs.pop("nblocks", 1)
        super().__init__(*args, **kwargs)

    def update_layer(self, adapter_name, r, *args, **kwargs):
        super().update_layer(adapter_name, r, *args, **kwargs)
        # B is block-diagonal (for column-parallel)
        layer = BlockDiagonalLinear(r, self.out_features, nblocks=self.nblocks, bias=False)  # type: ignore
        self.lora_B.update(nn.ModuleDict({adapter_name: layer}))  # type: ignore
