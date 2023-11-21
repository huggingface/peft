# coding=utf-8
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
import warnings
from typing import Any, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.lycoris_utils import LycorisLayer


class OFTLayer(nn.Module, LycorisLayer):
    # All names of layers that may contain adapter weights
    adapter_layer_names = ("oft_r",)
    # other_param_names is defined on parent class

    def __init__(self, base_layer: nn.Module):
        super().__init__()
        LycorisLayer.__init__(self, base_layer)

        # OFT info
        self.oft_r = nn.ParameterDict({})
        self.coft = {}
        self.eps = {}
        self.block_share = {}

    @property
    def _available_adapters(self) -> Set[str]:
        return {*self.oft_r}

    def create_adapter_parameters(self, adapter_name: str, r: int, shape: Tuple[int, ...], block_share: bool):
        if block_share:
            self.oft_r[adapter_name] = nn.Parameter(torch.empty(1, shape[1] // r, shape[1] // r))
        else:
            self.oft_r[adapter_name] = nn.Parameter(torch.empty(r, shape[1] // r, shape[1] // r))

    def reset_adapter_parameters(self, adapter_name: str):
        nn.init.zeros_(self.oft_r[adapter_name])

    def reset_adapter_parameters_random(self, adapter_name: str):
        nn.init.kaiming_uniform_(self.oft_r[adapter_name], a=math.sqrt(5))

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        alpha: float,
        rank_dropout: float,
        module_dropout: float,
        init_weights: bool,
        use_effective_conv2d: bool = False,
        coft: bool = False,
        eps: float = 6e-5,
        block_share: bool = False,
        **kwargs,
    ) -> None:
        """Internal function to create oft adapter

        Args:
            adapter_name (`str`): Name for the adapter to add.
            r (`int`): Rank for the added adapter.
            alpha (`float`): Alpha for the added adapter.
            rank_dropout (`float`): The dropout probability for rank dimension during training.
            module_dropout (`float`): The dropout probability for disabling adapter during training.
            init_weights (`bool`): Whether to initialize weights.
            use_effective_conv2d (`bool`, *optional*, defaults to `False`):
                Use parameter effective decomposition for Conv2d with ksize > 1.
            coft (`bool`): Whether to use the constrainted variant of OFT or not.
            eps (`float`): The control strength of COFT. The freedom of rotation. Only has an effect if `coft` is set to True.
            block_share (`bool`): Whether to share the OFT parameters between blocks or not.
        """

        self.r[adapter_name] = r
        self.alpha[adapter_name] = alpha
        self.scaling[adapter_name] = alpha / r
        self.rank_dropout[adapter_name] = rank_dropout
        self.module_dropout[adapter_name] = module_dropout
        self.coft[adapter_name] = coft
        self.eps[adapter_name] = eps
        self.block_share[adapter_name] = block_share

        # Determine shape of OFT weights
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            shape = tuple(base_layer.weight.shape)
        elif isinstance(base_layer, nn.Conv2d):
            use_effective_conv2d = use_effective_conv2d and base_layer.kernel_size != (1, 1)
            if use_effective_conv2d:
                shape = (base_layer.out_channels, base_layer.in_channels, *base_layer.kernel_size)
            else:
                shape = (
                    base_layer.out_channels,
                    base_layer.in_channels * base_layer.kernel_size[0] * base_layer.kernel_size[1],
                )
        else:
            raise TypeError(f"OFT is not implemented for base layers of type {type(base_layer).__name__}")

        # Create weights with provided shape
        self.create_adapter_parameters(adapter_name, r, shape, block_share)

        # Initialize weights
        if init_weights:
            self.reset_adapter_parameters(adapter_name)
        else:
            self.reset_adapter_parameters_random(adapter_name)

        # Move new weights to device
        weight = getattr(self.get_base_layer(), "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )
        if adapter_names is None:
            adapter_names = self.active_adapters

        for active_adapter in adapter_names:
            if active_adapter in self._available_adapters:
                base_layer = self.get_base_layer()

                orig_weights = base_layer.weight.data
                if isinstance(base_layer, nn.Linear):
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                elif isinstance(base_layer, nn.Conv2d):
                    orig_weights = orig_weights.view(
                        [
                            base_layer.kernel_size[0] * base_layer.kernel_size[1] * base_layer.in_channels,
                            base_layer.out_channels,
                        ]
                    )
                orig_weights = torch.mm(self.get_delta_weight(active_adapter), orig_weights)
                if isinstance(base_layer, nn.Linear):
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                elif isinstance(base_layer, nn.Conv2d):
                    orig_weights = orig_weights.view(
                        [
                            base_layer.out_channels,
                            base_layer.in_channels,
                            base_layer.kernel_size[0],
                            base_layer.kernel_size[1],
                        ]
                    )

                if not torch.isfinite(orig_weights).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )

                base_layer.weight.data = orig_weights
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self._available_adapters:
                base_layer = self.base_layer
                orig_weights = base_layer.weight.data
                if isinstance(base_layer, nn.Linear):
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                elif isinstance(base_layer, nn.Conv2d):
                    orig_weights = orig_weights.view(
                        [
                            base_layer.kernel_size[0] * base_layer.kernel_size[1] * base_layer.in_channels,
                            base_layer.out_channels,
                        ]
                    )
                delta_weight_inv = torch.inverse(self.get_delta_weight(active_adapter))
                orig_weights = torch.mm(orig_weights, delta_weight_inv)
                if isinstance(base_layer, nn.Linear):
                    orig_weights = torch.transpose(orig_weights, 0, 1)
                elif isinstance(base_layer, nn.Conv2d):
                    orig_weights = orig_weights.view(
                        [
                            base_layer.out_channels,
                            base_layer.in_channels,
                            base_layer.kernel_size[0],
                            base_layer.kernel_size[1],
                        ]
                    )
                self.base_layer.weight.data = orig_weights

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        scaling = self.scaling[adapter_name]
        rank = self.r[adapter_name]
        coft = self.coft[adapter_name]
        eps = self.eps[adapter_name]
        opt_r = self.oft_r[adapter_name]

        if coft:
            with torch.no_grad():
                opt_r.copy_(self._project_batch(opt_r, eps=eps))

        # Perform rank dropout during training - drop rows of addition weights
        rank_dropout = self.rank_dropout[adapter_name]
        if self.training and rank_dropout:
            drop = (torch.rand(opt_r.size(0)) > rank_dropout).to(opt_r.dtype)
            drop = drop.view(-1, *[1] * len(opt_r.shape[1:])).to(opt_r.device)
            # TODO: Investigate if there should be a scaler like in normal dropout during training
            # Original implementation doesn't have it
            # https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/modules/OFT.py#L193
            drop /= drop.mean()
            opt_r *= drop

        orth_rotate = self._cayley_batch(opt_r * scaling)
        weight = self._block_diagonal(orth_rotate, rank)

        return weight

    def _cayley_batch(self, data: torch.Tensor) -> torch.Tensor:
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.transpose(1, 2))
        I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)

        # Perform the Cayley parametrization
        Q = torch.bmm(I - skew, torch.inverse(I + skew))

        return Q

    def _block_diagonal(self, oft_r: torch.Tensor, rank: int) -> torch.Tensor:
        if oft_r.shape[0] == 1:
            # block share
            blocks = [oft_r[0, ...] for i in range(rank)]
        else:
            blocks = [oft_r[i, ...] for i in range(rank)]

        # Use torch.block_diag to create the block diagonal matrix
        A = torch.block_diag(*blocks)

        return A

    def _project_batch(self, oft_r, eps=1e-5):
        # scaling factor for each of the smaller block matrix
        eps = eps * 1 / torch.sqrt(torch.tensor(oft_r.shape[0]))
        I = (
            torch.zeros((oft_r.size(1), oft_r.size(1)), device=oft_r.device, dtype=oft_r.dtype)
            .unsqueeze(0)
            .expand_as(oft_r)
        )
        diff = oft_r - I
        norm_diff = torch.norm(oft_r - I, dim=(1, 2), keepdim=True)
        mask = (norm_diff <= eps).bool()
        out = torch.where(mask, oft_r, I + eps * (diff / norm_diff))
        return out

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            # Execute all the adapters
            for active_adapter in self.active_adapters:
                if active_adapter not in self._available_adapters:
                    continue

                module_dropout = self.module_dropout[active_adapter]

                if (not self.training) or (self.training and torch.rand(1) > module_dropout):
                    result = self._get_delta_activations(active_adapter, x, *args, **kwargs)
                else:
                    result = self.base_layer(x, *args, **kwargs)

        result = result.to(previous_dtype)
        return result


class Linear(OFTLayer):
    """OFT implemented in Linear layer"""

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str = "default",
        r: int = 0,
        alpha: float = 0.0,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        init_weights: bool = True,
        **kwargs,
    ):
        super().__init__(base_layer)

        # Create adapter and set it active
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, alpha, rank_dropout, module_dropout, init_weights, **kwargs)

    def _get_delta_activations(
        self, adapter_name: str, input: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        delta_weight = self.get_delta_weight(adapter_name)

        base_layer = self.get_base_layer()

        base_weight = base_layer.weight.data
        base_weight = torch.transpose(base_weight, 0, 1)
        weight = torch.mm(delta_weight, base_weight)
        weight = torch.transpose(weight, 0, 1)

        base_bias = base_layer.bias.data if base_layer.bias is not None else None

        return F.linear(input=input, weight=weight, bias=base_bias)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "oft." + rep


class Conv2d(OFTLayer):
    """OFT implemented in Conv2d layer"""

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str = "default",
        r: int = 0,
        alpha: float = 0.0,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        use_effective_conv2d: bool = False,
        init_weights: bool = True,
        **kwargs,
    ):
        super().__init__(base_layer)

        # Create adapter and set it active
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name, r, alpha, rank_dropout, module_dropout, init_weights, use_effective_conv2d, **kwargs
        )

    def _get_delta_activations(
        self, adapter_name: str, input: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        delta_weight = self.get_delta_weight(adapter_name)

        base_layer = self.get_base_layer()

        base_weight = base_layer.weight.data
        base_weight = base_weight.view(
            [base_layer.kernel_size[0] * base_layer.kernel_size[1] * base_layer.in_channels, base_layer.out_channels]
        )
        weight = torch.mm(delta_weight, base_weight)
        weight = weight.view(
            [base_layer.out_channels, base_layer.in_channels, base_layer.kernel_size[0], base_layer.kernel_size[1]]
        )

        base_bias = base_layer.bias.data if base_layer.bias is not None else None

        return F.conv2d(
            input,
            weight,
            bias=base_bias,
            stride=base_layer.stride,
            padding=base_layer.padding,
            dilation=base_layer.dilation,
            groups=base_layer.groups,
        )

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "oft." + rep
