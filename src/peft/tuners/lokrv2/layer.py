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

# weight decompose, fullm atrix and rslora remaining

import math
import warnings
from typing import Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lycoris.functional import lokr

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge


class LoKrLayerv2(BaseTunerLayer):
    # All names of layers that may contain adapter weights
    adapter_layer_names = (
        "lokr_w1",
        "lokr_w1_a",
        "lokr_w1_b",
        "lokr_w2",
        "lokr_w2_a",
        "lokr_w2_b",
        "lokr_t2",
    )
    # Other params which may contain adapter related keys
    other_param_names = ("r", "alpha", "scaling", "rank_dropout", "module_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.r = {}
        self.alpha = {}
        self.scale = {}
        self.scaling = {}
        self.rank_dropout = {}
        self.module_dropout = {}

        self.lokr_w1 = nn.ParameterDict({})
        self.lokr_w1_a = nn.ParameterDict({})
        self.lokr_w1_b = nn.ParameterDict({})
        self.lokr_w2 = nn.ParameterDict({})
        self.lokr_w2_a = nn.ParameterDict({})
        self.lokr_w2_b = nn.ParameterDict({})
        self.lokr_t2 = nn.ParameterDict({})

        base_layer = self.get_base_layer()
        self.kernel_size = None

        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            (
                in_features,
                out_features,
            ) = base_layer.in_channels, base_layer.out_channels
            self.kernel_size = base_layer.kernel_size
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features

        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    @property
    def _available_adapters(self) -> Set[str]:
        return {
            *self.lokr_w1,
            *self.lokr_w1_a,
            *self.lokr_w1_b,
            *self.lokr_w2,
            *self.lokr_w2_a,
            *self.lokr_w2_b,
            *self.lokr_t2,
        }

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        alpha: float,
        rank_dropout: float,
        module_dropout: float,
        init_weights: bool,
        use_effective_conv2d: bool,
        decompose_both: bool,
        decompose_factor: int,
        use_upstream: bool = False,
        **kwargs,
    ) -> None:
        """Internal function to create lokr adapter

        Args:
            adapter_name (`str`): Name for the adapter to add.
            r (`int`): Rank for the added adapter.
            alpha (`float`): Alpha for the added adapter.
            rank_dropout (`float`): The dropout probability for rank dimension during training
            module_dropout (`float`): The dropout probability for disabling adapter during training.
            init_weights (`bool`): Whether to initialize adapter weights.
            use_effective_conv2d (`bool`): Use parameter effective decomposition for Conv2d with ksize > 1.
            decompose_both (`bool`): Perform rank decomposition of left kronecker product matrix.
            decompose_factor (`int`): Kronecker product decomposition factor.
        """
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.alpha[adapter_name] = alpha
        self.scaling[adapter_name] = alpha / r
        self.rank_dropout[adapter_name] = rank_dropout
        self.module_dropout[adapter_name] = module_dropout
        base_layer = self.get_base_layer()

        in_m, in_n = self.factorization(self.in_features, decompose_factor)
        out_l, out_k = self.factorization(self.out_features, decompose_factor)

        if isinstance(base_layer, nn.Conv2d):  # For Conv2d
            shape = ((out_l, out_k), (in_m, in_n), *self.kernel_size)
            use_w2 = r >= max(shape[0][1], shape[1][1]) / 2
            use_effective_conv2d = use_effective_conv2d and self.kernel_size != (1, 1)
        else:
            shape = ((out_l, out_k), (in_m, in_n))
            use_w2 = not (r < max(shape[0][1], shape[1][1]) / 2)

        use_w1 = not (decompose_both and r < max(shape[0][0], shape[1][0]) / 2)

        if use_upstream:
            if self.kernel_size:
                dummy_weights = torch.rand((self.in_features, self.out_features, *self.kernel_size))
            else:
                dummy_weights = torch.rand((self.in_features, self.out_features))

            weights = lokr.weight_gen(dummy_weights, rank=r, decompose_both=decompose_both, **kwargs)
            attributes = [
                "lokr_w1",
                "lokr_w1_a",
                "lokr_w1_b",
                "lokr_w2",
                "lokr_w2_a",
                "lokr_w2_b",
                "lokr_t2",
            ]

            for attr, weight in zip(attributes, weights):
                if weight is not None:
                    setattr(self, f"{attr}[{adapter_name}]", weight)
        else:
            self.create_adapter_parameters(adapter_name, r, shape, use_w1, use_w2, use_effective_conv2d)
            self.reset_lokr_parameters(adapter_name, init_weights)

        # Move new weights to device
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def create_adapter_parameters(
        self,
        adapter_name: str,
        r: int,
        shape,
        use_w1: bool,
        use_w2: bool,
        use_effective_conv2d: bool,
    ):
        if use_w1:
            self.lokr_w1[adapter_name] = nn.Parameter(torch.empty(shape[0][0], shape[1][0]))
        else:
            self.lokr_w1_a[adapter_name] = nn.Parameter(torch.empty(shape[0][0], r))
            self.lokr_w1_b[adapter_name] = nn.Parameter(torch.empty(r, shape[1][0]))

        if len(shape) == 4:
            # Conv2d
            if use_w2:
                self.lokr_w2[adapter_name] = nn.Parameter(torch.empty(shape[0][1], shape[1][1], *shape[2:]))
            elif use_effective_conv2d:
                self.lokr_t2[adapter_name] = nn.Parameter(torch.empty(r, r, shape[2], shape[3]))
                self.lokr_w2_a[adapter_name] = nn.Parameter(torch.empty(r, shape[0][1]))  # b, 1-mode
                self.lokr_w2_b[adapter_name] = nn.Parameter(torch.empty(r, shape[1][1]))  # d, 2-mode
            else:
                self.lokr_w2_a[adapter_name] = nn.Parameter(torch.empty(shape[0][1], r))
                self.lokr_w2_b[adapter_name] = nn.Parameter(torch.empty(r, shape[1][1] * shape[2] * shape[3]))
        else:
            # Linear
            if use_w2:
                self.lokr_w2[adapter_name] = nn.Parameter(torch.empty(shape[0][1], shape[1][1]))
            else:
                self.lokr_w2_a[adapter_name] = nn.Parameter(torch.empty(shape[0][1], r))
                self.lokr_w2_b[adapter_name] = nn.Parameter(torch.empty(r, shape[1][1]))

    def reset_lokr_parameters(self, adapter_name: str, init_weights: bool):
        if init_weights:
            if adapter_name in self.lokr_w1:
                nn.init.zeros_(self.lokr_w1[adapter_name])
            else:
                nn.init.zeros_(self.lokr_w1_a[adapter_name])
                nn.init.kaiming_uniform_(self.lokr_w1_b[adapter_name], a=math.sqrt(5))
        else:
            if adapter_name in self.lokr_w1:
                nn.init.kaiming_uniform_(self.lokr_w1[adapter_name], a=math.sqrt(5))
            else:
                nn.init.kaiming_uniform_(self.lokr_w1_a[adapter_name], a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.lokr_w1_b[adapter_name], a=math.sqrt(5))

        if adapter_name in self.lokr_w2:
            nn.init.kaiming_uniform_(self.lokr_w2[adapter_name], a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(self.lokr_w2_a[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lokr_w2_b[adapter_name], a=math.sqrt(5))

        if adapter_name in self.lokr_t2:
            nn.init.kaiming_uniform_(self.lokr_t2[adapter_name], a=math.sqrt(5))

    def factorization(self, dimension: int, factor: int = -1) -> Tuple[int, int]:
        """Factorizes the provided number into the product of two numbers

        Args:
            dimension (`int`): The number that needs to be factorized.
            factor (`int`, optional):
                Factorization divider. The algorithm will try to output two numbers, one of each will be as close to
                the factor as possible. If -1 is provided, the decomposition algorithm would try to search dividers
                near the square root of the dimension. Defaults to -1.

        Returns:
            Tuple[`int`, `int`]: A tuple of two numbers, whose product is equal to the provided number. The first
            number is always less than or equal to the second.

        Example:
            ```py
            >>> factorization(256, factor=-1)
            (16, 16)

            >>> factorization(128, factor=-1)
            (8, 16)

            >>> factorization(127, factor=-1)
            (1, 127)

            >>> factorization(128, factor=4)
            (4, 32)
            ```
        """
        if factor > 0 and (dimension % factor) == 0:
            m = factor
            n = dimension // factor
            return m, n
        if factor == -1:
            factor = dimension
        m, n = 1, dimension
        length = m + n
        while m < n:
            new_m = m + 1
            while dimension % new_m != 0:
                new_m += 1
            new_n = dimension // new_m
            if new_m + new_n > length or new_m > factor:
                break
            else:
                m, n = new_m, new_n
        if m > n:
            n, m = m, n
        return m, n

    def make_weight_cp(self, t, wa, wb):
        rebuild2 = torch.einsum("i j k l, i p, j r -> p r k l", t, wa, wb)  # [c, d, k1, k2]
        return rebuild2

    def make_kron(self, w1, w2, scale=1.0):
        if len(w2.shape) == 4:
            w1 = w1.unsqueeze(2).unsqueeze(2)
        w2 = w2.contiguous()
        rebuild = torch.kron(w1, w2)

        return rebuild * scale


class Linear(nn.Linear, LoKrLayerv2):
    """LoKr implemented in Linear layer"""

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str = "default",
        r: int = 0,
        alpha: float = 0.0,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        init_weights: bool | str = True,
        fan_in_fan_out: bool = False,
        **kwargs,
    ):
        super(nn.Linear, self).__init__()
        LoKrLayerv2.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        # Create adapter and set it active
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, alpha, rank_dropout, module_dropout, init_weights, **kwargs)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)

        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self._available_adapters:
                base_layer = self.get_base_layer()
                # Note that safe_merge will be slower than the normal merge
                # because of the copy operation.
                if safe_merge:
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    orig_weights += delta_weight

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self._available_adapters:
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        # https://github.com/KohakuBlueleaf/LyCORIS/blob/e4259b870d3354a9615a96be61cb5d07455c58ea/lycoris/modules/lokr.py#L224
        if adapter_name in self.lokr_w1:
            w1 = self.lokr_w1[adapter_name]
        else:
            w1 = self.lokr_w1_a[adapter_name] @ self.lokr_w1_b[adapter_name]

        if adapter_name in self.lokr_w2:
            w2 = self.lokr_w2[adapter_name]
        elif adapter_name in self.lokr_t2:
            w2 = self.make_weight_cp(
                self.lokr_t2[adapter_name], self.lokr_w2_a[adapter_name], self.lokr_w2_b[adapter_name]
            )
        else:
            w2 = self.lokr_w2_a[adapter_name] @ self.lokr_w2_b[adapter_name]

        device = w1.device
        dtype = w1.dtype

        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)
        if cast_to_fp32:
            w1 = w1.float()
            w2 = w2.float()

        # Make weights with Kronecker product
        weight = self.make_kron(w1, w2)
        weight = weight.reshape(self.get_base_layer().weight.shape)

        # Perform rank dropout during training - drop rows of addition weights
        rank_dropout = self.rank_dropout[adapter_name]
        if self.training and rank_dropout:
            drop = (torch.rand(weight.size(0)) > rank_dropout).float()
            drop = drop.view(-1, *[1] * len(weight.shape[1:])).to(weight.device)
            # consider adapter name check
            # if self.kwargs["rank_dropout_scale"]:
            drop /= drop.mean()
            weight *= drop

        if cast_to_fp32:
            weight = weight.to(dtype=dtype)

        return weight

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)

            for active_adapter in self.active_adapters:
                if active_adapter not in self._available_adapters:
                    continue

                module_dropout = self.module_dropout[active_adapter]

                # Modify current execution weights
                if (not self.training) or (self.training and torch.rand(1) > module_dropout):
                    result = result + F.linear(x, self.get_delta_weight(active_adapter))

            result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lokr." + rep


class Conv2d(nn.Module, LoKrLayerv2):
    """LoKr implemented in Conv2d layer"""

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str = "default",
        r: int = 0,
        alpha: float = 0.0,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        use_effective_conv2d: bool = False,
        init_weights: bool = True,
        **kwargs,
    ):
        super().__init__()
        LoKrLayerv2.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        # Create adapter and set it active
        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name, r, alpha, rank_dropout, module_dropout, init_weights, use_effective_conv2d, **kwargs
        )

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)

        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self._available_adapters:
                base_layer = self.get_base_layer()
                # Note that safe_merge will be slower than the normal merge
                # because of the copy operation.
                if safe_merge:
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    orig_weights += delta_weight

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self._available_adapters:
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        if adapter_name in self.lokr_w1:
            w1 = self.lokr_w1[adapter_name]
        else:
            w1 = self.lokr_w1_a[adapter_name] @ self.lokr_w1_b[adapter_name]

        if adapter_name in self.lokr_w2:
            w2 = self.lokr_w2[adapter_name]
        elif adapter_name in self.lokr_t2:
            w2 = self.make_weight_cp(
                self.lokr_t2[adapter_name], self.lokr_w2_a[adapter_name], self.lokr_w2_b[adapter_name]
            )
        else:
            w2 = self.lokr_w2_a[adapter_name] @ self.lokr_w2_b[adapter_name]

        device = w1.device
        dtype = w1.dtype

        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)
        if cast_to_fp32:
            w1 = w1.float()
            w2 = w2.float()

        # Make weights with Kronecker product
        weight = self.make_kron(w1, w2)
        weight = weight.reshape(self.get_base_layer().weight.shape)

        # Perform rank dropout during training - drop rows of addition weights
        rank_dropout = self.rank_dropout[adapter_name]
        if self.training and rank_dropout:
            drop = (torch.rand(weight.size(0)) > rank_dropout).float()
            drop = drop.view(-1, *[1] * len(weight.shape[1:])).to(weight.device)
            # if self.kwargs["rank_dropout_scale"]:
            drop /= drop.mean()
            weight *= drop

        if cast_to_fp32:
            weight = weight.to(dtype=dtype)

        return weight

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            base_layer = self.get_base_layer()
            for active_adapter in self.active_adapters:
                if active_adapter not in self._available_adapters:
                    continue

                module_dropout = self.module_dropout[active_adapter]

                # Modify current execution weights
                if (not self.training) or (self.training and torch.rand(1) > module_dropout):
                    result = result + F.conv2d(
                        x,
                        self.get_delta_weight(active_adapter),
                        stride=base_layer.stride,
                        padding=base_layer.padding,
                        dilation=base_layer.dilation,
                        groups=base_layer.groups,
                    )

            result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lokr." + rep
