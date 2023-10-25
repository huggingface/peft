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
from typing import Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.lycoris_utils import LycorisLayer


class LoKrLayer(LycorisLayer, nn.Module):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = [
        "lokr_w1",
        "lokr_w1_a",
        "lokr_w1_b",
        "lokr_w2",
        "lokr_w2_a",
        "lokr_w2_b",
        "lokr_t2",
    ]

    def __init__(self):
        LycorisLayer.__init__(self)
        super(nn.Module, self).__init__()

        # LoKr info
        self.lokr_w1 = nn.ParameterDict({})
        self.lokr_w1_a = nn.ParameterDict({})
        self.lokr_w1_b = nn.ParameterDict({})
        self.lokr_w2 = nn.ParameterDict({})
        self.lokr_w2_a = nn.ParameterDict({})
        self.lokr_w2_b = nn.ParameterDict({})
        self.lokr_t2 = nn.ParameterDict({})

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

    def reset_adapter_parameters(self, adapter_name: str):
        if adapter_name in self.lokr_w1:
            nn.init.zeros_(self.lokr_w1[adapter_name])
        else:
            nn.init.zeros_(self.lokr_w1_a[adapter_name])
            nn.init.kaiming_uniform_(self.lokr_w1_b[adapter_name], a=math.sqrt(5))

        if adapter_name in self.lokr_w2:
            nn.init.kaiming_uniform_(self.lokr_w2[adapter_name], a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(self.lokr_w2_a[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lokr_w2_b[adapter_name], a=math.sqrt(5))

        if adapter_name in self.lokr_t2:
            nn.init.kaiming_uniform_(self.lokr_t2[adapter_name], a=math.sqrt(5))

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

        self.r[adapter_name] = r
        self.alpha[adapter_name] = alpha
        self.scaling[adapter_name] = alpha / r
        self.rank_dropout[adapter_name] = rank_dropout
        self.module_dropout[adapter_name] = module_dropout

        # Determine shape of LoKr weights
        if isinstance(self, nn.Linear):
            in_dim, out_dim = self.in_features, self.out_features

            in_m, in_n = factorization(in_dim, decompose_factor)
            out_l, out_k = factorization(out_dim, decompose_factor)
            shape = ((out_l, out_k), (in_m, in_n))  # ((a, b), (c, d)), out_dim = a*c, in_dim = b*d

            use_w1 = not (decompose_both and r < max(shape[0][0], shape[1][0]) / 2)
            use_w2 = not (r < max(shape[0][1], shape[1][1]) / 2)
            use_effective_conv2d = False
        elif isinstance(self, nn.Conv2d):
            in_dim, out_dim = self.in_channels, self.out_channels
            k_size = self.kernel_size

            in_m, in_n = factorization(in_dim, decompose_factor)
            out_l, out_k = factorization(out_dim, decompose_factor)
            shape = ((out_l, out_k), (in_m, in_n), *k_size)  # ((a, b), (c, d), *k_size)

            use_w1 = not (decompose_both and r < max(shape[0][0], shape[1][0]) / 2)
            use_w2 = r >= max(shape[0][1], shape[1][1]) / 2
            use_effective_conv2d = use_effective_conv2d and self.kernel_size != (1, 1)
        else:
            raise TypeError(f"LoKr is not implemented for {type(self).__name__} layer")

        # Create weights with provided shape
        self.create_adapter_parameters(adapter_name, r, shape, use_w1, use_w2, use_effective_conv2d)

        # Initialize weights
        if init_weights:
            self.reset_adapter_parameters(adapter_name)

        # Move new weights to device
        weight = getattr(self, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        # https://github.com/KohakuBlueleaf/LyCORIS/blob/e4259b870d3354a9615a96be61cb5d07455c58ea/lycoris/modules/lokr.py#L224
        if adapter_name in self.lokr_w1:
            w1 = self.lokr_w1[adapter_name]
        else:
            w1 = self.lokr_w1_a[adapter_name] @ self.lokr_w1_b[adapter_name]

        if adapter_name in self.lokr_w2:
            w2 = self.lokr_w2[adapter_name]
        elif adapter_name in self.lokr_t2:
            w2 = make_weight_cp(self.lokr_t2[adapter_name], self.lokr_w2_a[adapter_name], self.lokr_w2_b[adapter_name])
        else:
            w2 = self.lokr_w2_a[adapter_name] @ self.lokr_w2_b[adapter_name]

        # Make weights with Kronecker product
        weight = make_kron(w1, w2)
        weight = weight.reshape(self.weight.shape)

        # Perform rank dropout during training - drop rows of addition weights
        rank_dropout = self.rank_dropout[adapter_name]
        if self.training and rank_dropout:
            drop = (torch.rand(weight.size(0)) > rank_dropout).float()
            drop = drop.view(-1, *[1] * len(weight.shape[1:])).to(weight.device)
            drop /= drop.mean()
            weight *= drop

        return weight


class Linear(LoKrLayer, nn.Linear):
    """LoKr implemented in Linear layer"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        adapter_name: str = "default",
        r: int = 0,
        alpha: float = 0.0,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        **kwargs,
    ):
        init_weights = kwargs.pop("init_weights", True)
        self._init_empty_weights(nn.Linear, in_features, out_features, bias, device=device, dtype=dtype)

        LoKrLayer.__init__(self)

        # Create adapter and set it active
        self.update_layer(adapter_name, r, alpha, rank_dropout, module_dropout, init_weights, **kwargs)
        self.set_adapter(adapter_name)

    def _op(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return F.linear(input, weight, bias=self.bias)


class Conv2d(LoKrLayer, nn.Conv2d):
    """LoKr implemented in Conv2d layer"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        adapter_name: str = "default",
        r: int = 0,
        alpha: float = 0.0,
        rank_dropout: float = 0.0,
        module_dropout: float = 0.0,
        use_effective_conv2d: bool = False,
        **kwargs,
    ):
        init_weights = kwargs.pop("init_weights", True)
        self._init_empty_weights(
            nn.Conv2d,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

        LoKrLayer.__init__(self)

        # Create adapter and set it active
        self.update_layer(
            adapter_name, r, alpha, rank_dropout, module_dropout, init_weights, use_effective_conv2d, **kwargs
        )
        self.set_adapter(adapter_name)

    def _op(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            input,
            weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


# Below code is a direct copy from https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/modules/lokr.py#L11


def factorization(dimension: int, factor: int = -1) -> Tuple[int, int]:
    """Factorizes the provided number into the product of two numbers

    Args:
        dimension (`int`): The number that needs to be factorized.
        factor (`int`, optional):
            Factorization divider. The algorithm will try to output two numbers, one of each will be as close to the
            factor as possible. If -1 is provided, the decomposition algorithm would try to search dividers near the
            square root of the dimension. Defaults to -1.

    Returns:
        Tuple[`int`, `int`]: A tuple of two numbers, whose product is equal to the provided number. The first number is
        always less than or equal to the second.

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


def make_weight_cp(t, wa, wb):
    rebuild2 = torch.einsum("i j k l, i p, j r -> p r k l", t, wa, wb)  # [c, d, k1, k2]
    return rebuild2


def make_kron(w1, w2, scale=1.0):
    if len(w2.shape) == 4:
        w1 = w1.unsqueeze(2).unsqueeze(2)
    w2 = w2.contiguous()
    rebuild = torch.kron(w1, w2)

    return rebuild * scale
