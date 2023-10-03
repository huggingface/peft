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
from itertools import chain
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer


class LoKrLayer(BaseTunerLayer, nn.Module):
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
        super(nn.Module, self).__init__()

        # LoKr info
        self.r = {}
        self.alpha = {}
        self.scaling = {}
        self.lokr_w1 = nn.ParameterDict({})
        self.lokr_w1_a = nn.ParameterDict({})
        self.lokr_w1_b = nn.ParameterDict({})
        self.lokr_w2 = nn.ParameterDict({})
        self.lokr_w2_a = nn.ParameterDict({})
        self.lokr_w2_b = nn.ParameterDict({})
        self.lokr_t2 = nn.ParameterDict({})
        self.rank_dropout = {}
        self.module_dropout = {}

        # Tuner info
        self.merged = False
        self._disable_adapters = False
        self.merged_adapters = []

    def _init_empty_weights(self, cls, *args, **kwargs) -> None:
        # A helper method that allows to initialize the layer of the given class without spending time to initialize the
        # model weights. The implementation is inspired by
        # https://pytorch.org/docs/stable/generated/torch.nn.utils.skip_init.html but this function cannot be used
        # directly.
        # Instead of this approach, it would be possible to bypass the __init__ of the class but that runs the risk of
        # omitting important logic inside that __init__.
        kwargs = kwargs.copy()
        final_device = kwargs.pop("device", "cpu")
        cls.__init__(self, *args, device="meta", **kwargs)
        self.to_empty(device=final_device)

    @property
    def _available_adapters(self) -> Iterable[str]:
        return set(
            chain(
                self.lokr_w1.keys(),
                self.lokr_w1_a.keys(),
                self.lokr_w1_b.keys(),
                self.lokr_w2.keys(),
                self.lokr_w2_a.keys(),
                self.lokr_w2_b.keys(),
                self.lokr_t2.keys(),
            )
        )

    def create_lokr_parameters(
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

    def reset_lokr_parameters(self, adapter_name: str):
        if adapter_name in self.lokr_w1:
            nn.init.kaiming_uniform_(self.lokr_w1[adapter_name], a=math.sqrt(5))
        if adapter_name in self.lokr_w2:
            nn.init.kaiming_uniform_(self.lokr_w2[adapter_name], a=math.sqrt(5))
        if adapter_name in self.lokr_w1_a:
            nn.init.kaiming_uniform_(self.lokr_w1_a[adapter_name], a=math.sqrt(5))
            nn.init.zeros_(self.lokr_w1_b[adapter_name])
        if adapter_name in self.lokr_w2_a:
            nn.init.kaiming_uniform_(self.lokr_w2_a[adapter_name], a=math.sqrt(5))
            nn.init.zeros_(self.lokr_w2_b[adapter_name])
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
            shape (`Tuple[int, ...]`): Shape of weights to produce
            adapter_name (`str`): Name for the adapter to add
            r (`int`): Rank for the added adapter
            alpha (`float`): Alpha for the added adapter
            rank_dropout (`float`): The dropout probability for rank dimension during training
            module_dropout (`float`): The dropout probability for disabling adapter during training.
            init_weights (`bool`): Whether to initialize weights
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
            raise NotImplementedError(f"LoHa is not implemented for {type(self).__name__} layer")

        # Create weights with provided shape
        self.create_lokr_parameters(adapter_name, r, shape, use_w1, use_w2, use_effective_conv2d)

        # Initialize weights
        if init_weights:
            self.reset_lokr_parameters(adapter_name)

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
        # https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/modules/loha.py#L178
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

    def merge(self) -> None:
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )
        for active_adapter in self.active_adapters:
            # if active_adapter in self.hada_w1_a.keys():
            if active_adapter in self._available_adapters:
                self.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)
                self.merged = True

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            # if active_adapter in self.hada_w1_a.keys():
            if active_adapter in self._available_adapters:
                self.weight.data -= self.get_delta_weight(active_adapter)
                self.merged = False

    def _op(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self._op(x, self.weight)
        elif self.merged:
            result = self._op(x, self.weight)
        else:
            # Get base weights
            weight = self.weight.data

            # Execute all the adapters
            for active_adapter in self.active_adapters:
                # if active_adapter not in self.hada_w1_a.keys():
                if active_adapter not in self._available_adapters:
                    continue

                module_dropout = self.module_dropout[active_adapter]

                # Modify current execution weights
                if (not self.training) or (self.training and torch.rand(1) > module_dropout):
                    weight = weight + self.get_delta_weight(active_adapter)

            # Perform actual operation
            result = self._op(x, weight)

        result = result.to(previous_dtype)
        return result

    def scale_layer(self, scale_factor: float) -> None:
        if scale_factor != 1:
            for active_adapter in self.active_adapters:
                alpha = self.alpha[active_adapter]
                r = self.r[active_adapter]
                self.scaling[active_adapter] = (alpha / r) * scale_factor

    def unscale_layer(self) -> None:
        for active_adapter in self.active_adapters:
            alpha = self.alpha[active_adapter]
            r = self.r[active_adapter]
            self.scaling[active_adapter] = alpha / r


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
    """
    return a tuple of two value of input dimension decomposed by the number closest to factor second value is higher or
    equal than first value.

    In LoRA with Kroneckor Product, first value is a value for weight scale. secon value is a value for weight.

    Becuase of non-commutative property, A⊗B ≠ B⊗A. Meaning of two matrices is slightly different.

    examples) factor
        -1 2 4 8 16 ...
    127 -> 127, 1 127 -> 127, 1 127 -> 127, 1 127 -> 127, 1 127 -> 127, 1 128 -> 16, 8 128 -> 64, 2 128 -> 32, 4 128 ->
    16, 8 128 -> 16, 8 250 -> 125, 2 250 -> 125, 2 250 -> 125, 2 250 -> 125, 2 250 -> 125, 2 360 -> 45, 8 360 -> 180, 2
    360 -> 90, 4 360 -> 45, 8 360 -> 45, 8 512 -> 32, 16 512 -> 256, 2 512 -> 128, 4 512 -> 64, 8 512 -> 32, 16 1024 ->
    32, 32 1024 -> 512, 2 1024 -> 256, 4 1024 -> 128, 8 1024 -> 64, 16
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
