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
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from peft.tuners.lycoris_utils import LycorisLayer

from .config import LoHaConfig


def khatri_rao(w1_b: torch.Tensor, w2_b: torch.Tensor) -> torch.Tensor:
    """Column-wise Kronecker product: (r, d_in), (r, d_in) -> (r*r, d_in)."""
    return (w1_b.unsqueeze(1) * w2_b.unsqueeze(0)).reshape(-1, w1_b.shape[1])


def face_split(w1_a: torch.Tensor, w2_a: torch.Tensor) -> torch.Tensor:
    """Row-wise Kronecker (face-splitting) product: (d_out, r), (d_out, r) -> (d_out, r*r)."""
    return (w1_a.unsqueeze(2) * w2_a.unsqueeze(1)).reshape(w1_a.shape[0], -1)


class LoHaLayer(nn.Module, LycorisLayer):
    # All names of layers that may contain adapter weights
    adapter_layer_names = ("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b", "hada_t1", "hada_t2")
    other_param_names = (
        "r",
        "alpha",
        "scaling",
        "rank_dropout",
        "rank_dropout_scale",
        "module_dropout",
        "use_khatri_rao",
    )

    def __init__(self, base_layer: nn.Module):
        super().__init__()
        LycorisLayer.__init__(self, base_layer)

        # LoHa info
        self.hada_w1_a = nn.ParameterDict({})
        self.hada_w1_b = nn.ParameterDict({})
        self.hada_w2_a = nn.ParameterDict({})
        self.hada_w2_b = nn.ParameterDict({})
        self.hada_t1 = nn.ParameterDict({})
        self.hada_t2 = nn.ParameterDict({})
        self.use_khatri_rao = {}

    @property
    def _available_adapters(self) -> set[str]:
        return {*self.hada_w1_a, *self.hada_w1_b, *self.hada_w2_a, *self.hada_w2_b, *self.hada_t1, *self.hada_t2}

    def create_adapter_parameters(self, adapter_name: str, r: int, shape: tuple[int, ...]):
        # https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/modules/loha.py#L130C9-L143C75
        if len(shape) == 4:  # Conv2d
            self.hada_t1[adapter_name] = nn.Parameter(torch.empty(r, r, shape[2], shape[3]))
            self.hada_w1_a[adapter_name] = nn.Parameter(torch.empty(r, shape[0]))  # out_dim, 1-mode
            self.hada_w1_b[adapter_name] = nn.Parameter(torch.empty(r, shape[1]))  # in_dim , 2-mode

            self.hada_t2[adapter_name] = nn.Parameter(torch.empty(r, r, shape[2], shape[3]))
            self.hada_w2_a[adapter_name] = nn.Parameter(torch.empty(r, shape[0]))  # out_dim, 1-mode
            self.hada_w2_b[adapter_name] = nn.Parameter(torch.empty(r, shape[1]))  # in_dim , 2-mode
        elif len(shape) == 3:  # Conv1d
            self.hada_t1[adapter_name] = nn.Parameter(torch.empty(r, r, shape[2], 1))
            self.hada_w1_a[adapter_name] = nn.Parameter(torch.empty(r, shape[0]))  # out_dim, 1-mode
            self.hada_w1_b[adapter_name] = nn.Parameter(torch.empty(r, shape[1]))  # in_dim , 2-mode

            self.hada_t2[adapter_name] = nn.Parameter(torch.empty(r, r, shape[2], 1))
            self.hada_w2_a[adapter_name] = nn.Parameter(torch.empty(r, shape[0]))  # out_dim, 1-mode
            self.hada_w2_b[adapter_name] = nn.Parameter(torch.empty(r, shape[1]))  # in_dim , 2-mode
        else:  # Linear
            self.hada_w1_a[adapter_name] = nn.Parameter(torch.empty(shape[0], r))
            self.hada_w1_b[adapter_name] = nn.Parameter(torch.empty(r, shape[1]))

            self.hada_w2_a[adapter_name] = nn.Parameter(torch.empty(shape[0], r))
            self.hada_w2_b[adapter_name] = nn.Parameter(torch.empty(r, shape[1]))

    def reset_adapter_parameters(self, adapter_name: str):
        # Original implementation performs initialization with normal distribution
        # https://github.com/KohakuBlueleaf/LyCORIS/blob/3549fdef8f564761d68b695a08ef88b1122fdedc/lycoris/modules/loha.py#L158

        # FedPara paper proposes to perform He initialization, let's stick with it
        # It is enough to initialize only single matrix with zeros to make adapter do nothing after initialization
        if adapter_name in self.hada_w1_a.keys():
            nn.init.kaiming_uniform_(self.hada_w1_a[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.hada_w1_b[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.hada_w2_a[adapter_name], a=math.sqrt(5))
            nn.init.zeros_(self.hada_w2_b[adapter_name])
        if adapter_name in self.hada_t1.keys():
            nn.init.kaiming_uniform_(self.hada_t1[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.hada_t2[adapter_name], a=math.sqrt(5))

    def reset_adapter_parameters_random(self, adapter_name: str):
        # Original implementation performs initialization with normal distribution
        # https://github.com/KohakuBlueleaf/LyCORIS/blob/3549fdef8f564761d68b695a08ef88b1122fdedc/lycoris/modules/loha.py#L158

        # FedPara paper proposes to perform He initialization, let's stick with it
        # It is enough to initialize only single matrix with zeros to make adapter do nothing after initialization
        if adapter_name in self.hada_w1_a.keys():
            nn.init.kaiming_uniform_(self.hada_w1_a[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.hada_w1_b[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.hada_w2_a[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.hada_w2_b[adapter_name], a=math.sqrt(5))
        if adapter_name in self.hada_t1.keys():
            nn.init.kaiming_uniform_(self.hada_t1[adapter_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.hada_t2[adapter_name], a=math.sqrt(5))

    def reset_adapter_parameters_lora_anchored(self, adapter_name: str):
        # LoRA-anchored initialization for the Khatri-Rao path:
        #   - the first factor pair is initialized like a plain LoRA pair (hada_w1_a takes the role of lora_B and is
        #     zeroed, hada_w1_b takes the role of lora_A)
        #   - the second factor pair is anchored so that hada_w2_a @ hada_w2_b is the all-ones matrix, thus the delta
        #     weight starts out as (hada_w1_a @ hada_w1_b) * 1, i.e. exactly a plain LoRA delta
        # The small noise on the remaining rows of hada_w2_b gives the other columns of hada_w2_a a gradient path once
        # hada_w1_a @ hada_w1_b != 0, so higher-rank directions are learned gradually.
        nn.init.zeros_(self.hada_w1_a[adapter_name])
        nn.init.kaiming_uniform_(self.hada_w1_b[adapter_name], a=math.sqrt(5))
        w2_a = self.hada_w2_a[adapter_name]
        w2_b = self.hada_w2_b[adapter_name]
        nn.init.zeros_(w2_a)
        w2_a.data[:, 0] = 1.0
        nn.init.zeros_(w2_b)
        w2_b.data[0] = 1.0
        if w2_b.shape[0] > 1:
            w2_b.data[1:].normal_(std=w2_b.shape[1] ** -0.5)

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        alpha: float,
        config: LoHaConfig,
        **kwargs,
    ) -> None:
        """Internal function to create loha adapter

        Args:
            adapter_name (`str`): Name for the adapter to add.
            r (`int`): Rank for the added adapter.
            alpha (`float`): Alpha for the added adapter.
            config (`LoHaConfig`): The adapter configuration for this layer.
        """
        rank_dropout = config.rank_dropout
        module_dropout = config.module_dropout
        init_weights = config.init_weights
        use_effective_conv2d = config.use_effective_conv2d
        use_khatri_rao = config.use_khatri_rao
        inference_mode = config.inference_mode

        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.alpha[adapter_name] = alpha
        self.scaling[adapter_name] = alpha / r
        self.rank_dropout[adapter_name] = rank_dropout
        self.module_dropout[adapter_name] = module_dropout

        # Determine shape of LoHa weights
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            shape = tuple(base_layer.weight.shape)
        elif isinstance(base_layer, nn.Conv2d):
            # For 1x1 convolutions, disable effective_conv2d to avoid unnecessary tensor reshaping overhead.
            # Since 1x1 convolutions are essentially pointwise operations (matrix multiplications),
            # they can be more efficiently handled with the flattened weight representation,
            # similar to how Linear layers work. This optimization reduces computational cost
            # without affecting the mathematical equivalence of the operation.
            use_effective_conv2d = use_effective_conv2d and base_layer.kernel_size != (1, 1)
            in_channels = base_layer.in_channels // base_layer.groups
            if use_effective_conv2d:
                shape = (base_layer.out_channels, in_channels, *base_layer.kernel_size)
            else:
                shape = (
                    base_layer.out_channels,
                    in_channels * base_layer.kernel_size[0] * base_layer.kernel_size[1],
                )
        elif isinstance(base_layer, nn.Conv1d):
            # For Conv1d with kernel_size=1, disable effective_conv2d for the same optimization reasons
            # as 1x1 Conv2d. Kernel size 1 means no spatial/temporal context, making it equivalent
            # to a Linear layer applied across the channel dimension. Using flattened representation
            # avoids unnecessary reshaping and improves computational efficiency.
            use_effective_conv2d = use_effective_conv2d and base_layer.kernel_size[0] != 1
            in_channels = base_layer.in_channels // base_layer.groups
            if use_effective_conv2d:
                shape = (base_layer.out_channels, in_channels, base_layer.kernel_size[0])
            else:
                shape = (
                    base_layer.out_channels,
                    in_channels * base_layer.kernel_size[0],
                )
        else:
            raise TypeError(f"LoHa is not implemented for base layers of type {type(base_layer).__name__}")

        # The Khatri-Rao identity does not apply to the Tucker decomposition (i.e. the effective conv2d case), so
        # store the flag resolved per adapter: it indicates whether the factored path is actually used, which can
        # deviate from the config value.
        self.use_khatri_rao[adapter_name] = use_khatri_rao and (len(shape) == 2)

        # Create weights with provided shape
        self.create_adapter_parameters(adapter_name, r, shape)

        # Initialize weights
        if init_weights:
            if self.use_khatri_rao[adapter_name]:
                self.reset_adapter_parameters_lora_anchored(adapter_name)
            else:
                self.reset_adapter_parameters(adapter_name)
        else:
            self.reset_adapter_parameters_random(adapter_name)

        # Move new weights to device
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=inference_mode)

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        # https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/modules/loha.py#L178
        if adapter_name in self.hada_t1.keys():
            weight = make_weight_cp(
                self.hada_t1[adapter_name],
                self.hada_w1_a[adapter_name],
                self.hada_w1_b[adapter_name],
                self.hada_t2[adapter_name],
                self.hada_w2_a[adapter_name],
                self.hada_w2_b[adapter_name],
                scale=torch.tensor(self.scaling[adapter_name]),
            )
        else:
            weight = make_weight(
                self.hada_w1_a[adapter_name],
                self.hada_w1_b[adapter_name],
                self.hada_w2_a[adapter_name],
                self.hada_w2_b[adapter_name],
                scale=torch.tensor(self.scaling[adapter_name]),
            )

        base_layer = self.get_base_layer()

        # Reshape to match base layer shape
        weight = weight.reshape(base_layer.weight.shape)

        # Perform rank dropout during training - drop rows of addition weights
        rank_dropout = self.rank_dropout[adapter_name]
        if self.training and rank_dropout:
            drop = (torch.rand(weight.size(0)) > rank_dropout).to(weight.dtype)
            drop = drop.view(-1, *[1] * len(weight.shape[1:])).to(weight.device)
            # TODO: Investigate if there should be a scaler like in normal dropout during training
            # Original implementation doesn't have it
            # https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/modules/loha.py#L193
            drop /= drop.mean()
            weight *= drop

        return weight

    def get_effective_AB(self, adapter_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return factor-sized matrices (A, B) such that scaling * B @ A equals the 2D delta weight.

        This uses the Khatri-Rao identity (w1_a @ w1_b) * (w2_a @ w2_b) = face_split(w1_a, w2_a) @ khatri_rao(w1_b,
        w2_b), i.e. LoHa is an exact rank r² LoRA with A of shape (r², d_in) and B of shape (d_out, r²). The scaling is
        not included in the returned factors. This identity does not apply to the Tucker decomposition, so it cannot be
        used for adapters created with `use_effective_conv2d=True`.

        """
        if adapter_name in self.hada_t1.keys():
            raise ValueError(
                "get_effective_AB does not support the Tucker decomposition (i.e. use_effective_conv2d=True)"
            )

        A = khatri_rao(self.hada_w1_b[adapter_name], self.hada_w2_b[adapter_name])
        B = face_split(self.hada_w1_a[adapter_name], self.hada_w2_a[adapter_name])

        # Perform rank dropout during training; masking the rows of B is equivalent to masking the rows of the delta
        # weight as done in get_delta_weight
        rank_dropout = self.rank_dropout[adapter_name]
        if self.training and rank_dropout:
            drop = (torch.rand(B.size(0)) > rank_dropout).to(B.dtype)
            drop = drop.view(-1, 1).to(B.device)
            drop /= drop.mean()
            B = B * drop

        return A, B

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

            # Execute all the adapters
            for active_adapter in self.active_adapters:
                if active_adapter not in self._available_adapters:
                    continue

                module_dropout = self.module_dropout[active_adapter]

                # Modify current execution weights
                if (not self.training) or (self.training and torch.rand(1) > module_dropout):
                    result = result + self._get_delta_activations(active_adapter, x, *args, **kwargs)

        result = result.to(previous_dtype)
        return result


class Linear(LoHaLayer):
    """LoHa implemented in Linear layer"""

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        config: LoHaConfig,
        r: int = 0,
        alpha: float = 0.0,
        **kwargs,
    ):
        super().__init__(base_layer)

        # Create adapter and set it active
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, alpha, config=config, **kwargs)

    def _get_delta_activations(
        self, adapter_name: str, input: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        if self.use_khatri_rao[adapter_name]:
            # factored execution as a rank r² LoRA, the full delta weight is never materialized
            A, B = self.get_effective_AB(adapter_name)
            input = self._cast_input_dtype(input, A.dtype)
            return self.scaling[adapter_name] * F.linear(F.linear(input, A), B)

        delta_weight = self.get_delta_weight(adapter_name)
        input = self._cast_input_dtype(input, delta_weight.dtype)
        # don't add bias here, because the bias is already included in the output of the base_layer
        return F.linear(input, delta_weight)

    def supports_lora_conversion(self, adapter_name: str = "default") -> bool:
        return True

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "loha." + rep


class Conv2d(LoHaLayer):
    """LoHa implemented in Conv2d layer"""

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        config: LoHaConfig,
        r: int = 0,
        alpha: float = 0.0,
        **kwargs,
    ):
        super().__init__(base_layer)

        # Create adapter and set it active
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, alpha, config=config, **kwargs)

    def _get_delta_activations(
        self, adapter_name: str, input: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        base_layer = self.get_base_layer()
        if self.use_khatri_rao[adapter_name]:
            # Factored execution: the flattened delta weight is B @ A, so the delta conv splits into a conv with the
            # rows of A as kernels, followed by a 1x1 conv with B. For grouped convs, A is tiled so that the inputs of
            # each group are projected onto the same r² effective kernels.
            A, B = self.get_effective_AB(adapter_name)
            input = self._cast_input_dtype(input, A.dtype)
            groups = base_layer.groups
            weight_A = A.view(-1, *base_layer.weight.shape[1:])
            if groups > 1:
                weight_A = weight_A.repeat(groups, 1, 1, 1)
            hidden = F.conv2d(
                input,
                weight_A,
                stride=base_layer.stride,
                padding=base_layer.padding,
                dilation=base_layer.dilation,
                groups=groups,
            )
            weight_B = B.unsqueeze(-1).unsqueeze(-1)
            result = self.scaling[adapter_name] * F.conv2d(hidden, weight_B, groups=groups)
        else:
            delta_weight = self.get_delta_weight(adapter_name)
            input = self._cast_input_dtype(input, delta_weight.dtype)
            # don't add bias here, because the bias is already included in the output of the base_layer
            result = F.conv2d(
                input,
                delta_weight,
                stride=base_layer.stride,
                padding=base_layer.padding,
                dilation=base_layer.dilation,
                groups=base_layer.groups,
            )
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "loha." + rep


class Conv1d(LoHaLayer):
    """LoHa implemented in Conv1d layer"""

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        config: LoHaConfig,
        r: int = 0,
        alpha: float = 0.0,
        **kwargs,
    ):
        super().__init__(base_layer)

        # Create adapter and set it active
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, alpha, config=config, **kwargs)

    def _get_delta_activations(
        self, adapter_name: str, input: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        base_layer = self.get_base_layer()
        if self.use_khatri_rao[adapter_name]:
            # Factored execution, see the comment in Conv2d._get_delta_activations
            A, B = self.get_effective_AB(adapter_name)
            input = self._cast_input_dtype(input, A.dtype)
            groups = base_layer.groups
            weight_A = A.view(-1, *base_layer.weight.shape[1:])
            if groups > 1:
                weight_A = weight_A.repeat(groups, 1, 1)
            hidden = F.conv1d(
                input,
                weight_A,
                stride=base_layer.stride,
                padding=base_layer.padding,
                dilation=base_layer.dilation,
                groups=groups,
            )
            weight_B = B.unsqueeze(-1)
            result = self.scaling[adapter_name] * F.conv1d(hidden, weight_B, groups=groups)
        else:
            delta_weight = self.get_delta_weight(adapter_name)
            input = self._cast_input_dtype(input, delta_weight.dtype)
            # don't add bias here, because the bias is already included in the output of the base_layer
            result = F.conv1d(
                input,
                delta_weight,
                stride=base_layer.stride,
                padding=base_layer.padding,
                dilation=base_layer.dilation,
                groups=base_layer.groups,
            )
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "loha." + rep


# Below code is a direct copy from https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/modules/loha.py#L9


class HadaWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w1a, w1b, w2a, w2b, scale=torch.tensor(1)):  # noqa: B008
        ctx.save_for_backward(w1a, w1b, w2a, w2b, scale)
        diff_weight = ((w1a @ w1b) * (w2a @ w2b)) * scale
        return diff_weight

    @staticmethod
    def backward(ctx, grad_out):
        (w1a, w1b, w2a, w2b, scale) = ctx.saved_tensors
        grad_out = grad_out * scale
        temp = grad_out * (w2a @ w2b)
        grad_w1a = temp @ w1b.T
        grad_w1b = w1a.T @ temp

        temp = grad_out * (w1a @ w1b)
        grad_w2a = temp @ w2b.T
        grad_w2b = w2a.T @ temp

        del temp
        return grad_w1a, grad_w1b, grad_w2a, grad_w2b, None


class HadaWeightCP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t1, w1a, w1b, t2, w2a, w2b, scale=torch.tensor(1)):  # noqa: B008
        ctx.save_for_backward(t1, w1a, w1b, t2, w2a, w2b, scale)

        rebuild1 = torch.einsum("i j k l, j r, i p -> p r k l", t1, w1b, w1a)
        rebuild2 = torch.einsum("i j k l, j r, i p -> p r k l", t2, w2b, w2a)

        return rebuild1 * rebuild2 * scale

    @staticmethod
    def backward(ctx, grad_out):
        (t1, w1a, w1b, t2, w2a, w2b, scale) = ctx.saved_tensors
        grad_out = grad_out * scale

        temp = torch.einsum("i j k l, j r -> i r k l", t2, w2b)
        rebuild = torch.einsum("i j k l, i r -> r j k l", temp, w2a)

        grad_w = rebuild * grad_out
        del rebuild

        grad_w1a = torch.einsum("r j k l, i j k l -> r i", temp, grad_w)
        grad_temp = torch.einsum("i j k l, i r -> r j k l", grad_w, w1a.T)
        del grad_w, temp

        grad_w1b = torch.einsum("i r k l, i j k l -> r j", t1, grad_temp)
        grad_t1 = torch.einsum("i j k l, j r -> i r k l", grad_temp, w1b.T)
        del grad_temp

        temp = torch.einsum("i j k l, j r -> i r k l", t1, w1b)
        rebuild = torch.einsum("i j k l, i r -> r j k l", temp, w1a)

        grad_w = rebuild * grad_out
        del rebuild

        grad_w2a = torch.einsum("r j k l, i j k l -> r i", temp, grad_w)
        grad_temp = torch.einsum("i j k l, i r -> r j k l", grad_w, w2a.T)
        del grad_w, temp

        grad_w2b = torch.einsum("i r k l, i j k l -> r j", t2, grad_temp)
        grad_t2 = torch.einsum("i j k l, j r -> i r k l", grad_temp, w2b.T)
        del grad_temp
        return grad_t1, grad_w1a, grad_w1b, grad_t2, grad_w2a, grad_w2b, None


def make_weight(w1a, w1b, w2a, w2b, scale):
    return HadaWeight.apply(w1a, w1b, w2a, w2b, scale)


def make_weight_cp(t1, w1a, w1b, t2, w2a, w2b, scale):
    return HadaWeightCP.apply(t1, w1a, w1b, t2, w2a, w2b, scale)
