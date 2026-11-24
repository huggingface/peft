# Copyright 2026-present the HuggingFace Inc. team.
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
from typing import Optional

import torch
from torch import nn

from peft.tuners.tuners_utils import BaseTunerLayer, _get_in_out_features, check_adapters_to_merge
from peft.utils import quantization_extra_repr, resolve_quantization_backend
from peft.utils.other import transpose

from .config import DummyConfig


class DummyLayer(BaseTunerLayer):
    # List all names of layers that may contain (trainable) adapter weights. Note that the names contain the prefix of
    # the method ("dummy_"), which is relied upon when filtering adapter weights from a state dict.
    adapter_layer_names = ("dummy_lora_A", "dummy_lora_B")
    # All names of other adapter-related attributes (no trainable parameters)
    other_param_names = ("r", "scaling", "dummy_lora_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.quantization_backend = resolve_quantization_backend(
            self.get_base_layer(), get_apply_tensor_subclass=kwargs.get("get_apply_tensor_subclass")
        )
        self.dummy_lora_A = nn.ModuleDict({})
        self.dummy_lora_B = nn.ModuleDict({})
        self.dummy_lora_dropout = nn.ModuleDict({})
        self.r = {}
        self.scaling = {}

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        in_features, out_features = _get_in_out_features(base_layer)
        if (in_features is None) or (out_features is None):
            raise TypeError(f"Unsupported layer type {type(base_layer)}")
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    def update_layer(
        self,
        adapter_name: str,
        config: DummyConfig,
        inference_mode: bool = False,
        **kwargs,
    ) -> None:
        """Internal function to create a dummy adapter and its weights based
        on its name and config. Will be called by __init__ and when adding additional adapters.

        Args:
            adapter_name (`str`): Name for the adapter to add.
            config ([`DummyConfig`]): The Dummy configuration object holding the layer hyperparameters.
            inference_mode (`bool`, defaults to `False`): Whether to start the adapter in inference mode.
        """
        r = config.r
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r[adapter_name] = r
        self.scaling[adapter_name] = config.lora_alpha / r
        self.dummy_lora_dropout[adapter_name] = nn.Dropout(p=config.lora_dropout)

        self._create_adapter_modules(adapter_name, r, config)
        self._init_adapter_weights(adapter_name, config)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=inference_mode)

    def _create_adapter_modules(self, adapter_name: str, r: int, config: DummyConfig) -> None:
        # Implemented by subclasses; creates the low-rank update modules for the given adapter.
        raise NotImplementedError

    def _init_adapter_weights(self, adapter_name: str, config: DummyConfig) -> None:
        """Initialize the adapter weights.

        With the default init_weights=True, one of the two factors is randomly initialized but the other is zero, so
        the adapter starts out as an exact identity transform -- the adapted layer behaves exactly like the base layer
        until the adapter has been trained. This is the convention that most PEFT methods follow. With
        init_weights=False, both factors are randomly initialized, so the adapter has a non-zero effect right away;
        this is mainly useful for testing.
        """
        nn.init.kaiming_uniform_(self.dummy_lora_A[adapter_name].weight, a=math.sqrt(5))
        if config.init_weights:
            nn.init.zeros_(self.dummy_lora_B[adapter_name].weight)
        else:
            nn.init.kaiming_uniform_(self.dummy_lora_B[adapter_name].weight, a=math.sqrt(5))

    def set_scale(self, adapter: str, scale: float) -> None:
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale

    def get_delta_weight(self, adapter: str) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter, i.e. the low-rank update expressed in the shape and orientation
        of the base layer weight, ready to be added to it.

        Args:
            adapter (`str`):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.dummy_lora_B[adapter].weight.device
        dtype = self.dummy_lora_A[adapter].weight.dtype

        # In case users want to merge adapter weights that are in (b)float16 while being on CPU, we need to cast the
        # weights to float32, perform the merge, and then cast back. Historically, this was a compatibility measure:
        # (b)float16 matmuls were simply not supported on CPU. Nowadays they are, but they can still be much slower
        # than fp32 and accumulate in reduced precision, which would degrade the merged weight. The temporary fp32
        # copy is affordable on CPU, where memory is abundant, unlike on accelerators.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_a = self.dummy_lora_A[adapter].weight
        weight_b = self.dummy_lora_B[adapter].weight

        if cast_to_fp32:
            weight_a = weight_a.float()
            weight_b = weight_b.float()

        output_tensor = self._compose_delta_weight(weight_a, weight_b)
        # The delta weight is a freshly computed tensor that nothing else references, so apply the scaling in place to
        # avoid allocating another full-size tensor.
        output_tensor.mul_(self.scaling[adapter])

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

        return output_tensor

    def _compose_delta_weight(self, weight_a: torch.Tensor, weight_b: torch.Tensor) -> torch.Tensor:
        # Implemented by subclasses; combines the low-rank factors into a full-size delta weight.
        raise NotImplementedError

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights.

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs but is generally slower and needs more memory. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.dummy_lora_A.keys():
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge because of the copy operation.
                    orig_weight = self.get_base_weight().clone()
                    orig_weight += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weight).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    if self.quantization_backend is not None:
                        self.set_base_weight(orig_weight)
                    else:
                        # Write into the existing weight in place instead of rebinding the data: with FSDP2, the
                        # module can hold a temporary unsharded copy of the weight, and rebinding (via set_base_weight)
                        # would only update that copy, which is then discarded at the next reshard, losing the merge.
                        self.get_base_weight().copy_(orig_weight)
                else:
                    # For plain weights, get_base_weight returns the existing weight tensor, so this writes the delta
                    # into it in place. For quantized weights, a dequantized copy is returned and set_base_weight
                    # re-quantizes it afterwards.
                    orig_weight = self.get_base_weight()
                    orig_weight += self.get_delta_weight(active_adapter)
                    self.set_base_weight(orig_weight)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.dummy_lora_A.keys():
                orig_weight = self.get_base_weight()
                orig_weight -= self.get_delta_weight(active_adapter)
                self.set_base_weight(orig_weight)


class Linear(nn.Module, DummyLayer):
    """Dummy implemented in a dense layer, supporting `torch.nn.Linear` and `transformers.pytorch_utils.Conv1D`."""

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        config: DummyConfig,
        **kwargs,
    ) -> None:
        super().__init__()
        DummyLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = config.fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, config=config)

    def _create_adapter_modules(self, adapter_name: str, r: int, config: DummyConfig) -> None:
        self.dummy_lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.dummy_lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)

    def _compose_delta_weight(self, weight_a: torch.Tensor, weight_b: torch.Tensor) -> torch.Tensor:
        # weight_a has shape (r, in_features), weight_b has shape (out_features, r); transpose for Conv1D layers, which
        # store the weight like (fan_in, fan_out)
        return transpose(weight_b @ weight_a, self.fan_in_fan_out)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            # We don't have to explicitly check if there are unmerged active adapters since this is explicitly
            # forbidden to do. Merging requires the merged adapter to be active and each affected layer must merge all
            # of its applicable active adapters in active order.
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            orig_dtype = result.dtype

            # Some backends (bnb 4bit) need the base result cloned before it enters the adapter computation, as
            # backprop can fail on a manipulated view of the quantized kernel's output; for the others this is a no-op.
            if self.quantization_backend is not None:
                result = self.quantization_backend.maybe_clone_base_result(result)

            for active_adapter in self.active_adapters:
                if active_adapter not in self.dummy_lora_A.keys():
                    continue
                a = self.dummy_lora_A[active_adapter]
                b = self.dummy_lora_B[active_adapter]
                dropout = self.dummy_lora_dropout[active_adapter]
                x = self._cast_input_dtype(x, a.weight.dtype)
                result = result + b(dropout(a(x))) * self.scaling[active_adapter]
            result = result.to(orig_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "dummy." + rep

    def extra_repr(self) -> str:
        return quantization_extra_repr(self)


class _ConvNd(nn.Module, DummyLayer):
    """Dummy implemented in a convNd layer, supporting `torch.nn.Conv1d`, `torch.nn.Conv2d`, and `torch.nn.Conv3d`."""

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        config: DummyConfig,
        **kwargs,
    ) -> None:
        super().__init__()
        DummyLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, config=config)

    def _create_adapter_modules(self, adapter_name: str, r: int, config: DummyConfig) -> None:
        base_layer = self.get_base_layer()
        kernel_dim = base_layer.weight.dim()

        # Background on grouped convolutions: grouping effectively splits one conv layer into several independent conv
        # layers, each getting 1/groups part of the input and output channels. The weight of each group only connects
        # its share of the input channels to its share of the output channels, which is why the weight has the shape
        # (out_channels, in_channels // groups, *kernel_size).
        #
        # The adapter follows the same split: A is itself a grouped convolution with the same kernel as the base layer,
        # mapping in_channels // groups to r // groups channels per group, and B is a grouped 1x1 convolution mapping
        # r // groups to out_channels // groups channels per group. This requires r to be divisible by groups. With
        # this choice, the low-rank update of each group is simply B_g @ A_g, so the full delta weight can be
        # reconstructed per group for merging.
        if r % base_layer.groups != 0:
            raise ValueError(
                f"Targeting a {base_layer.__class__.__name__} with groups={base_layer.groups} and rank {r}. "
                "Currently, support is limited to conv layers where the rank is divisible by groups. "
                "Either choose a different rank or do not target this specific layer."
            )

        conv_layer = type(base_layer)
        self.dummy_lora_A[adapter_name] = conv_layer(
            base_layer.in_channels,
            r,
            base_layer.kernel_size,
            base_layer.stride,
            base_layer.padding,
            dilation=base_layer.dilation,
            padding_mode=base_layer.padding_mode,
            groups=base_layer.groups,
            bias=False,
        )
        self.dummy_lora_B[adapter_name] = conv_layer(
            r,
            base_layer.out_channels,
            (1,) * (kernel_dim - 2),
            (1,) * (kernel_dim - 2),
            groups=base_layer.groups,
            bias=False,
        )

    def _compose_delta_weight(self, weight_a: torch.Tensor, weight_b: torch.Tensor) -> torch.Tensor:
        # weight_a has shape (r, in_channels // groups, *kernel_size), weight_b has shape
        # (out_channels, r // groups, *(1,) * (kernel_dim - 2)). The delta weight of each group is B_g @ A_g, so the
        # factors are reshaped to (groups, r // groups, ...) and combined with a single batched matrix multiplication.
        base_layer = self.get_base_layer()
        groups = base_layer.groups
        rank_per_group = weight_a.shape[0] // groups

        a = weight_a.reshape(groups, rank_per_group, -1)
        b = weight_b.reshape(groups, -1, rank_per_group)
        output_tensor = torch.bmm(b, a)

        return output_tensor.reshape(base_layer.weight.shape)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            orig_dtype = result.dtype

            for active_adapter in self.active_adapters:
                if active_adapter not in self.dummy_lora_A.keys():
                    continue
                a = self.dummy_lora_A[active_adapter]
                b = self.dummy_lora_B[active_adapter]
                dropout = self.dummy_lora_dropout[active_adapter]
                x = self._cast_input_dtype(x, a.weight.dtype)
                result = result + b(dropout(a(x))) * self.scaling[active_adapter]
            result = result.to(orig_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "dummy." + rep

    def extra_repr(self) -> str:
        return quantization_extra_repr(self)


class Conv1d(_ConvNd):
    # Dummy implemented in a conv1d layer
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.get_base_layer().weight.dim() == 3:
            raise ValueError(f"Conv1d layer kernel must have 3 dimensions, not {self.get_base_layer().weight.dim()}")


class Conv2d(_ConvNd):
    # Dummy implemented in a conv2d layer
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.get_base_layer().weight.dim() == 4:
            raise ValueError(f"Conv2d layer kernel must have 4 dimensions, not {self.get_base_layer().weight.dim()}")


class Conv3d(_ConvNd):
    # Dummy implemented in a conv3d layer
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.get_base_layer().weight.dim() == 5:
            raise ValueError(f"Conv3d layer kernel must have 5 dimensions, not {self.get_base_layer().weight.dim()}")


class Embedding(nn.Module, DummyLayer):
    """Dummy implemented in an embedding layer, supporting `torch.nn.Embedding`."""

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        config: DummyConfig,
        **kwargs,
    ) -> None:
        super().__init__()
        DummyLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, config=config)

    def _create_adapter_modules(self, adapter_name: str, r: int, config: DummyConfig) -> None:
        # A is an embedding that maps a token id to r dimensions, B is a linear layer mapping those r dimensions to the
        # embedding dimension, mirroring the structure of the Linear layer adapter.
        self.dummy_lora_A[adapter_name] = nn.Embedding(self.in_features, r)
        self.dummy_lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)

    def _init_adapter_weights(self, adapter_name: str, config: DummyConfig) -> None:
        r"""Initialize the adapter weights.

        The roles of A and B are swapped compared to the Linear layer: A (the embedding lookup) is initialized with
        zeros, B with random values instead of the other way around. Either factor being zero makes the delta weight
        zero, so the adapter is still an identity transform at init. The swap matters for training: for a Linear layer,
        the input values multiply into the gradient of the upstream factor:
          dL/dA = x.T @ (upstream @ B)
        So it picks up pace on its own once B has moved away from zero.

        An embedding input only selects rows of A, so the gradient of A is proportional to B:
          dL/dA = \sum_i (upstream[i] @ B)
        Therefore gradients would stay vanishingly small for many steps if B started at zero.
        Initializing A with zero instead scales the gradient with random values from B instead of zero and gives
        larger values for the gradients.

        With init_weights=False, both factors are randomly initialized, giving the adapter a non-zero effect right away.
        """
        nn.init.zeros_(self.dummy_lora_A[adapter_name].weight)
        if config.init_weights:
            nn.init.normal_(self.dummy_lora_B[adapter_name].weight)
        else:
            nn.init.normal_(self.dummy_lora_A[adapter_name].weight)
            nn.init.normal_(self.dummy_lora_B[adapter_name].weight)

    def _compose_delta_weight(self, weight_a: torch.Tensor, weight_b: torch.Tensor) -> torch.Tensor:
        # weight_a is the lookup table of A with shape (num_embeddings, r), weight_b has shape (embedding_dim, r),
        # stored like a nn.Linear weight. The low-rank update of a token i is B applied to the i-th row of A, so the
        # delta weight is A @ B.T, which directly has the shape (num_embeddings, embedding_dim) of the base weight.
        return weight_a @ weight_b.t()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            orig_dtype = result.dtype

            for active_adapter in self.active_adapters:
                if active_adapter not in self.dummy_lora_A.keys():
                    continue
                a = self.dummy_lora_A[active_adapter]
                b = self.dummy_lora_B[active_adapter]
                dropout = self.dummy_lora_dropout[active_adapter]
                # Note: no input dtype cast here, unlike in Linear: x contains token ids that must stay integers.
                result = result + b(dropout(a(x))) * self.scaling[active_adapter]
            result = result.to(orig_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "dummy." + rep

    def extra_repr(self) -> str:
        return quantization_extra_repr(self)
