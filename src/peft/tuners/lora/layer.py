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
from __future__ import annotations

import math
import warnings
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import svd_lowrank
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import (
    dequantize_module_weight,
    gather_params_ctx,
    get_bnb_param_type,
    skip_init_on_device,
)
from peft.utils.other import transpose

from .config import LoraConfig


class LoraVariant:
    """
    Base class for LoRA variants, e.g. DoRA.

    This class should be subclassed and the methods below should be implemented accordingly. The methods should be
    implemented as static methods, this makes it easier to combine variants.

    Note for developers: These methods are prone to change and should thus considered to be "private". Use at your own
    discretion.
    """

    @staticmethod
    def init(module: LoraLayer, adapter_name: str) -> None:
        """Initialization code for the LoRA variant, it's called within `update_layer`"""
        raise NotImplementedError

    @staticmethod
    def merge_safe(module: LoraLayer, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        """Safe merging of the weights from `merge(..., safe_merge=True)`, should return a new tensor"""
        raise NotImplementedError

    @staticmethod
    def merge_unsafe(module: LoraLayer, active_adapter: str, orig_weight: torch.Tensor) -> None:
        """Unsafe merging of the weights from `merge(..., safe_merge=False)`, should modify the weight in-place"""

    @staticmethod
    def unmerge(module: LoraLayer, active_adapter: str, orig_weight: torch.Tensor) -> torch.Tensor:
        """Remove the adapter weights from the original weights, then return them"""

    @staticmethod
    def forward(module: LoraLayer, active_adapter: str, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the LoRA variant, should return the overall result (not just the diff)

        Args:
            module (LoraLayer): The module on which the forward pass is called
            active_adapter (str): The name of the active adapter
            x (torch.Tensor): The input to the forward call
            result (torch.Tensor): The result from the base model
        """
        raise NotImplementedError


class LoraLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names: tuple[str, ...] = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names: tuple[str, ...] = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Module, ephemeral_gpu_offload: bool = False, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.use_dora: dict[str, bool] = {}  # not actively used anymore after #2443, keep it for BC
        self.lora_bias: dict[str, bool] = {}
        self.lora_magnitude_vector = torch.nn.ModuleDict()  # for DoRA
        self._caches: dict[str, Any] = {}
        self.ephemeral_gpu_offload: bool = ephemeral_gpu_offload
        # flag to enable/disable casting of input to weight dtype during forward call
        self.cast_input_dtype_enabled: bool = True
        self.lora_variant: dict[str, LoraVariant] = {}
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv1d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Conv3d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif isinstance(base_layer, nn.MultiheadAttention):
            if not base_layer._qkv_same_embed_dim:
                raise ValueError(f"Only same dim for query/key/value is supported as of now for {self.__class__}.")
            in_features, out_features = base_layer.embed_dim, 3 * base_layer.embed_dim
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        elif hasattr(base_layer, "codebooks") and base_layer.__class__.__name__ == "QuantizedLinear":
            # AQLM QuantLinear
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "w_bit") and base_layer.__class__.__name__ == "WQLinear_GEMM":
            # Awq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif base_layer.__class__.__name__ == "EetqLinear":
            # Eetq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "W_q") and base_layer.__class__.__name__ == "HQQLinear":
            # HQQ layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif base_layer.__class__.__name__ == "PatchedLinear":
            # INC layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            # possibly support user provided custom layer types using dynamic dispatch
            if hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
                in_features, out_features = base_layer.in_features, base_layer.out_features
            else:
                in_features, out_features = None, None
            warnings.warn(
                f"Unsupported layer type '{type(base_layer)}' encountered, proceed at your own risk.", UserWarning
            )

        self.in_features = in_features
        self.out_features = out_features

    def resolve_lora_variant(self, *, use_dora: bool, **kwargs) -> Optional[LoraVariant]:
        """Return a matching LoRA variant for this layer type.

        Given the init arguments of this layer, return the correct LoRA variant, if any. E.g., if `use_dora=True`, this
        method should return the DoRA variant for the given layer.

        If there is no fitting variant, return None.

        Note: If this layer type does not support the LoRA variant at all, please raise an error during __init__ as is
        convention, and not here.

        """
        return None

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        use_dora: bool = False,
        use_qalora: bool = False,
        lora_bias: bool = False,
        qalora_group_size: int = 32,
        **kwargs,
    ):
        # collect the kwargs
        kwargs = locals().copy()
        del kwargs["self"]

        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        lora_variant = self.resolve_lora_variant(
            use_dora=use_dora, use_qalora=use_qalora, qalora_group_size=qalora_group_size
        )
        if lora_variant is not None:
            self.lora_variant[adapter_name] = lora_variant

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=lora_bias)
        self.lora_bias[adapter_name] = lora_bias

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        self.use_dora[adapter_name] = use_dora

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.pissa_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.startswith("corda"):
            with gather_params_ctx(self.get_base_layer().weight):
                self.corda_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
            with gather_params_ctx(self.get_base_layer().weight):
                self.olora_init(adapter_name)
        elif init_lora_weights == "loftq":
            with gather_params_ctx(self.get_base_layer().weight):
                self.loftq_init(adapter_name)
        elif init_lora_weights == "eva":
            nn.init.zeros_(self.lora_B[adapter_name].weight)
        elif init_lora_weights == "orthogonal":
            with gather_params_ctx(self.get_base_layer().weight):
                self.orthogonal_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        # call this before init of the lora variants
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if adapter_name in self.lora_variant:
            self.lora_variant[adapter_name].init(self, **kwargs)

        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            if init_lora_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            elif init_lora_weights.lower() == "gaussian":
                nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")
            nn.init.zeros_(self.lora_B[adapter_name].weight)
            if self.lora_bias[adapter_name]:
                nn.init.zeros_(self.lora_B[adapter_name].bias)
        if adapter_name in self.lora_embedding_A.keys():
            # Initialize A to zeros and B the same way as the default for nn.Embedding, see:
            # https://github.com/microsoft/LoRA/blob/4c0333854cb905966f8cc4e9a74068c1e507c7b7/loralib/layers.py#L59-L60
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])
            if self.lora_bias[adapter_name]:
                # embeddings are not supported at the moment, but still adding this for consistency
                nn.init.zeros_(self.lora_embedding_B[adapter_name].bias)

    def olora_init(self, adapter_name):
        base_layer = self.get_base_layer()
        orig_weight = base_layer.weight
        bnb_param_type = get_bnb_param_type(orig_weight)
        dtype = orig_weight.dtype

        if bnb_param_type:
            # check without importing bitsandbytes and robust to bnb_4bit_quant_storage=float*
            weight_tensor = dequantize_module_weight(base_layer)
        elif dtype in [torch.float32, torch.float16, torch.bfloat16]:
            weight_tensor = orig_weight
        else:
            raise TypeError(f"Unsupported data type for the base layer. Got {dtype}.")

        scale_factor = self.scaling[adapter_name]
        r = self.r[adapter_name]
        weight_tensor = weight_tensor.to(torch.float32)
        Q, R = torch.linalg.qr(weight_tensor.data)

        Qr, Rr = Q[:, :r], R[:r]

        self.lora_A[adapter_name].weight.data = Rr.contiguous()
        self.lora_B[adapter_name].weight.data = Qr.contiguous()

        weight_tensor.data -= scale_factor * self.lora_B[adapter_name].weight @ self.lora_A[adapter_name].weight
        if bnb_param_type == "4bit":
            weight_tensor = orig_weight.__class__(
                weight_tensor,
                quant_type=orig_weight.quant_type,
                quant_storage=orig_weight.quant_storage,
                compress_statistics=orig_weight.compress_statistics,
                module=orig_weight.module,
            ).to(orig_weight.device)
            base_layer.weight = weight_tensor
        elif bnb_param_type == "8bit":
            weight_tensor = orig_weight.__class__(
                weight_tensor,
                requires_grad=orig_weight.requires_grad,
                has_fp16_weights=orig_weight.has_fp16_weights,
            ).to(orig_weight.device)
            base_layer.weight = weight_tensor
        else:
            weight_tensor = weight_tensor.to(dtype)
            base_layer.weight.data = weight_tensor

    def pissa_init(self, adapter_name, init_lora_weights):
        weight = self.get_base_layer().weight
        dtype = weight.dtype
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise TypeError(
                "Please initialize PiSSA under float32, float16, or bfloat16. "
                "Subsequently, re-quantize the residual model to help minimize quantization errors."
            )
        weight = transpose(weight.to(torch.float32), self.fan_in_fan_out)
        if init_lora_weights == "pissa":
            # USV^T = W <-> VSU^T = W^T, where W^T = weight.data in R^{out_channel, in_channel},
            V, S, Uh = torch.linalg.svd(weight.data, full_matrices=False)
            Vr = V[:, : self.r[adapter_name]]
            Sr = S[: self.r[adapter_name]]
            Sr /= self.scaling[adapter_name]
            Uhr = Uh[: self.r[adapter_name]]
        elif len(init_lora_weights.split("_niter_")) == 2:
            Vr, Sr, Ur = svd_lowrank(
                weight.data, self.r[adapter_name], niter=int(init_lora_weights.split("_niter_")[-1])
            )
            Sr /= self.scaling[adapter_name]
            Uhr = Ur.t()
        else:
            raise ValueError(
                f"init_lora_weights should be 'pissa' or 'pissa_niter_[number of iters]', got {init_lora_weights} instead."
            )

        lora_A = torch.diag(torch.sqrt(Sr)) @ Uhr
        lora_B = Vr @ torch.diag(torch.sqrt(Sr))
        self.lora_A[adapter_name].weight.data = lora_A
        self.lora_B[adapter_name].weight.data = lora_B
        weight = weight.data - self.scaling[adapter_name] * lora_B @ lora_A
        weight = transpose(weight.to(dtype), self.fan_in_fan_out)
        self.get_base_layer().weight.data = weight

    def corda_init(self, adapter_name, init_lora_weights):
        linear = self.get_base_layer()
        weight = linear.weight
        dtype = weight.dtype
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise TypeError(
                "Please initialize CorDA under float32, float16, or bfloat16. "
                "Subsequently, re-quantize the residual model to help minimize quantization errors."
            )
        weight = weight.to(torch.float32)
        out_dim = weight.data.size(0)
        in_dim = weight.data.size(1)

        # Calculate WC from covariance matrix
        if not hasattr(linear, "eigens"):
            raise ValueError(
                "`eigens` attribute not found for layer, please run `preprocess_corda` first. "
                "More information can be found at examples/corda_finetuning/README.md."
            )
        eigens = linear.eigens
        U = eigens.U_WC
        S = eigens.S_WC
        V = eigens.V_WC
        r = self.r[adapter_name]

        # nan or inf check
        if torch.isnan(S).any() or torch.isinf(S).any():
            raise ValueError(
                "Invalid value found in matrix S. Please file an issue at https://github.com/huggingface/peft/issues."
            )
        if torch.isnan(U).any() or torch.isinf(U).any():
            raise ValueError(
                "Invalid value found in matrix U. Please file an issue at https://github.com/huggingface/peft/issues."
            )
        if torch.isnan(V).any() or torch.isinf(V).any():
            raise ValueError(
                "Invalid value found in matrix V. Please file an issue at https://github.com/huggingface/peft/issues."
            )

        # Sanity check
        if U.size(0) != out_dim or U.size(1) != r:
            raise ValueError(
                f"Matrix U size mismatch: {U.size()} vs. ({out_dim}, {r}). Please make sure the `lora_config` and "
                "`model` argument of `preprocess_corda` is consistent with `get_peft_model`. If you're using cache "
                "in `preprocess_corda`, please make sure the cache is built with the same model and LoRA rank."
            )
        if S.size(0) != r:
            raise ValueError(
                f"Matrix S size mismatch: {S.size()} vs. ({r},). Please make sure the `lora_config` and `model` argument "
                "of `preprocess_corda` is consistent with `get_peft_model`. If you're using cache in `preprocess_corda`, "
                "please make sure the cache is built with the same model and LoRA rank."
            )
        if V.size(0) != in_dim or V.size(1) != r:
            raise ValueError(
                f"Matrix V size mismatch: {V.size()} vs. ({in_dim}, {r}). Please make sure the `lora_config` and "
                "`model` argument of `preprocess_corda` is consistent with `get_peft_model`. If you're using cache "
                "in `preprocess_corda`, please make sure the cache is built with the same model and LoRA rank."
            )

        # Apply alpha
        S /= self.scaling[adapter_name]

        # Init lora_A and lora_B weights
        lora_A = V.t().mul(S.sqrt().view(-1, 1)).contiguous()
        lora_B = U.mul(S.sqrt()).contiguous()
        self.lora_A[adapter_name].weight.data = lora_A
        self.lora_B[adapter_name].weight.data = lora_B
        weight = weight.data - self.scaling[adapter_name] * lora_B @ lora_A
        weight = weight.to(dtype)
        self.get_base_layer().weight.data = weight

        # Remove redundant fields
        del linear.eigens

    def loftq_init(self, adapter_name):
        from peft.utils.loftq_utils import loftq_init

        weight = self.get_base_layer().weight
        kwargs = {
            "num_bits": self.kwargs.get("loftq_bits", 4),
            "reduced_rank": self.r[adapter_name],
            "num_iter": self.kwargs.get("loftq_iter", 1),
        }

        qweight, lora_A, lora_B = loftq_init(weight, **kwargs)
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            self.lora_A[adapter_name].weight.data = lora_A
            self.lora_B[adapter_name].weight.data = lora_B
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            self.lora_embedding_A[adapter_name].weight.data = lora_A
            self.lora_embedding_B[adapter_name].weight.data = lora_B
        self.get_base_layer().weight.data = qweight

    @torch.no_grad()
    def orthogonal_init(self, adapter_name):
        # https://datta0.github.io/posts/rethink-lora-init/#orthogonal-initialisation
        rank = self.r[adapter_name]
        if rank % 2 != 0:
            raise ValueError(f"Orthogonal initialization requires the LoRA rank to be even, got {rank} instead.")

        X = torch.randn(rank, rank)
        Q, _ = torch.linalg.qr(X)
        q_odd = Q[0::2, :]  # Odd rows
        q_even = Q[1::2, :]  # Even rows
        dtype = self.get_base_layer().weight.dtype
        lora_A = torch.randn(self.in_features, rank // 2).mm(q_odd).T / 10.0
        lora_B = torch.randn(rank // 2, self.out_features).T.mm(q_even) / 10.0
        self.lora_A[adapter_name].weight = nn.Parameter(lora_A.contiguous().to(dtype))
        self.lora_B[adapter_name].weight = nn.Parameter(lora_B.contiguous().to(dtype))

    def _cache_store(self, key: str, value: Any) -> None:
        self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key)
        return value

    def set_scale(self, adapter: str, scale: float | int) -> None:
        """Set the scale of the given adapter to the initial scale multiplied by the provided factor

        The initial scale is determined by the configured `r` (rank) and `lora_alpha`.
        """
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float | int) -> None:
        """Multiply the current scale of all active adapters by the provided factor"""
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale: Optional[float | int] = None) -> None:
        """Divide the current scale of all active adapters by the provided factor. If `scale=None` is passed, reset to
        initial scale

        The initial scale is determined by the configured `r` (rank) and `lora_alpha`.

        """
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            if scale is None:
                self.scaling[active_adapter] = self.lora_alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)

        if self.merged:
            # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
            # adapters. Therefore, it is better to raise an error in this case.
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)

        # DoRA is not supported (yet), check that it's not being used. Don't check "__base__", as this is the
        # placeholder for the base model.
        unique_adapters = {name for name in adapter_names if name != "__base__"}
        for adapter_name in unique_adapters:
            if self.use_dora.get(adapter_name, False):
                msg = "Cannot pass `adapter_names` when DoRA is enabled."
                raise ValueError(msg)

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_A.keys():
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]].to(lora_A.weight.dtype)
            lora_output = lora_B(lora_A(dropout(sub_batch))) * scaling
            result[sub_batch_indices_list[i]] += lora_output.to(torch_result_dtype)

        return result


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class Linear(nn.Module, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def resolve_lora_variant(self, *, use_dora: bool, **kwargs) -> Optional[LoraVariant]:
        if not use_dora:
            return None

        from .variants import DoraLinearVariant

        return DoraLinearVariant()

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
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weight = base_layer.weight.data.clone()
                    orig_dtype = orig_weight.dtype
                    if active_adapter not in self.lora_variant:  # vanilla LoRA
                        delta_weight = self.get_delta_weight(active_adapter)
                        orig_weight += delta_weight.to(orig_dtype)
                    else:
                        orig_weight = self.lora_variant[active_adapter].merge_safe(self, active_adapter, orig_weight)

                    if not torch.isfinite(orig_weight).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weight

                    if self.lora_bias[active_adapter]:
                        new_bias = base_layer.bias + self.lora_B[active_adapter].bias * self.scaling[active_adapter]
                        if not torch.isfinite(new_bias).all():
                            raise ValueError(
                                f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                            )
                        base_layer.bias.data = new_bias.to(orig_dtype)

                else:
                    if active_adapter not in self.lora_variant:  # vanilla LoRA
                        delta_weight = self.get_delta_weight(active_adapter)
                        base_layer.weight.data += delta_weight
                    else:
                        self.lora_variant[active_adapter].merge_unsafe(self, active_adapter, base_layer.weight)

                    if self.lora_bias[active_adapter]:
                        base_layer.bias.data += self.lora_B[active_adapter].bias * self.scaling[active_adapter]

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
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                if active_adapter not in self.lora_variant:  # vanilla LoRA
                    orig_dtype = weight.dtype
                    delta_weight = self.get_delta_weight(active_adapter)
                    weight.data -= delta_weight.to(orig_dtype)
                else:
                    unmerged = self.lora_variant[active_adapter].unmerge(self, active_adapter, weight)
                    weight.data = unmerged

                if self.lora_bias[active_adapter]:
                    self.get_base_layer().bias.data -= self.lora_B[active_adapter].bias * self.scaling[active_adapter]

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            lora_A_keys = self.lora_A.keys()
            for active_adapter in self.active_adapters:
                if active_adapter not in lora_A_keys:
                    continue

                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = self._cast_input_dtype(x, lora_A.weight.dtype)
                if active_adapter not in self.lora_variant:  # vanilla LoRA
                    result = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    result = self.lora_variant[active_adapter].forward(
                        self,
                        active_adapter=active_adapter,
                        x=x,
                        result=result,
                    )

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


class Embedding(nn.Module, LoraLayer):
    # LoRA implemented in a Embedding layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        if lora_bias:
            # lora_bias=True is not supported (yet) for embedding layers, as they use nn.Parameter
            raise ValueError(f"lora_bias={lora_bias} is not supported for {self.__class__.__name__}.")

        super().__init__()
        LoraLayer.__init__(self, base_layer)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias,
        )

    def resolve_lora_variant(self, *, use_dora: bool, **kwargs) -> Optional[LoraVariant]:
        if not use_dora:
            return None

        from .variants import DoraEmbeddingVariant

        return DoraEmbeddingVariant()

    def update_layer(
        self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora, lora_bias
    ):
        # collect the kwargs
        kwargs = locals().copy()
        del kwargs["self"]

        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        lora_variant = self.resolve_lora_variant(use_dora=use_dora)
        if lora_variant is not None:
            self.lora_variant[adapter_name] = lora_variant

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        weight_A = torch.randn((r, self.in_features))
        weight_B = torch.randn((self.out_features, r))
        self.lora_embedding_A[adapter_name] = nn.Parameter(weight_A)
        self.lora_embedding_B[adapter_name] = nn.Parameter(weight_B)
        self.lora_bias[adapter_name] = lora_bias

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        self.use_dora[adapter_name] = use_dora

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        # call this before init of the lora variants
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if adapter_name in self.lora_variant:
            self.lora_variant[adapter_name].init(self, **kwargs)

        self.set_adapter(self.active_adapters)

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
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_embedding_A.keys():
                base_layer = self.get_base_layer()
                orig_dtype = base_layer.weight.dtype
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weight = base_layer.weight.data.clone()
                    if active_adapter not in self.lora_variant:  # vanilla LoRA
                        orig_weight += self.get_delta_weight(active_adapter).to(orig_dtype)
                    else:
                        orig_weight = self.lora_variant[active_adapter].merge_safe(self, active_adapter, orig_weight)

                    if not torch.isfinite(orig_weight).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weight
                else:
                    if active_adapter not in self.lora_variant:  # vanilla LoRA
                        base_layer.weight.data += self.get_delta_weight(active_adapter).to(orig_dtype)
                    else:
                        self.lora_variant[active_adapter].merge_unsafe(self, active_adapter, base_layer.weight)
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
            orig_dtype = self.get_base_layer().weight.dtype
            if active_adapter in self.lora_embedding_A.keys():
                weight = self.get_base_layer().weight
                if active_adapter not in self.lora_variant:  # vanilla LoRA
                    weight.data -= self.get_delta_weight(active_adapter).to(orig_dtype)
                else:
                    unmerged = self.lora_variant[active_adapter].unmerge(self, active_adapter, weight)
                    weight.data = unmerged

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_embedding_B[adapter].device
        dtype = self.lora_embedding_A[adapter].dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.lora_embedding_A[adapter]
        weight_B = self.lora_embedding_B[adapter]

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, True) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_embedding_A[adapter] = weight_A.to(dtype)
            self.lora_embedding_B[adapter] = weight_B.to(dtype)

        return output_tensor

    def _mixed_batch_forward(
        self, x: torch.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any
    ) -> torch.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_embedding_A.keys():
                continue

            embedding_A = self.lora_embedding_A[active_adapter].T
            embedding_B = self.lora_embedding_B[active_adapter].T
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]]
            after_A = self._embed(sub_batch, embedding_A)
            result[sub_batch_indices_list[i]] += (after_A @ embedding_B) * scaling

        return result

    def _embed(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        base_layer = self.get_base_layer()
        return F.embedding(
            input,
            weight,
            padding_idx=base_layer.padding_idx,
            max_norm=base_layer.max_norm,
            norm_type=base_layer.norm_type,
            scale_grad_by_freq=base_layer.scale_grad_by_freq,
            sparse=base_layer.sparse,
        )

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # TODO: no dtype conversion here, unlike in Linear, is that correct?
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_embedding_A:
                    continue

                if active_adapter not in self.lora_variant:  # vanilla LoRA
                    embedding_A = self.lora_embedding_A[active_adapter].T
                    embedding_B = self.lora_embedding_B[active_adapter].T
                    scaling = self.scaling[active_adapter]
                    after_A = self._embed(x, embedding_A)
                    result = result + (after_A @ embedding_B) * scaling
                else:
                    result = self.lora_variant[active_adapter].forward(
                        self,
                        active_adapter=active_adapter,
                        x=x,
                        result=result,
                    )
            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


class _ConvNd(nn.Module, LoraLayer):
    # Lora implemented in a conv(2,3)d layer
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer)

        if base_layer.groups > 1:
            warnings.warn("LoRA adapter added to ConvNd layer with groups > 1. Merging is not supported.")

        if r % base_layer.groups != 0:
            raise ValueError(
                f"Targeting a {base_layer.__class__.__name__} with groups={base_layer.groups} and rank {r}. "
                "Currently, support is limited to conv layers where the rank is divisible by groups. "
                "Either choose a different rank or do not target this specific layer."
            )

        self._active_adapter = adapter_name
        self._kernel_dim = base_layer.weight.dim()

        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias,
        )

    def update_layer(
        self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora, lora_bias
    ):
        # collect the kwargs
        kwargs = locals().copy()
        del kwargs["self"]

        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        lora_variant = self.resolve_lora_variant(use_dora=use_dora)
        if lora_variant is not None:
            self.lora_variant[adapter_name] = lora_variant

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        base_layer = self.get_base_layer()
        kernel_size = base_layer.kernel_size
        stride = base_layer.stride
        padding = base_layer.padding
        conv_layer = type(base_layer)
        out_kernel = out_stride = (1,) * (self._kernel_dim - 2)
        self.lora_A[adapter_name] = conv_layer(self.in_features, r, kernel_size, stride, padding, bias=False)
        self.lora_B[adapter_name] = conv_layer(
            r, self.out_features, out_kernel, out_stride, groups=base_layer.groups, bias=lora_bias
        )
        self.lora_bias[adapter_name] = lora_bias

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        self.use_dora[adapter_name] = use_dora

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        # call this before init of the lora variants
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if adapter_name in self.lora_variant:
            self.lora_variant[adapter_name].init(self, **kwargs)

        self.set_adapter(self.active_adapters)

    def _get_dora_factor_view(self):
        return (-1,) + (1,) * (self._kernel_dim - 1)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights inside the base weights

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
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                orig_dtype = base_layer.weight.dtype

                if base_layer.groups > 1:
                    # https://github.com/huggingface/peft/pull/2403
                    raise NotImplementedError("Merging is not supported for _ConvNd layers with groups > 1!")

                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weight = base_layer.weight.data.clone()
                    if active_adapter not in self.lora_variant:  # vanilla LoRA
                        delta_weight = self.get_delta_weight(active_adapter)
                        orig_weight += delta_weight.to(orig_dtype)
                    else:
                        orig_weight = self.lora_variant[active_adapter].merge_safe(self, active_adapter, orig_weight)

                    if not torch.isfinite(orig_weight).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weight

                    if self.lora_bias[active_adapter]:
                        new_bias = base_layer.bias + self.lora_B[active_adapter].bias * self.scaling[active_adapter]
                        if not torch.isfinite(new_bias).all():
                            raise ValueError(
                                f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                            )
                        base_layer.bias.data = new_bias.to(orig_dtype)

                else:
                    if active_adapter not in self.lora_variant:  # vanilla LoRA
                        delta_weight = self.get_delta_weight(active_adapter)
                        base_layer.weight.data += delta_weight.to(orig_dtype)
                    else:
                        self.lora_variant[active_adapter].merge_unsafe(self, active_adapter, base_layer.weight)

                    if self.lora_bias[active_adapter]:
                        base_layer.bias.data += self.lora_B[active_adapter].bias * self.scaling[active_adapter]

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
            if active_adapter in self.lora_A.keys():
                weight = self.get_base_layer().weight
                if active_adapter not in self.lora_variant:  # vanilla LoRA
                    orig_dtype = weight.dtype
                    delta_weight = self.get_delta_weight(active_adapter)
                    weight.data -= delta_weight.to(orig_dtype)
                else:
                    unmerged = self.lora_variant[active_adapter].unmerge(self, active_adapter, weight)
                    weight.data = unmerged

                if self.lora_bias[active_adapter]:
                    self.get_base_layer().bias.data -= self.lora_B[active_adapter].bias * self.scaling[active_adapter]

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_A[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        # https://github.com/bmaltais/kohya_ss/blob/feb6728762a8f463d15ba936d189d4c3abfaa1ab/networks/lora.py#L117
        if self.get_base_layer().weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            output_tensor = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(
                3
            ) * self.scaling[adapter]
        else:
            output_tensor = self.conv_fn(weight_A.transpose(0, 1), weight_B)

            if self.get_base_layer().groups > 1:
                output_tensor = output_tensor * self.scaling[adapter]
            else:
                output_tensor = output_tensor.transpose(0, 1) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)

        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = self._cast_input_dtype(x, lora_A.weight.dtype)

                if active_adapter not in self.lora_variant:  # vanilla LoRA
                    result = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    result = self.lora_variant[active_adapter].forward(
                        self,
                        active_adapter=active_adapter,
                        x=x,
                        result=result,
                    )

            result = result.to(torch_result_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


class Conv2d(_ConvNd):
    # Lora implemented in a conv2d layer
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._kernel_dim == 4:
            raise ValueError(f"Conv2d layer kernel must have 4 dimensions, not {self._kernel_dim}")
        self.conv_fn = F.conv2d

    def resolve_lora_variant(self, *, use_dora: bool, **kwargs) -> Optional[LoraVariant]:
        if not use_dora:
            return None

        from .variants import DoraConv2dVariant

        return DoraConv2dVariant()


class Conv1d(_ConvNd):
    # Lora implemented in a conv1d layer
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._kernel_dim == 3:
            raise ValueError(f"Conv1d layer kernel must have 3 dimensions, not {self._kernel_dim}")
        self.conv_fn = F.conv1d

    def resolve_lora_variant(self, *, use_dora: bool, **kwargs) -> Optional[LoraVariant]:
        if not use_dora:
            return None

        from .variants import DoraConv1dVariant

        return DoraConv1dVariant()


class Conv3d(_ConvNd):
    # Lora implemented in a conv3d layer
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._kernel_dim == 5:
            raise ValueError(f"Conv3d layer kernel must have 5 dimensions, not {self._kernel_dim}")
        self.conv_fn = F.conv3d

    def resolve_lora_variant(self, *, use_dora: bool, **kwargs) -> Optional[LoraVariant]:
        if not use_dora:
            return None

        from .variants import DoraConv3dVariant

        return DoraConv3dVariant()


class MultiheadAttention(nn.Module, LoraLayer):
    """LoRA implemented in a multihead attention layer

    This is currently only implemented for the case of `_qkv_same_embed_dim = True`, i.e. query, key, and value having
    the same dimension.

    Note: LoRA is applied to both the in_proj (query/key/value) and out_proj. There is currently no way to specify only
    one of them. Don't try to apply LoRA to the out_proj of MultiheadAttention by targeting that layer specifically,
    since the forward method of that layer is not being used, hence the LoRA adapter would be ignored.

    This is a little bit hacky because of the way that MultiheadAttention is implemented in PyTorch: There are no
    `nn.Linear` layers which we can hook onto or, in case of output projection, `.forward` is not used. This
    implementation works around these problems by merging the weights before the forward call and unmerging them after
    the forward call.
    """

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        # TODO work with separate weights
        if not getattr(base_layer, "_qkv_same_embed_dim", True):
            # default for this value appears to be True:
            # https://github.com/pytorch/pytorch/blob/701ba5203fe68d55d655bd4d6c008be94cf34ea5/torch/nn/modules/activation.py#L1128-L1130
            raise ValueError(
                f"Only same embed for query/key/value is supported as of now for {self.__class__.__name__}."
            )
        if use_dora:
            # TODO: probably not so hard to implement
            raise ValueError(f"{self.__class__.__name__} does not support DoRA (yet), please set use_dora to False")

        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)

        # Note: LoRA is applied to both in_proj and out_proj. There is currently no way to only specify one of them.
        if isinstance(base_layer.out_proj, nn.Linear):
            self.base_layer.out_proj = Linear(
                base_layer.out_proj,
                adapter_name,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_lora_weights=init_lora_weights,
                use_rslora=use_rslora,
                use_dora=use_dora,
                **kwargs,
            )
        else:
            raise ValueError(f"out_proj must be an instance of nn.Linear for {self.__class__.__name__}.")

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora)

    @property
    def embed_dim(self) -> int:
        return self.get_base_layer().embed_dim

    @property
    def kdim(self) -> Optional[int]:
        return self.get_base_layer().kdim

    @property
    def vdim(self) -> Optional[int]:
        return self.get_base_layer().vdim

    @property
    def _qkv_same_embed_dim(self) -> bool:
        return self.get_base_layer()._qkv_same_embed_dim

    @property
    def num_heads(self) -> int:
        return self.get_base_layer().num_heads

    @property
    def dropout(self) -> float:
        return self.get_base_layer().dropout

    @property
    def batch_first(self) -> bool:
        return self.get_base_layer().batch_first

    @property
    def head_dim(self) -> int:
        return self.get_base_layer().head_dim

    @property
    def in_proj_weight(self) -> nn.Parameter:
        return self.get_base_layer().in_proj_weight

    @property
    def in_proj_bias(self) -> nn.Parameter:
        return self.get_base_layer().in_proj_bias

    @property
    def out_proj(self) -> nn.Module:
        return self.get_base_layer().out_proj.get_base_layer()

    @property
    def bias_k(self) -> Optional[nn.Parameter]:
        return self.get_base_layer().bias_k

    @property
    def bias_v(self) -> Optional[nn.Parameter]:
        return self.get_base_layer().bias_v

    def merge_masks(self, *args, **kwargs) -> tuple[Optional[torch.Tensor], Optional[int]]:
        return self.get_base_layer().merge_masks(*args, **kwargs)

    @property
    def add_zero_attn(self) -> bool:
        return self.get_base_layer().add_zero_attn

    def update_layer(self, *args, **kwargs) -> None:
        super().update_layer(*args, **kwargs)
        # Note: LoRA is applied to both in_proj and out_proj. There is currently no way to only specify one of them.
        self.base_layer.out_proj.update_layer(*args, **kwargs)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        # Implementation follows this:
        # https://github.com/Baijiong-Lin/LoRA-Torch/blob/4bfed6820b64fcf47064c30f30606a190a4f0d2e/loratorch/layers.py#L73-L79
        # Notably, instead of mutating the weight, we delete the original weight and replace it by the merged weight
        # TODO: work with separate weights
        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                orig_dtype = base_layer.out_proj.weight.dtype
                if safe_merge:
                    # TODO: work with separate weights
                    # merging in_proj (nn.Parameter)
                    orig_weight_in = base_layer.in_proj_weight.data.detach().clone()
                    orig_weight_in += self.get_delta_weight(active_adapter).to(orig_dtype)
                    if not torch.isfinite(orig_weight_in).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    # merging out_proj (subclass of nn.Linear)
                    orig_weight_out = base_layer.out_proj.weight.data.detach().clone()
                    orig_weight_out += base_layer.out_proj.get_delta_weight(active_adapter).to(orig_dtype)
                    if not torch.isfinite(orig_weight_out).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    # unregister parameter implicitly and overwrite using merged weights; gradients are computed after
                    # forward and, thus, after unmerging (see forward()), therefore this is safe to do.
                    del base_layer.in_proj_weight
                    base_layer.in_proj_weight = orig_weight_in

                    del base_layer.out_proj.get_base_layer().weight
                    base_layer.out_proj.get_base_layer().weight = orig_weight_out
                    base_layer.out_proj.merge(adapter_names=[active_adapter])
                else:
                    # merging in_proj (nn.Parameter)
                    # TODO: work with separate weights
                    delta_weight = self.get_delta_weight(active_adapter).to(orig_dtype)
                    weight_merged = base_layer.in_proj_weight.data.detach() + delta_weight

                    # unregister parameter implicitly and overwrite using merged weights; gradients are computed after
                    # forward and, thus, after unmerging (see forward()), therefore this is safe to do.
                    del base_layer.in_proj_weight
                    base_layer.in_proj_weight = weight_merged

                    # merging out_proj (subclass of nn.Linear)
                    delta_weight = base_layer.out_proj.get_delta_weight(active_adapter).to(orig_dtype)
                    weight_merged = base_layer.out_proj.weight.data.detach() + delta_weight
                    del base_layer.out_proj.get_base_layer().weight
                    base_layer.out_proj.get_base_layer().weight = weight_merged
                    base_layer.out_proj.merge(adapter_names=[active_adapter])
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        # TODO work with separate weights
        base_layer = self.get_base_layer()
        orig_dtype = base_layer.out_proj.base_layer.weight.dtype
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                # Ensure that requires_grad=False for the base weights after unmerging. This may not matter since
                # requires_grad was False when the optimizer was initialized, but still let's try to be correct here.

                # in_proj
                delta_weight = self.get_delta_weight(active_adapter).to(orig_dtype)
                old_weight = base_layer.in_proj_weight.data - delta_weight
                del base_layer.in_proj_weight
                base_layer.register_parameter("in_proj_weight", nn.Parameter(old_weight, requires_grad=False))

                # out_proj
                delta_weight = base_layer.out_proj.get_delta_weight(active_adapter).to(orig_dtype)
                old_weight = base_layer.out_proj.base_layer.weight.data - delta_weight
                del base_layer.out_proj.base_layer.weight
                base_layer.out_proj.base_layer.register_parameter(
                    "weight", nn.Parameter(old_weight, requires_grad=False)
                )

        self.get_base_layer().out_proj.unmerge()

    def unload_and_optionally_merge_module(
        self, merge: bool, safe_merge: bool, adapter_names: Optional[list[str]]
    ) -> nn.MultiheadAttention:
        """
        Merging and unloading of the MultiheadAttention module

        This requires an extra step for MultiheadAttention, which is why there is this special method instead of
        relying on the normal merge_and_unload code path.
        """
        if merge:
            self.merge(safe_merge=safe_merge, adapter_names=adapter_names)
        base_layer = self.get_base_layer()

        # extra steps: re-register weights, take care of out_proj layer
        # in_proj
        weight = base_layer.in_proj_weight
        del base_layer.in_proj_weight
        base_layer.register_parameter("in_proj_weight", nn.Parameter(weight.data, requires_grad=weight.requires_grad))

        # out_proj
        out_proj_layer = base_layer.out_proj.get_base_layer()
        weight = out_proj_layer.weight
        del out_proj_layer.weight
        out_proj_layer.register_parameter("weight", nn.Parameter(weight.data, requires_grad=weight.requires_grad))

        base_layer.out_proj = out_proj_layer
        return base_layer

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = (weight_B @ weight_A) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def _check_forward_args(self, x, *args, **kwargs):
        if "adapter_names" in kwargs:
            raise TypeError(f"lora.{self.__class__.__name__} does not support mixed adapter batches.")
        super()._check_forward_args(x, *args, **kwargs)

    def forward(self, query: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = query.dtype
        self._check_forward_args(query, *args, **kwargs)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(query, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(query, *args, **kwargs)
        else:
            out_proj = self.get_base_layer().out_proj
            if out_proj.active_adapters != self.active_adapters:
                # We have a case that in_proj and out_proj have diverging merged adapters. We cannot
                # really deal with this correctly, thus it's better to raise than possibly create a hard to debug mess
                cls_name = self.get_base_layer().__class__.__name__
                raise ValueError(
                    f"The out_proj layer of {cls_name} has merged layers but {cls_name} itself doesn't; please ensure "
                    "that either both or none have merged layers"
                )

            # Merge all adapters that are active for this module, i.e. the LoRA weights for in_proj and out_proj.
            # in_proj uses nn.Parameters, therefore, there is no forward method to be used and we have to explicitly
            # merge for the LoRA weights to have an effect:
            # https://github.com/pytorch/pytorch/blob/6ebb26d572d5fcdc6ac0d1297bdf8d1eb5d20722/torch/nn/modules/activation.py#L1020
            # For out_proj, we have an nn.Linear (or rather: NonDynamicallyQuantizableLinear), but its forward method
            # is not used:
            # https://github.com/pytorch/pytorch/blob/6ebb26d572d5fcdc6ac0d1297bdf8d1eb5d20722/torch/nn/modules/activation.py#L1267-L1271
            # Therefore, its LoRA weights also need to be merged to have an effect.
            active_adapters = [a for a in self.active_adapters if a in self.lora_A]
            try:
                self.merge(adapter_names=active_adapters)
                result = self.base_layer(query, *args, **kwargs)
            finally:
                # it's safe to call unmerge(), which unmerges all adapters, because we checked that not self.merged,
                # i.e. there is was no merged layer before
                self.unmerge()

        result = (result[0].to(previous_dtype), result[1].to(previous_dtype) if result[1] is not None else result[1])
        return result

    # The decorator is needed in case low_cpu_mem_usage=True is used, as we don't want the base layer weights to be
    # moved to meta device. This requires the use of PEFT's implementation of init_empty_weight instead of using the one
    # from accelerate.
    @skip_init_on_device
    def _restore_weights(self):
        # Restore the weights as registered parameters on the base layer.
        # This is necessary because the way that weights are merged/unmerged (which is necessary for forward to work
        # correctly), the Module "forgets" these attributes. Therefore, we need to call register_parameter explicitly.
        # We cannot call register_parameter for merging/unmerging because that cuts them off from the autograd graph.
        # Note that this is hacky, since we need to ensure that _restore_weights is called by each method that needs it.

        # in_proj
        # TODO work with separate weights
        base_layer = self.get_base_layer()
        weight = base_layer.in_proj_weight
        del base_layer.in_proj_weight
        base_layer.register_parameter("in_proj_weight", nn.Parameter(weight.data, requires_grad=weight.requires_grad))

        # out_proj
        base_layer = base_layer.out_proj.get_base_layer()
        weight = base_layer.weight
        del base_layer.weight
        base_layer.register_parameter("weight", nn.Parameter(weight.data, requires_grad=weight.requires_grad))

    def state_dict(self, *args, **kwargs):
        self._restore_weights()
        return super().state_dict(*args, **kwargs)

    def named_modules(self, *args, **kwargs):
        # Note: no need to also implement modules(), as modules() calls named_modules() under the hood
        self._restore_weights()
        return super().named_modules(*args, **kwargs)

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: LoraConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Embedding):
        embedding_kwargs = kwargs.copy()
        embedding_kwargs.pop("fan_in_fan_out", None)
        embedding_kwargs.update(lora_config.loftq_config)
        new_module = Embedding(target, adapter_name, **embedding_kwargs)
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        kwargs.update(lora_config.loftq_config)
        new_module = Conv2d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Conv3d):
        kwargs.update(lora_config.loftq_config)
        new_module = Conv3d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, nn.Conv1d):
        kwargs.update(lora_config.loftq_config)
        new_module = Conv1d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.MultiheadAttention):
        kwargs.update(lora_config.loftq_config)
        new_module = MultiheadAttention(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        new_module = Linear(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        if not kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
        kwargs.update(lora_config.loftq_config)
        new_module = Linear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)

    return new_module
