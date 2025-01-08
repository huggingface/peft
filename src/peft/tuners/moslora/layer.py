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
from accelerate.utils.imports import is_xpu_available
from torch import svd_lowrank
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import dequantize_module_weight, gather_params_ctx, get_bnb_param_type
from peft.utils.other import transpose


from peft.tuners.lora import LoraLayer, Embedding, Conv2d, Conv3d
from .config import MoSLoraConfig


class MoSLoraLayer(LoraLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_mixer", "lora_embedding_A", "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Module, ephemeral_gpu_offload: bool = False, **kwargs) -> None:
        super().__init__(base_layer)
        self.use_moslora: dict[str, bool] = {}
        self.lora_mixer = nn.ModuleDict({}) # for MoSLoRA

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        use_moslora,
        use_dora: bool = False,
        lora_bias: bool = False,
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

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
        
        # moslora init
        if isinstance(use_moslora, str):
            self.lora_mixer[adapter_name] = nn.Linear(r, r, bias=False) 
            self.use_moslora[adapter_name] = use_moslora
        elif use_moslora is True:
            self.lora_mixer[adapter_name] = nn.Linear(r, r, bias=False) 
            self.use_moslora[adapter_name] = True
        else:
            self.use_moslora[adapter_name] = False

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
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)
        # call this before dora_init
        self._move_adapter_to_device_of_base_layer(adapter_name)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

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

            if self.use_moslora[adapter_name] is True:
                nn.init.kaiming_uniform_(self.lora_mixer[adapter_name].weight, a=math.sqrt(5)) # default = kai
            elif isinstance(self.use_moslora[adapter_name], str):
                if self.use_moslora[adapter_name].lower() == "kai":
                    nn.init.kaiming_uniform_(self.lora_mixer[adapter_name].weight, a=math.sqrt(5))
                elif self.use_moslora[adapter_name].lower() == "orth":
                    nn.init.orthogonal_(self.lora_mixer[adapter_name].weight)
                else:
                    raise ValueError(f"Unknown initialization method for mixer {self.use_moslora[adapter_name]}")

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
            
            if isinstance(self.use_moslora[active_adapter], str) or self.use_moslora[active_adapter] is True:
                lora_mixer = self.lora_mixer[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]].to(lora_A.weight.dtype)

            if isinstance(self.use_moslora[active_adapter], str) or self.use_moslora[active_adapter] is True:
                lora_output = lora_B(lora_mixer(lora_A(dropout(sub_batch)))) * scaling
            else:
                lora_output = lora_B(lora_A(dropout(sub_batch))) * scaling

            result[sub_batch_indices_list[i]] += lora_output.to(torch_result_dtype)

        return result


class MoSLinear(nn.Module, MoSLoraLayer):
    # MoSLora implemented in a dense layer
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
        use_moslora: Union[bool, str] = True,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        MoSLoraLayer.__init__(self, base_layer, **kwargs)

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
            use_moslora=use_moslora,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

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
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        orig_weights += delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(orig_weights, transpose(delta_weight, self.fan_in_fan_out), scaling=1)
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                        orig_weights = dora_factor * (orig_weights + delta_weight)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights

                    if self.lora_bias[active_adapter]:
                        new_bias = base_layer.bias + self.lora_B[active_adapter].bias
                        if not torch.isfinite(new_bias).all():
                            raise ValueError(
                                f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                            )
                        base_layer.bias.data = new_bias

                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        base_layer.weight.data += delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = (
                            self.lora_magnitude_vector[active_adapter]
                            .get_weight_norm(
                                base_layer.weight, transpose(delta_weight, self.fan_in_fan_out), scaling=1
                            )
                            .detach()
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                        new_weight = dora_factor * (base_layer.weight.data + delta_weight)
                        base_layer.weight.data = new_weight

                    if self.lora_bias[active_adapter]:
                        base_layer.bias.data += self.lora_B[active_adapter].bias

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
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight.data -= delta_weight
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    weight_orig = weight.data / dora_factor.view(-1, 1) - delta_weight
                    weight.data = weight_orig

                if self.lora_bias[active_adapter]:
                    self.get_base_layer().bias.data -= self.lora_B[active_adapter].bias

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

        if isinstance(self.use_moslora[adapter], str) or self.use_moslora[adapter] is True:
            weight_mixer = self.lora_mixer[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()
            if isinstance(self.use_moslora[adapter], str) or self.use_moslora[adapter] is True:
                weight_mixer = weight_mixer.float()

        if isinstance(self.use_moslora[adapter], str) or self.use_moslora[adapter] is True:
            output_tensor = transpose(weight_B @ weight_mixer @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]
        else:
            output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

            if isinstance(self.use_moslora[adapter], str) or self.use_moslora[adapter] is True:
                self.lora_mixer[adapter].weight.data = weight_mixer.to(dtype)

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
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                if isinstance(self.use_moslora[active_adapter], str) or self.use_moslora[active_adapter] is True:
                    lora_mixer = self.lora_mixer[active_adapter]

                if not self.use_dora[active_adapter]:
                    if isinstance(self.use_moslora[active_adapter], str) or self.use_moslora[active_adapter] is True:
                        result = result + lora_B(lora_mixer(lora_A(dropout(x)))) * scaling
                    else:
                        result = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    if isinstance(dropout, nn.Identity) or not self.training:
                        base_result = result
                    else:
                        x = dropout(x)
                        base_result = None

                    result = result + self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                        base_result=base_result,
                    )

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "moslora." + rep



def dispatch_default(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: MoSLoraConfig,
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
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        new_module = MoSLinear(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        if not kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to False but the target module is `Conv1D`. " "Setting fan_in_fan_out to True."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
        kwargs.update(lora_config.loftq_config)
        new_module = MoSLinear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)

    return new_module