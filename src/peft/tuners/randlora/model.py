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

import math
import warnings
from typing import Union

import torch
import torch.nn as nn
from accelerate.utils.imports import is_bf16_available
from transformers.pytorch_utils import Conv1D

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import (
    TRANSFORMERS_MODELS_TO_RANDLORA_TARGET_MODULES_MAPPING,
)

from .._buffer_dict import BufferDict
from ..tuners_utils import _maybe_include_all_linear_layers
from .config import RandLoraConfig
from .layer import Linear, RandLoraLayer


def _kaiming_init(
    tensor_or_shape: Union[torch.Tensor, tuple[int, ...]],
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Kaiming Uniform Initialisation adapted to accept a `torch.Generator` object for PRNG.

    Args:
        tensor_or_shape (`Union[torch.Tensor, tuple[int, ...]]`):
            Tensor to initialise, or shape of new tensor to create and then initialise.
        generator: (`torch.Generator`):
            Generator object that manages the state of the PRNG algorithm in use.

    Returns:
        `torch.Tensor`: The initialised tensor.
    """
    if isinstance(tensor_or_shape, tuple):
        tensor = torch.empty(
            tensor_or_shape,
            dtype=torch.bfloat16 if is_bf16_available() else torch.float16,
        )
    else:
        tensor = tensor_or_shape

    with torch.no_grad():
        basis = torch.nn.init.kaiming_uniform_(tensor, a=math.sqrt(5), generator=generator)
        return basis


class RandLoraModel(BaseTuner):
    """
    Creates a RandLoRA model from a pretrained transformers model.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`RandLoraConfig`]): The configuration of the RandLora model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    Returns:
        `torch.nn.Module`: The RandLora model.

    Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import RandLoraConfig, get_peft_model

        >>> base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        >>> config = RandLoraConfig(r=32)
        >>> model = get_peft_model(base_model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`RandLoraConfig`]): The configuration of the RandLora model.
    """

    prefix: str = "randlora_"
    tuner_layer_cls = RandLoraLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_RANDLORA_TARGET_MODULES_MAPPING

    def _find_dim(self, config) -> tuple[int, int]:
        """
        Finds the largest input and output dimensions across linear layers that have been wrapped with RandLora.

        This will be used for determining the size of the shared randlora_A and randlora_B matrices.
        """
        model_config = self.get_model_config(self.model)

        peft_config = self._prepare_adapter_config(config, model_config)
        peft_config = _maybe_include_all_linear_layers(peft_config, self.model)

        largest_shape = None
        for key, module in self.model.named_modules():
            if not self._check_target_module_exists(peft_config, key):
                continue

            if isinstance(module, nn.Linear):
                module_shape = module.out_features, module.in_features
            elif isinstance(module, Conv1D):
                module_shape = module.weight.ds_shape if hasattr(module.weight, "ds_shape") else module.weight.shape
                module_shape = module_shape[::-1]
            else:
                continue

            if largest_shape is None:
                largest_shape = module_shape
                continue

            if module_shape != largest_shape:
                largest_shape = tuple(max(a, b) for a, b in zip(largest_shape, module_shape))

        if largest_shape is None:
            msg = "No layers types compatible with RandLora were found. Please check `peft_config.target_modules`."
            raise ValueError(msg)

        return largest_shape

    def _init_randlora_A_randlora_B_sparse(self, config: RandLoraConfig, adapter_name: str, sparsity: int = 3) -> None:
        """
        Sparse random projections as described in https://cs-people.bu.edu/evimaria/cs565/kdd-rp.pdf
        """

        linear_out_dim, linear_in_dim = self._find_dim(config)
        max_dim, min_dim = max(linear_out_dim, linear_in_dim), min(linear_out_dim, linear_in_dim)

        # use of persistent to exclude randlora_A and randlora_B from the state dict if we choose not to save them.
        self.randlora_A = BufferDict({}, persistent=config.save_projection)
        self.randlora_B = BufferDict({}, persistent=config.save_projection)

        # deterministic init of randlora_A and randlora_B if we know the key
        generator = torch.Generator(device="cpu").manual_seed(config.projection_prng_key)

        # The gamma matrix is applied on A meaning it can be unique (shared) across the n scaling matrices.
        # We also set randlora_A as the smallest matrix to reduce trainable parameters.
        randlora_A = torch.rand((config.r, 1, min_dim), generator=generator)

        # Number of bases to ensure full rank
        num_bases = min_dim / config.r
        num_bases = int(num_bases) if num_bases.is_integer() else int(num_bases) + 1  # Ensure full rank
        randlora_B = torch.rand((max_dim, num_bases, config.r), generator=generator)

        # The current implementation is a proof of concept and does take into consideration
        # the sparsity to reduce memory usage or speed up compute
        randlora_B_sparse = torch.zeros(randlora_B.shape)
        randlora_A_sparse = torch.zeros(randlora_A.shape)
        randlora_B_sparse[randlora_B < 1 / (2 * sparsity)] = -1
        randlora_B_sparse[randlora_B > 1 - 1 / (2 * sparsity)] = 1
        randlora_A_sparse[randlora_A < 1 / (2 * sparsity)] = -1
        randlora_A_sparse[randlora_A > 1 - 1 / (2 * sparsity)] = 1

        # Std normalization is empirically found to be the best
        randlora_A, randlora_B = (
            randlora_A_sparse / randlora_A_sparse.std(),
            randlora_B_sparse / randlora_B_sparse.std(),
        )
        self.randlora_A[adapter_name] = randlora_A
        self.randlora_B[adapter_name] = randlora_B

    def _init_randlora_A_randlora_B(self, config: RandLoraConfig, adapter_name: str) -> None:
        linear_out_dim, linear_in_dim = self._find_dim(config)
        max_dim, min_dim = max(linear_out_dim, linear_in_dim), min(linear_out_dim, linear_in_dim)

        # use of persistent to exclude randlora_A and randlora_B from the state dict if we choose not to save them.
        self.randlora_A = BufferDict({}, persistent=config.save_projection)
        self.randlora_B = BufferDict({}, persistent=config.save_projection)

        # deterministic init of randlora_A and randlora_B if we know the key
        generator = torch.Generator(device="cpu").manual_seed(config.projection_prng_key)

        # The gamma matrix is applied on A meaning it can be unique (shared) across the n scaling matrices.
        # We also set randlora_A as the smallest matrix to reduce trainable parameters.
        randlora_A = _kaiming_init((config.r, 1, min_dim), generator=generator)

        # Ensure full rank
        num_bases = min(linear_out_dim, linear_in_dim) / config.r
        num_bases = int(num_bases) if num_bases.is_integer() else int(num_bases) + 1
        randlora_B = torch.cat(
            [_kaiming_init((max_dim, 1, config.r), generator=generator) for _ in range(num_bases)], dim=1
        )

        # Std normalization is empirically found to be the best
        randlora_A, randlora_B = randlora_A / randlora_A.std(), randlora_B / randlora_B.std()
        self.randlora_A[adapter_name] = randlora_A
        self.randlora_B[adapter_name] = randlora_B

    def _pre_injection_hook(self, model: nn.Module, config: RandLoraConfig, adapter_name: str) -> None:
        if config.very_sparse:
            linear_out_dim, linear_in_dim = self._find_dim(config)
            self._init_randlora_A_randlora_B_sparse(
                config, adapter_name, sparsity=math.sqrt(min(linear_out_dim, linear_in_dim))
            )
        elif config.sparse:
            self._init_randlora_A_randlora_B_sparse(config, adapter_name, sparsity=3)
        else:
            self._init_randlora_A_randlora_B(config, adapter_name)

    def _check_new_adapter_config(self, config: RandLoraConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        super()._check_new_adapter_config(config)

        for existing_config in self.peft_config.values():
            if existing_config is config:
                # skip the current config
                continue

            if existing_config.projection_prng_key != config.projection_prng_key:
                raise ValueError(
                    f"RandLora PRNG initialisation key must be the same for all adapters. Got {config.projection_prng_key=} but "
                    f"previous config had {existing_config.projection_prng_key}."
                )

        save_project_unique_values = sorted({config.save_projection for config in self.peft_config.values()})
        if len(save_project_unique_values) > 1:
            raise ValueError(
                "RandLora projection weights must be saved for all adapters or none, but got multiple different values: "
                f"{save_project_unique_values}"
            )

    def _create_and_replace(
        self,
        randlora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        r = randlora_config.r
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": r,
            "randlora_alpha": randlora_config.randlora_alpha,
            "randlora_dropout": randlora_config.randlora_dropout,
            "fan_in_fan_out": randlora_config.fan_in_fan_out,
            "init_weights": randlora_config.init_weights,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
        }
        kwargs["bias"] = bias
        if isinstance(target, Linear):
            target.update_layer(
                adapter_name,
                self.randlora_A,
                self.randlora_B,
                r,
                randlora_config.randlora_alpha,
                randlora_config.randlora_dropout,
                randlora_config.init_weights,
            )
        else:
            new_module = self._create_new_module(
                randlora_config, self.randlora_A, self.randlora_B, adapter_name, target, **kwargs
            )
            if adapter_name not in self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(randlora_config, randlora_A, randlora_B, adapter_name, target, **kwargs):
        # avoid eager bnb import
        if is_bnb_available():
            import bitsandbytes as bnb

            from .bnb import Linear8bitLt

        if is_bnb_4bit_available():
            from .bnb import Linear4bit

        bias = kwargs.pop("bias", False)
        loaded_in_8bit = kwargs.get("loaded_in_8bit", False)
        loaded_in_4bit = kwargs.get("loaded_in_4bit", False)

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if loaded_in_8bit and isinstance(target_base_layer, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target_base_layer.state.has_fp16_weights,
                    "threshold": target_base_layer.state.threshold,
                    "index": target_base_layer.index,
                }
            )
            return Linear8bitLt(target, adapter_name, randlora_A, randlora_B, **eightbit_kwargs)
        elif loaded_in_4bit and isinstance(target_base_layer, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target_base_layer.compute_dtype,
                    "compress_statistics": target_base_layer.weight.compress_statistics,
                    "quant_type": target_base_layer.weight.quant_type,
                }
            )
            return Linear4bit(target, adapter_name, randlora_A, randlora_B, **fourbit_kwargs)
        elif isinstance(target_base_layer, torch.nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = randlora_config.fan_in_fan_out = False
        elif isinstance(target_base_layer, Conv1D):
            kwargs["is_target_conv_1d_layer"] = True
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = randlora_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `transformers.pytorch_utils.Conv1D`."
            )
        new_module = Linear(
            target,
            randlora_A,
            randlora_B,
            adapter_name,
            bias=bias,
            **kwargs,
        )

        return new_module
