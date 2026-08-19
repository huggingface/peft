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

import warnings

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import (
    TRANSFORMERS_MODELS_TO_PVERA_TARGET_MODULES_MAPPING,
)

from .._buffer_dict import BufferDict
from ..tuners_utils import _maybe_include_all_linear_layers
from .config import PveraConfig
from .layer import Linear, PveraLayer


class PveraModel(BaseTuner):
    """
    Creates Probabilistic Vector-based Random Matrix Adaptation (PVeRA) model from a pretrained transformers model.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`PveraConfig`]): The configuration of the PVeRA model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    Returns:
        `torch.nn.Module`: The PVeRA model.

    Example:

        ```py
        >>> from transformers import AutoModel
        >>> from peft import PveraConfig, get_peft_model

        >>> base_model = AutoModel.from_pretrained("facebook/dinov2-base")
        >>> config = PveraConfig(r=128, sample_at_inference=False)
        >>> model = get_peft_model(base_model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`PveraConfig`]): The configuration of the PVeRA model.
    """

    prefix: str = "pvera_lambda_"
    tuner_layer_cls = PveraLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_PVERA_TARGET_MODULES_MAPPING

    def _find_dim(self, config) -> tuple[int, int]:
        """
        Finds the largest input and output dimensions across linear layers that have been wrapped with PVeRA.

        This will be used for determining the size of the shared pvera_A and pvera_B matrices.
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
            msg = "No layers types compatible with PVeRA were found. Please check `peft_config.target_modules`."
            raise ValueError(msg)

        return largest_shape

    def _init_pvera_A_pvera_B(self, config: PveraConfig, adapter_name: str) -> None:
        linear_out_dim, linear_in_dim = self._find_dim(config)

        # use of persistent to exclude pvera_A and pvera_B from the state dict if we choose not to save them.
        self.pvera_A = BufferDict({}, persistent=config.save_projection)
        self.pvera_B = BufferDict({}, persistent=config.save_projection)

        # deterministic init of pvera_A and pvera_B if we know the key
        generator = torch.Generator(device="cpu").manual_seed(config.projection_prng_key)
        pvera_A = torch.nn.init.kaiming_uniform_(torch.empty(config.r * 2, linear_in_dim), generator=generator)
        pvera_B = torch.nn.init.kaiming_uniform_(torch.empty(linear_out_dim, config.r), generator=generator)

        self.pvera_A[adapter_name] = pvera_A
        self.pvera_B[adapter_name] = pvera_B

    def _pre_injection_hook(self, model: nn.Module, config: PveraConfig, adapter_name: str) -> None:
        self._init_pvera_A_pvera_B(config, adapter_name)

    def _check_new_adapter_config(self, config: PveraConfig) -> None:
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
                    f"PVeRA PRNG initialisation key must be the same for all adapters. Got {config.projection_prng_key=} but "
                    f"previous config had {existing_config.projection_prng_key}."
                )

        save_project_unique_values = {config.save_projection for config in self.peft_config.values()}
        if len(save_project_unique_values) > 1:
            raise ValueError(
                "PVeRA projection weights must be saved for all adapters or none, but got multiple different values: "
                f"{save_project_unique_values}"
            )

    def _create_and_replace(
        self,
        pvera_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        r = pvera_config.r
        bias = hasattr(target, "bias") and target.bias is not None
        sample_at_inference = self._resolve_sample_at_inference(pvera_config, current_key)
        kwargs = {
            "r": r,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
            "sample_at_inference": sample_at_inference,
        }
        kwargs["bias"] = bias

        if isinstance(target, Linear):
            target.update_layer(
                adapter_name,
                pvera_A=self.pvera_A,
                pvera_B=self.pvera_B,
                r=r,
                config=pvera_config,
                sample_at_inference=sample_at_inference,
            )
        else:
            new_module = self._create_new_module(
                pvera_config, self.pvera_A, self.pvera_B, adapter_name, target, **kwargs
            )
            if adapter_name not in self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _resolve_sample_at_inference(pvera_config, current_key: str) -> bool:
        """Resolve `sample_at_inference` for the module at `current_key`.

        The config value is either a bool that applies to every module, or a dict mapping module names to bools, in
        which case modules that are not listed default to `False`.
        """
        if isinstance(pvera_config.sample_at_inference, bool):
            return pvera_config.sample_at_inference
        return pvera_config.sample_at_inference.get(current_key, False)

    @staticmethod
    def _create_new_module(pvera_config, pvera_A, pvera_B, adapter_name, target, **kwargs):
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
            return Linear8bitLt(target, adapter_name, pvera_A, pvera_B, config=pvera_config, **eightbit_kwargs)
        elif loaded_in_4bit and isinstance(target_base_layer, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target_base_layer.compute_dtype,
                    "compress_statistics": target_base_layer.weight.compress_statistics,
                    "quant_type": target_base_layer.weight.quant_type,
                }
            )
            return Linear4bit(target, adapter_name, pvera_A, pvera_B, config=pvera_config, **fourbit_kwargs)
        elif isinstance(target_base_layer, torch.nn.Linear):
            if pvera_config.fan_in_fan_out:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                pvera_config.fan_in_fan_out = False
        elif isinstance(target_base_layer, Conv1D):
            if not pvera_config.fan_in_fan_out:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True."
                )
                pvera_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `transformers.pytorch_utils.Conv1D`."
            )

        new_module = Linear(
            target,
            pvera_A,
            pvera_B,
            adapter_name,
            config=pvera_config,
            **kwargs,
        )

        return new_module

    @classmethod
    def _get_adapter_state_dict(cls, model, config, adapter_name, state_dict, unwanted_adapter_names):
        to_return = super()._get_adapter_state_dict(model, config, adapter_name, state_dict, unwanted_adapter_names)
        # Each layer holds a reference to the shared projections, so the state dict contains a duplicate of them for
        # every layer. Remove all of them here; the canonical model-level entries ("base_model.pvera_A.<adapter>" etc.)
        # are only added back after the explicit save_projection check below.
        to_return = {k: v for k, v in to_return.items() if (".pvera_A." not in k) and (".pvera_B." not in k)}
        if config.save_projection:
            # TODO: adding pvera_A and pvera_B to `self.get_base_layer` would
            # make name to match here difficult to predict.
            if f"base_model.pvera_A.{adapter_name}" not in state_dict:
                raise ValueError(
                    "Model was initialised to not save pvera_A and pvera_B but config now specifies to save projection!"
                    + " Set `config.save_projection` to `False`."
                )
            to_return["base_model.pvera_A." + adapter_name] = state_dict["base_model.pvera_A." + adapter_name]
            to_return["base_model.pvera_B." + adapter_name] = state_dict["base_model.pvera_B." + adapter_name]
        return to_return

    @classmethod
    def _remap_adapter_state_dict_for_load(cls, model, config, adapter_name, state_dict):
        # note that the remapping renames the projection keys from e.g. "base_model.pvera_A" (checkpoint format) to
        # "base_model.pvera_A.{adapter_name}" (model format)
        peft_model_state_dict = super()._remap_adapter_state_dict_for_load(model, config, adapter_name, state_dict)
        if config.save_projection and f"base_model.pvera_A.{adapter_name}" not in peft_model_state_dict:
            raise ValueError(
                "Specified to load pvera_A and pvera_B from state dictionary however they were not present!"
            )
        elif not config.save_projection and "base_model.pvera_A" in peft_model_state_dict:
            # note: with save_projection=False, the projection buffers are non-persistent and thus have no model state
            # dict entry to be remapped to, so the checkpoint key is still in its unsuffixed form here
            warnings.warn(
                "Specified to not load pvera_A and pvera_B from state dictionary however they are present in state"
                " dictionary! Consider using them to ensure checkpoint loading is correct on all platforms using"
                " `peft_config.save_projection = True`"
            )
        elif not config.save_projection:  # and no vera_A in state dictionary
            warnings.warn(
                "Specified to not load pvera_A and pvera_B from state dictionary. This means we will be relying on"
                " PRNG initialisation to restore these projections using `config.projection_prng_key`, which may"
                " not be accurate on all system configurations."
            )
        return peft_model_state_dict
