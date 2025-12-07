# Copyright 2024-present the HuggingFace Inc. team.
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
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_VBLORA_TARGET_MODULES_MAPPING

from .config import VBLoRAConfig
from .layer import Linear, VBLoRALayer


class VBLoRAModel(BaseTuner):
    """
    Creates VBLoRA model from a pretrained transformers model.

    The method is described in detail in https://huggingface.co/papers/2405.15179.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`VBLoRAConfig`]): The configuration of the VBLoRA model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    Returns:
        `torch.nn.Module`: The VBLoRA model.

    Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import VBLoRAConfig, get_peft_model

        >>> base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        >>> config = VBLoRAConfig(
        ...     task_type="SEQ_CLS",
        ...     r=4,
        ...     target_modules=["fc1", "fc2", "k_proj", "out_proj", "q_proj", "v_proj"],
        ...     num_vectors=60,
        ...     vector_length=256,
        ...     save_only_topk_weights=True,
        ... )
        >>> model = get_peft_model(base_model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`VBLoRAConfig`]): The configuration of the VBLoRAConfig model.
    """

    prefix: str = "vblora_"
    tuner_layer_cls = VBLoRALayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_VBLORA_TARGET_MODULES_MAPPING

    def _init_vblora_vector_bank(self, config: VBLoRAConfig, adapter_name: str) -> None:
        vblora_vector_bank = torch.zeros(config.num_vectors, config.vector_length)
        torch.nn.init.uniform_(vblora_vector_bank, -config.init_vector_bank_bound, config.init_vector_bank_bound)
        self.vblora_vector_bank[adapter_name] = vblora_vector_bank

    def _pre_injection_hook(self, model: nn.Module, config: VBLoRAConfig, adapter_name: str) -> None:
        self.vblora_vector_bank = nn.ParameterDict({})

    def _create_and_replace(
        self,
        vblora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "fan_in_fan_out": vblora_config.fan_in_fan_out,
            "bias": bias,
        }
        self._init_vblora_vector_bank(vblora_config, adapter_name)
        # TODO: add quantization support

        if isinstance(target, Linear):
            target.update_layer(
                adapter_name=adapter_name,
                vblora_vector_bank=self.vblora_vector_bank,
                r=vblora_config.r,
                topk=vblora_config.topk,
                num_vectors=vblora_config.num_vectors,
                vector_length=vblora_config.vector_length,
                vblora_dropout=vblora_config.vblora_dropout,
                init_logits_std=vblora_config.init_logits_std,
            )
        else:
            new_module = self._create_new_module(
                vblora_config=vblora_config,
                vblora_vector_bank=self.vblora_vector_bank,
                adapter_name=adapter_name,
                target=target,
                **kwargs,
            )
            if adapter_name not in self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(vblora_config, vblora_vector_bank, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = vblora_config.fan_in_fan_out = False
        elif isinstance(target_base_layer, Conv1D):
            kwargs["is_target_conv_1d_layer"] = True
            if not kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True."
                )
                kwargs["fan_in_fan_out"] = vblora_config.fan_in_fan_out = True
        else:
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `transformers.pytorch_utils.Conv1D`."
            )
        new_module = Linear(
            base_layer=target,
            vblora_vector_bank=vblora_vector_bank,
            adapter_name=adapter_name,
            r=vblora_config.r,
            num_vectors=vblora_config.num_vectors,
            vector_length=vblora_config.vector_length,
            topk=vblora_config.topk,
            vblora_dropout=vblora_config.vblora_dropout,
            init_logits_std=vblora_config.init_logits_std,
            **kwargs,
        )

        return new_module

    def get_nb_savable_parameters(self, adapter="default") -> tuple[int, int]:
        r"""
        Returns the number of savable VB-LoRA parameters and other savable parameters.
        """
        logits_params = 0
        vector_bank_params = 0
        other_params = 0
        for name, param in self.named_parameters():
            if "vblora_logits" in name:
                logits_params += param.numel()
            elif "vblora_vector_bank" in name:
                vector_bank_params += param.numel()
            elif param.requires_grad:
                other_params += param.numel()
        if self.peft_config[adapter].save_only_topk_weights:
            num_vectors = self.peft_config[adapter].num_vectors
            factor = 1  # factor to count float32-equivalent parameters
            if num_vectors < 2**8:
                factor = 0.25
            elif num_vectors < 2**15:
                factor = 0.5
            elif num_vectors < 2**31:
                factor = 1
            else:
                factor = 2
            topk_weight_params = (
                logits_params / self.peft_config[adapter].num_vectors * (self.peft_config[adapter].topk - 1)
            )
            topk_indices_params = (
                logits_params / self.peft_config[adapter].num_vectors * self.peft_config[adapter].topk * factor
            )
            vblora_params = int(vector_bank_params + topk_weight_params + topk_indices_params)
        else:
            vblora_params = vector_bank_params + logits_params
        return vblora_params, other_params

    def print_savable_parameters(self) -> None:
        r"""
        Prints the number of savable VB-LoRA parameters and total savable parameters.
        """
        vblora_params, other_params = self.get_nb_savable_parameters()
        print(
            f"VB-LoRA params to-be-saved (float32-equivalent): {vblora_params:,d} "
            f"|| total params to-be-saved: {(vblora_params + other_params):,d}"
        )
