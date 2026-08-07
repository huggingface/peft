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
from torch import nn
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
                config=vblora_config,
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
            if vblora_config.fan_in_fan_out:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                vblora_config.fan_in_fan_out = False
        elif isinstance(target_base_layer, Conv1D):
            kwargs["is_target_conv_1d_layer"] = True
            if not vblora_config.fan_in_fan_out:
                warnings.warn(
                    "fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True."
                )
                vblora_config.fan_in_fan_out = True
        else:
            raise TypeError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `transformers.pytorch_utils.Conv1D`."
            )
        new_module = Linear(
            base_layer=target,
            vblora_vector_bank=vblora_vector_bank,
            adapter_name=adapter_name,
            config=vblora_config,
            r=vblora_config.r,
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

    @classmethod
    def _get_adapter_state_dict(cls, model, config, adapter_name, state_dict, unwanted_adapter_names):
        to_return = {}
        # choose the most efficient dtype for indices
        if config.num_vectors < 2**8:
            indices_dtype = torch.uint8
        elif config.num_vectors < 2**15:
            indices_dtype = torch.int16
        elif config.num_vectors < 2**31:
            indices_dtype = torch.int32
        else:
            indices_dtype = torch.int64
        if config.save_only_topk_weights:
            # in save_only_topk_weights mode, we save topk_indices and topk_weights for parameter efficiency
            for k in state_dict:
                if "vblora_logits" in k:
                    logits, indices = state_dict[k].topk(config.topk)
                    to_return.update({k + "_topk_indices": indices.to(dtype=indices_dtype)})
                    to_return.update({k + "_topk_weights": torch.softmax(logits, dim=-1)[:, :, :-1].contiguous()})
        else:
            to_return = {k: state_dict[k] for k in state_dict if "vblora_logits" in k}
        to_return["base_model.vblora_vector_bank." + adapter_name] = state_dict[
            "base_model.vblora_vector_bank." + adapter_name
        ]
        to_return.update(cls._get_learnable_bias_state_dict(model, state_dict, config))
        return to_return

    @classmethod
    def _remove_adapter_name_from_key(cls, key, adapter_name):
        if "." in key and not key.endswith(f".{adapter_name}"):
            key_without_suffix, _, suffix = key.rpartition(".")
            if suffix.startswith(f"{adapter_name}_"):
                # special case: VBLoRA creates keys that require this replacement:
                # base_model.model.lin0.vblora_logits_A.default_topk_indices =>
                # base_model.model.lin0.vblora_logits_A_topk_indices
                return key_without_suffix + "_" + suffix.removeprefix(f"{adapter_name}_")
        return super()._remove_adapter_name_from_key(key, adapter_name)

    @classmethod
    def _remap_adapter_state_dict_for_load(cls, model, config, adapter_name, state_dict):
        if config.save_only_topk_weights:
            num_vectors, _ = model.vblora_vector_bank[adapter_name].shape
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                # in save_only_topk_weights mode, only topk_indices and topk_weights are saved
                # note that topk_indices and topk_weights serve as an efficient representation of the logits
                # so we need to recover the logits from the topk_indices and topk_weights
                if "_topk_indices" in k:
                    v = state_dict[k].to(torch.long)
                    original_key = k.replace("_topk_indices", "")
                    # find the corresponding topk_weights from the state_dict
                    topk_weights = state_dict[k.replace("_topk_indices", "_topk_weights")]
                    # as we only save the first k-1 topk_weights, here we recover the last one
                    topk_weights = torch.cat([topk_weights, 1 - topk_weights.sum(-1, keepdim=True)], dim=-1)
                    # convert the weights to logits
                    topk_logits = torch.log(topk_weights)
                    matrix = (
                        torch.zeros([*(topk_logits.shape[:-1]), num_vectors])
                        .fill_(float("-inf"))
                        .to(topk_logits.device)
                        .scatter(-1, v, topk_logits)
                    )
                    # add logits to the state_dict
                    state_dict[original_key] = matrix
                    # delete the topk_indices and topk_weights from the state_dict
                    del state_dict[k]
                    del state_dict[k.replace("_topk_indices", "_topk_weights")]
        return super()._remap_adapter_state_dict_for_load(model, config, adapter_name, state_dict)
