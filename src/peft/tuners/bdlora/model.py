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

import operator
from typing import Optional

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.utils import get_quantization_config
from peft.utils.other import get_pattern_key

from ..lora import LoraLayer, LoraModel
from ..lora.aqlm import dispatch_aqlm
from ..lora.awq import dispatch_awq
from ..lora.eetq import dispatch_eetq
from ..lora.gptq import dispatch_gptq
from ..lora.hqq import dispatch_hqq
from ..lora.inc import dispatch_inc
from ..lora.layer import Linear, ParamWrapper, dispatch_default
from ..lora.torchao import dispatch_torchao
from ..lora.tp_layer import dispatch_megatron
from .layer import ColumnParallelLinearLora, RowParallelLinearLora


class BdLoraModel(LoraModel):
    prefix: str = "lora_"

    @staticmethod
    def _create_new_module(bdlora_config, adapter_name, target, target_name, **kwargs):
        # Replacing this function is easier than replacing the _create_and_replace method
        # ---
        # Read the user-defined lists from the config
        row_patterns = bdlora_config.row_sharded_modules or []
        column_patterns = bdlora_config.column_sharded_modules or []

        # Check if the current module's name matches any pattern
        is_row_module = any(pattern in target_name for pattern in row_patterns)
        is_column_module = any(pattern in target_name for pattern in column_patterns)

        if is_row_module:
            LayerToUse = RowParallelLinearLora
        elif is_column_module:
            LayerToUse = ColumnParallelLinearLora
        else:
            LayerToUse = Linear  # We assume that the module should be unsharded in this case
        dispatchers = []

        def bdlora_dispatch_func(target, adapter_name, lora_config, **kwargs):
            kwargs["nblocks"] = lora_config.nblocks
            layer = LayerToUse(target, adapter_name, **kwargs)
            return layer

        dispatchers.append(bdlora_dispatch_func)

        # avoid eager bnb import
        if is_bnb_available():
            from peft.tuners.lora.bnb import dispatch_bnb_8bit

            dispatchers.append(dispatch_bnb_8bit)

        if is_bnb_4bit_available():
            from peft.tuners.lora.bnb import dispatch_bnb_4bit

            dispatchers.append(dispatch_bnb_4bit)

        dispatchers.extend(
            [
                dispatch_eetq,
                dispatch_aqlm,
                dispatch_awq,
                dispatch_gptq,
                dispatch_hqq,
                dispatch_inc,
                dispatch_torchao,
                dispatch_megatron,
                dispatch_default,
            ]
        )

        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(target, adapter_name, lora_config=bdlora_config, **kwargs)
            if new_module is not None:  # first match wins
                break

        if new_module is None:
            # no module could be matched
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv1d`, `torch.nn.Conv2d`, `torch.nn.Conv3d`, "
                "`transformers.pytorch_utils.Conv1D`, `torch.nn.MultiheadAttention.`."
            )

        return new_module

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        *,
        parameter_name: Optional[str] = None,
    ) -> None:
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        if lora_config.target_parameters:
            # Right now, unfortunately, we don't support multiple adapters with target_parameters on the same model.
            other_configs_use_target_params = any(
                conf.target_parameters for key, conf in self.peft_config.items() if key != adapter_name
            )
            if other_configs_use_target_params:
                raise ValueError(
                    f"Adding a LoRA config with `target_parameters={lora_config.target_parameters}` but there are "
                    "already other LoRA adapters on this model that use `target_parameters`. At the moment, only "
                    "one LoRA adapter per model with `target_parameters` is allowed."
                )

        # Regexp matching - Find key which matches current target_name in patterns provided
        r_key = get_pattern_key(lora_config.rank_pattern.keys(), current_key)
        alpha_key = get_pattern_key(lora_config.alpha_pattern.keys(), current_key)
        r = lora_config.rank_pattern.get(r_key, lora_config.r)
        alpha = lora_config.alpha_pattern.get(alpha_key, lora_config.lora_alpha)

        kwargs = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "use_rslora": lora_config.use_rslora,
            "use_dora": lora_config.use_dora,
            "use_alora": lora_config.alora_invocation_tokens is not None,
            "use_qalora": lora_config.use_qalora,
            "qalora_group_size": lora_config.qalora_group_size,
            "ephemeral_gpu_offload": lora_config.runtime_config.ephemeral_gpu_offload,
            "lora_bias": lora_config.lora_bias,
            "arrow_config": lora_config.arrow_config,
            "loaded_in_8bit": getattr(self.model, "is_loaded_in_8bit", False),
            "loaded_in_4bit": getattr(self.model, "is_loaded_in_4bit", False),
            "parameter_name": parameter_name,
        }

        # for torchao merging, we need the get_apply_tensor_subclass from the quantization config
        try:
            kwargs["get_apply_tensor_subclass"] = operator.attrgetter(
                "hf_quantizer.quantization_config.get_apply_tensor_subclass"
            )(self.model)
        except AttributeError:
            pass

        quant_methods = ["gptq", "aqlm", "awq"]
        for quant_method in quant_methods:
            quantization_config = get_quantization_config(self.model, method=quant_method)
            if quantization_config is not None:
                kwargs[f"{quant_method}_quantization_config"] = quantization_config

        # note: AdaLoraLayer is a subclass of LoraLayer, we need to exclude it
        from peft.tuners.adalora import AdaLoraLayer

        # if the target is a ParamWrapper, we nest it to allow targeting multiple nn.Parameter on the same module
        wrap_target_param = isinstance(target, ParamWrapper) and (adapter_name in target.lora_A)
        if isinstance(target, LoraLayer) and not isinstance(target, AdaLoraLayer) and not wrap_target_param:
            target.update_layer(
                adapter_name,
                r,
                lora_alpha=alpha,
                lora_dropout=lora_config.lora_dropout,
                init_lora_weights=lora_config.init_lora_weights,
                use_rslora=lora_config.use_rslora,
                use_dora=lora_config.use_dora,
                lora_bias=lora_config.lora_bias,
                arrow_config=lora_config.arrow_config,
                inference_mode=lora_config.inference_mode,
            )
        else:
            if isinstance(target, ParamWrapper) and (parameter_name == target.parameter_name):
                raise ValueError(
                    "Trying to target the same nn.Parameter twice, this should not happen. Please open an issue on the "
                    "PEFT repo: https://github.com/huggingface/peft/issues"
                )
            device_map = self.model.hf_device_map if hasattr(self.model, "hf_device_map") else None
            new_module = self._create_new_module(
                lora_config, adapter_name, target, target_name, device_map=device_map, **kwargs
            )
            new_module.requires_grad_(True)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
