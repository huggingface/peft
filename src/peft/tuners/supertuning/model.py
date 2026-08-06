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
from __future__ import annotations

import torch

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import TRANSFORMERS_MODELS_TO_SUPERTUNING_TARGET_MODULES_MAPPING

from .layer import Linear, SupertuningLayer


class SupertuningModel(BaseTuner):
    """Super-Tuning tuner (arXiv:2607.09287).

    Freezes the base weights and trains only a sparse support of scalar entries selected by weight magnitude (paper's
    best single-mechanism configuration; data-free). When ``config.r`` is set, additionally allocates LoRA A/B
    parameters composed additively with the sparse support — the paper's Supra hybrid.

    Args:
        model ([`~transformers.PreTrainedModel`]): The base model to adapt.
        config ([`SupertuningConfig`]): The Supertuning configuration.
        adapter_name (`str`): The adapter name. Defaults to ``"default"``.
        low_cpu_mem_usage (`bool`, *optional*): Create empty adapter weights on the meta device to speed up loading.

    Returns:
        `torch.nn.Module`: The Supertuning-wrapped model.

    Example (pure Super):

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import SupertuningConfig, get_peft_model

        >>> base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
        >>> config = SupertuningConfig(target_modules=["q_proj", "v_proj"], sparsity=0.99)
        >>> model = get_peft_model(base, config)
        ```

    Example (Supra hybrid):

        ```py
        >>> config = SupertuningConfig(
        ...     target_modules=["q_proj", "v_proj"], sparsity=0.99, r=8, lora_alpha=16,
        ... )
        >>> model = get_peft_model(base, config)
        ```

    Paper: https://arxiv.org/abs/2607.09287
    """

    prefix: str = "supertuning_"
    tuner_layer_cls = SupertuningLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_SUPERTUNING_TARGET_MODULES_MAPPING

    @staticmethod
    def _create_new_module(supertuning_config, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            new_module = Linear(target, adapter_name, config=supertuning_config, **kwargs)
        else:
            raise TypeError(
                f"Target module {target} is not supported. Currently, only `torch.nn.Linear` is supported."
            )
        return new_module

    def _create_and_replace(
        self,
        supertuning_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        kwargs = {}

        if isinstance(target, SupertuningLayer):
            target.update_layer(
                adapter_name,
                config=supertuning_config,
            )
        else:
            new_module = self._create_new_module(supertuning_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # Adding an additional adapter: it is not automatically trainable.
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def get_trainable_parameters_count(self, adapter_name: str = "default") -> dict:
        """Report per-adapter trainable-parameter accounting for reporting / test assertions."""
        total_params = 0
        sparse_params = 0
        lora_params = 0

        for _, module in self.model.named_modules():
            if isinstance(module, Linear):
                base_layer = module.get_base_layer()
                total_params += base_layer.weight.numel()

                if adapter_name in module.supertuning_values.keys():
                    sparse_params += int(module.supertuning_values[adapter_name].numel())
                if adapter_name in module.lora_A.keys():
                    lora_params += int(
                        module.lora_A[adapter_name].numel() + module.lora_B[adapter_name].numel()
                    )

        trainable = sparse_params + lora_params
        return {
            "total_parameters": total_params,
            "sparse_parameters": sparse_params,
            "lora_parameters": lora_params,
            "trainable_parameters": trainable,
            "sparsity": 1.0 - (trainable / total_params if total_params > 0 else 0),
        }
