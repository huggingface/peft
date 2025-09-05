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

# The implementation is based on "Parameter-Efficient Orthogonal Finetuning
# via Butterfly Factorization" (https://huggingface.co/papers/2311.06243) in ICLR 2024.

import warnings

import torch

from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
)
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
)

from .layer import BOFTLayer, Conv2d, Linear


class BOFTModel(BaseTuner):
    """
    Creates BOFT and OFT model from a pretrained transformers model. Paper: https://huggingface.co/papers/2311.06243
    https://huggingface.co/papers/2306.07280

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`BOFTConfig`]): The configuration of the BOFT model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    Returns:
        `torch.nn.Module`: The BOFT model.

    Example::

        >>> import transformers >>> from transformers import AutoModelForSeq2SeqLM, BOFTConfig >>> from peft import
        BOFTConfig, get_peft_model

        >>> config = BOFTConfig( ... boft_block_size=8, ... boft_n_butterfly_factor=1, ... target_modules=["query",
        "value", "key", "output.dense", "mlp.fc1", "mlp.fc2"], ... boft_dropout=0.1, ... bias="boft_only", ...
        modules_to_save=["classifier"], ... )

        >>> model = transformers.Dinov2ForImageClassification.from_pretrained( ... "facebook/dinov2-large", ...
        num_labels=100, ... ) >>> boft_model = get_peft_model(model, config)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`BOFTConfig`]): The configuration of the BOFT model.
    """

    prefix: str = "boft_"
    base_layer_cls = BOFTLayer

    def _create_and_replace(
        self,
        boft_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "boft_block_size": boft_config.boft_block_size,
            "boft_block_num": boft_config.boft_block_num,
            "boft_n_butterfly_factor": boft_config.boft_n_butterfly_factor,
            "boft_dropout": boft_config.boft_dropout,
            "fan_in_fan_out": boft_config.fan_in_fan_out,
            "init_weights": boft_config.init_weights,
        }
        kwargs["bias"] = bias

        # If it is not a BOFTLayer, create a new module, else update it with new adapters
        if not isinstance(target, BOFTLayer):
            new_module = self._create_new_module(boft_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
        else:
            target.update_layer(
                adapter_name,
                boft_block_size=boft_config.boft_block_size,
                boft_block_num=boft_config.boft_block_num,
                boft_n_butterfly_factor=boft_config.boft_n_butterfly_factor,
                boft_dropout=boft_config.boft_dropout,
                init_weights=boft_config.init_weights,
            )

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        meta = torch.device("meta")
        # dispatch to correct device
        for name, module in new_module.named_modules():
            if self.prefix in name:
                if not any(p.device == meta for p in module.parameters()):
                    module.to(child.weight.device)

    @staticmethod
    def _create_new_module(boft_config, adapter_name, target, **kwargs):
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
                kwargs["fan_in_fan_out"] = boft_config.fan_in_fan_out = False
            new_module = Linear(target, adapter_name, **kwargs)
        elif isinstance(target_base_layer, torch.nn.Conv2d):
            new_module = Conv2d(target, adapter_name, **kwargs)
        else:
            raise ValueError(
                f"Target module {target} is not supported. "
                "Currently, only `torch.nn.Linear` and `torch.nn.Conv2d` are supported."
            )

        return new_module

    def set_adapter(self, adapter_name):
        self.set_auxiliary_adapters(adapter_name)
        for module in self.model.modules():
            if isinstance(module, BOFTLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config
