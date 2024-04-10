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

import json
import os
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
from safetensors.torch import save_model  # type: ignore
from transformers import PreTrainedModel

from peft.tuners.lora.model import LoraModel
from peft.tuners.tuners_utils import BaseTuner
from peft.utils.peft_types import PeftType

from .. import lora
from .classifier import InhibitorFlagPayload, Number, XLoraClassifier
from .config import XLoraConfig
from .layer import XLoRAConv2dLayer, XLoRAEmbeddingLayer, XLoRALinearLayer


@staticmethod
def apply_scalings_to_x(x: torch.Tensor, scalings_layer: torch.Tensor, adapter: int) -> torch.Tensor:
    # scalings_layer = [batch_size, seq_len, n_classes]
    scalings = scalings_layer[:, :, adapter].unsqueeze(-1)
    # scalings_layer = [batch_size, seq_len, 1]
    return x * scalings


def convert_layers_to_xlora(
    base: nn.Module,  # PeftModel
    xloramodel: nn.Module,  # XLoraModel
    config: XLoraConfig,
) -> (int, torch.device | None):
    """
    Returns the number of swapped layers.
    """
    total_swapped = 0

    device = None
    for module in base.modules():
        if isinstance(module, lora.Linear):
            device = module.lora_A[next(iter(module.lora_A))].weight.device
            new_layer = XLoRALinearLayer(
                model=xloramodel,
                target=module,
                target_forward=module.forward,
                layer_number=total_swapped,
                config=config,
            )
            module.forward = new_layer.forward  # type: ignore[method-assign]
            total_swapped += 1
        elif isinstance(module, lora.Embedding):
            device = module.lora_A[next(iter(module.lora_A))].weight.device
            new_layer = XLoRAEmbeddingLayer(
                model=xloramodel,
                target=module,
                target_forward=module.forward,
                layer_number=total_swapped,
                config=config,
            )
            module.forward = new_layer.forward  # type: ignore[method-assign]
            total_swapped += 1
        elif isinstance(module, lora.Conv2d):
            device = module.lora_A[next(iter(module.lora_A))].weight.device
            new_layer = XLoRAConv2dLayer(
                model=xloramodel,
                target=module,
                target_forward=module.forward,
                layer_number=total_swapped,
                config=config,
            )
            module.forward = new_layer.forward  # type: ignore[method-assign]
            total_swapped += 1

    return (total_swapped, device)


class XLoraModel(BaseTuner):
    """
    Creates an X-LoRA (Mixture of LoRA experts), model from a pretrained transformers model. Currently,
    this X-LoRA implementation only works with models with a transformer architecture.

    The method is described in detail in https://arxiv.org/abs/2402.07148.

    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`XLoraConfig`]): The configuration of the Lora model.
        adapter_name (`str`): The name of the adapter, does not affect the LoRA adapter names.

    Returns:
        `torch.nn.Module`: The X-LoRA model.

    Example:
        ```py
        >>> from transformers import AutoModelForCausalLM, AutoConfig
        >>> from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_int8_training

        >>> model_config = AutoConfig.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        >>> config = XLoraConfig(
        ...     task_type="CAUSAL_LM",
        ...     hidden_size=model_config.hidden_size,
        ...     xlora_depth=4,
        ...     adapters={
        ...         "adapter_1": "./path/to/the/checkpoint/",
        ...         "adapter_2": "./path/to/the/checkpoint/",
        ...         "adapter_n": "./path/to/the/checkpoint/",
        ...     },
        ... )

        >>> model = AutoModelForCausalLM.from_pretrained(
        ...     "mistralai/Mistral-7B-Instruct-v0.1",
        ...     trust_remote_code=True,
        ...     use_flash_attention_2=False,
        ...     device_map="cuda:0",
        ...     torch_dtype=torch.bfloat16,
        ... )
        >>> model = prepare_model_for_int8_training(model)
        >>> xlora_model = get_peft_model(model, config)
        ```
    """

    def __init__(
        self,
        model: nn.Module,
        config: Union[dict[str, XLoraConfig], XLoraConfig],
        adapter_name: str,
    ) -> None:
        self.lora_model = LoraModel(model, config, adapter_name)
        super().__init__(model, config, adapter_name)

    def post_init_lora(
        self,
        model: nn.Module,
        peft_config: XLoraConfig,
        adapter_name: str,
        model_peft: nn.Module,
    ) -> None:
        # model_peft: PeftModel
        self.xlora_config = peft_config

        if hasattr(model.config, "use_cache") and not model.config.use_cache:
            raise ValueError("`use_cache` must be False")

        use_trainable_adapters = peft_config.use_trainable_adapters
        adapters_items = peft_config.adapters.items()

        # For load_adapter to think we are a LoraModel
        model_peft.peft_type = PeftType.LORA

        for name, model_id in adapters_items:
            model_peft.load_adapter(model_id, name, is_trainable=use_trainable_adapters)

        self.lora_model.delete_adapter(adapter_name)

        self.lora_model.set_adapter(list(peft_config.adapters.keys()))
        model_peft.active_adapter = name
        model_peft.peft_type = PeftType.XLORA

        def hook(module, *args, **kwargs) -> None:
            args_real = args[0]
            kwargs_real: dict = args[1]
            kwargs_real.update(kwargs)

            xlora_classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore

            if "_xlora_classifier_inhibitor_flag" in kwargs_real:
                payload: InhibitorFlagPayload = kwargs_real["_xlora_classifier_inhibitor_flag"]

                del kwargs_real["_xlora_classifier_inhibitor_flag"]

                self.internal_xlora_scalings = torch.full(  # type: ignore
                    (payload.batch_size, payload.seq_len, xlora_classifier.n_layers, xlora_classifier.n_classes),
                    payload.override_scaling_pass_value,
                )

                return

            xlora_scalings = xlora_classifier.forward(
                *args_real,
                **kwargs_real,
            )
            # Set the scalings
            self.internal_xlora_scalings = xlora_scalings

        model.register_forward_pre_hook(hook, with_kwargs=True, prepend=True)

        self._freeze_all_adapters()

        total_swapped, device = convert_layers_to_xlora(
            model_peft,
            self,
            peft_config,
        )

        n_classes = len(peft_config.adapters)
        xlora_classifier = XLoraClassifier(model_peft, peft_config, n_classes, total_swapped, device)

        # Setup the model internal state
        self.internal_xlora_classifier = xlora_classifier
        self.internal_xlora_scalings = None  # type: ignore

    def _freeze_all_adapters(self):
        self.eval()
        if not self.xlora_config.use_trainable_adapters:
            for name, param in self.named_parameters():
                if "lora_" in name:
                    param.requires_grad = False

    def generate(self, *args, **kwargs):
        # Rely on LoraModel.__getattr__
        res = self.model.generate(*args, **kwargs)  # type: ignore
        # TODO(EricLBuehler): Evaluate effectiveness and performance degradation
        self._freeze_all_adapters()
        return res

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        active_adapters = []
        copy = self.lora_model.active_adapters.copy()
        for name in self.lora_model.active_adapters:
            if not isinstance(self.lora_model.peft_config[name], XLoraConfig):
                active_adapters.append(name)
        self.lora_model.active_adapter = active_adapters
        if self.xlora_config.use_trainable_adapters:
            super()._mark_only_adapters_as_trainable(model)

        self.lora_model.active_adapter = copy

    def _save_pretrained_hook(
        self,
        save_directory: str,
        safe_serialization: bool = True,
        is_main_process: bool = True,
        **kwargs: Any,
    ) -> None:
        conf = self.xlora_config.__dict__.copy()

        # So that the adapters are unloadable and the user is forced to set them for from_pretrained
        conf["adapters"] = None
        with open(os.path.join(save_directory, "xlora_config.json"), "w") as f:
            json.dump(conf, f)

    def forward(self, *args, **kwargs):
        return self.lora_model.model(*args, **kwargs)  # Important to *call* the model

    def set_topk_lora(self, value: Optional[int]):
        """
        Sparsely select the specified top_k LoRA experts instead of the default dense method. Set to None to use dense. This is reflected in the config.
        """
        classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore
        classifier.config.top_k_lora = value

    def set_global_scaling_weight(self, weight: float):
        """
        Set the global LoRA weight, a scalar to multiply the output of each LoRA adapter by. This is by default 1. This is reflected in the config.
        """
        classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore
        classifier.config.global_scaling_weight = weight

    def get_global_scaling_weight(self) -> float:
        """
        Get the global LoRA weight.
        """
        classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore
        return classifier.config.global_scaling_weight

    def get_latest_scalings(self) -> Optional[torch.Tensor]:
        """
        Returns the latest scalings prediction, or None if no scalings have been predicted. The tensor is of shape (batch_size, seq_len, n_layers, n_classes).
        """
        return self.internal_xlora_scalings

    def get_scalings_log(self) -> List[torch.Tensor]:
        """
        Returns a shallow (only copying the list itself not the tensors) copy of the list containing the scalings log. Editing the list does not change the underlying log.
        The tensors are of shape (batch_size, seq_len, n_layers, n_classes). The seq_len dim may vary with input dimension.
        """
        classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore
        return classifier.log_scalings.copy()

    def enable_scalings_logging(self):
        """
        Enable scalings logging.
        """
        classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore
        classifier.scalings_logging = True

    def disable_scalings_logging(self):
        """
        Disable scalings logging, without clearing the log.
        """
        classifier: XLoraClassifier = self.model.internal_xlora_classifier  # type: ignore
        classifier.scalings_logging = False

    def clear_scalings_log(self):
        """
        Clear the scalings log.
        """
        classifier: XLoraClassifier = self.model.internal_xlora_classifier  # type: ignore
        classifier.log_scalings.clear()

    def flush_log_scalings(self, path: str):
        """
        Write the scalings log (a tensor of shape (num_logged, batch_size, seq_len, n_layers, n_classes)) to the specified path.
        If the tensor cannot be constructed, multiple files are written containing tensors of shape
        (num_logged, batch_size, seq_len, n_layers, n_classes) such that each file contains one sequence length. Additionally a JSON
        file is outputted containing the mapping from each sequence log file to the index of the contained tensor so that one may reconstruct
        the log order.

        The file specified should not contain an extension.
        """
        classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore
        classifier._flush_log_scalings(path)

    def set_use_trainable_adapters(self, use_trainable_adapters: bool):
        """
        Set the adapters to trainable or not trainable.

        This is reflected in the config.
        """
        for name, param in self.named_parameters():
            if "lora_" in name:
                param.requires_grad = use_trainable_adapters

        self.xlora_config.use_trainable_adapters = use_trainable_adapters

    def get_use_trainable_adapters(self) -> bool:
        """
        Get the trainable or not trainable state of the adapters.
        """
        return self.xlora_config.use_trainable_adapters