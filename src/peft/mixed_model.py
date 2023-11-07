# coding=utf-8
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

import inspect
import os
from contextlib import contextmanager
from typing import Any, Optional, Union

import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.hooks import AlignDevicesHook, add_hook_to_module
from accelerate.utils import get_balanced_memory
from huggingface_hub import ModelCard, ModelCardData, hf_hub_download
from torch import nn
from transformers.utils import PushToHubMixin

from peft.tuners.lycoris.model import COMPATIBLE_TUNER_TYPES

from . import __version__
from .config import PeftConfig
from .tuners import (
    AdaLoraModel,
    IA3Model,
    LoHaModel,
    LoKrModel,
    LoraModel,
    LycorisModel,
)
from .utils import (
    PeftType,
    _set_adapter,
    _set_trainable,
    infer_device,
    load_peft_weights,
    set_peft_model_state_dict,
)


PEFT_TYPE_TO_MODEL_MAPPING = {
    PeftType.LORA: LoraModel,
    PeftType.LOHA: LoHaModel,
    PeftType.LOKR: LoKrModel,
    PeftType.ADALORA: AdaLoraModel,
    PeftType.IA3: IA3Model,
}


def _prepare_model_for_gradient_checkpointing(model: nn.Module) -> None:
    r"""
    Prepares the model for gradient checkpointing if necessary
    """
    if not getattr(model, "is_gradient_checkpointing", True):
        return model

    if not (
        getattr(model, "is_loaded_in_8bit", False)
        or getattr(model, "is_loaded_in_4bit", False)
        or getattr(model, "is_quantized", False)
    ):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        elif hasattr(model, "get_input_embeddings"):

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


def _check_config_compatible(peft_config: PeftConfig) -> None:
    if peft_config.peft_type not in COMPATIBLE_TUNER_TYPES:
        raise ValueError(
            f"The provided `peft_type` '{peft_config.peft_type}' is not compatible with the `PeftMixedModel`."
            f"Compatible types are: {COMPATIBLE_TUNER_TYPES}"
        )


class PeftMixedModel(PushToHubMixin, torch.nn.Module):
    """
    TODO
    """

    def __init__(self, model: nn.Module, peft_config: PeftConfig, adapter_name: str = "default") -> None:
        super().__init__()
        _check_config_compatible(peft_config)
        _prepare_model_for_gradient_checkpointing(model)
        self.modules_to_save = None
        self.base_model = LycorisModel(model, {adapter_name: peft_config}, adapter_name)
        assert self.base_model.active_adapter == adapter_name  # FIXME
        self.set_modules_to_save(peft_config, adapter_name)

        self.config = getattr(model, "config", {"model_type": "custom"})

        # the `pretraining_tp` is set for some models to simulate Tensor Parallelism during inference to avoid
        # numerical differences, https://github.com/pytorch/pytorch/issues/76232 - to avoid any unexpected
        # behavior we disable that in this line.
        if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "pretraining_tp"):
            self.base_model.config.pretraining_tp = 1

    @property
    def peft_config(self) -> dict[str, PeftConfig]:
        return self.base_model.peft_config

    @property
    def active_adapter(self) -> str:
        return self.base_model.active_adapter

    @property
    def active_adapters(self) -> list[str]:
        return self.base_model.active_adapters

    def get_nb_trainable_parameters(self):
        r"""
        Returns the number of trainable parameters and number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params, all_param = self.get_nb_trainable_parameters()

        print(
            f"trainable params: {trainable_params:,d} || "
            f"all params: {all_param:,d} || "
            f"trainable%: {100 * trainable_params / all_param:.4f}"
        )

    # def __getattr__(self, name: str):
    #     """Forward missing attributes to the wrapped module."""
    #     try:
    #         return super().__getattr__(name)  # defer to nn.Module's logic
    #     except AttributeError:
    #         return getattr(self.base_model, name)

    def forward(self, *args: Any, **kwargs: Any):
        """
        Forward pass of the model.
        """
        return self.base_model(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any):
        """
        Generate output.
        """
        return self.base_model.generate(*args, **kwargs)

    @contextmanager
    def disable_adapter(self):
        """
        Disables the adapter module.
        """
        try:
            self.base_model.disable_adapter_layers()
            yield
        finally:
            self.base_model.enable_adapter_layers()

    def add_adapter(self, adapter_name: str, peft_config: PeftConfig):
        _check_config_compatible(peft_config)

        try:
            self.peft_config[adapter_name] = peft_config
            self.base_model.inject_adapter(self, adapter_name)
        except Exception:  # somthing went wrong, roll back
            if adapter_name in self.peft_config:
                del self.peft_config[adapter_name]
            raise

        self.set_modules_to_save(peft_config, adapter_name)

    def set_modules_to_save(self, peft_config: PeftConfig, adapter_name: str) -> None:
        if (modules_to_save := getattr(peft_config, "modules_to_save", None)) is None:
            return

        if self.modules_to_save is None:
            self.modules_to_save = set(modules_to_save)
        else:
            self.modules_to_save.update(modules_to_save)
        _set_trainable(self, adapter_name)

    @classmethod
    def _split_kwargs(cls, kwargs: dict[str, Any]):
        _kwargs_not_in_hf_hub_download_signature = ("use_auth_token",)
        hf_hub_download_kwargs = {}
        other_kwargs = {}

        for key, value in kwargs.items():
            if key in inspect.signature(hf_hub_download).parameters or key in _kwargs_not_in_hf_hub_download_signature:
                hf_hub_download_kwargs[key] = value
            else:
                other_kwargs[key] = value

        return hf_hub_download_kwargs, other_kwargs

    def _set_hf_device_map(self, kwargs):
        hf_device_map = getattr(self.base_model, "hf_device_map", {})
        device_map_contains_cpu_or_disk = set(hf_device_map.values()) & {"cpu", "disk"}
        if not device_map_contains_cpu_or_disk:
            return

        device_map = kwargs.get("device_map", "auto")
        max_memory = kwargs.get("max_memory", None)
        offload_dir = kwargs.get("offload_folder", None)
        offload_index = kwargs.get("offload_index", None)

        dispatch_model_kwargs = {}
        # Safety checker for previous `accelerate` versions
        # `offload_index` was introduced in https://github.com/huggingface/accelerate/pull/873/
        if "offload_index" in inspect.signature(dispatch_model).parameters:
            dispatch_model_kwargs["offload_index"] = offload_index

        no_split_module_classes = self.base_model._no_split_modules

        if device_map != "sequential":
            max_memory = get_balanced_memory(
                self,
                max_memory=max_memory,
                no_split_module_classes=no_split_module_classes,
                low_zero=(device_map == "balanced_low_0"),
            )
        if isinstance(device_map, str):
            device_map = infer_auto_device_map(
                self, max_memory=max_memory, no_split_module_classes=no_split_module_classes
            )
        dispatch_model(
            self,
            device_map=device_map,
            offload_dir=offload_dir,
            **dispatch_model_kwargs,
        )
        hook = AlignDevicesHook(io_same_device=True)
        add_hook_to_module(self.get_base_model(), hook)

    def load_adapter(
        self, model_id: str, adapter_name: str, is_trainable: bool = False, **kwargs: Any
    ) -> tuple[list[str], list[str]]:
        from .mapping import PEFT_TYPE_TO_CONFIG_MAPPING

        hf_hub_download_kwargs, kwargs = self._split_kwargs(kwargs)
        torch_device = infer_device()

        if adapter_name not in self.peft_config:
            # load the config
            cls = PEFT_TYPE_TO_CONFIG_MAPPING[PeftConfig._get_peft_type(model_id, **hf_hub_download_kwargs)]
            peft_config = cls.from_pretrained(model_id, **hf_hub_download_kwargs)
            _check_config_compatible(peft_config)

            peft_config.inference_mode = not is_trainable
            self.add_adapter(adapter_name, peft_config)

        adapters_weights = load_peft_weights(model_id, device=torch_device, **hf_hub_download_kwargs)

        # load the weights into the model
        load_result = set_peft_model_state_dict(self, adapters_weights, adapter_name=adapter_name)
        self._set_hf_device_map(kwargs)

        # Set model in evaluation mode to deactivate Dropout modules by default
        if not is_trainable:
            self.eval()
        return load_result

    def set_adapter(self, adapter_name: Union[str, list[str]]) -> None:
        """
        Sets the active adapter.
        """
        if isinstance(adapter_name, str):
            adapter_name = [adapter_name]

        mismatched = set(adapter_name) - set(self.peft_config.keys())
        if mismatched:
            raise ValueError(
                f"Adapter(s) {sorted(mismatched)} not found, available adapters: {sorted(self.peft_config.keys())}"
            )

        self.base_model.set_adapter(adapter_name)
        _set_adapter(self, adapter_name)

    def create_or_update_model_card(self, output_dir: str):
        """
        Updates or create model card to include information about peft:
        1. Adds `peft` library tag
        2. Adds peft version
        3. Adds base model info
        4. Adds quantization information if it was used
        """

        filename = os.path.join(output_dir, "README.md")

        card = ModelCard.load(filename) if os.path.exists(filename) else ModelCard.from_template(ModelCardData())

        card.data["library_name"] = "peft"
        model_config = self.config
        if hasattr(model_config, "to_dict"):
            model_config = model_config.to_dict()
        if model_config.get("model_type", "custom") != "custom":
            card.data["base_model"] = model_config["_name_or_path"]

        lines = card.text.splitlines()

        quantization_config = None
        if hasattr(self.config, "quantization_config"):
            quantization_config = self.config.quantization_config.to_dict()
        training_config_text = ""
        # Adds quantization information if it was used
        if quantization_config is not None:
            training_config_text += "\nThe following `bitsandbytes` quantization config was used during training:\n"
            training_config_text += "\n".join([f"- {name}: {value}" for name, value in quantization_config.items()])
            training_config_text += "\n"

        training_procedure_heading = "## Training procedure\n"
        if training_procedure_heading in lines:
            lines.insert(lines.index(training_procedure_heading) + 2, training_config_text)
        else:
            lines.append(f"{training_procedure_heading}\n{training_config_text}")

        # Adds peft version
        framework_block_heading = "### Framework versions\n"
        if framework_block_heading in lines:
            lines.insert(lines.index(framework_block_heading) + 2, f"- PEFT {__version__}\n")
        else:
            lines.append(f"{framework_block_heading}\n\n- PEFT {__version__}\n")

        card.text = "\n".join(lines)
        card.save(filename)

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = False,
        selected_adapters: Optional[list[str]] = None,
        **kwargs: Any,
    ):
        raise NotImplementedError("TODO")

    @classmethod
    def from_pretrained(
        cls,
        model: nn.Module,
        model_id: str | os.PathLike,
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: Optional[PeftConfig] = None,
        **kwargs: Any,
    ):
        raise NotImplementedError("TODO")
