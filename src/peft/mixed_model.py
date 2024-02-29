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

import os
from contextlib import contextmanager
from typing import Any, Optional, Union

import torch
from accelerate.hooks import remove_hook_from_submodules
from torch import nn
from transformers.utils import PushToHubMixin

from peft.tuners.mixed import COMPATIBLE_TUNER_TYPES

from .config import PeftConfig
from .peft_model import PeftModel
from .tuners import (
    AdaLoraModel,
    IA3Model,
    LoHaModel,
    LoKrModel,
    LoraModel,
    MixedModel,
    OFTModel,
)
from .utils import PeftType, _set_adapter, _set_trainable


PEFT_TYPE_TO_MODEL_MAPPING = {
    PeftType.LORA: LoraModel,
    PeftType.LOHA: LoHaModel,
    PeftType.LOKR: LoKrModel,
    PeftType.ADALORA: AdaLoraModel,
    PeftType.IA3: IA3Model,
    PeftType.OFT: OFTModel,
}


def _prepare_model_for_gradient_checkpointing(model: nn.Module) -> None:
    r"""
    Prepares the model for gradient checkpointing if necessary
    """
    # Note: same as PeftModel._prepare_model_for_gradient_checkpointing
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
            f"The provided `peft_type` '{peft_config.peft_type.value}' is not compatible with the `PeftMixedModel`. "
            f"Compatible types are: {COMPATIBLE_TUNER_TYPES}"
        )


class PeftMixedModel(PushToHubMixin, torch.nn.Module):
    """
    PeftMixedModel for loading mixing different types of adapters for inference.

    This class does not support loading/saving, and it shouldn't usually be initialized directly. Instead, use
    `get_peft_model` with the argument `mixed=True`.

    <Tip>

    Read the [Mixed adapter types](https://huggingface.co/docs/peft/en/developer_guides/mixed_models) guide to learn
    more about using different adapter types.

    </Tip>

    Example:

    ```py
    >>> from peft import get_peft_model

    >>> base_model = ...  # load the base model, e.g. from transformers
    >>> peft_model = PeftMixedModel.from_pretrained(base_model, path_to_adapter1, "adapter1").eval()
    >>> peft_model.load_adapter(path_to_adapter2, "adapter2")
    >>> peft_model.set_adapter(["adapter1", "adapter2"])  # activate both adapters
    >>> peft_model(data)  # forward pass using both adapters
    ```

    Args:
        model (`torch.nn.Module`):
            The model to be tuned.
        config (`PeftConfig`):
            The config of the model to be tuned. The adapter type must be compatible.
        adapter_name (`str`, `optional`, defaults to `"default"`):
            The name of the first adapter.
    """

    def __init__(self, model: nn.Module, peft_config: PeftConfig, adapter_name: str = "default") -> None:
        super().__init__()
        _check_config_compatible(peft_config)
        _prepare_model_for_gradient_checkpointing(model)
        self.modules_to_save = None
        self.base_model = MixedModel(model, {adapter_name: peft_config}, adapter_name)
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
        # note: same as PeftModel.get_nb_trainable_parameters
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
        # note: same as PeftModel.print_trainable_parameters
        trainable_params, all_param = self.get_nb_trainable_parameters()

        print(
            f"trainable params: {trainable_params:,d} || "
            f"all params: {all_param:,d} || "
            f"trainable%: {100 * trainable_params / all_param:.4f}"
        )

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.base_model, name)

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
        except Exception:  # something went wrong, roll back
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

    def set_adapter(self, adapter_name: Union[str, list[str]]) -> None:
        """
        Sets the active adapter(s) for the model.

        Note that the order in which the adapters are applied during the forward pass may not be the same as the order
        in which they are passed to this function. Instead, the order during the forward pass is determined by the
        order in which the adapters were loaded into the model. The active adapters only determine which adapters are
        active during the forward pass, but not the order in which they are applied.

        Additionally, this function will set the specified adapters to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (`str` or `List[str]`):
                The name of the adapter(s) to be activated.
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

    def delete_adapter(self, adapter_name: Union[str, list[str]]) -> None:
        if isinstance(adapter_name, str):
            adapter_name = [adapter_name]

        mismatched = set(adapter_name) - set(self.peft_config.keys())
        if mismatched:
            raise ValueError(
                f"Adapter(s) {sorted(mismatched)} not found, available adapters: {sorted(self.peft_config.keys())}"
            )

        self.base_model.delete_adapter(adapter_name)

    def merge_and_unload(self, *args: Any, **kwargs: Any):
        r"""
        This method merges the adapter layers into the base model. This is needed if someone wants to use the base
        model as a standalone model.

        Args:
            progressbar (`bool`):
                whether to show a progressbar indicating the unload and merge process
            safe_merge (`bool`):
                whether to activate the safe merging check to check if there is any potential Nan in the adapter
                weights
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        return self.base_model.merge_and_unload(*args, **kwargs)

    def unload(self, *args: Any, **kwargs: Any):
        """
        Gets back the base model by removing all the adapter modules without merging. This gives back the original base
        model.
        """
        return self.base_model.unload(*args, **kwargs)

    @classmethod
    def _split_kwargs(cls, kwargs: dict[str, Any]):
        return PeftModel._split_kwargs(kwargs)

    def load_adapter(self, model_id: str, adapter_name: str, *args: Any, **kwargs: Any):
        output = PeftModel.load_adapter(self, model_id, adapter_name, *args, **kwargs)
        # TODO: not quite clear why this is necessary but tests fail without it
        self.set_adapter(self.active_adapters)
        return output

    def create_or_update_model_card(self, output_dir: str):
        raise NotImplementedError(f"Model card creation is not supported for {self.__class__.__name__} (yet).")

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = False,
        selected_adapters: Optional[list[str]] = None,
        **kwargs: Any,
    ):
        raise NotImplementedError(f"Saving is not supported for {self.__class__.__name__} (yet).")

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
        r"""
        Instantiate a PEFT mixed model from a pretrained model and loaded PEFT weights.

        Note that the passed `model` may be modified inplace.

        Args:
            model (`nn.Module`):
                The model to be adapted.
            model_id (`str` or `os.PathLike`):
                The name of the PEFT configuration to use. Can be either:
                    - A string, the `model id` of a PEFT configuration hosted inside a model repo on the Hugging Face
                      Hub.
                    - A path to a directory containing a PEFT configuration file saved using the `save_pretrained`
                      method (`./my_peft_config_directory/`).
            adapter_name (`str`, *optional*, defaults to `"default"`):
                The name of the adapter to be loaded. This is useful for loading multiple adapters.
            is_trainable (`bool`, *optional*, defaults to `False`):
                Whether the adapter should be trainable or not. If `False`, the adapter will be frozen and use for
                inference
            config ([`~peft.PeftConfig`], *optional*):
                The configuration object to use instead of an automatically loaded configuration. This configuration
                object is mutually exclusive with `model_id` and `kwargs`. This is useful when configuration is already
                loaded before calling `from_pretrained`.
            kwargs: (`optional`):
                Additional keyword arguments passed along to the specific PEFT configuration class.
        """
        # note: adapted from PeftModel.from_pretrained
        from .mapping import PEFT_TYPE_TO_CONFIG_MAPPING

        # load the config
        if config is None:
            config = PEFT_TYPE_TO_CONFIG_MAPPING[
                PeftConfig._get_peft_type(
                    model_id,
                    subfolder=kwargs.get("subfolder", None),
                    revision=kwargs.get("revision", None),
                    cache_dir=kwargs.get("cache_dir", None),
                    use_auth_token=kwargs.get("use_auth_token", None),
                )
            ].from_pretrained(model_id, **kwargs)
        elif isinstance(config, PeftConfig):
            config.inference_mode = not is_trainable
        else:
            raise ValueError(f"The input config must be a PeftConfig, got {config.__class__}")

        # note: this is different from PeftModel.from_pretrained
        if config.peft_type not in PEFT_TYPE_TO_MODEL_MAPPING:
            raise ValueError(f"Adapter of type {config.peft_type} is not supported for mixed models.")

        if (getattr(model, "hf_device_map", None) is not None) and len(
            set(model.hf_device_map.values()).intersection({"cpu", "disk"})
        ) > 0:
            remove_hook_from_submodules(model)

        if config.is_prompt_learning and is_trainable:
            # note: should not be possible to reach, but just in case
            raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
        else:
            config.inference_mode = not is_trainable

        # note: this is different from PeftModel.from_pretrained, we always return a PeftMixedModel
        model = cls(model, config, adapter_name)
        model.load_adapter(model_id, adapter_name, is_trainable=is_trainable, **kwargs)
        return model
