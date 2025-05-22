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

import copy
from contextlib import contextmanager
from functools import partial
from typing import Optional, Union

import torch
import torch.nn as nn

from peft.tuners.lora.layer import LoraLayer
from peft.tuners.lora.model import LoraModel
from peft.tuners.tuners_utils import BaseTuner
from peft.utils.constants import DUMMY_TARGET_MODULES
from peft.utils.save_and_load import set_peft_model_state_dict

from .. import lora
from .classifier import XLoraClassifier
from .config import XLoraConfig
from .layer import XLoraConv2dLayer, XLoraEmbeddingLayer, XLoraLinearLayer


def convert_layers_to_xlora(
    base: nn.Module,  # PeftModel
    xloramodel: nn.Module,  # XLoraModel
    config: XLoraConfig,
) -> tuple[int, torch.device | None]:
    """
    Returns the number of swapped layers.
    """
    total_swapped = 0
    all_layers = []

    device = None
    for module in base.modules():
        # Check the exact type because classes like OPTLearnedPositionalEmbedding inherit from nn.Embedding
        if isinstance(module, lora.Linear):
            device = module.lora_A[next(iter(module.lora_A))].weight.device
            new_layer = XLoraLinearLayer(
                model=xloramodel,
                target=module,
                target_forward=module.forward,
                layer_number=total_swapped,
                config=config,
            )
            all_layers.append(new_layer)
            module.forward = new_layer.forward  # type: ignore[method-assign]
            total_swapped += 1
        elif isinstance(module, lora.Embedding):
            device = module.lora_embedding_A[next(iter(module.lora_embedding_A))].device
            new_layer = XLoraEmbeddingLayer(
                model=xloramodel,
                target=module,
                target_forward=module.forward,
                layer_number=total_swapped,
                config=config,
            )
            all_layers.append(new_layer)
            module.forward = new_layer.forward  # type: ignore[method-assign]
            total_swapped += 1
        elif isinstance(module, lora.Conv2d):
            device = module.lora_A[next(iter(module.lora_A))].weight.device
            new_layer = XLoraConv2dLayer(
                model=xloramodel,
                target=module,
                target_forward=module.forward,
                layer_number=total_swapped,
                config=config,
            )
            all_layers.append(new_layer)
            module.forward = new_layer.forward  # type: ignore[method-assign]
            total_swapped += 1

    return (total_swapped, device)


def _load_adapter_into_lora_model(
    lora_model: LoraModel,
    adapter_name: str,
    model_id: str,
    torch_device: Optional[str] = None,
    ephemeral_gpu_offload: bool = False,
    autocast_adapter_dtype: bool = True,
    subfolder: Optional[str] = None,
    **kwargs,
):
    """
    This method emulates the behavior of `PeftModel.from_pretrained`. Updates to `PeftModel.from_pretrained` may need
    to be reflected here.

    All params pertain to the adapter (adapter name, model id, `i` is the adapter number in 0 indexing).
    """
    from peft.peft_model import PeftModel
    from peft.tuners.lora.config import LoraConfig
    from peft.utils.other import infer_device
    from peft.utils.save_and_load import load_peft_weights

    hf_hub_download_kwargs, kwargs = PeftModel._split_kwargs(kwargs)
    if torch_device is None:
        torch_device = infer_device()

    if adapter_name not in lora_model.peft_config:
        # load the config
        lora_peft_config = LoraConfig.from_pretrained(
            model_id,
            ephemeral_gpu_offload=ephemeral_gpu_offload,
            subfolder=subfolder,
            **hf_hub_download_kwargs,
        )
        lora_peft_config.inference_mode = False
        lora_model.peft_config[adapter_name] = lora_peft_config
        lora_model.inject_adapter(lora_model.model, adapter_name)

    adapter_weights = load_peft_weights(model_id, device=torch_device, subfolder=subfolder, **hf_hub_download_kwargs)
    new_adapter_weights = {}
    # Rework the keys to contain the adapter numbers
    for old_key in adapter_weights.keys():
        key: str = old_key
        # Remove all the prefixes until we have model.<...>
        while not (key.startswith("model.") and not key.startswith("model.model.")):
            key = key[key.find(".") + 1 :]
        # We always want model.model
        key = "model." + key
        new_adapter_weights[key] = adapter_weights[old_key]

    # load the weights into the model
    ignore_mismatched_sizes = kwargs.get("ignore_mismatched_sizes", False)
    load_result = set_peft_model_state_dict(
        lora_model,
        new_adapter_weights,
        adapter_name=adapter_name,
        ignore_mismatched_sizes=ignore_mismatched_sizes,
    )
    if len(load_result.unexpected_keys) > 0:
        raise ValueError(
            f"Got unexpected keys! Please raise an issue and tag @EricLBuehler.\n\nunexpected_keys={load_result.unexpected_keys}"
        )

    if hasattr(lora_model, "_cast_adapter_dtype"):
        lora_model._cast_adapter_dtype(adapter_name=adapter_name, autocast_adapter_dtype=autocast_adapter_dtype)


class XLoraModel(BaseTuner):
    """
    Creates an X-LoRA (Mixture of LoRA experts), model from a pretrained transformers model. Currently, this X-LoRA
    implementation only works with models with a transformer architecture.

    The method is described in detail in https://huggingface.co/papers/2402.07148.

    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`XLoraConfig`]): The configuration of the Lora model.
        adapter_name (`str`): The name of the adapter, does not affect the LoRA adapter names.

    Returns:
        `torch.nn.Module`: The X-LoRA model.

    Example:
        ```py
        >>> from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
        >>> from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

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
        >>> int8_config = BitsAndBytesConfig(load_in_8bit=True)
        >>> model = AutoModelForCausalLM.from_pretrained(
        ...     "mistralai/Mistral-7B-Instruct-v0.1",
        ...     trust_remote_code=True,
        ...     attn_implementation="flash_attention_2",
        ...     device_map="cuda:0",
        ...     torch_dtype=torch.bfloat16,
        ...     quantization_config=int8_config,
        ... )
        >>> model = prepare_model_for_kbit_training(4)
        >>> xlora_model = get_peft_model(model, config)
        ```
    """

    def __init__(
        self,
        model: nn.Module,
        config: Union[dict[str, XLoraConfig], XLoraConfig],
        adapter_name: str,
        torch_device: Optional[str] = None,
        ephemeral_gpu_offload: bool = False,
        autocast_adapter_dtype: bool = True,
        **kwargs,
    ) -> None:
        """
        Create a new X-LoRA model

        Args:
            model (`nn.Module`):
                Base model to apply X-LoRA to.
            config: ([`XLoraConfig`]):
                X-LoRA configuration object.
            adapter_name: (`str`):
                Adapter name for the X-LoRA adapter.
            torch_device (`str`, *optional*, defaults to None):
                (For loading the LoRA adapters) The device to load the adapter on. If `None`, the device will be
                inferred.
            ephemeral_gpu_offload (`bool`, *optional*, defaults to `False`):
                (For loading the LoRA adapters) Whether to use ephemeral GPU offloading for partially loaded modules.
                Defaults to `False`.
            autocast_adapter_dtype (`bool`, *optional*, defaults to `True`):
                (For loading the LoRA adapters) Whether to autocast the adapter dtype. Defaults to `True`. Right now,
                this will only cast adapter weights using float16 and bfloat16 to float32, as this is typically
                required for stable training, and only affect select PEFT tuners.
            kwargs: (`optional`):
                (For loading the LoRA adapters) Additional arguments to modify the way the adapter is loaded, e.g. the
                token for Hugging Face Hub.
        """

        nn.Module.__init__(self)

        if isinstance(config, dict):
            conf = config[adapter_name]
        else:
            conf = config

        # Create an empty LoraModel
        base_lora_config = copy.copy(conf)
        base_lora_config.target_modules = DUMMY_TARGET_MODULES
        # Imitate a LoraConfig, fields might need to be updated if LoraConfig is updated
        base_lora_config.layer_replication = None
        base_lora_config.bias = "none"
        lora_model = LoraModel(model, base_lora_config, adapter_name)

        self.xlora_config = conf
        self.lora_model = lora_model

        peft_config = conf

        if hasattr(model.config, "use_cache") and model.config.use_cache:
            raise ValueError("`use_cache` must be False")

        adapters_items = peft_config.adapters.items()
        if hasattr(self.xlora_config, "_subfolders"):
            adapters_items = zip(peft_config.adapters.items(), self.xlora_config._subfolders)
        else:
            adapters_items = peft_config.adapters.items()

        if hasattr(self.xlora_config, "_subfolders"):
            for i, (_adapter_name, model_id), subfolder in enumerate(adapters_items):
                _load_adapter_into_lora_model(
                    lora_model=self.lora_model,
                    adapter_name=str(i),
                    model_id=model_id,
                    torch_device=torch_device,
                    ephemeral_gpu_offload=ephemeral_gpu_offload,
                    autocast_adapter_dtype=autocast_adapter_dtype,
                    subfolder=subfolder,
                    **kwargs,
                )
        else:
            for i, (_adapter_name, model_id) in enumerate(adapters_items):
                _load_adapter_into_lora_model(
                    lora_model=self.lora_model,
                    adapter_name=str(i),
                    model_id=model_id,
                    torch_device=torch_device,
                    ephemeral_gpu_offload=ephemeral_gpu_offload,
                    autocast_adapter_dtype=autocast_adapter_dtype,
                    subfolder=None,
                    **kwargs,
                )

        self.lora_model.set_adapter(list(peft_config.adapters.keys()))

        self._maybe_freeze_all_adapters()

        total_swapped, device = convert_layers_to_xlora(
            model,
            self,
            peft_config,
        )

        n_classes = len(peft_config.adapters)
        xlora_classifier = XLoraClassifier(model, peft_config, n_classes, total_swapped, device)

        # Setup the model internal state
        self.internal_xlora_classifier = xlora_classifier
        self.internal_xlora_scalings = None  # type: ignore
        # Controlled by enable_adapter_layers or disable_adapter_layers
        self.disabled = False

    def _maybe_freeze_all_adapters(self):
        self.eval()
        if not self.xlora_config.use_trainable_adapters:
            for name, param in self.named_parameters():
                if "lora_" in name:
                    param.requires_grad = False

    def generate(self, *args, **kwargs):
        kwargs["use_cache"] = False
        res = self.lora_model.generate(*args, **kwargs)  # type: ignore
        #  This is necessary because we use PeftModel.disable_adapter() which reenables the adapters
        self._maybe_freeze_all_adapters()
        return res

    @contextmanager
    def _enable_peft_forward_hooks(self, *generate_args, **generate_kwargs):
        def scalings_injection_hook(target, args, kwargs, scalings):
            # pre-forward hook to inject the adapter_names argument when using mixed adapter batches inference
            kwargs["scalings"] = scalings
            return args, kwargs

        handles_to_remove = None

        def pre_forward(module, *args, **kwargs):
            nonlocal handles_to_remove

            # =========================== Forward pass with "dummy" scalings ==================

            args_real = args[0]
            kwargs_real = args[1]
            kwargs_real.update(kwargs)

            dummy_scalings = self.internal_xlora_classifier.make_dummy_scalings(*args_real, **kwargs_real)

            hook_handles = []
            for module in self.modules():
                if isinstance(module, LoraLayer):
                    pre_forward = partial(scalings_injection_hook, scalings=dummy_scalings)
                    handle = module.register_forward_pre_hook(pre_forward, with_kwargs=True)
                    hook_handles.append(handle)

            with torch.no_grad():
                self.lora_model.disable_adapter_layers()

                try:
                    scaling_pass_kwargs = kwargs_real.copy()
                    scaling_pass_kwargs["output_hidden_states"] = True
                    scaling_pass_kwargs["return_dict"] = True
                    try:
                        base_output = self.lora_model.model.forward(*args_real, **scaling_pass_kwargs)
                    finally:
                        # Clean everything up
                        for handle in hook_handles:
                            handle.remove()
                finally:
                    self.lora_model.enable_adapter_layers()

            xlora_scalings = self.internal_xlora_classifier(result=base_output, *args_real, **kwargs_real)

            # =========================== Real forward pass with calculated scalings ==================

            hook_handles = []
            for module in self.modules():
                if isinstance(module, LoraLayer):
                    pre_forward = partial(scalings_injection_hook, scalings=xlora_scalings)
                    handle = module.register_forward_pre_hook(pre_forward, with_kwargs=True)
                    hook_handles.append(handle)

            handles_to_remove = hook_handles

        if not self.disabled:
            forward_handle = self.lora_model.model.register_forward_pre_hook(pre_forward, with_kwargs=True)

        # Run the forward pass: first the scaling pass in the hook, and then with the base model
        yield

        if not self.disabled:
            # TODO(EricLBuehler): If we get a forward exception, we may have multiple forward hooks.
            for handle in handles_to_remove:
                handle.remove()
            forward_handle.remove()

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "lora_model":  # see #1892: prevent infinite recursion if class is not initialized
                raise
            return getattr(self.lora_model, name)

    @staticmethod
    def _prepare_adapter_config(peft_config, _model_config):
        # Handle X-LoRA case
        return peft_config

    """
    Does nothing. X-LoRA needs adapters to be frozen.
    """

    def _mark_only_adapters_as_trainable(self) -> None: ...

    """
    This enables the X-LoRA adapter.
    """

    def enable_adapter_layers(self) -> None:
        self.disabled = False

    """
    This diasables the X-LoRA adapter.
    """

    def disable_adapter_layers(self) -> None:
        self.disabled = True

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        # Does nothing because XLoraModel has no target modules
        pass

    @staticmethod
    def _check_target_module_exists(lora_config, key):
        # Does nothing because XLoraModel has no target modules
        return False

    def forward(self, *args, **kwargs):
        return self.lora_model.model(*args, **kwargs)

    def set_topk_lora(self, value: Optional[int]):
        """
        Sparsely select the specified top_k LoRA experts instead of the default dense method. Set to None to use dense.
        This is reflected in the config.
        """
        classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore
        classifier.config.top_k_lora = value

    def set_global_scaling_weight(self, weight: float):
        """
        Set the global LoRA weight, a scalar to multiply the output of each LoRA adapter by. This is by default 1. This
        is reflected in the config.
        """
        classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore
        classifier.config.global_scaling_weight = weight

    def set_scaling_pass_value(self, value: float | None):
        """
        Set the scaling pass value, the value to set the scalings to during the scaling pass. If the value is None, the
        scaling pass value will be 1/n where n is the number of adapters.
        """
        classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore
        classifier._set_override_scaling_pass_value(value)

    def get_global_scaling_weight(self) -> float:
        """
        Get the global LoRA weight.
        """
        classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore
        return classifier.config.global_scaling_weight

    def get_latest_scalings(self) -> Optional[torch.Tensor]:
        """
        Returns the latest scalings prediction, or None if no scalings have been predicted. The tensor is of shape
        (batch_size, seq_len, n_layers, n_classes).
        """
        return self.internal_xlora_scalings

    def get_scalings_log(self) -> list[torch.Tensor]:
        """
        Returns a shallow (only copying the list itself not the tensors) copy of the list containing the scalings log.
        Editing the list does not change the underlying log. The tensors are of shape (batch_size, seq_len, n_layers,
        n_classes). The seq_len dim may vary with input dimension.
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
        classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore
        classifier.scalings_logging = False

    def clear_scalings_log(self):
        """
        Clear the scalings log.
        """
        classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore
        classifier.log_scalings.clear()

    def get_bucketed_scalings_log(self) -> dict[int, tuple[list[int], list[torch.Tensor]]]:
        """
        Returns bucketed scalings, bucketed by seq_len. Each value consists of the positions (the first) and the
        associated tensors. The positions are paired with the associated tensors and give the position in the scaling
        log.
        """
        classifier: XLoraClassifier = self.internal_xlora_classifier  # type: ignore
        return classifier._get_bucketed_scalings()
