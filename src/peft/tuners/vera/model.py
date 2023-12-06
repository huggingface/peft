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
import math
import warnings
from dataclasses import asdict
from enum import Enum
from functools import partial
from typing import Union

import torch
from torch import nn
from torch.nn.init import _calculate_correct_fan
from tqdm import tqdm
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists
from peft.utils import (
    TRANSFORMERS_MODELS_TO_VERA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    _get_submodules,
)

from .config import VeraConfig
from .layer import Embedding, Linear, VeraLayer


def _vera_forward_hook(module, args, kwargs, vera_A, vera_B):
    kwargs["vera_A"] = vera_A
    kwargs["vera_B"] = vera_B
    return args, kwargs


def _kaiming_init(
    tensor_or_shape: Union[torch.Tensor, tuple],
    generator: torch.Generator,
):
    if isinstance(tensor_or_shape, tuple):
        tensor = torch.empty(tensor_or_shape)
    else:
        tensor = tensor_or_shape
    fan = _calculate_correct_fan(tensor, "fan_in")
    gain = math.sqrt(2)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std

    with torch.no_grad():
        return tensor.uniform_(-bound, bound, generator=generator)


class VeraModel(BaseTuner):
    """
    Creates Vector-based Random Matrix Adaptation (Vera) model from a pretrained transformers model.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`VeraConfig`]): The configuration of the Vera model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The Vera model.

    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM
        >>> from peft import VeraModel, VeraConfig

        >>> config = VeraConfig(
        ...     task_type="SEQ_2_SEQ_LM",
        ...     r=8,
        ...     target_modules=["q", "v"],
        ...     vera_dropout=0.01,
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> vera_model = VeraModel(model, config, "default")
        ```

        ```py
        >>> import transformers
        >>> from peft import VeraConfig, PeftModel, get_peft_model

        >>> target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
        >>> config = VeraConfig(
        ...     r=4, target_modules=target_modules, vera_dropout=0.1, bias="none", task_type="CAUSAL_LM"
        ... )

        >>> model = transformers.GPTJForCausalLM.from_pretrained(
        ...     "kakaobrain/kogpt",
        ...     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
        ...     pad_token_id=tokenizer.eos_token_id,
        ...     use_cache=False,
        ...     device_map={"": rank},
        ...     torch_dtype=torch.float16,
        ...     load_in_8bit=True,
        ... )
        >>> model = prepare_model_for_int8_training(model)
        >>> vera_model = get_peft_model(model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`VeraConfig`]): The configuration of the Vera model.
    """

    def _find_first_dim(self) -> int:
        """
        Finds the first linear and embedding that has been wrapped with Vera, and extract the input and output
        dimension.

        This will be used for determining the size of the shared vera_A and vera_B matrices.

        This will throw an error if there are multiple layers of the same type with different shapes.
        """
        first_linear, first_embedding = None, None
        for module in self.model.modules():
            if isinstance(module, Linear):
                module_shape = tuple(module.weight.shape)
                if module.fan_in_fan_out:
                    module_shape = module_shape[::-1]

                if first_linear is not None and module_shape != first_linear:
                    raise ValueError(
                        f"Multiple target linear layers with different dimensions were specified! Vera only supports a single dimension size. Got '{module_shape}' expected '{first_linear}"
                    )
                first_linear = module_shape

            elif isinstance(module, Embedding):
                if first_embedding is not None and tuple(module.weight.shape) != first_embedding:
                    raise ValueError(
                        "Multiple target embedding layers with different dimensions or vocabulary sizes were specified! Vera only supports a single size."
                    )
                first_embedding = tuple(module.weight.shape)

            if first_linear is not None and first_embedding is not None:
                return first_linear, first_embedding

        return first_linear, first_embedding

    def __init__(self, model, config, adapter_name) -> None:
        super().__init__(model, config, adapter_name)
        config = config[adapter_name]

        first_linear, first_embedding = self._find_first_dim()

        if first_embedding is not None:
            first_embedding_vocab_size, first_embedding_dim = first_embedding
        if first_linear is not None:
            first_linear_out_dim, first_linear_in_dim = first_linear

        # deterministic init of vera_A and vera_B if we know the key
        generator = torch.Generator(device="cpu").manual_seed(config.projection_prng_key)
        if first_linear is not None:
            vera_A = _kaiming_init((config.r, first_linear_in_dim), generator=generator)
            vera_B = _kaiming_init((first_linear_out_dim, config.r), generator=generator)

            # use of persistent to exclude vera_A and vera_B from the state dict
            # if we choose not to save them.
            self.register_buffer("vera_A", vera_A, persistent=config.save_projection)
            self.register_buffer("vera_B", vera_B, persistent=config.save_projection)
        else:
            self.vera_A = None
            self.vera_B = None

        # as above, but for embedding layer if at least one has been wrapped with Vera.
        if first_embedding is not None:
            vera_embedding_A = torch.randn((config.r, first_embedding_vocab_size), generator=generator)
            vera_embedding_B = torch.randn((first_embedding_dim, config.r), generator=generator)
            self.register_buffer("vera_embedding_A", vera_embedding_A, persistent=config.save_projection)
            self.register_buffer("vera_embedding_B", vera_embedding_B, persistent=config.save_projection)
        else:
            self.vera_embedding_A = None
            self.vera_embedding_B = None

        if not config.save_projection:
            warnings.warn(
                "Specified to not save vera_A and vera_B within the state dictionary, instead they will be restored using the PRNG key store in `config.projection_prng_key`. Consider setting `config.save_projection` to `True` to guarantee restoring the checkpoint correctly on all system configurations."
            )

        self.to(self.dtype)

    def _check_new_adapter_config(self, config: VeraConfig) -> None:
        """
        A helper method to check the config when a new adapter is being added.

        Raise a ValueError if there is something wrong with the config or if it conflicts with existing adapters.

        """
        # the below todo is copied from LoRA
        # TODO: there should be a check if any of the existing adapters actually has bias != "none", or else the check
        # does not fully correspond to the error message.
        if (len(self.peft_config) > 1) and (config.bias != "none"):
            raise ValueError(
                f"{self.__class__.__name__} supports only 1 adapter with bias. When using multiple adapters, "
                "set bias to 'none' for all adapters."
            )

        # TODO: allow this in conjunction with save_projection=True
        if config.projection_prng_key is None:
            raise ValueError("Vera PRNG initialisation key cannot be `None`. Set `VeraConfig.projection_prng_key`.")

    @staticmethod
    def _check_target_module_exists(vera_config, key):
        return check_target_module_exists(vera_config, key)

    def _create_and_replace(
        self,
        vera_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        r = vera_config.r
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": r,
            "vera_dropout": vera_config.vera_dropout,
            "fan_in_fan_out": vera_config.fan_in_fan_out,
            "init_vera_weights": vera_config.init_vera_weights,
        }

        # TODO: add back once we have quant support
        # kwargs["loaded_in_8bit"] = False
        # kwargs["loaded_in_4bit"] = False
        kwargs["bias"] = bias

        # TODO: add in quant?
        # quantization_config = get_quantization_config(self.model, method="gptq")
        # if quantization_config is not None:
        # kwargs["gptq_quantization_config"] = quantization_config

        # the below todo is copied from LoRA
        # TODO: better deal with that
        if isinstance(target, VeraLayer) and isinstance(target, (torch.nn.Embedding, torch.nn.sparse.Embedding)):
            target.update_layer_embedding(
                adapter_name,
                r,
                vera_config.vera_dropout,
                vera_config.init_vera_weights,
                d_initial=vera_config.d_initial,
            )

        elif isinstance(target, VeraLayer):
            target.update_layer(
                adapter_name,
                r,
                vera_config.vera_dropout,
                vera_config.init_vera_weights,
                d_initial=vera_config.d_initial,
            )
        else:
            new_module = self._create_new_module(vera_config, adapter_name, target, **kwargs)
            if adapter_name != self.active_adapter:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _replace_module(parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        # the below todo is copied from LoRA
        # TODO: layers with base_layer don't need the weight to be copied, as they have a reference already
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

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "vera_" in name:
                module.to(child.weight.device)
            if "ranknum" in name:
                module.to(child.weight.device)

    def _mark_only_adapters_as_trainable(self) -> None:
        for n, p in self.model.named_parameters():
            if "vera_lambda_" not in n:
                p.requires_grad = False

        for active_adapter in self.active_adapters:
            bias = self.peft_config[active_adapter].bias
            if bias == "none":
                continue

            if bias == "all":
                for n, p in self.model.named_parameters():
                    if "bias" in n:
                        p.requires_grad = True
            elif bias == "vera_only":
                for m in self.model.modules():
                    if isinstance(m, VeraLayer) and hasattr(m, "bias") and m.bias is not None:
                        m.bias.requires_grad = True
            else:
                raise NotImplementedError(f"Requested bias: {bias}, is not implemented.")

    @staticmethod
    def _create_new_module(vera_config, adapter_name, target, **kwargs):
        bias = kwargs.pop("bias", False)

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Embedding):
            embedding_kwargs = kwargs.copy()
            embedding_kwargs.pop("fan_in_fan_out", None)
            new_module = Embedding(
                target,
                adapter_name,
                d_initial=vera_config.d_initial,
                **embedding_kwargs,
            )
        else:
            if isinstance(target_base_layer, torch.nn.Linear):
                if kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                    kwargs["fan_in_fan_out"] = vera_config.fan_in_fan_out = False
            elif isinstance(target_base_layer, Conv1D):
                kwargs["is_target_conv_1d_layer"] = True
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                        "Setting fan_in_fan_out to True."
                    )
                    kwargs["fan_in_fan_out"] = vera_config.fan_in_fan_out = True
            else:
                raise ValueError(
                    f"Target module {target} is not supported. Currently, only the following modules are supported: "
                    "`torch.nn.Linear`, `torch.nn.Embedding`, `transformers.pytorch_utils.Conv1D`."
                )
            new_module = Linear(
                target,
                adapter_name,
                bias=bias,
                d_initial=vera_config.d_initial,
                **kwargs,
            )

        return new_module

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        for active_adapter in self.active_adapters:
            val = self.peft_config[active_adapter].bias
            if val != "none":
                msg = (
                    f"Careful, disabling adapter layers with bias configured to be '{val}' does not produce the same "
                    "output as the the base model would without adaption."
                )
                warnings.warn(msg)
        self._set_adapter_layers(enabled=False)

    def set_adapter(self, adapter_name):
        for module in self.model.modules():
            if isinstance(module, VeraLayer):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    if isinstance(module, Linear):
                        module.unmerge(self.vera_A, self.vera_B)
                    elif isinstance(module, Embedding):
                        module.unmerge(self.vera_embedding_A, self.vera_embedding_B)
                module.set_adapter(adapter_name)

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_VERA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_VERA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def _unload_and_optionally_merge(self, merge=True, progressbar: bool = False, safe_merge: bool = False):
        key_list = [key for key, _ in self.model.named_modules() if "vera" not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue

            if hasattr(target, "base_layer"):
                if merge:
                    target.merge(self.vera_A, self.vera_B, safe_merge=safe_merge)

                self._replace_module(parent, target_name, target.get_base_layer(), target)
            elif isinstance(target, ModulesToSaveWrapper):
                # save any additional trainable modules part of `modules_to_save`
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

    def delete_adapter(self, adapter_name: str):
        """
        Deletes an existing adapter.

        Args:
            adapter_name (str): Name of the adapter to be deleted.
        """
        if adapter_name not in list(self.peft_config.keys()):
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]

        key_list = [key for key, _ in self.model.named_modules() if "vera" not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, VeraLayer):
                for attr in [
                    "r",
                    "vera_A",
                    "vera_B",
                    "vera_lambda_b",
                    "vera_lambda_d",
                    "vera_embedding_A",
                    "vera_embedding_B",
                    "vera_dropout",
                ]:
                    if adapter_name in getattr(target, attr):
                        getattr(target, attr).pop(adapter_name)
                if adapter_name in target.active_adapters:
                    resetting_active_adapter = (
                        list(self.peft_config.keys())[0] if len(self.peft_config) > 0 else "default"
                    )
                    warnings.warn(
                        f"Adapter {adapter_name} was active which is now deleted. Setting active adapter to {resetting_active_adapter}. "
                    )
                    target.set_adapter(resetting_active_adapter)

    def merge_and_unload(self, progressbar: bool = False, safe_merge: bool = False):
        r"""
        This method merges the Vera layers into the base model. This is needed if someone wants to use the base model
        as a standalone model.

        Args:
            progressbar (`bool`):
                whether to show a progressbar indicating the unload and merge process
            safe_merge (`bool`):
                whether to activate the safe merging check to check if there is any potential Nan in the adapter
                weights

        Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import PeftModel

        >>> base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-40b")
        >>> peft_model_id = "smangrul/falcon-40B-int4-peft-lora-sfttrainer-sample"
        >>> model = PeftModel.from_pretrained(base_model, peft_model_id)
        >>> merged_model = model.merge_and_unload()
        ```
        """
        return self._unload_and_optionally_merge(progressbar=progressbar, safe_merge=safe_merge)

    def unload(self):
        """
        Gets back the base model by removing all the Vera modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)

    def _add_forward_hooks(self):
        """
        Adds pre-forward hooks to all Vera modules in order to inject the shared vera_A and vera_B without adding them
        to each module's state dictionary.
        """
        hook_handles = []
        for module in self.modules():
            if isinstance(module, Linear):
                pre_forward = partial(_vera_forward_hook, vera_A=self.vera_A, vera_B=self.vera_B)
                handle = module.register_forward_pre_hook(pre_forward, with_kwargs=True)
                hook_handles.append(handle)

            elif isinstance(module, Embedding):
                pre_forward = partial(_vera_forward_hook, vera_A=self.vera_embedding_A, vera_B=self.vera_embedding_B)
                handle = module.register_forward_pre_hook(pre_forward, with_kwargs=True)
                hook_handles.append(handle)

        return hook_handles

    def forward(self, *args, **kwargs):
        hook_handles = self._add_forward_hooks()
        try:
            outputs = super().forward(*args, **kwargs)
        finally:
            # regardless of success or failure of forward pass, we should remove
            # handles to restore the original model.
            for handle in hook_handles:
                handle.remove()

        return outputs
