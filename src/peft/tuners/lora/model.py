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

import math
import operator
import warnings
from contextlib import contextmanager
from dataclasses import replace
from functools import partial, reduce
from typing import Literal, Optional

import torch
from torch import nn
from transformers.modeling_layers import GradientCheckpointingLayer

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    replicate_layers,
)
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    AuxiliaryTrainingWrapper,
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_submodules,
    get_peft_model_state_dict,
    get_quantization_config,
)
from peft.utils.merge_utils import dare_linear, dare_ties, magnitude_prune, task_arithmetic, ties
from peft.utils.other import get_pattern_key

from .aqlm import dispatch_aqlm
from .awq import dispatch_awq
from .config import LoraConfig
from .eetq import dispatch_eetq
from .gptq import dispatch_gptq
from .hqq import dispatch_hqq
from .inc import dispatch_inc
from .layer import Conv2d, LoraLayer, ParamWrapper, dispatch_default
from .torchao import dispatch_torchao
from .tp_layer import dispatch_megatron


def _adapter_names_pre_forward_hook(target, args, kwargs, adapter_names):
    # pre-forward hook to inject the adapter_names argument when using mixed adapter batches inference
    kwargs["adapter_names"] = adapter_names
    return args, kwargs


def _alora_offsets_pre_forward_hook(target, args, kwargs, alora_offsets):
    kwargs["alora_offsets"] = alora_offsets
    return args, kwargs


def _get_encoder(model: nn.Module) -> nn.Module | None:
    """Check if the model has an encoder and if it has, returns it; otherwise returns None"""
    if not hasattr(model, "get_encoder"):
        return None

    encoder = model.get_encoder()
    # https://github.com/huggingface/transformers/pull/42156
    # new logic in transformers v5: all PretrainedModels return a model here, but it is self if there is no encoder
    if encoder is model:
        return None
    return encoder


class LoraModel(BaseTuner):
    """
    Creates Low Rank Adapter (LoRA) model from a pretrained transformers model.

    The method is described in detail in https://huggingface.co/papers/2106.09685.

    Args:
        model ([`torch.nn.Module`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example:

        ```py
        >>> from transformers import AutoModelForSeq2SeqLM
        >>> from peft import LoraModel, LoraConfig

        >>> config = LoraConfig(
        ...     task_type="SEQ_2_SEQ_LM",
        ...     r=8,
        ...     lora_alpha=32,
        ...     target_modules=["q", "v"],
        ...     lora_dropout=0.01,
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> lora_model = LoraModel(model, config, "default")
        ```

        ```py
        >>> import torch
        >>> import transformers
        >>> from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

        >>> rank = ...
        >>> target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
        >>> config = LoraConfig(
        ...     r=4, lora_alpha=16, target_modules=target_modules, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
        ... )
        >>> quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)

        >>> tokenizer = transformers.AutoTokenizer.from_pretrained(
        ...     "kakaobrain/kogpt",
        ...     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
        ...     bos_token="[BOS]",
        ...     eos_token="[EOS]",
        ...     unk_token="[UNK]",
        ...     pad_token="[PAD]",
        ...     mask_token="[MASK]",
        ... )
        >>> model = transformers.GPTJForCausalLM.from_pretrained(
        ...     "kakaobrain/kogpt",
        ...     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
        ...     pad_token_id=tokenizer.eos_token_id,
        ...     use_cache=False,
        ...     device_map={"": rank},
        ...     torch_dtype=torch.float16,
        ...     quantization_config=quantization_config,
        ... )
        >>> model = prepare_model_for_kbit_training(model)
        >>> lora_model = get_peft_model(model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    prefix: str = "lora_"
    tuner_layer_cls = LoraLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

    def _prepare_model(self, peft_config: LoraConfig, model: nn.Module):
        r"""
        A private method to modify the model structure before adapter is applied.

        Args:
            peft_config (`PeftConfig`):
                The prepared adapter config.
            model (`nn.Module`):
                The model that is going to be adapted.
        """
        if peft_config.layer_replication:
            replicate_layers(model, peft_config.layer_replication)

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
            "use_bdlora": lora_config.use_bdlora,
            "target_name": current_key,
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
                use_bdlora=lora_config.use_bdlora,
                inference_mode=lora_config.inference_mode,
                target_name=current_key,
            )
        else:
            if isinstance(target, ParamWrapper) and (parameter_name == target.parameter_name):
                raise ValueError(
                    "Trying to target the same nn.Parameter twice, this should not happen. Please open an issue on the "
                    "PEFT repo: https://github.com/huggingface/peft/issues"
                )
            device_map = self.model.hf_device_map if hasattr(self.model, "hf_device_map") else None
            new_module = self._create_new_module(lora_config, adapter_name, target, device_map=device_map, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def _replace_module(self, parent, child_name, new_module, child):
        # override in LoraModel to handle quantized weights properly

        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        meta = torch.device("meta")
        # dispatch to correct device
        for name, module in new_module.named_modules():
            if (self.prefix in name) or ("ranknum" in name):
                if hasattr(child, "qweight"):
                    weight = child.qweight
                elif hasattr(child, "W_q"):
                    weight = child.W_q
                elif hasattr(child, "weight"):
                    weight = child.weight
                elif getattr(child, "in_proj_weight", None) is not None:  # MHA
                    weight = child.in_proj_weight
                else:
                    weight = next(child.parameters())
                if not any(p.device == meta for p in module.parameters()):
                    module.to(weight.device)

    @staticmethod
    def _create_new_module(lora_config, adapter_name, target, **kwargs):
        # Collect dispatcher functions to decide what backend to use for the replaced LoRA layer. The order matters,
        # because the first match is always used. Therefore, the default layers should be checked last.
        dispatchers = []

        if lora_config._custom_modules:
            # Experimental custom LoRA module support. Allows users to pass a custom mapping for unsupported layer
            # types by impelementing their own LoRA layers.
            def dynamic_dispatch_func(target, adapter_name, lora_config, **kwargs):
                new_module = None

                if isinstance(target, BaseTunerLayer):
                    target_base_layer = target.get_base_layer()
                else:
                    target_base_layer = target

                for key, custom_cls in lora_config._custom_modules.items():
                    if isinstance(target_base_layer, key):
                        new_module = custom_cls(target, adapter_name, **kwargs)
                        break

                return new_module

            dispatchers.append(dynamic_dispatch_func)

        # avoid eager bnb import
        if is_bnb_available():
            from .bnb import dispatch_bnb_8bit

            dispatchers.append(dispatch_bnb_8bit)

        if is_bnb_4bit_available():
            from .bnb import dispatch_bnb_4bit

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
            new_module = dispatcher(target, adapter_name, lora_config=lora_config, **kwargs)
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

    @contextmanager
    def _enable_peft_forward_hooks(self, *args, **kwargs):
        # If adapter_names is passed as an argument, we inject it into the forward arguments.
        adapter_names = kwargs.pop("adapter_names", None)
        alora_offsets = kwargs.pop("alora_offsets", None)

        if adapter_names is None and alora_offsets is None:
            # nothing to do
            yield
            return
        hook_handles = []

        if alora_offsets is not None:
            for n, layer in self.named_modules():
                # gradient checkpointing layer are executed concurrently to the 'normal' forward call
                # (in the backward step the gradient checkpointing layer's forward will be executed again).
                # this means that when the gradient checkpointing layer is called, the _enable_peft_forward_hooks
                # context manager is long gone. to be consistent with the normal forward we need to register the pre
                # hooks for this concurrent forward call as well.
                #
                # Note that this will lead to double application of whatever the callbacks do in normal forward.
                # Make sure that whatever change is done, can be applied more than once without harm (idempotency).
                if isinstance(layer, GradientCheckpointingLayer) and layer.gradient_checkpointing:

                    def forward_pre_hook(name, module, inputs, **kwargs):
                        for submodule in module.modules():
                            if isinstance(submodule, LoraLayer):
                                handle = submodule.register_forward_pre_hook(
                                    partial(_alora_offsets_pre_forward_hook, alora_offsets=kwargs["alora_offsets"]),
                                    with_kwargs=True,
                                )
                                module._peft_gradient_checkpointing_forward_hooks.append(handle)

                    def backward_hook(name, module, *grad_output, **kwargs):
                        while module._peft_gradient_checkpointing_forward_hooks:
                            module._peft_gradient_checkpointing_forward_hooks.pop().remove()

                    if getattr(layer, "_peft_gradient_checkpointing_forward_hooks", []):
                        raise ValueError(
                            "Multiple invocations of PEFT forward hooks before .backward() with enabled gradient "
                            "checkpointing. Disable gradient checkpointing or only call forward once per backward."
                        )
                    layer._peft_gradient_checkpointing_forward_hooks = []
                    handle = layer.register_forward_pre_hook(partial(forward_pre_hook, n, alora_offsets=alora_offsets))
                    layer._peft_gradient_checkpointing_forward_hooks.append(handle)
                    handle = layer.register_full_backward_hook(partial(backward_hook, n))
                    layer._peft_gradient_checkpointing_forward_hooks.append(handle)
                if isinstance(layer, LoraLayer):
                    pre_forward = partial(_alora_offsets_pre_forward_hook, alora_offsets=alora_offsets)
                    handle = layer.register_forward_pre_hook(pre_forward, with_kwargs=True)
                    hook_handles.append(handle)
        num_beams = kwargs.get("num_beams", None)
        uses_beam_search = isinstance(num_beams, int) and (num_beams > 1)
        if uses_beam_search:
            if alora_offsets is not None:
                raise ValueError("Beam search not yet supported for aLoRA.")
        if adapter_names is not None:
            if self.training:
                raise ValueError("Cannot pass `adapter_names` when the model is in training mode.")

            # Check that users only passed actually existing adapters.
            # Note: We cannot do this on the layer level, as each individual layer may not have each adapter. Still, we want
            # to check that there is at least one layer with the given name, or else something like typos can easily slip.
            expected_adapters = set()
            for layer in self.modules():
                if isinstance(layer, LoraLayer):
                    expected_adapters |= layer.lora_A.keys()
                    expected_adapters |= layer.lora_embedding_A.keys()
            unique_adapters = {name for name in adapter_names if name != "__base__"}
            unexpected_adapters = unique_adapters - expected_adapters
            if unexpected_adapters:
                raise ValueError(
                    f"Trying to infer with non-existing adapter(s): {', '.join(sorted(unexpected_adapters))}"
                )

            # deal with beam search
            original_adapter_names = adapter_names[:]
            if uses_beam_search:
                if not isinstance(adapter_names, (list, tuple)):
                    raise TypeError(f"Got adapter names of type {type(adapter_names)}, expected a list of str.")
                # When there is beam search, the inputs are repeated n times, thus we repeat each adapter name n times and
                # then flatten the nested list. For encoder-decoder models, this extended list should not be applied to the
                # encoder part. Further below, the original argument is thus restored for the encoder.
                adapter_names = sum(([n] * kwargs["num_beams"] for n in adapter_names), [])

            for module in self.modules():
                if isinstance(module, LoraLayer) or isinstance(module, AuxiliaryTrainingWrapper):
                    pre_forward = partial(_adapter_names_pre_forward_hook, adapter_names=adapter_names)
                    handle = module.register_forward_pre_hook(pre_forward, with_kwargs=True)
                    hook_handles.append(handle)

            encoder = _get_encoder(self.model)
            if uses_beam_search and (encoder is not None):
                # For encoder-decoder models, even when applying beam search, the encoder part of the model should not use
                # the extended adapter_names. This is because the encoder still uses the original, non-extended samples.
                for module in encoder.modules():
                    if isinstance(module, LoraLayer) or isinstance(module, AuxiliaryTrainingWrapper):
                        # Add another hook to overwrite the kwargs with the original adapter names -- this is easier than
                        # trying to exclude the encoder.
                        pre_forward = partial(_adapter_names_pre_forward_hook, adapter_names=original_adapter_names)
                        handle = module.register_forward_pre_hook(pre_forward, with_kwargs=True)
                        hook_handles.append(handle)

        yield

        for handle in hook_handles:
            handle.remove()

    def _check_merge_allowed(self):
        """Verify that the configuration supports merging.

        Currently gptq quantization and replicated layers do not support merging.
        """
        super()._check_merge_allowed()
        if getattr(self.model, "quantization_method", None) == "gptq":
            raise ValueError("Cannot merge LORA layers when the model is gptq quantized")
        if self.peft_config.get("layer_replication"):
            raise ValueError("Cannot merge LORA layers when base model layers are replicated")

    def _prepare_adapter_config(self, peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] in self.target_module_mapping:
                peft_config.target_modules = set(self.target_module_mapping[model_config["model_type"]])
            elif not peft_config.target_parameters:
                raise ValueError("Please specify `target_modules` or `target_parameters`in `peft_config`")
        return peft_config

    def _check_add_weighted_adapter(
        self, adapters: list[str], combination_type: str, svd_rank: int | None
    ) -> tuple[str, int, str]:
        """
        Helper function to check if the arguments to add_weighted_adapter are valid and compatible with the underlying
        model.
        """
        for adapter in adapters:
            if adapter not in list(self.peft_config.keys()):
                raise ValueError(f"Adapter {adapter} does not exist")

        for adapter in adapters:
            if self.peft_config[adapter].target_parameters:
                raise ValueError(
                    f"add_weighted_adapter does not support targeting nn.Parameter (problematic adapter '{adapter}')"
                )

        # If more than one of the adapters targets the same module with modules_to_save, raise an error, as these
        # modules cannot be merged. First, find the ModulesToSaveWrapper instances in the model, then check if they
        # have modules for the adapters to be merged.
        modules_to_save_wrappers = [module for module in self.modules() if isinstance(module, ModulesToSaveWrapper)]
        problematic_wrappers = [
            wrapper
            for wrapper in modules_to_save_wrappers
            if sum(adapter in wrapper.modules_to_save for adapter in adapters) > 1
        ]
        if problematic_wrappers:
            raise ValueError(
                "Cannot add weighted adapters if they target the same module with modules_to_save, but found "
                f"{len(problematic_wrappers)} such instance(s)."
            )

        # if there is only one adapter, we can only use linear merging
        combination_type = "linear" if len(adapters) == 1 else combination_type

        adapters_ranks: list[int] = [
            # When allocating tensors for the new adapter, we need the maximum possible rank to not overflow
            config.r if not config.rank_pattern else max(config.r, *config.rank_pattern.values())
            for config in (self.peft_config[adapter] for adapter in adapters)
        ]

        if combination_type in ("linear", "ties", "dare_ties", "dare_linear", "magnitude_prune"):
            # all adapters ranks should be same, new rank is just this value
            if len(set(adapters_ranks)) != 1:
                raise ValueError(
                    "All adapters must have the same r value when using combination_type linear, ties, dare_ties or "
                    "dare_linear."
                )
            new_rank = adapters_ranks[0]
        elif combination_type == "cat":
            # adapters ranks may be different, new rank is sum of all ranks
            # be careful, because output adapter rank may be really big if mixing a lot of adapters
            new_rank = sum(adapters_ranks)
        elif combination_type.endswith("svd"):
            # new rank is the max of all ranks of the adapters if not provided
            new_rank = svd_rank or max(adapters_ranks)
        else:
            raise ValueError(f"Invalid combination_type: {combination_type}")

        target_module_types = [type(self.peft_config[adapter].target_modules) for adapter in adapters]
        if not target_module_types:
            raise ValueError(f"Found no adapter matching the names in {adapters}")
        if len(set(target_module_types)) > 1:
            raise ValueError(
                "all adapter configs should follow the same target modules type. "
                "Combining adapters with `target_modules` type being a mix of list/set and string is not supported."
            )

        if target_module_types[0] is str:
            new_target_modules = "|".join(f"({self.peft_config[adapter].target_modules})" for adapter in adapters)
        elif target_module_types[0] is set:
            new_target_modules = reduce(
                operator.or_, (self.peft_config[adapter].target_modules for adapter in adapters)
            )
        else:
            raise TypeError(f"Invalid type {target_module_types[0]} found in target_modules")

        return combination_type, new_rank, new_target_modules

    def add_weighted_adapter(
        self,
        adapters: list[str],
        weights: list[float],
        adapter_name: str,
        combination_type: str = "svd",
        svd_rank: int | None = None,
        svd_clamp: int | None = None,
        svd_full_matrices: bool = True,
        svd_driver: str | None = None,
        density: float | None = None,
        majority_sign_method: Literal["total", "frequency"] = "total",
    ) -> None:
        """
        This method adds a new adapter by merging the given adapters with the given weights.

        When using the `cat` combination_type you should be aware that rank of the resulting adapter will be equal to
        the sum of all adapters ranks. So it's possible that the mixed adapter may become too big and result in OOM
        errors.

        Args:
            adapters (`list`):
                List of adapter names to be merged.
            weights (`list`):
                List of weights for each adapter. Weights can be positive or negative, allowing for both addition and
                subtraction of adapter effects.
            adapter_name (`str`):
                Name of the new adapter.
            combination_type (`str`):
                The merging type can be one of [`svd`, `linear`, `cat`, `ties`, `ties_svd`, `dare_ties`, `dare_linear`,
                `dare_ties_svd`, `dare_linear_svd`, `magnitude_prune`, `magnitude_prune_svd`]. When using the `cat`
                combination_type, the rank of the resulting adapter is equal to the sum of all adapters ranks (the
                mixed adapter may be too big and result in OOM errors).
            svd_rank (`int`, *optional*):
                Rank of output adapter for svd. If None provided, will use max rank of merging adapters.
            svd_clamp (`float`, *optional*):
                A quantile threshold for clamping SVD decomposition output. If None is provided, do not perform
                clamping. Defaults to None.
            svd_full_matrices (`bool`, *optional*):
                Controls whether to compute the full or reduced SVD, and consequently, the shape of the returned
                tensors U and Vh. Defaults to True.
            svd_driver (`str`, *optional*):
                Name of the cuSOLVER method to be used. This keyword argument only works when merging on CUDA. Can be
                one of [None, `gesvd`, `gesvdj`, `gesvda`]. For more info please refer to `torch.linalg.svd`
                documentation. Defaults to None.
            density (`float`, *optional*):
                Value between 0 and 1. 0 means all values are pruned and 1 means no values are pruned. Should be used
                with [`ties`, `ties_svd`, `dare_ties`, `dare_linear`, `dare_ties_svd`, `dare_linear_svd`,
                `magnintude_prune`, `magnitude_prune_svd`]
            majority_sign_method (`str`):
                The method, should be one of ["total", "frequency"], to use to get the magnitude of the sign values.
                Should be used with [`ties`, `ties_svd`, `dare_ties`, `dare_ties_svd`]
        """

        if adapter_name in list(self.peft_config.keys()):
            return

        combination_type, new_rank, new_target_modules = self._check_add_weighted_adapter(
            adapters=adapters,
            combination_type=combination_type,
            svd_rank=svd_rank,
        )

        self.peft_config[adapter_name] = replace(
            self.peft_config[adapters[0]],
            r=new_rank,
            lora_alpha=new_rank,
            target_modules=new_target_modules,
            alpha_pattern={},
            rank_pattern={},
        )
        self.inject_adapter(self.model, adapter_name)

        # Do we really need that?
        _freeze_adapter(self.model, adapter_name)

        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        for key in key_list:
            _, target, _ = _get_submodules(self.model, key)
            if isinstance(target, LoraLayer):
                if adapter_name in target.lora_A:
                    target_lora_A = target.lora_A[adapter_name].weight
                    target_lora_B = target.lora_B[adapter_name].weight
                elif adapter_name in target.lora_embedding_A:
                    target_lora_A = target.lora_embedding_A[adapter_name]
                    target_lora_B = target.lora_embedding_B[adapter_name]
                else:
                    continue

                target_lora_A.data = target_lora_A.data * 0.0
                target_lora_B.data = target_lora_B.data * 0.0
                if combination_type == "cat":
                    loras_A, loras_B = [], []
                    for adapter, weight in zip(adapters, weights):
                        if adapter in target.lora_A:
                            current_adapter_lora_A = target.lora_A[adapter].weight
                            current_adapter_lora_B = target.lora_B[adapter].weight
                        elif adapter in target.lora_embedding_A:
                            current_adapter_lora_A = target.lora_embedding_A[adapter]
                            current_adapter_lora_B = target.lora_embedding_B[adapter]
                        else:
                            continue
                        loras_A.append(current_adapter_lora_A.data * weight * target.scaling[adapter])
                        loras_B.append(current_adapter_lora_B.data)

                    if len(loras_A) == 0:
                        raise ValueError("No matching LoRAs found. Please raise an issue on GitHub.")
                    loras_A = torch.cat(loras_A, dim=0)
                    loras_B = torch.cat(loras_B, dim=1)
                    target_lora_A.data[: loras_A.shape[0], :] = loras_A
                    target_lora_B.data[:, : loras_B.shape[1]] = loras_B
                elif combination_type in [
                    "svd",
                    "ties_svd",
                    "dare_linear_svd",
                    "dare_ties_svd",
                    "magnitude_prune_svd",
                ]:
                    target_lora_A.data, target_lora_B.data = self._svd_generalized_task_arithmetic_weighted_adapter(
                        combination_type,
                        adapters,
                        weights,
                        new_rank,
                        target,
                        target_lora_A,
                        target_lora_B,
                        density,
                        majority_sign_method,
                        svd_clamp,
                        full_matrices=svd_full_matrices,
                        driver=svd_driver,
                    )
                elif combination_type in ["linear", "ties", "dare_linear", "dare_ties", "magnitude_prune"]:
                    target_lora_A.data, target_lora_B.data = self._generalized_task_arithmetic_weighted_adapter(
                        combination_type, adapters, weights, target, density, majority_sign_method
                    )

    def _svd_generalized_task_arithmetic_weighted_adapter(
        self,
        combination_type,
        adapters,
        weights,
        new_rank,
        target,
        target_lora_A,
        target_lora_B,
        density,
        majority_sign_method,
        clamp=None,
        full_matrices=True,
        driver=None,
    ):
        valid_adapters = []
        valid_weights = []
        is_embedding = any(adapter in target.lora_embedding_A for adapter in adapters)
        for adapter, weight in zip(adapters, weights):
            if adapter in target.lora_A or adapter in target.lora_embedding_A:
                valid_adapters.append(adapter)
                valid_weights.append(weight * target.scaling[adapter])

        # if no valid adapter, nothing to do
        if len(valid_adapters) == 0:
            raise ValueError("No matching LoRAs found. Please raise an issue on Github.")
        delta_weight = [target.get_delta_weight(adapter) for adapter in valid_adapters]
        valid_weights = torch.tensor(valid_weights).to(delta_weight[0].device)
        if combination_type == "svd":
            delta_weight = task_arithmetic(delta_weight, valid_weights)
        elif combination_type == "ties_svd":
            delta_weight = ties(delta_weight, valid_weights, density, majority_sign_method)
        elif combination_type == "dare_linear_svd":
            delta_weight = dare_linear(delta_weight, valid_weights, density)
        elif combination_type == "dare_ties_svd":
            delta_weight = dare_ties(delta_weight, valid_weights, density, majority_sign_method)
        elif combination_type == "magnitude_prune_svd":
            delta_weight = magnitude_prune(delta_weight, valid_weights, density)
        else:
            raise ValueError(f"Invalid value passed to combination type: {combination_type}")

        conv2d = isinstance(target, Conv2d)
        if conv2d:
            conv2d_1x1 = target.weight.size()[2:4] == (1, 1)
            if not conv2d_1x1:
                delta_weight = delta_weight.flatten(start_dim=1)
            else:
                delta_weight = delta_weight.squeeze()
        if (hasattr(target, "fan_in_fan_out") and target.fan_in_fan_out) or is_embedding:
            delta_weight = delta_weight.T

        # based on https://github.com/kohya-ss/sd-scripts/blob/main/networks/svd_merge_lora.py#L114-L131
        U, S, Vh = torch.linalg.svd(delta_weight, full_matrices=full_matrices, driver=driver)
        U = U[:, :new_rank]
        S = S[:new_rank]
        U = U @ torch.diag(S)
        Vh = Vh[:new_rank, :]
        if clamp is not None:
            dist = torch.cat([U.flatten(), Vh.flatten()])
            hi_val = torch.quantile(dist, clamp)
            low_val = -hi_val
            U = U.clamp(low_val, hi_val)
            Vh = Vh.clamp(low_val, hi_val)
        if conv2d:
            U = U.reshape(target_lora_B.data.shape)
            Vh = Vh.reshape(target_lora_A.data.shape)
        return Vh, U

    def _generalized_task_arithmetic_weighted_adapter(
        self,
        combination_type,
        adapters,
        weights,
        target,
        density,
        majority_sign_method,
    ):
        # account weights for LoRA A and B layers.
        valid_weights_A = []
        valid_weights_B = []
        lora_A_deltas = []
        lora_B_deltas = []
        for adapter, weight in zip(adapters, weights):
            if adapter in target.lora_A:
                current_adapter_lora_A = target.lora_A[adapter].weight
                current_adapter_lora_B = target.lora_B[adapter].weight
            elif adapter in target.lora_embedding_A:
                current_adapter_lora_A = target.lora_embedding_A[adapter]
                current_adapter_lora_B = target.lora_embedding_B[adapter]
            else:
                continue
            # Support negative weights: take absolute value for sqrt, then apply sign
            weight_with_scaling = weight * target.scaling[adapter]
            sign = 1 if weight_with_scaling >= 0 else -1
            # apply sign only on one side of the weights, otherwise negative signs negate
            valid_weights_A.append(math.sqrt(abs(weight_with_scaling)) * sign)
            valid_weights_B.append(math.sqrt(abs(weight_with_scaling)))
            lora_A_deltas.append(current_adapter_lora_A.data)
            lora_B_deltas.append(current_adapter_lora_B.data)
        valid_weights_A = torch.tensor(valid_weights_A).to(lora_A_deltas[0].device)
        valid_weights_B = torch.tensor(valid_weights_B).to(lora_B_deltas[0].device)
        valid_weights = [valid_weights_A, valid_weights_B]
        lora_deltas = [lora_A_deltas, lora_B_deltas]
        dtype = lora_A_deltas[0].dtype
        for i, task_tensors in enumerate(lora_deltas):
            if combination_type == "linear":
                lora_deltas[i] = task_arithmetic(task_tensors, valid_weights[i])
            elif combination_type == "ties":
                lora_deltas[i] = ties(task_tensors, valid_weights[i], density, majority_sign_method)
            elif combination_type == "dare_linear":
                lora_deltas[i] = dare_linear(task_tensors, valid_weights[i], density)
            elif combination_type == "dare_ties":
                lora_deltas[i] = dare_ties(task_tensors, valid_weights[i], density, majority_sign_method)
            elif combination_type == "magnitude_prune":
                lora_deltas[i] = magnitude_prune(task_tensors, valid_weights[i], density)
            else:
                raise ValueError("Invalid combination type")
        lora_deltas = [delta.to(dtype) for delta in lora_deltas]
        return lora_deltas

    def subtract_mutated_init(self, output_state_dict: dict[str, torch.Tensor], adapter_name: str, kwargs=None):
        """
        This function can calculate the updates of the PiSSA/CorDA/OLoRA by comparing the parameters of the
        PiSSA/CorDA/OLoRA adapter in `output_state_dict` with the initial values of PiSSA/CorDA/OLoRA in
        `adapter_name`, thus converting PiSSA/CorDA/OLoRA to LoRA.
        """
        for name, param in self.model.named_parameters():
            if (
                param.data.dtype != torch.float32
                and param.data.dtype != torch.float16
                and param.data.dtype != torch.bfloat16
            ) and adapter_name.startswith("pissa"):
                warnings.warn(
                    r"Note that Quant(W_res) + AB != Quant(W) + \Delta(AB); "
                    "the converted LoRA, when combined with W or Quant(W), may introduce a certain gap in the fine-tuned model. "
                    "Therefore, we recommend directly using the Quant(W_res) in conjunction with the PiSSA adapter. "
                )
        mutated_init_state_dict = get_peft_model_state_dict(
            self,
            state_dict=kwargs.get("state_dict", None),
            adapter_name=adapter_name,
        )
        tensors_lora = {}
        for name in output_state_dict.keys():
            ## W = W^{res} + A_0 \times B_0,
            ## W + \Delta W = W^{res} + A \times B,
            ## \Delta W = A \times B - A_0 \times B_0 = [A | A_0] \times [B | -B_0]^T = A'B'.
            if "lora_A" in name:
                tensors_lora[name] = torch.cat(
                    [output_state_dict[name], mutated_init_state_dict[".".join(name.split(".")[1:])]], dim=0
                )
            elif "lora_B" in name:
                tensors_lora[name] = torch.cat(
                    [output_state_dict[name], -mutated_init_state_dict[".".join(name.split(".")[1:])]], dim=1
                )

        return tensors_lora

    def _add_modules_to_tie(self, peft_config, tied_weight_keys):
        modules_to_save = set(getattr(peft_config, "modules_to_save", []) or [])
        missing_keys = set(tied_weight_keys) - modules_to_save

        peft_config.modules_to_tie = missing_keys
