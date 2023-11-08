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
import operator
import re
import warnings
from dataclasses import asdict, replace
from enum import Enum
from functools import reduce
from itertools import chain

import torch
from torch import nn

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.lora import LoraModel
from peft.utils import (
    TRANSFORMERS_MODELS_TO_LOFTQ_TARGET_MODULES_MAPPING,
    get_auto_gptq_quant_linear,
)

from .config import LoftQConfig
from peft.tuners.lora import Linear, QuantLinear
from .utils import loftq_init

if is_bnb_available():
    import bitsandbytes as bnb
    from peft.tuners.lora import Linear8bitLt

if is_bnb_4bit_available():
    from peft.tuners.lora import Linear4bit


class LoftQModel(LoraModel):
    """
    Creates LoftQ model from a pretrained transformers model.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.
        adapter_name (`str`): The name of the adapter, defaults to `"default"`.

    Returns:
        `torch.nn.Module`: The Lora model with LoftQ initialization.

    Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from peft import LoftQConfig, PeftModel, get_peft_model

        >>> target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        >>> config = LoftQConfig(
        ...     r=4, lora_alpha=16, target_modules=target_modules, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM",
        ...     bits=4, num_iters=1, fake_quant=False,
        ... )

        >>> model = AutoModelForCausalLM.from_pretrained(
        ...     "LoftQ/Llama-2-7b-hf-bit4-rank64",
        ...     use_cache=False,
        ...     torch_dtype=torch.float16,
        ...     load_in_4bit=True,
        ... )
        >>> loftq_model = get_peft_model(model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    def __init__(self, model, config, adapter_name) -> None:
        super().__init__(model, config, adapter_name)

    @staticmethod
    def _replace_module(parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer
        elif hasattr(child, "quant_linear_module"):
            child = child.quant_linear_module

        # TODO: layers with base_layer don't need the weight to be copied, as they have a reference already
        if not hasattr(new_module, "base_layer") and hasattr(child, "bias"):
            # for fake quantization
            new_module.bias = child.bias

        # handle quantized models downloaded from HuggingFace model hub
        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state
            new_module.to(child.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(child.weight.device)
            if "ranknum" in name:
                module.to(child.weight.device)

    @staticmethod
    def _create_new_module(loftq_config, adapter_name, target, **kwargs):
        gptq_quantization_config = kwargs.get("gptq_quantization_config", None)
        AutoGPTQQuantLinear = get_auto_gptq_quant_linear(gptq_quantization_config)

        loaded_in_8bit = kwargs.pop("loaded_in_8bit", False)
        loaded_in_4bit = kwargs.pop("loaded_in_4bit", False)
        bias = kwargs.pop("bias", False)

        if loftq_config.bits == 8 or loaded_in_8bit:
            if isinstance(target, bnb.nn.Linear8bitLt):
                eightbit_kwargs = kwargs.copy()
                eightbit_kwargs.update(
                    {
                        "has_fp16_weights": target.state.has_fp16_weights,
                        "memory_efficient_backward": target.state.memory_efficient_backward,
                        "threshold": target.state.threshold,
                        "index": target.index,
                    }
                )
                new_module = Linear8bitLt(adapter_name, target, **eightbit_kwargs)
            elif isinstance(target, nn.Linear):
                # Apply LoftQ algorithm to original 32/16-bit models for the first time
                if loftq_config.loftq_init:
                    qweight, lora_A, lora_B = loftq_init(target.weight,
                                                         loftq_config.bits,
                                                         loftq_config.r,
                                                         loftq_config.num_iter)
                    if loftq_config.fake_quant:
                        new_module = Linear(adapter_name, target.in_features, target.out_features, bias=bias, **kwargs)
                        new_module.weight = nn.Parameter(qweight, requires_grad=False)
                    else:
                        base_layer = bnb.nn.Linear8bitLt(target.in_features, target.out_features,
                                                         bias=bias, threshold=0.6)
                        base_layer.weight = bnb.nn.Int8Params(qweight.to("cpu"), requires_grad=False).to(
                            target.weight.device
                        )
                        new_module = Linear8bitLt(adapter_name, base_layer, **kwargs)

                    new_module.lora_A[adapter_name].weight.data = lora_A
                    new_module.lora_B[adapter_name].weight.data = lora_B
                else:
                    # in case users didn't specify load_in_4bit=True and the module is not bnb.nn.Linear4bit
                    if not loftq_config.fake_quant:
                        warnings.warn("You didn't pass `load_in_8bit=True`. "
                                      "Convert it to 8-bit model automatically because you passed `fake_quant=False`")
                        base_layer = bnb.nn.Linear8bitLt(target.in_features, target.out_features,
                                                         bias=bias, threshold=0.6)
                        base_layer.weight = bnb.nn.Int8Params(target.weight.data.to("cpu"), requires_grad=False).to(
                            target.weight.device
                        )
                        new_module = Linear8bitLt(adapter_name, base_layer, **kwargs)
                    # intentionally use 32-bit model, the fake quantization
                    else:
                        new_module = Linear(adapter_name, target.in_features, target.out_features, bias=bias, **kwargs)
                        new_module.weight = nn.Parameter(target.weight, requires_grad=False)
            else:
                raise ValueError(f"Unsupported Linear type: {type(target)}")

        elif loftq_config.bits in [2, 4] or loaded_in_4bit:
            # Download 4-bit models from HuggingFace model hub
            if isinstance(target, bnb.nn.Linear4bit):
                assert is_bnb_4bit_available()
                fourbit_kwargs = kwargs.copy()
                fourbit_kwargs.update(
                    {
                        "compute_dtype": target.compute_dtype,
                        "compress_statistics": target.weight.compress_statistics,
                        "quant_type": target.weight.quant_type,
                    }
                )
                new_module = Linear4bit(adapter_name, target, **fourbit_kwargs)

            elif isinstance(target, nn.Linear):
                # Apply LoftQ algorithm to original 32/16-bit models for the first time
                if loftq_config.loftq_init:
                    qweight, lora_A, lora_B = loftq_init(target.weight,
                                                         loftq_config.bits,
                                                         loftq_config.r,
                                                         loftq_config.num_iters)
                    if loftq_config.fake_quant:
                        new_module = Linear(adapter_name, target.in_features, target.out_features, bias=bias, **kwargs)
                        new_module.weight = nn.Parameter(qweight, requires_grad=False)
                    else:
                        assert is_bnb_4bit_available()
                        base_layer = bnb.nn.Linear4bit(target.in_features, target.out_features, bias=bias,
                                                       compress_statistics=False, quant_type='nf4', device='cpu')
                        base_layer.weight = bnb.nn.Params4bit(qweight.to("cpu"), requires_grad=False).to(
                            target.weight.device
                        )
                        new_module = Linear4bit(adapter_name, base_layer, **kwargs)
                    new_module.lora_A[adapter_name].weight.data = lora_A
                    new_module.lora_B[adapter_name].weight.data = lora_B
                # Download from HuggingFace model hub
                else:
                    # in case users didn't specify load_in_4bit=True and the module is not bnb.nn.Linear4bit
                    if not loftq_config.fake_quant:
                        assert is_bnb_4bit_available()
                        warnings.warn("You didn't pass `load_in_4bit=True`. "
                                      "Convert it to 4-bit model automatically because you passed `fake_quant=False`")
                        base_layer = bnb.nn.Linear4bit(target.in_features, target.out_features, bias=bias,
                                                       compress_statistics=False, quant_type='nf4', device='cpu')
                        base_layer.weight = bnb.nn.Params4bit(target.weight.data.to("cpu"), requires_grad=False).to(
                            target.weight.device
                        )
                        new_module = Linear4bit(adapter_name, base_layer, **kwargs)

                    # intentionally use 32-bit model, the fake quantization
                    else:
                        new_module = Linear(adapter_name, target.in_features, target.out_features, bias=bias, **kwargs)
                        new_module.weight = nn.Parameter(target.weight, requires_grad=False)
            else:
                raise ValueError(f"Unsupported Linear type: {type(target)}")

        elif AutoGPTQQuantLinear is not None and isinstance(target, AutoGPTQQuantLinear):
            new_module = QuantLinear(adapter_name, target, **kwargs)
            target.weight = target.qweight
        elif isinstance(target, torch.nn.Embedding):
            raise NotImplementedError("Embedding layers are not supported yet")
        elif isinstance(target, torch.nn.Conv2d):
            raise NotImplementedError("Conv2d layers are not supported yet")
        else:
            raise ValueError(f"Unsupported layer type: {type(target)}")

        return new_module

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LOFTQ_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_LOFTQ_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config
