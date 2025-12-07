# Copyright 2025-present the HuggingFace Inc. team.
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

import operator
from contextlib import contextmanager
from functools import partial
from typing import Optional

import torch
from torch import nn

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, replicate_layers
from peft.utils import AuxiliaryTrainingWrapper
from peft.utils.other import get_pattern_key

from ...utils.constants import TRANSFORMERS_MODELS_TO_HIRA_TARGET_MODULES_MAPPING
from .config import HiraConfig
from .layer import HiraLayer, dispatch_default


def _adapter_names_pre_forward_hook(target, args, kwargs, adapter_names):
    # pre-forward hook to inject the adapter_names argument when using mixed adapter batches inference
    kwargs["adapter_names"] = adapter_names
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


class HiraModel(BaseTuner):
    """
    Creates HiRA Adapter model from a pretrained transformers model.

    The method is described in detail in https://openreview.net/pdf?id=TwJrTz9cRS.

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
        >>> from peft import HiraModel, HiraConfig

        >>> config = HiraConfig(
        ...     task_type="SEQ_2_SEQ_LM",
        ...     r=32,
        ...     target_modules=["q", "v"],
        ...     hira_dropout=0.01,
        ... )

        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> hira_model = HiraModel(model, config, "default")
        ```

        ```py
        >>> import torch
        >>> import transformers
        >>> from peft import HiraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

        >>> rank = ...
        >>> target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
        >>> config = HiraConfig(r=32, target_modules=target_modules, hira_dropout=0.1, task_type="CAUSAL_LM")
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
        >>> hira_model = get_peft_model(model, config)
        ```

    **Attributes**:
        - **model** ([`~transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    prefix: str = "hira_"
    tuner_layer_cls = HiraLayer
    target_module_mapping = TRANSFORMERS_MODELS_TO_HIRA_TARGET_MODULES_MAPPING

    def _create_and_replace(
        self,
        hira_config,
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

        # Regexp matching - Find key which matches current target_name in patterns provided
        r_key = get_pattern_key(hira_config.r_pattern.keys(), current_key)
        r = hira_config.r_pattern.get(r_key, hira_config.r)

        kwargs = {
            "r": r,
            "hira_dropout": hira_config.hira_dropout,
            "fan_in_fan_out": hira_config.fan_in_fan_out,
            "init_weights": hira_config.init_weights,
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

        if isinstance(target, HiraLayer):
            target.update_layer(
                adapter_name,
                r,
                hira_dropout=hira_config.hira_dropout,
                init_weights=hira_config.init_weights,
            )
        else:
            device_map = self.model.hf_device_map if hasattr(self.model, "hf_device_map") else None
            new_module = self._create_new_module(hira_config, adapter_name, target, device_map=device_map, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def _replace_module(self, parent, child_name, new_module, child):
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
    def _create_new_module(hira_config: HiraConfig, adapter_name, target, **kwargs):
        # Collect dispatcher functions to decide what backend to use for the replaced HiRA layer. The order matters,
        # because the first match is always used. Therefore, the default layers should be checked last.
        dispatchers = []

        # avoid eager bnb import
        if is_bnb_available():
            from .bnb import dispatch_bnb_8bit

            dispatchers.append(dispatch_bnb_8bit)

        if is_bnb_4bit_available():
            from .bnb import dispatch_bnb_4bit

            dispatchers.append(dispatch_bnb_4bit)
        dispatchers.extend(
            [
                dispatch_default,
            ]
        )

        new_module = None
        for dispatcher in dispatchers:
            new_module = dispatcher(target, adapter_name, hira_config=hira_config, **kwargs)
            if new_module is not None:  # first match wins
                break

        if new_module is None:
            # no module could be matched
            raise ValueError(
                f"Target module {target} is not supported. Currently, only the following modules are supported: "
                "`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv1d`, `torch.nn.Conv2d`, `torch.nn.Conv3d`, "
                "`transformers.pytorch_utils.Conv1D`."
            )

        return new_module

    @contextmanager
    def _enable_peft_forward_hooks(self, *args, **kwargs):
        # If adapter_names is passed as an argument, we inject it into the forward arguments.
        adapter_names = kwargs.pop("adapter_names", None)

        if adapter_names is None:
            # nothing to do
            yield
            return

        hook_handles = []
        num_beams = kwargs.get("num_beams", None)
        uses_beam_search = isinstance(num_beams, int) and (num_beams > 1)

        if self.training:
            raise ValueError("Cannot pass `adapter_names` when the model is in training mode.")

        # Check that users only passed actually existing adapters.
        # Note: We cannot do this on the layer level, as each individual layer may not have each adapter. Still, we want
        # to check that there is at least one layer with the given name, or else something like typos can easily slip.
        expected_adapters = set()
        for layer in self.modules():
            if isinstance(layer, HiraLayer):
                expected_adapters |= layer.hira_A.keys()
                expected_adapters |= layer.hira_embedding_A.keys()
        unique_adapters = {name for name in adapter_names if name != "__base__"}
        unexpected_adapters = unique_adapters - expected_adapters
        if unexpected_adapters:
            raise ValueError(f"Trying to infer with non-existing adapter(s): {', '.join(sorted(unexpected_adapters))}")

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
            if isinstance(module, HiraLayer) or isinstance(module, AuxiliaryTrainingWrapper):
                pre_forward = partial(_adapter_names_pre_forward_hook, adapter_names=adapter_names)
                handle = module.register_forward_pre_hook(pre_forward, with_kwargs=True)
                hook_handles.append(handle)

        encoder = _get_encoder(self.model)
        if uses_beam_search and (encoder is not None):
            # For encoder-decoder models, even when applying beam search, the encoder part of the model should not use
            # the extended adapter_names. This is because the encoder still uses the original, non-extended samples.
            for module in encoder.modules():
                if isinstance(module, HiraLayer) or isinstance(module, AuxiliaryTrainingWrapper):
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
            raise ValueError("Cannot merge HiRA layers when the model is gptq quantized")

    def _prepare_adapter_config(self, peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] in self.target_module_mapping:
                peft_config.target_modules = set(self.target_module_mapping[model_config["model_type"]])
            else:
                raise ValueError("Please specify `target_modules` in `peft_config`")
        return peft_config
