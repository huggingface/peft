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

from .config import PeftType, PromptLearningConfig


def get_peft_model_state_dict(model, state_dict=None, adapter_name="default"):
    """
    Get the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model. When using torch.nn.DistributedDataParallel, DeepSpeed or FSDP,
        the model should be the underlying model/unwrapped model (i.e. model.module).
        state_dict (`dict`, *optional*, defaults to `None`):
            The state dict of the model. If not provided, the state dict of the model
        will be used.
    """
    config = model.peft_config[adapter_name]
    if state_dict is None:
        state_dict = model.state_dict()
    if config.peft_type in (PeftType.LORA, PeftType.ADALORA):
        # to_return = lora_state_dict(model, bias=model.peft_config.bias)
        # adapted from `https://github.com/microsoft/LoRA/blob/main/loralib/utils.py`
        # to be used directly with the state dict which is necessary when using DeepSpeed or FSDP
        bias = config.bias
        if bias == "none":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k}
        elif bias == "all":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k}
        elif bias == "lora_only":
            to_return = {}
            for k in state_dict:
                if "lora_" in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split("lora_")[0] + "bias"
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError
        to_return = {k: v for k, v in to_return.items() if (("lora_" in k and adapter_name in k) or ("bias" in k))}
        if config.peft_type == PeftType.ADALORA:
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                rank_pattern = {k.replace(f".{adapter_name}", ""): v for k, v in rank_pattern.items()}
                config.rank_pattern = rank_pattern
                to_return = model.resize_state_dict_by_rank_pattern(rank_pattern, to_return, adapter_name)

    elif config.peft_type == PeftType.ADAPTION_PROMPT:
        to_return = {k: state_dict[k] for k in state_dict if k.split(".")[-1].startswith("adaption_")}
    elif isinstance(config, PromptLearningConfig):
        to_return = {}
        if config.inference_mode:
            prompt_embeddings = model.prompt_encoder[adapter_name].embedding.weight
        else:
            prompt_embeddings = model.get_prompt_embedding_to_save(adapter_name)
        to_return["prompt_embeddings"] = prompt_embeddings
    else:
        raise NotImplementedError
    if model.modules_to_save is not None:
        for key, value in state_dict.items():
            if any(f"{module_name}.modules_to_save.{adapter_name}" in key for module_name in model.modules_to_save):
                to_return[key.replace("modules_to_save.", "")] = value

    to_return = {k.replace(f".{adapter_name}", ""): v for k, v in to_return.items()}
    return to_return


def set_peft_model_state_dict(model, peft_model_state_dict, adapter_name="default"):
    """
    Set the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model.
        peft_model_state_dict (`dict`): The state dict of the Peft model.
    """
    config = model.peft_config[adapter_name]
    state_dict = {}
    if model.modules_to_save is not None:
        for key, value in peft_model_state_dict.items():
            if any(module_name in key for module_name in model.modules_to_save):
                for module_name in model.modules_to_save:
                    if module_name in key:
                        key = key.replace(module_name, f"{module_name}.modules_to_save.{adapter_name}")
                        break
            state_dict[key] = value
    else:
        state_dict = peft_model_state_dict

    if config.peft_type in (PeftType.LORA, PeftType.ADALORA):
        peft_model_state_dict = {}
        for k, v in state_dict.items():
            if "lora_" in k:
                suffix = k.split("lora_")[1]
                if "." in suffix:
                    suffix_to_replace = ".".join(suffix.split(".")[1:])
                    k = k.replace(suffix_to_replace, f"{adapter_name}.{suffix_to_replace}")
                else:
                    k = f"{k}.{adapter_name}"
                peft_model_state_dict[k] = v
            else:
                peft_model_state_dict[k] = v
        if config.peft_type == PeftType.ADALORA:
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                model.resize_modules_by_rank_pattern(rank_pattern, adapter_name)
    elif isinstance(config, PromptLearningConfig) or config.peft_type == PeftType.ADAPTION_PROMPT:
        peft_model_state_dict = state_dict
    else:
        raise NotImplementedError

    load_result = model.load_state_dict(peft_model_state_dict, strict=False)
    if isinstance(config, PromptLearningConfig):
        model.prompt_encoder[adapter_name].embedding.load_state_dict(
            {"weight": peft_model_state_dict["prompt_embeddings"]}, strict=True
        )
    return load_result
