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

import torch
from transformers import AutoModelForCausalLM

from peft import LoraConfig, PromptTuningConfig, TaskType, get_peft_model
from peft.utils import infer_device

from .testing_utils import hub_online_once


def test_get_base_model_state_dict_matches():
    # Test to check whether all the keys in the base model match to the keys
    # of the lora wrapped model when calling get_base_model_state_dict method
    model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
    torch_device = infer_device()
    with hub_online_once(model_id):
        base_model = AutoModelForCausalLM.from_pretrained(model_id).to(torch_device)
    base_model_keys = set(base_model.state_dict().keys())
    lora_config = LoraConfig(r=4, lora_alpha=2, target_modules="all-linear", lora_dropout=0.1)
    peft_model = get_peft_model(base_model, lora_config)
    new_state_dict = set(peft_model.get_base_model_state_dict().keys())
    assert base_model_keys == new_state_dict


def test_get_base_model_state_dict_values_match():
    # Test that the actual tensor values match the original base model weights
    model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
    torch_device = infer_device()
    with hub_online_once(model_id):
        base_model = AutoModelForCausalLM.from_pretrained(model_id).to(torch_device)
    original_state_dict = {k: v.clone() for k, v in base_model.state_dict().items()}

    lora_config = LoraConfig(r=4, lora_alpha=2, target_modules="all-linear")
    peft_model = get_peft_model(base_model, lora_config)

    extracted_state_dict = peft_model.get_base_model_state_dict()

    for key in original_state_dict:
        assert key in extracted_state_dict
        assert torch.allclose(original_state_dict[key], extracted_state_dict[key])


def test_get_base_model_state_dict_with_modules_to_save():
    # Test that modules_to_save are handled correctly (filters .modules_to_save.
    # keys and transforms .original_module. keys back to original format)
    model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
    torch_device = infer_device()
    with hub_online_once(model_id):
        base_model = AutoModelForCausalLM.from_pretrained(model_id).to(torch_device)

    base_model_keys = set(base_model.state_dict().keys())

    lora_config = LoraConfig(
        r=4,
        lora_alpha=2,
        target_modules="all-linear",
        modules_to_save=["lm_head"],
    )
    peft_model = get_peft_model(base_model, lora_config)

    extracted_keys = set(peft_model.get_base_model_state_dict().keys())
    assert base_model_keys == extracted_keys


def test_get_base_model_state_dict_with_multiple_adapters():
    # Test that base model state dict is correctly extracted when multiple
    # adapters are present, ensuring all adapter params are filtered out
    model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
    torch_device = infer_device()
    with hub_online_once(model_id):
        base_model = AutoModelForCausalLM.from_pretrained(model_id).to(torch_device)

    base_model_keys = set(base_model.state_dict().keys())

    lora_config_1 = LoraConfig(r=4, lora_alpha=2, target_modules=["q_proj", "v_proj"])
    peft_model = get_peft_model(base_model, lora_config_1, adapter_name="adapter1")

    lora_config_2 = LoraConfig(r=8, lora_alpha=4, target_modules=["k_proj", "out_proj"])
    peft_model.add_adapter("adapter2", lora_config_2)

    extracted_keys = set(peft_model.get_base_model_state_dict().keys())
    assert base_model_keys == extracted_keys


def test_get_base_model_state_dict_prompt_learning():
    # Test with prompt learning method
    model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
    torch_device = infer_device()
    with hub_online_once(model_id):
        base_model = AutoModelForCausalLM.from_pretrained(model_id).to(torch_device)

    base_model_keys = set(base_model.state_dict().keys())

    prompt_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=10,
    )
    peft_model = get_peft_model(base_model, prompt_config)

    extracted_keys = set(peft_model.get_base_model_state_dict().keys())
    assert base_model_keys == extracted_keys
