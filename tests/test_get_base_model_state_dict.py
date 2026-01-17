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

import pytest
import torch
from transformers import AutoModelForCausalLM

from peft import (
    BOFTConfig,
    IA3Config,
    LoHaConfig,
    LoKrConfig,
    LoraConfig,
    OFTConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    TaskType,
    get_peft_model,
)
from peft.utils import infer_device

from .testing_utils import hub_online_once


MODEL_ID = "peft-internal-testing/tiny-random-OPTForCausalLM"


def get_peft_configs():
    """Return a list of (config_name, config) tuples for parametrized testing"""
    return [
        ("lora", LoraConfig(r=4, lora_alpha=2, target_modules=["q_proj", "v_proj"])),
        ("lora_all_linear", LoraConfig(r=4, lora_alpha=2, target_modules="all-linear")),
        ("loha", LoHaConfig(r=4, alpha=2, target_modules=["q_proj", "v_proj"])),
        ("lokr", LoKrConfig(r=4, alpha=2, target_modules=["q_proj", "v_proj"])),
        ("ia3", IA3Config(target_modules=["q_proj", "v_proj", "fc1"], feedforward_modules=["fc1"])),
        ("oft", OFTConfig(oft_block_size=4, target_modules=["q_proj", "v_proj"])),
        ("boft", BOFTConfig(boft_block_size=4, target_modules=["q_proj", "v_proj"])),
    ]


def get_prompt_learning_configs():
    """Return a list of (config_name, config) tuples for prompt learning methods"""
    return [
        ("prompt_tuning", PromptTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=10)),
        ("prefix_tuning", PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=10)),
    ]


@pytest.fixture
def base_model():
    """Fixture that provides a fresh base model for each test."""
    torch_device = infer_device()
    with hub_online_once(MODEL_ID):
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(torch_device)
    return model


@pytest.mark.parametrize("config_name,peft_config", get_peft_configs(), ids=[c[0] for c in get_peft_configs()])
def test_get_base_model_state_dict_keys_match(base_model, config_name, peft_config):
    """Test that get_base_model_state_dict returns keys matching the original base model."""
    base_model_keys = set(base_model.state_dict().keys())
    peft_model = get_peft_model(base_model, peft_config)
    extracted_keys = set(peft_model.get_base_model_state_dict().keys())
    assert base_model_keys == extracted_keys, f"Key mismatch for {config_name}"


@pytest.mark.parametrize("config_name,peft_config", get_peft_configs(), ids=[c[0] for c in get_peft_configs()])
def test_get_base_model_state_dict_values_match(config_name, peft_config):
    """Test that the tensor values match the original base model weights."""
    torch_device = infer_device()
    with hub_online_once(MODEL_ID):
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(torch_device)

    original_state_dict = {k: v.clone() for k, v in base_model.state_dict().items()}

    peft_model = get_peft_model(base_model, peft_config)
    extracted_state_dict = peft_model.get_base_model_state_dict()

    for key in original_state_dict:
        assert key in extracted_state_dict, f"Missing key {key} for {config_name}"
        assert torch.allclose(original_state_dict[key], extracted_state_dict[key]), (
            f"Value mismatch for key {key} in {config_name}"
        )


@pytest.mark.parametrize("config_name,peft_config", get_peft_configs(), ids=[c[0] for c in get_peft_configs()])
def test_set_base_model_state_dict_roundtrip(config_name, peft_config):
    """Test that get followed by set preserves the weights."""
    torch_device = infer_device()
    with hub_online_once(MODEL_ID):
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(torch_device)

    peft_model = get_peft_model(base_model, peft_config)

    # Get original base weights
    original_base_state_dict = {k: v.clone() for k, v in peft_model.get_base_model_state_dict().items()}

    # Modify base weights by adding noise
    with torch.no_grad():
        for key, param in peft_model.base_model.model.named_parameters():
            if "base_layer" in key or (
                not any(adapter_key in key for adapter_key in ["lora_", "hada_", "lokr_", "ia3_", "oft_", "boft_"])
            ):
                param.add_(torch.randn_like(param) * 0.1)

    # Restore using set_base_model_state_dict
    result = peft_model.set_base_model_state_dict(original_base_state_dict)
    assert len(result.missing_keys) == 0, f"Missing keys for {config_name}: {result.missing_keys}"
    assert len(result.unexpected_keys) == 0, f"Unexpected keys for {config_name}: {result.unexpected_keys}"

    # Verify weights match
    restored_state_dict = peft_model.get_base_model_state_dict()
    for key in original_base_state_dict:
        assert torch.allclose(original_base_state_dict[key], restored_state_dict[key]), (
            f"Roundtrip failed for key {key} in {config_name}"
        )


@pytest.mark.parametrize(
    "config_name,peft_config", get_prompt_learning_configs(), ids=[c[0] for c in get_prompt_learning_configs()]
)
def test_get_base_model_state_dict_prompt_learning(config_name, peft_config):
    """Test get_base_model_state_dict with prompt learning methods."""
    torch_device = infer_device()
    with hub_online_once(MODEL_ID):
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(torch_device)

    base_model_keys = set(base_model.state_dict().keys())

    peft_model = get_peft_model(base_model, peft_config)
    extracted_keys = set(peft_model.get_base_model_state_dict().keys())

    assert base_model_keys == extracted_keys, f"Key mismatch for {config_name}"


@pytest.mark.parametrize(
    "config_name,peft_config", get_prompt_learning_configs(), ids=[c[0] for c in get_prompt_learning_configs()]
)
def test_set_base_model_state_dict_prompt_learning(config_name, peft_config):
    """Test set_base_model_state_dict with prompt learning methods."""
    torch_device = infer_device()
    with hub_online_once(MODEL_ID):
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(torch_device)

    original_state_dict = {k: v.clone() for k, v in base_model.state_dict().items()}

    peft_model = get_peft_model(base_model, peft_config)

    result = peft_model.set_base_model_state_dict(original_state_dict)
    assert len(result.missing_keys) == 0, f"Missing keys for {config_name}"
    assert len(result.unexpected_keys) == 0, f"Unexpected keys for {config_name}"


@pytest.mark.parametrize("config_name,peft_config", get_peft_configs(), ids=[c[0] for c in get_peft_configs()])
def test_set_base_model_state_dict_strict_missing_keys(base_model, config_name, peft_config):
    """Test that strict=True raises error for missing keys across PEFT methods."""
    peft_model = get_peft_model(base_model, peft_config)

    state_dict = peft_model.get_base_model_state_dict()
    removed_key = list(state_dict.keys())[0]
    del state_dict[removed_key]

    with pytest.raises(RuntimeError, match="Missing key"):
        peft_model.set_base_model_state_dict(state_dict, strict=True)

    result = peft_model.set_base_model_state_dict(state_dict, strict=False)
    assert removed_key in result.missing_keys, f"Missing key not reported for {config_name}"


@pytest.mark.parametrize("config_name,peft_config", get_peft_configs(), ids=[c[0] for c in get_peft_configs()])
def test_set_base_model_state_dict_strict_unexpected_keys(base_model, config_name, peft_config):
    """Test that strict=True raises error for unexpected keys across PEFT methods."""
    peft_model = get_peft_model(base_model, peft_config)

    state_dict = peft_model.get_base_model_state_dict()
    state_dict["unexpected.weight"] = torch.zeros(10)

    with pytest.raises(RuntimeError, match="Unexpected key"):
        peft_model.set_base_model_state_dict(state_dict, strict=True)

    result = peft_model.set_base_model_state_dict(state_dict, strict=False)
    assert "unexpected.weight" in result.unexpected_keys, f"Unexpected key not reported for {config_name}"


def test_get_base_model_state_dict_with_modules_to_save():
    """Test that modules_to_save are handled correctly."""
    torch_device = infer_device()
    with hub_online_once(MODEL_ID):
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(torch_device)

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


def test_set_base_model_state_dict_with_modules_to_save():
    """Test that modules_to_save are handled correctly during set."""
    torch_device = infer_device()
    with hub_online_once(MODEL_ID):
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(torch_device)

    original_state_dict = {k: v.clone() for k, v in base_model.state_dict().items()}

    lora_config = LoraConfig(
        r=4,
        lora_alpha=2,
        target_modules="all-linear",
        modules_to_save=["lm_head"],
    )
    peft_model = get_peft_model(base_model, lora_config)

    result = peft_model.set_base_model_state_dict(original_state_dict)
    assert len(result.missing_keys) == 0
    assert len(result.unexpected_keys) == 0


def test_get_base_model_state_dict_with_multiple_adapters():
    """Test with multiple adapters of different types."""
    torch_device = infer_device()
    with hub_online_once(MODEL_ID):
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(torch_device)

    base_model_keys = set(base_model.state_dict().keys())

    lora_config_1 = LoraConfig(r=4, lora_alpha=2, target_modules=["q_proj", "v_proj"])
    peft_model = get_peft_model(base_model, lora_config_1, adapter_name="adapter1")

    lora_config_2 = LoraConfig(r=8, lora_alpha=4, target_modules=["k_proj", "out_proj"])
    peft_model.add_adapter("adapter2", lora_config_2)

    extracted_keys = set(peft_model.get_base_model_state_dict().keys())
    assert base_model_keys == extracted_keys


def test_get_base_model_state_dict_nested_base_layer():
    """Test that deeply nested .base_layer.base_layer. patterns are correctly unwrapped."""
    torch_device = infer_device()
    with hub_online_once(MODEL_ID):
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(torch_device)

    base_model_keys = set(base_model.state_dict().keys())

    lora_config = LoraConfig(r=4, lora_alpha=2, target_modules="all-linear")
    peft_model = get_peft_model(base_model, lora_config)

    peft_state_dict = peft_model.base_model.model.state_dict()

    # Create a modified version with nested .base_layer.base_layer.
    nested_state_dict = {}
    for key, value in peft_state_dict.items():
        if ".base_layer." in key:
            nested_key = key.replace(".base_layer.", ".base_layer.base_layer.")
            nested_state_dict[nested_key] = value
        else:
            nested_state_dict[key] = value

    original_method = peft_model.base_model.model.state_dict
    peft_model.base_model.model.state_dict = lambda: nested_state_dict
    result = peft_model.get_base_model_state_dict()
    for key in result.keys():
        assert ".base_layer." not in key
    assert set(result.keys()) == base_model_keys


def test_set_base_model_state_dict_nested_base_layer():
    """Test that set works correctly when model has deeply nested .base_layer. patterns."""
    torch_device = infer_device()
    with hub_online_once(MODEL_ID):
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(torch_device)

    original_state_dict = {k: v.clone() for k, v in base_model.state_dict().items()}

    lora_config = LoraConfig(r=4, lora_alpha=2, target_modules="all-linear")
    peft_model = get_peft_model(base_model, lora_config)

    peft_state_dict = peft_model.base_model.model.state_dict()

    nested_state_dict = {}
    for key, value in peft_state_dict.items():
        if ".base_layer." in key:
            nested_key = key.replace(".base_layer.", ".base_layer.base_layer.")
            nested_state_dict[nested_key] = value
        else:
            nested_state_dict[key] = value

    original_method = peft_model.base_model.model.state_dict
    peft_model.base_model.model.state_dict = lambda: nested_state_dict

    result = peft_model.set_base_model_state_dict(original_state_dict)
    assert len(result.missing_keys) == 0
    assert len(result.unexpected_keys) == 0


def test_get_base_model_state_dict_filters_trainable_tokens():
    """Test that .trainable_tokens_ entries are filtered out from the state dict."""
    torch_device = infer_device()
    with hub_online_once(MODEL_ID):
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(torch_device)

    base_model_keys = set(base_model.state_dict().keys())

    lora_config = LoraConfig(r=4, lora_alpha=2, target_modules="all-linear")
    peft_model = get_peft_model(base_model, lora_config)

    peft_state_dict = dict(peft_model.base_model.model.state_dict())

    # add fake trainable_tokens entries
    peft_state_dict["model.embed_tokens.trainable_tokens_default"] = torch.zeros(10, 128)
    peft_state_dict["model.embed_tokens.trainable_tokens_other"] = torch.zeros(10, 128)

    original_method = peft_model.base_model.model.state_dict
    peft_model.base_model.model.state_dict = lambda: peft_state_dict
    result = peft_model.get_base_model_state_dict()
    for key in result.keys():
        assert "trainable_tokens" not in key
    assert set(result.keys()) == base_model_keys


def test_set_base_model_state_dict_with_trainable_tokens():
    """Test that set works correctly when model has .trainable_tokens_ entries."""
    torch_device = infer_device()
    with hub_online_once(MODEL_ID):
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(torch_device)

    original_state_dict = {k: v.clone() for k, v in base_model.state_dict().items()}

    lora_config = LoraConfig(r=4, lora_alpha=2, target_modules="all-linear")
    peft_model = get_peft_model(base_model, lora_config)

    peft_state_dict = dict(peft_model.base_model.model.state_dict())

    peft_state_dict["model.embed_tokens.trainable_tokens_default"] = torch.zeros(10, 128)

    original_method = peft_model.base_model.model.state_dict
    peft_model.base_model.model.state_dict = lambda: peft_state_dict

    result = peft_model.set_base_model_state_dict(original_state_dict)
    assert len(result.missing_keys) == 0
    assert len(result.unexpected_keys) == 0
