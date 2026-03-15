# Copyright 2026-present the HuggingFace Inc. team.
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

import copy

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

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
    TrainableTokensConfig,
    VBLoRAConfig,
    VeraConfig,
    get_peft_model,
)
from peft.utils import infer_device

from .testing_utils import hub_online_once


CAUSAL_LM_MODEL_ID = "peft-internal-testing/tiny-random-OPTForCausalLM"
SEQ2SEQ_MODEL_ID = "peft-internal-testing/tiny-random-T5ForConditionalGeneration-calibrated"


def get_peft_configs():
    """Return a list of (config_name, config) tuples for parametrized testing.

    Uses ``"all-linear"`` so the same configs work with both causal LM (OPT) and seq2seq (T5) models. Configs that
    cannot handle arbitrary layer dimensions (BOFT, VB-LoRA) are tested separately with causal-LM-only target modules.
    """
    return [
        ("lora", LoraConfig(r=4, lora_alpha=2, target_modules="all-linear")),
        ("lora_trainable_tokens", LoraConfig(r=4, trainable_token_indices=[0, 1], target_modules="all-linear")),
        ("trainable_tokens", TrainableTokensConfig(token_indices=[0, 1])),
        ("loha", LoHaConfig(r=4, alpha=2, target_modules="all-linear")),
        ("lokr", LoKrConfig(r=4, alpha=2, target_modules="all-linear")),
        ("ia3", IA3Config(target_modules="all-linear", feedforward_modules="all-linear")),
        ("oft", OFTConfig(oft_block_size=4, target_modules="all-linear")),
        ("vera", VeraConfig(r=4, target_modules="all-linear")),
    ]


def get_causal_only_configs():
    """Return configs that only work with causal LM models due to dimension constraints."""
    return [
        ("boft", BOFTConfig(boft_block_size=4, target_modules=["q_proj", "v_proj"])),
        ("vblora", VBLoRAConfig(r=4, target_modules=["q_proj", "v_proj"], num_vectors=50, vector_length=2)),
    ]


def get_prompt_learning_configs():
    """Return a list of (config_name, config) tuples for prompt learning methods."""
    return [
        ("prompt_tuning", PromptTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=10)),
        ("prefix_tuning", PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=10)),
    ]


class TestGetBaseModelStateDict:
    """Tests for get_base_model_state_dict and set_base_model_state_dict."""

    torch_device = infer_device()

    @pytest.mark.parametrize("model_id", [CAUSAL_LM_MODEL_ID, SEQ2SEQ_MODEL_ID])
    @pytest.mark.parametrize("config_name,peft_config", get_peft_configs(), ids=[c[0] for c in get_peft_configs()])
    def test_keys_match(self, model_id, config_name, peft_config):
        """Test that get_base_model_state_dict returns keys matching the original base model."""
        peft_config = copy.deepcopy(peft_config)
        model_cls = AutoModelForSeq2SeqLM if "T5" in model_id else AutoModelForCausalLM
        with hub_online_once(model_id):
            base_model = model_cls.from_pretrained(model_id).to(self.torch_device)

        base_model_keys = set(base_model.state_dict().keys())
        peft_model = get_peft_model(base_model, peft_config)
        extracted_keys = set(peft_model.get_base_model_state_dict().keys())
        assert base_model_keys == extracted_keys, f"Key mismatch for {config_name} on {model_id}"

    @pytest.mark.parametrize("model_id", [CAUSAL_LM_MODEL_ID, SEQ2SEQ_MODEL_ID])
    @pytest.mark.parametrize("config_name,peft_config", get_peft_configs(), ids=[c[0] for c in get_peft_configs()])
    def test_values_match(self, model_id, config_name, peft_config):
        """Test that the tensor values match the original base model weights."""
        peft_config = copy.deepcopy(peft_config)
        model_cls = AutoModelForSeq2SeqLM if "T5" in model_id else AutoModelForCausalLM
        with hub_online_once(model_id):
            base_model = model_cls.from_pretrained(model_id).to(self.torch_device)

        original_state_dict = {k: v.clone() for k, v in base_model.state_dict().items()}

        peft_model = get_peft_model(base_model, peft_config)
        extracted_state_dict = peft_model.get_base_model_state_dict()

        for key in original_state_dict:
            assert key in extracted_state_dict, f"Missing key {key} for {config_name}"
            assert torch.allclose(original_state_dict[key], extracted_state_dict[key]), (
                f"Value mismatch for key {key} in {config_name}"
            )

    @pytest.mark.parametrize("config_name,peft_config", get_peft_configs(), ids=[c[0] for c in get_peft_configs()])
    def test_roundtrip(self, config_name, peft_config):
        """Test that get followed by set preserves the weights."""
        peft_config = copy.deepcopy(peft_config)
        with hub_online_once(CAUSAL_LM_MODEL_ID):
            base_model = AutoModelForCausalLM.from_pretrained(CAUSAL_LM_MODEL_ID).to(self.torch_device)

        peft_model = get_peft_model(base_model, peft_config)

        original_base_state_dict = {k: v.clone() for k, v in peft_model.get_base_model_state_dict().items()}

        # Modify base weights by adding noise
        with torch.no_grad():
            for param in peft_model.base_model.model.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        result = peft_model.set_base_model_state_dict(original_base_state_dict)
        assert len(result.missing_keys) == 0, f"Missing keys for {config_name}: {result.missing_keys}"
        assert len(result.unexpected_keys) == 0, f"Unexpected keys for {config_name}: {result.unexpected_keys}"

        restored_state_dict = peft_model.get_base_model_state_dict()
        for key in original_base_state_dict:
            assert torch.allclose(original_base_state_dict[key], restored_state_dict[key]), (
                f"Roundtrip failed for key {key} in {config_name}"
            )

    @pytest.mark.parametrize(
        "config_name,peft_config",
        get_prompt_learning_configs(),
        ids=[c[0] for c in get_prompt_learning_configs()],
    )
    def test_prompt_learning_keys(self, config_name, peft_config):
        """Test get_base_model_state_dict with prompt learning methods."""
        peft_config = copy.deepcopy(peft_config)
        with hub_online_once(CAUSAL_LM_MODEL_ID):
            base_model = AutoModelForCausalLM.from_pretrained(CAUSAL_LM_MODEL_ID).to(self.torch_device)

        base_model_keys = set(base_model.state_dict().keys())
        peft_model = get_peft_model(base_model, peft_config)
        extracted_keys = set(peft_model.get_base_model_state_dict().keys())
        assert base_model_keys == extracted_keys, f"Key mismatch for {config_name}"

    @pytest.mark.parametrize(
        "config_name,peft_config",
        get_prompt_learning_configs(),
        ids=[c[0] for c in get_prompt_learning_configs()],
    )
    def test_prompt_learning_set(self, config_name, peft_config):
        """Test set_base_model_state_dict with prompt learning methods."""
        peft_config = copy.deepcopy(peft_config)
        with hub_online_once(CAUSAL_LM_MODEL_ID):
            base_model = AutoModelForCausalLM.from_pretrained(CAUSAL_LM_MODEL_ID).to(self.torch_device)

        original_state_dict = {k: v.clone() for k, v in base_model.state_dict().items()}
        peft_model = get_peft_model(base_model, peft_config)

        result = peft_model.set_base_model_state_dict(original_state_dict)
        assert len(result.missing_keys) == 0, f"Missing keys for {config_name}"
        assert len(result.unexpected_keys) == 0, f"Unexpected keys for {config_name}"

    @pytest.mark.parametrize("config_name,peft_config", get_peft_configs(), ids=[c[0] for c in get_peft_configs()])
    def test_strict_missing_keys(self, config_name, peft_config):
        """Test that strict=True raises error for missing keys."""
        peft_config = copy.deepcopy(peft_config)
        with hub_online_once(CAUSAL_LM_MODEL_ID):
            base_model = AutoModelForCausalLM.from_pretrained(CAUSAL_LM_MODEL_ID).to(self.torch_device)

        peft_model = get_peft_model(base_model, peft_config)
        state_dict = peft_model.get_base_model_state_dict()
        removed_key = list(state_dict.keys())[0]
        del state_dict[removed_key]

        with pytest.raises(RuntimeError, match="Missing key"):
            peft_model.set_base_model_state_dict(state_dict, strict=True)

        result = peft_model.set_base_model_state_dict(state_dict, strict=False)
        assert removed_key in result.missing_keys, f"Missing key not reported for {config_name}"

    @pytest.mark.parametrize("config_name,peft_config", get_peft_configs(), ids=[c[0] for c in get_peft_configs()])
    def test_strict_unexpected_keys(self, config_name, peft_config):
        """Test that strict=True raises error for unexpected keys."""
        peft_config = copy.deepcopy(peft_config)
        with hub_online_once(CAUSAL_LM_MODEL_ID):
            base_model = AutoModelForCausalLM.from_pretrained(CAUSAL_LM_MODEL_ID).to(self.torch_device)

        peft_model = get_peft_model(base_model, peft_config)
        state_dict = peft_model.get_base_model_state_dict()
        state_dict["unexpected.weight"] = torch.zeros(10)

        with pytest.raises(RuntimeError, match="Unexpected key"):
            peft_model.set_base_model_state_dict(state_dict, strict=True)

        result = peft_model.set_base_model_state_dict(state_dict, strict=False)
        assert "unexpected.weight" in result.unexpected_keys, f"Unexpected key not reported for {config_name}"

    def test_modules_to_save_get(self):
        """Test that modules_to_save are handled correctly."""
        with hub_online_once(CAUSAL_LM_MODEL_ID):
            base_model = AutoModelForCausalLM.from_pretrained(CAUSAL_LM_MODEL_ID).to(self.torch_device)

        base_model_keys = set(base_model.state_dict().keys())
        lora_config = LoraConfig(r=4, lora_alpha=2, target_modules="all-linear", modules_to_save=["lm_head"])
        peft_model = get_peft_model(base_model, lora_config)

        extracted_keys = set(peft_model.get_base_model_state_dict().keys())
        assert base_model_keys == extracted_keys

    def test_modules_to_save_set(self):
        """Test that modules_to_save are handled correctly during set."""
        with hub_online_once(CAUSAL_LM_MODEL_ID):
            base_model = AutoModelForCausalLM.from_pretrained(CAUSAL_LM_MODEL_ID).to(self.torch_device)

        original_state_dict = {k: v.clone() for k, v in base_model.state_dict().items()}
        lora_config = LoraConfig(r=4, lora_alpha=2, target_modules="all-linear", modules_to_save=["lm_head"])
        peft_model = get_peft_model(base_model, lora_config)

        result = peft_model.set_base_model_state_dict(original_state_dict)
        assert len(result.missing_keys) == 0
        assert len(result.unexpected_keys) == 0

    @pytest.mark.parametrize(
        "config_name,peft_config", get_causal_only_configs(), ids=[c[0] for c in get_causal_only_configs()]
    )
    def test_keys_match_causal_only(self, config_name, peft_config):
        """Test get_base_model_state_dict for configs with dimension constraints (causal LM only)."""
        peft_config = copy.deepcopy(peft_config)
        with hub_online_once(CAUSAL_LM_MODEL_ID):
            base_model = AutoModelForCausalLM.from_pretrained(CAUSAL_LM_MODEL_ID).to(self.torch_device)

        base_model_keys = set(base_model.state_dict().keys())
        peft_model = get_peft_model(base_model, peft_config)
        extracted_keys = set(peft_model.get_base_model_state_dict().keys())
        assert base_model_keys == extracted_keys, f"Key mismatch for {config_name}"

    @pytest.mark.parametrize(
        "config_name,peft_config", get_causal_only_configs(), ids=[c[0] for c in get_causal_only_configs()]
    )
    def test_values_match_causal_only(self, config_name, peft_config):
        """Test values match for configs with dimension constraints (causal LM only)."""
        peft_config = copy.deepcopy(peft_config)
        with hub_online_once(CAUSAL_LM_MODEL_ID):
            base_model = AutoModelForCausalLM.from_pretrained(CAUSAL_LM_MODEL_ID).to(self.torch_device)

        original_state_dict = {k: v.clone() for k, v in base_model.state_dict().items()}
        peft_model = get_peft_model(base_model, peft_config)
        extracted_state_dict = peft_model.get_base_model_state_dict()

        for key in original_state_dict:
            assert key in extracted_state_dict, f"Missing key {key} for {config_name}"
            assert torch.allclose(original_state_dict[key], extracted_state_dict[key]), (
                f"Value mismatch for key {key} in {config_name}"
            )

    @pytest.mark.parametrize(
        "config_name,peft_config", get_causal_only_configs(), ids=[c[0] for c in get_causal_only_configs()]
    )
    def test_roundtrip_causal_only(self, config_name, peft_config):
        """Test roundtrip for configs with dimension constraints (causal LM only)."""
        peft_config = copy.deepcopy(peft_config)
        with hub_online_once(CAUSAL_LM_MODEL_ID):
            base_model = AutoModelForCausalLM.from_pretrained(CAUSAL_LM_MODEL_ID).to(self.torch_device)

        peft_model = get_peft_model(base_model, peft_config)
        original_base_state_dict = {k: v.clone() for k, v in peft_model.get_base_model_state_dict().items()}

        with torch.no_grad():
            for param in peft_model.base_model.model.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        result = peft_model.set_base_model_state_dict(original_base_state_dict)
        assert len(result.missing_keys) == 0, f"Missing keys for {config_name}: {result.missing_keys}"
        assert len(result.unexpected_keys) == 0, f"Unexpected keys for {config_name}: {result.unexpected_keys}"

        restored_state_dict = peft_model.get_base_model_state_dict()
        for key in original_base_state_dict:
            assert torch.allclose(original_base_state_dict[key], restored_state_dict[key]), (
                f"Roundtrip failed for key {key} in {config_name}"
            )

    def test_multiple_adapters(self):
        """Test with multiple adapters of different types."""
        with hub_online_once(CAUSAL_LM_MODEL_ID):
            base_model = AutoModelForCausalLM.from_pretrained(CAUSAL_LM_MODEL_ID).to(self.torch_device)

        base_model_keys = set(base_model.state_dict().keys())

        lora_config_1 = LoraConfig(r=4, lora_alpha=2, target_modules=["q_proj", "v_proj"])
        peft_model = get_peft_model(base_model, lora_config_1, adapter_name="adapter1")

        lora_config_2 = LoraConfig(r=8, lora_alpha=4, target_modules=["k_proj", "out_proj"])
        peft_model.add_adapter("adapter2", lora_config_2)

        extracted_keys = set(peft_model.get_base_model_state_dict().keys())
        assert base_model_keys == extracted_keys
