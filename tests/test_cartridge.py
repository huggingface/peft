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

import warnings

import pytest
import torch
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from peft import (
    CartridgeConfig,
    PeftConfig,
    PeftModel,
    compose_cartridge_adapters,
    get_peft_model,
    initialize_kv_prefix_from_past_key_values,
    load_peft_weights,
    prompt_embeddings_from_past_key_values,
)
from peft.tuners import PrefixTuningConfig

from .testing_utils import hub_online_once


TINY_CAUSAL_LM = "peft-internal-testing/tiny-random-OPTForCausalLM"


@pytest.fixture
def model_id():
    return TINY_CAUSAL_LM


@pytest.fixture
def base_model(model_id):
    with hub_online_once(model_id):
        return AutoModelForCausalLM.from_pretrained(model_id)


def test_cartridge_offsets_position_ids_in_forward(monkeypatch, base_model):
    base = base_model
    peft_config = CartridgeConfig(num_virtual_tokens=4, num_frozen_tokens=1, task_type="CAUSAL_LM")
    model = get_peft_model(base, peft_config)

    captured = {}

    def fake_forward(*args, **kwargs):
        captured["position_ids"] = kwargs.get("position_ids")
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        batch, seq_len = input_ids.shape
        logits = torch.zeros((batch, seq_len, base.config.vocab_size), device=input_ids.device)
        return CausalLMOutputWithPast(logits=logits)

    monkeypatch.setattr(model.base_model, "forward", fake_forward)

    input_ids = torch.randint(0, base.config.vocab_size, (1, 3))
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    _ = model(input_ids=input_ids, position_ids=position_ids)

    assert captured["position_ids"] is not None
    assert torch.equal(captured["position_ids"], position_ids + peft_config.num_virtual_tokens)


def test_cartridge_prefill_4d_mask_uses_cache_position(monkeypatch, base_model):
    base = base_model
    peft_config = CartridgeConfig(num_virtual_tokens=4, num_frozen_tokens=1, task_type="CAUSAL_LM")
    model = get_peft_model(base, peft_config)

    captured = {}

    def fake_create_attention_mask(
        model,
        *,
        model_input,
        attention_mask,
        past_key_values,
        cache_position,
        batch_size,
        sequence_length,
        position_ids,
    ):
        captured["cache_position"] = cache_position
        return attention_mask

    monkeypatch.setattr("peft.peft_model.create_attention_mask", fake_create_attention_mask)

    input_ids = torch.randint(0, base.config.vocab_size, (1, 2))
    attention_mask_4d = torch.ones((1, 1, input_ids.shape[1], input_ids.shape[1]))
    cache_position = torch.arange(input_ids.shape[1])

    def fake_prepare_inputs_for_generation(*args, **kwargs):
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask_4d,
            "cache_position": cache_position,
            "past_key_values": None,
        }

    model.base_model_prepare_inputs_for_generation = fake_prepare_inputs_for_generation
    _ = model.prepare_inputs_for_generation(input_ids)

    assert captured["cache_position"] is not None
    assert torch.equal(captured["cache_position"], cache_position)


@pytest.mark.parametrize("num_frozen_tokens", [0, 2])
def test_cartridge_forward_and_save_load(tmp_path, num_frozen_tokens, base_model, model_id):
    base = base_model
    peft_config = CartridgeConfig(num_virtual_tokens=4, num_frozen_tokens=num_frozen_tokens, task_type="CAUSAL_LM")
    model = get_peft_model(base, peft_config)

    assert model.active_peft_config.peft_type.value == "CARTRIDGE"
    if num_frozen_tokens:
        assert model.prompt_encoder[model.active_adapter].frozen_embedding is not None
        assert model.prompt_encoder[model.active_adapter].frozen_embedding.requires_grad is False
    else:
        assert model.prompt_encoder[model.active_adapter].frozen_embedding is None
    assert model.prompt_encoder[model.active_adapter].trainable_embedding.requires_grad is True

    input_ids = torch.randint(0, base.config.vocab_size, (1, 8))
    out = model(input_ids=input_ids)
    assert out.logits.shape[:2] == (1, 8)

    model.prompt_encoder[model.active_adapter].trainable_embedding.data.fill_(3.0)
    if num_frozen_tokens:
        model.prompt_encoder[model.active_adapter].frozen_embedding.data.fill_(7.0)

    model.save_pretrained(tmp_path)
    with hub_online_once(model_id):
        base2 = AutoModelForCausalLM.from_pretrained(model_id)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        loaded = PeftModel.from_pretrained(base2, tmp_path)
        assert not any("Found missing adapter keys" in str(warning.message) for warning in w)
    out2 = loaded(input_ids=input_ids)
    assert out2.logits.shape == out.logits.shape
    assert torch.allclose(
        loaded.prompt_encoder[loaded.active_adapter].trainable_embedding,
        torch.full_like(loaded.prompt_encoder[loaded.active_adapter].trainable_embedding, 3.0),
    )
    if num_frozen_tokens:
        assert torch.allclose(
            loaded.prompt_encoder[loaded.active_adapter].frozen_embedding,
            torch.full_like(loaded.prompt_encoder[loaded.active_adapter].frozen_embedding, 7.0),
        )
    else:
        assert loaded.prompt_encoder[loaded.active_adapter].frozen_embedding is None


def test_cartridge_init_from_past_key_values_and_compose(tmp_path, base_model, model_id):
    base = base_model
    peft_config = CartridgeConfig(num_virtual_tokens=4, num_frozen_tokens=1, task_type="CAUSAL_LM")
    model = get_peft_model(base, peft_config)

    # Prefill on the *base* model and use the cache prefix as initialization.
    input_ids = torch.randint(0, base.config.vocab_size, (1, 12))
    with model.disable_adapter():
        outputs = model(input_ids=input_ids, use_cache=True)
    prompt_embeddings = initialize_kv_prefix_from_past_key_values(
        model, past_key_values=outputs.past_key_values, num_virtual_tokens=4
    )
    assert prompt_embeddings.shape[0] == 4
    assert model.prompt_encoder[model.active_adapter].weight.device == prompt_embeddings.device
    assert torch.allclose(model.prompt_encoder[model.active_adapter].weight, prompt_embeddings)

    a1 = tmp_path / "a1"
    a2 = tmp_path / "a2"
    out_dir = tmp_path / "composed"
    model.save_pretrained(a1)

    with hub_online_once(model_id):
        base2 = AutoModelForCausalLM.from_pretrained(model_id)
    model2 = get_peft_model(base2, peft_config)
    with model2.disable_adapter():
        outputs2 = model2(input_ids=input_ids, use_cache=True)
    _ = initialize_kv_prefix_from_past_key_values(
        model2, past_key_values=outputs2.past_key_values, num_virtual_tokens=4
    )
    model2.save_pretrained(a2)

    compose_cartridge_adapters([a1, a2], output_path=out_dir)
    cfg = PeftConfig.from_pretrained(out_dir)
    assert cfg.peft_type.value == "CARTRIDGE"
    assert cfg.num_virtual_tokens == 8
    w = load_peft_weights(out_dir, device="cpu")
    assert w["prompt_embeddings"].shape[0] == 8


def test_cartridge_prompt_embeddings_from_past_key_values_matches_init(base_model):
    base = base_model
    peft_config = CartridgeConfig(num_virtual_tokens=4, num_frozen_tokens=0, task_type="CAUSAL_LM")
    model = get_peft_model(base, peft_config)

    input_ids = torch.randint(0, base.config.vocab_size, (1, 10))
    with model.disable_adapter():
        outputs = model(input_ids=input_ids, use_cache=True)

    pe = prompt_embeddings_from_past_key_values(outputs.past_key_values, num_virtual_tokens=4)
    assert pe.shape[0] == 4

    pe2 = initialize_kv_prefix_from_past_key_values(
        model, past_key_values=outputs.past_key_values, num_virtual_tokens=4
    )
    assert pe.device == pe2.device
    assert torch.allclose(pe, pe2)


@pytest.mark.parametrize("num_frozen_tokens", [0, 2])
def test_cartridge_inference_mode_disables_grads_and_forward_works(num_frozen_tokens, base_model):
    base = base_model
    peft_config = CartridgeConfig(
        num_virtual_tokens=4,
        num_frozen_tokens=num_frozen_tokens,
        task_type="CAUSAL_LM",
        inference_mode=True,
    )
    model = get_peft_model(base, peft_config)

    enc = model.prompt_encoder[model.active_adapter]
    # In `inference_mode=True`, PEFT should mark adapter parameters as non-trainable (no gradients) so users can
    # safely run forward/generation without accidentally updating or tracking grads for the CARTRIDGE parameters.
    assert enc.trainable_embedding.requires_grad is False
    if num_frozen_tokens:
        assert enc.frozen_embedding is not None
        assert enc.frozen_embedding.requires_grad is False
    else:
        assert enc.frozen_embedding is None

    input_ids = torch.randint(0, base.config.vocab_size, (1, 6))
    out = model(input_ids=input_ids)
    assert out.logits.shape[:2] == (1, 6)


def test_cartridge_gradient_checkpointing_raises(base_model):
    base = base_model
    base.gradient_checkpointing_enable()
    peft_config = CartridgeConfig(num_virtual_tokens=4, num_frozen_tokens=0, task_type="CAUSAL_LM")

    with pytest.raises(ValueError, match="does not work with gradient checkpointing"):
        _ = get_peft_model(base, peft_config)


def test_prefix_tuning_can_be_initialized_from_past_key_values_when_no_projection(base_model):
    base = base_model
    peft_config = PrefixTuningConfig(num_virtual_tokens=4, task_type="CAUSAL_LM")
    model = get_peft_model(base, peft_config)

    input_ids = torch.randint(0, base.config.vocab_size, (1, 10))
    with model.disable_adapter():
        outputs = model(input_ids=input_ids, use_cache=True)

    pe = prompt_embeddings_from_past_key_values(outputs.past_key_values, num_virtual_tokens=4)
    pe2 = initialize_kv_prefix_from_past_key_values(
        model, past_key_values=outputs.past_key_values, num_virtual_tokens=4
    )
    assert pe.device == pe2.device
    assert torch.allclose(pe, pe2)
    assert model.prompt_encoder[model.active_adapter].embedding.weight.device == pe.device
    assert torch.allclose(model.prompt_encoder[model.active_adapter].embedding.weight, pe)
