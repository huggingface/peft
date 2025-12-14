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

import tempfile
import warnings

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from peft import (
    CartridgeConfig,
    PeftConfig,
    PeftModel,
    compose_cartridge_adapters,
    get_peft_model,
    initialize_cartridge_from_past_key_values,
    load_peft_weights,
    prompt_embeddings_from_past_key_values,
)


def _make_tiny_gpt2():
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=16, vocab_size=101)
    return GPT2LMHeadModel(cfg)


def test_cartridge_forward_and_save_load():
    base = _make_tiny_gpt2()
    peft_config = CartridgeConfig(num_virtual_tokens=4, num_frozen_tokens=2, task_type="CAUSAL_LM")
    model = get_peft_model(base, peft_config)

    assert model.active_peft_config.peft_type.value == "CARTRIDGE"
    assert model.prompt_encoder[model.active_adapter].frozen_embedding.requires_grad is False
    assert model.prompt_encoder[model.active_adapter].trainable_embedding.requires_grad is True

    input_ids = torch.randint(0, base.config.vocab_size, (1, 8))
    out = model(input_ids=input_ids)
    assert out.logits.shape[:2] == (1, 8)

    with tempfile.TemporaryDirectory() as tmp:
        model.prompt_encoder[model.active_adapter].trainable_embedding.data.fill_(3.0)
        if model.prompt_encoder[model.active_adapter].frozen_embedding is not None:
            model.prompt_encoder[model.active_adapter].frozen_embedding.data.fill_(7.0)

        model.save_pretrained(tmp)
        base2 = _make_tiny_gpt2()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loaded = PeftModel.from_pretrained(base2, tmp)
            assert not any("Found missing adapter keys" in str(warning.message) for warning in w)
        out2 = loaded(input_ids=input_ids)
        assert out2.logits.shape == out.logits.shape
        assert torch.allclose(
            loaded.prompt_encoder[loaded.active_adapter].trainable_embedding,
            torch.full_like(loaded.prompt_encoder[loaded.active_adapter].trainable_embedding, 3.0),
        )
        if loaded.prompt_encoder[loaded.active_adapter].frozen_embedding is not None:
            assert torch.allclose(
                loaded.prompt_encoder[loaded.active_adapter].frozen_embedding,
                torch.full_like(loaded.prompt_encoder[loaded.active_adapter].frozen_embedding, 7.0),
            )


def test_cartridge_init_from_past_key_values_and_compose():
    base = _make_tiny_gpt2()
    peft_config = CartridgeConfig(num_virtual_tokens=4, num_frozen_tokens=1, task_type="CAUSAL_LM")
    model = get_peft_model(base, peft_config)

    # Prefill on the *base* model and use the cache prefix as initialization.
    input_ids = torch.randint(0, base.config.vocab_size, (1, 12))
    with model.disable_adapter():
        outputs = model(input_ids=input_ids, use_cache=True)
    prompt_embeddings = initialize_cartridge_from_past_key_values(
        model, past_key_values=outputs.past_key_values, num_virtual_tokens=4
    )
    assert prompt_embeddings.shape[0] == 4
    assert torch.allclose(model.prompt_encoder[model.active_adapter].weight.cpu(), prompt_embeddings.cpu())

    with tempfile.TemporaryDirectory() as tmp:
        a1 = f"{tmp}/a1"
        a2 = f"{tmp}/a2"
        out_dir = f"{tmp}/composed"
        model.save_pretrained(a1)

        base2 = _make_tiny_gpt2()
        model2 = get_peft_model(base2, peft_config)
        with model2.disable_adapter():
            outputs2 = model2(input_ids=input_ids, use_cache=True)
        _ = initialize_cartridge_from_past_key_values(
            model2, past_key_values=outputs2.past_key_values, num_virtual_tokens=4
        )
        model2.save_pretrained(a2)

        compose_cartridge_adapters([a1, a2], output_path=out_dir)
        cfg = PeftConfig.from_pretrained(out_dir)
        assert cfg.peft_type.value == "CARTRIDGE"
        assert cfg.num_virtual_tokens == 8
        w = load_peft_weights(out_dir, device="cpu")
        assert w["prompt_embeddings"].shape[0] == 8


def test_cartridge_prompt_embeddings_from_past_key_values_matches_init():
    base = _make_tiny_gpt2()
    peft_config = CartridgeConfig(num_virtual_tokens=4, num_frozen_tokens=0, task_type="CAUSAL_LM")
    model = get_peft_model(base, peft_config)

    input_ids = torch.randint(0, base.config.vocab_size, (1, 10))
    with model.disable_adapter():
        outputs = model(input_ids=input_ids, use_cache=True)

    pe = prompt_embeddings_from_past_key_values(outputs.past_key_values, num_virtual_tokens=4)
    assert pe.shape[0] == 4

    pe2 = initialize_cartridge_from_past_key_values(
        model, past_key_values=outputs.past_key_values, num_virtual_tokens=4
    )
    assert torch.allclose(pe.cpu(), pe2.cpu())


def test_cartridge_inference_mode_disables_grads_and_forward_works():
    base = _make_tiny_gpt2()
    peft_config = CartridgeConfig(
        num_virtual_tokens=4, num_frozen_tokens=2, task_type="CAUSAL_LM", inference_mode=True
    )
    model = get_peft_model(base, peft_config)

    enc = model.prompt_encoder[model.active_adapter]
    assert enc.trainable_embedding.requires_grad is False
    if enc.frozen_embedding is not None:
        assert enc.frozen_embedding.requires_grad is False

    input_ids = torch.randint(0, base.config.vocab_size, (1, 6))
    out = model(input_ids=input_ids)
    assert out.logits.shape[:2] == (1, 6)


def test_cartridge_gradient_checkpointing_raises():
    base = _make_tiny_gpt2()
    base.gradient_checkpointing_enable()
    peft_config = CartridgeConfig(num_virtual_tokens=4, num_frozen_tokens=0, task_type="CAUSAL_LM")

    try:
        _ = get_peft_model(base, peft_config)
    except ValueError as exc:
        assert "does not work with gradient checkpointing" in str(exc)
    else:
        raise AssertionError("Expected CARTRIDGE to raise with gradient checkpointing enabled.")
