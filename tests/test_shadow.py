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
import json
import platform

import pytest
import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    Cache,
    LlamaConfig,
    LlamaModel,
)
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput

from peft import (
    PeftModel,
    ShadowConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft.tuners.shadow import DetachedShadowModel, ShadowCache
from peft.tuners.shadow.diffusion_models import DetachedFluxShadowModel
from peft.tuners.shadow.layers import ShadowLayer
from peft.utils.constants import SAFETENSORS_WEIGHTS_NAME

from .testing_utils import hub_online_once


LLAMA_CAUSAL_MODEL_ID = "peft-internal-testing/tiny-random-LlamaForCausalLM"
LLAMA_SEQCLS_MODEL_ID = "trl-internal-testing/tiny-LlamaForSequenceClassification-3.2"


def make_llama_causal():
    with hub_online_once(LLAMA_CAUSAL_MODEL_ID):
        return AutoModelForCausalLM.from_pretrained(LLAMA_CAUSAL_MODEL_ID)


def make_llama_seqcls(num_labels=3):
    with hub_online_once(LLAMA_SEQCLS_MODEL_ID):
        return AutoModelForSequenceClassification.from_pretrained(
            LLAMA_SEQCLS_MODEL_ID, num_labels=num_labels, ignore_mismatched_sizes=True
        )


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.mark.skipif(platform.system() != "Linux", reason="Run FSDP tests only on Linux")
@pytest.mark.skipif(not dist.is_available(), reason="These tests require torch.distributed")
class TestShadowFsdpStateDict:
    @pytest.fixture(autouse=True)
    def fsdp_process_group(self):
        dist.init_process_group(
            backend="gloo",
            store=dist.HashStore(),
            rank=0,
            world_size=1,
        )
        try:
            yield
        finally:
            dist.destroy_process_group()

    def test_save_load_roundtrip_fsdp_wrapped(self):
        config = ShadowConfig(task_type="CAUSAL_LM", init_weights=False, shadow_num_hidden_layers=2)
        model = get_peft_model(make_llama_causal(), config).eval()
        model = FSDP(model, use_orig_params=True, device_id=torch.device("cpu"))
        input_ids = torch.randint(0, 128, (2, 6))

        with torch.no_grad():
            output_before = model(input_ids=input_ids).logits

        state_dict = get_peft_model_state_dict(model)
        assert any(".shadow_backbone." in key for key in state_dict)
        assert not any("_fsdp_wrapped_module" in key for key in state_dict)
        del model

        model = get_peft_model(make_llama_causal(), config).eval()
        set_peft_model_state_dict(model, state_dict)
        with torch.no_grad():
            output_after = model(input_ids=input_ids).logits

        assert torch.allclose(output_after, output_before, atol=1e-6)


class TestShadowCausalLM:
    def test_auxiliary_loss_trains_shadow_backbone(self):
        # With auxiliary_loss_weight > 0, the shadow backbone must receive gradients.
        model = get_peft_model(make_llama_causal(), ShadowConfig(task_type="CAUSAL_LM", auxiliary_loss_weight=0.5))
        ids = torch.randint(0, 128, (2, 6))
        out = model(input_ids=ids, labels=ids.clone())
        out.loss.backward()
        backbone = model.base_model.shadow_backbone["default"]
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in backbone.parameters())

    def test_forward_exposes_shadow_loss(self):
        # The (unweighted) shadow-path loss is exposed for logging/inspection both on the output (`shadow_loss`) and on
        # the tuner (`last_shadow_loss`). Both are detached so logging does not retain the graph; training still flows
        # through the live aux term inside `output.loss`. `last_shadow_loss` is the DDP/FSDP-friendly place to read the
        # metric, because wrappers may drop non-field output attributes.
        model = get_peft_model(make_llama_causal(), ShadowConfig(task_type="CAUSAL_LM", auxiliary_loss_weight=0.5))
        ids = torch.randint(0, 128, (2, 6))
        out = model(input_ids=ids, labels=ids.clone())
        assert out.shadow_loss is not None
        assert out.shadow_loss.numel() == 1
        assert not out.shadow_loss.requires_grad
        assert model.base_model.last_shadow_loss is out.shadow_loss
        # The live (non-detached) aux term is inside `output.loss`, so backward still trains the shadow backbone.
        out.loss.backward()
        backbone = model.base_model.shadow_backbone["default"]
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in backbone.parameters())
        # Without labels there is no shadow loss to report.
        assert getattr(model(input_ids=ids), "shadow_loss", None) is None
        assert model.base_model.last_shadow_loss is None

    def test_auxiliary_loss_trains_the_detached_shadow_model(self):
        # The auxiliary loss trains the standalone shadow prediction head(s^(0)) -- exactly what unload_shadow()
        # computes -- so after training the detached shadow model's own loss must go down. (Regression: the aux loss
        # must use the initial shadow state s^(0), not the final s^(L) which does not exist standalone.)
        model = get_peft_model(
            make_llama_causal(),
            ShadowConfig(task_type="CAUSAL_LM", auxiliary_loss_weight=1.0, modules_to_save=["lm_head"]),
        )
        ids = torch.randint(0, 128, (4, 8))
        labels = ids.clone()

        def detached_loss():
            detached = model.base_model.unload_shadow()
            detached.eval()
            with torch.no_grad():
                return float(detached(input_ids=ids, labels=labels).loss)

        before = detached_loss()
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
        model.train()
        for _ in range(30):
            opt.zero_grad()
            model(input_ids=ids, labels=labels).loss.backward()
            opt.step()
        assert detached_loss() < before - 0.1

    def test_untrained_adapter_is_noop(self):
        # shadow_up is zero-initialized, so an untrained shadow adapter must not change the base output.
        base = make_llama_causal()
        base.eval()
        ids = torch.randint(0, 128, (2, 6))
        with torch.no_grad():
            before = base(input_ids=ids).logits

        model = get_peft_model(base, ShadowConfig(task_type="CAUSAL_LM"))
        model.eval()
        with torch.no_grad():
            on = model(input_ids=ids).logits
            with model.disable_adapter():
                off = model(input_ids=ids).logits
        assert torch.allclose(on, off, atol=1e-6)
        assert torch.allclose(on, before, atol=1e-6)
        assert torch.allclose(off, before, atol=1e-6)

    def test_switches_adapters_but_rejects_multiple_active_adapters(self):
        # init_weights=False so the adapters are not no-ops: switching must change the logits, and switching back must
        # restore them. Also check that only the active adapter's model-level shadow modules are trainable.
        cfg = {"task_type": "CAUSAL_LM", "init_weights": False, "shadow_num_hidden_layers": 2}
        model = get_peft_model(make_llama_causal(), ShadowConfig(**cfg))
        model.add_adapter("other", ShadowConfig(**cfg))
        model.eval()
        ids = torch.randint(0, 128, (2, 6))
        tuner = model.base_model

        def shadow_trainable(adapter_name):
            return any(
                p.requires_grad
                for container in (tuner.shadow_backbone, tuner.shadow_projection, tuner.shadow_head)
                if adapter_name in container
                for p in container[adapter_name].parameters()
            )

        assert model.active_adapters == ["default"]
        assert shadow_trainable("default") and not shadow_trainable("other")

        with torch.no_grad():
            default_out = model(input_ids=ids).logits

        model.set_adapter("other")
        assert model.active_adapters == ["other"]
        assert shadow_trainable("other") and not shadow_trainable("default")
        with torch.no_grad():
            other_out = model(input_ids=ids).logits

        model.set_adapter("default")
        assert model.active_adapters == ["default"]
        assert shadow_trainable("default") and not shadow_trainable("other")
        with torch.no_grad():
            default_again = model(input_ids=ids).logits

        assert not torch.allclose(default_out, other_out, atol=1e-5)
        assert torch.allclose(default_out, default_again, atol=1e-6)

        with pytest.raises(ValueError, match="exactly one active adapter"):
            model.base_model.set_adapter(["default", "other"])
        assert model.active_adapters == ["default"]

    def test_switching_adapters_moves_trainability_to_the_new_shadow_modules(self):
        # The model-level shadow backbone/projection/head hang off the tuner, not off the base model, so the generic
        # BaseTuner.set_adapter traversal does not reach them. Without an explicit sync, switching adapters would leave
        # the old backbone trainable and the new one frozen, i.e. silently train the wrong weights.
        model = get_peft_model(make_llama_causal(), ShadowConfig(task_type="CAUSAL_LM", shadow_num_hidden_layers=2))
        model.add_adapter("other", ShadowConfig(task_type="CAUSAL_LM", shadow_num_hidden_layers=2))
        tuner = model.base_model

        def trainable(container, adapter_name):
            return any(p.requires_grad for p in container[adapter_name].parameters())

        assert trainable(tuner.shadow_backbone, "default")
        assert not trainable(tuner.shadow_backbone, "other")

        model.set_adapter("other")
        assert not trainable(tuner.shadow_backbone, "default")
        assert trainable(tuner.shadow_backbone, "other")

        # Only the newly activated adapter's backbone may receive gradients.
        ids = torch.randint(0, 128, (2, 6))
        model(input_ids=ids, labels=ids.clone()).loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in tuner.shadow_backbone["other"].parameters())
        assert all(p.grad is None for p in tuner.shadow_backbone["default"].parameters())

        # inference_mode freezes the activated adapter instead of training it.
        model.set_adapter("default", inference_mode=True)
        assert not trainable(tuner.shadow_backbone, "default")
        assert not trainable(tuner.shadow_backbone, "other")

    def test_merge_raises(self):
        # ShadowPEFT cannot be merged: it must raise an explicit error rather than silently doing the wrong thing.
        model = get_peft_model(make_llama_causal(), ShadowConfig(task_type="CAUSAL_LM"))
        with pytest.raises(NotImplementedError):
            model.merge_and_unload()
        with pytest.raises(NotImplementedError):
            model.base_model.merge_adapter()

    def test_save_includes_shadow_modules_but_not_frozen_head(self, tmp_path):
        # For causal LM the shadow path reuses the frozen base LM head (not a stored shadow_head). Saving therefore
        # omits `.shadow_head.*`; after reload the head is rebuilt by resolving to the new base model's lm_head.
        base = make_llama_causal()
        base_sd = copy.deepcopy(base.state_dict())
        model = get_peft_model(base, ShadowConfig(task_type="CAUSAL_LM", init_weights=False))
        ids = torch.randint(0, 128, (2, 6))
        model.eval()

        assert "default" not in model.base_model.shadow_head
        assert model.base_model._resolve_shadow_head("default") is model.get_output_embeddings()
        with torch.no_grad():
            ref = model.base_model.unload_shadow()(input_ids=ids).logits

        model.save_pretrained(tmp_path)
        with safe_open(tmp_path / SAFETENSORS_WEIGHTS_NAME, framework="pt") as f:
            keys = list(f.keys())
        assert any(".shadow_backbone." in key for key in keys)
        assert any("shadow_down" in key for key in keys)
        assert any("shadow_update_transform" in key for key in keys)
        assert not any(".shadow_head." in key for key in keys)

        base2 = make_llama_causal()
        base2.load_state_dict(base_sd)
        loaded = PeftModel.from_pretrained(base2, tmp_path)
        loaded.eval()
        # The checkpoint had no shadow_head; the LM head is taken from the reloaded base model.
        assert "default" not in loaded.base_model.shadow_head
        assert loaded.base_model._resolve_shadow_head("default") is loaded.get_output_embeddings()
        with torch.no_grad():
            got = loaded.base_model.unload_shadow()(input_ids=ids).logits
        assert torch.allclose(ref, got, atol=1e-6)

    def test_save_includes_trainable_lm_head(self, tmp_path):
        # With modules_to_save=["lm_head"], the causal-LM head is trained and stored through PEFT's normal
        # modules_to_save path (as `lm_head` keys), not as a ShadowPEFT `shadow_head` module.
        model = get_peft_model(
            make_llama_causal(),
            ShadowConfig(task_type="CAUSAL_LM", modules_to_save=["lm_head"]),
        )
        head = model.get_output_embeddings()
        assert any(p.requires_grad for p in head.parameters())
        model.save_pretrained(tmp_path)
        with safe_open(tmp_path / SAFETENSORS_WEIGHTS_NAME, framework="pt") as f:
            keys = list(f.keys())
        assert any("lm_head" in key for key in keys)
        assert not any(".shadow_head." in key for key in keys)

    def test_requires_two_layers(self):
        # A single decoder block means the shadow carrier has no loop to ride; injection needs >= 2 blocks.
        # Target only one block of the tiny 2-layer model so entry == exit.
        model = get_peft_model(make_llama_causal(), ShadowConfig(task_type="CAUSAL_LM", layers_to_transform=[0]))
        ids = torch.randint(0, 128, (2, 6))
        # A single wrapped block still runs (entry == exit); just assert it does not crash.
        out = model(input_ids=ids, labels=ids.clone())
        assert out.logits.shape == (2, 6, model.config.vocab_size)

    def test_wrapped_layer_delegates_base_attributes(self):
        # The base model's forward reads attributes off the decoder block it iterates over (e.g. newer transformers
        # Qwen3 reads `decoder_layer.attention_type`). The wrapping ShadowLayer must expose the base block's attributes.
        base = make_llama_causal()
        for layer in base.model.layers:
            layer.some_marker_attr = "full_attention"
        model = get_peft_model(base, ShadowConfig(task_type="CAUSAL_LM"))
        wrapped = next(m for m in model.modules() if isinstance(m, ShadowLayer))
        assert wrapped.some_marker_attr == "full_attention"
        ids = torch.randint(0, 128, (2, 6))
        model(input_ids=ids, labels=ids.clone()).loss.backward()


class TestShadowKVCache:
    """Dual KV cache (base + shadow) incremental decode must match full-sequence recompute."""

    def test_generate_cached_matches_uncached(self):
        model = get_peft_model(make_llama_causal(), ShadowConfig(task_type="CAUSAL_LM", init_weights=False))
        model.eval()
        ids = torch.randint(0, 128, (2, 5))
        with torch.no_grad():
            cached = model.generate(input_ids=ids, max_new_tokens=5, use_cache=True, do_sample=False)
            uncached = model.generate(input_ids=ids, max_new_tokens=5, use_cache=False, do_sample=False)
        assert torch.equal(cached, uncached)

    def test_generate_cached_matches_uncached_with_projection(self):
        # Mirror shadow with a smaller hidden size must still align under caching.
        base = make_llama_causal()
        model = get_peft_model(
            base,
            ShadowConfig(
                task_type="CAUSAL_LM",
                shadow_num_hidden_layers=1,
                shadow_hidden_size=base.config.hidden_size // 2,
                init_weights=False,
            ),
        )
        model.eval()
        ids = torch.randint(0, 128, (1, 4))
        with torch.no_grad():
            cached = model.generate(input_ids=ids, max_new_tokens=6, use_cache=True, do_sample=False)
            uncached = model.generate(input_ids=ids, max_new_tokens=6, use_cache=False, do_sample=False)
        assert torch.equal(cached, uncached)

    def test_prefill_returns_shadow_cache(self):
        model = get_peft_model(make_llama_causal(), ShadowConfig(task_type="CAUSAL_LM"))
        model.eval()
        ids = torch.randint(0, 128, (1, 4))
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True)
        assert isinstance(out.past_key_values, ShadowCache)
        assert isinstance(out.past_key_values, Cache)
        assert not out.past_key_values.is_compileable
        assert len(out.past_key_values) == len(out.past_key_values.base)
        assert out.past_key_values.get_seq_length() == 4
        assert out.past_key_values.base is not None
        assert out.past_key_values.shadow is not None
        assert out.past_key_values.shadow.get_seq_length() == 4

    def test_prefill_then_decode_logits_match_full_sequence(self):
        model = get_peft_model(make_llama_causal(), ShadowConfig(task_type="CAUSAL_LM", init_weights=False))
        model.eval()
        ids = torch.randint(0, 128, (1, 4))
        next_token = torch.tensor([[77]])
        full_ids = torch.cat([ids, next_token], dim=1)
        with torch.no_grad():
            out_prefill = model(input_ids=ids, use_cache=True)
            assert isinstance(out_prefill.past_key_values, ShadowCache)
            out_decode = model(input_ids=next_token, past_key_values=out_prefill.past_key_values, use_cache=True)
            out_full = model(input_ids=full_ids, use_cache=False)
        assert torch.allclose(out_decode.logits[:, -1, :], out_full.logits[:, -1, :], atol=1e-4)

    def test_step_by_step_logits_alignment(self):
        # Prefill once, then decode token-by-token with the dual cache; each step's logits must match a full-sequence
        # recompute of the growing prefix (proves inject/update stay correct under incremental decoding).
        model = get_peft_model(make_llama_causal(), ShadowConfig(task_type="CAUSAL_LM", init_weights=False))
        model.eval()
        prefix = torch.randint(0, 128, (1, 3))
        max_new_tokens = 5

        with torch.no_grad():
            out = model(input_ids=prefix, use_cache=True)
            cache = out.past_key_values
            assert isinstance(cache, ShadowCache)
            cached_logits = out.logits[:, -1, :]
            full_logits = model(input_ids=prefix, use_cache=False).logits[:, -1, :]
            assert torch.allclose(cached_logits, full_logits, atol=1e-4)

            generated = []
            for step in range(max_new_tokens):
                next_tok = cached_logits.argmax(dim=-1, keepdim=True)
                generated.append(next_tok.item())
                full_ids = torch.cat([prefix] + [torch.tensor([[t]]) for t in generated], dim=1)
                full_logits = model(input_ids=full_ids, use_cache=False).logits[:, -1, :]
                if step < max_new_tokens - 1:
                    out = model(input_ids=next_tok, past_key_values=cache, use_cache=True)
                    cache = out.past_key_values
                    cached_logits = out.logits[:, -1, :]
                    assert torch.allclose(cached_logits, full_logits, atol=1e-4), f"mismatch at step {step}"

            # Manual greedy path should agree with generate(use_cache=True) up to an early EOS stop.
            gen = model.generate(input_ids=prefix, max_new_tokens=max_new_tokens, use_cache=True, do_sample=False)
            gen_tokens = gen[0, prefix.shape[1] :].tolist()
            # Guard against a trivial pass if generation immediately hits EOS.
            assert len(gen_tokens) >= 2
            assert generated[: len(gen_tokens)] == gen_tokens

    def test_uncached_forward_does_not_return_shadow_cache(self):
        model = get_peft_model(make_llama_causal(), ShadowConfig(task_type="CAUSAL_LM"))
        model.eval()
        ids = torch.randint(0, 128, (1, 5))
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=False)
        # With caching off we must not wrap a dual ShadowCache. Depending on transformers version the base model
        # may return `past_key_values=None` or an empty Cache; either is fine as long as nothing is stored.
        assert not isinstance(getattr(out, "past_key_values", None), ShadowCache)
        assert out.past_key_values is None or (
            hasattr(out.past_key_values, "get_seq_length") and out.past_key_values.get_seq_length() == 0
        )

    def test_disable_adapter_uses_plain_base_cache(self):
        model = get_peft_model(make_llama_causal(), ShadowConfig(task_type="CAUSAL_LM"))
        model.eval()
        ids = torch.randint(0, 128, (1, 4))
        with torch.no_grad():
            with model.disable_adapter():
                out = model(input_ids=ids, use_cache=True)
        # Shadow path inactive: past should be a plain base cache, not a ShadowCache.
        assert not isinstance(out.past_key_values, ShadowCache)
        assert out.past_key_values is not None
        assert out.past_key_values.get_seq_length() == 4

    def test_explicit_shadow_model_cached_generate(self, tmp_path):
        # Unlike `test_generate_cached_matches_uncached_with_projection` (which uses the default `shadow_model="mirror"`
        # builder with a smaller hidden size), this exercises loading an explicit external shadow backbone via
        # `shadow_model=<path>` (`_load_shadow_backbone`) and checks that dual-cache generate still matches uncached.

        base = make_llama_causal()
        shadow_hidden = base.config.hidden_size // 2
        shadow_cfg = LlamaConfig(
            vocab_size=base.config.vocab_size,
            hidden_size=shadow_hidden,
            intermediate_size=2 * shadow_hidden,
            num_hidden_layers=1,
            num_attention_heads=base.config.num_attention_heads,
            num_key_value_heads=getattr(base.config, "num_key_value_heads", base.config.num_attention_heads),
            max_position_embeddings=base.config.max_position_embeddings,
        )
        LlamaModel(shadow_cfg).save_pretrained(tmp_path)
        model = get_peft_model(
            base,
            ShadowConfig(task_type="CAUSAL_LM", shadow_model=str(tmp_path), init_weights=False),
        )
        model.eval()
        ids = torch.randint(0, 128, (1, 4))
        with torch.no_grad():
            cached = model.generate(input_ids=ids, max_new_tokens=4, use_cache=True, do_sample=False)
            uncached = model.generate(input_ids=ids, max_new_tokens=4, use_cache=False, do_sample=False)
        assert torch.equal(cached, uncached)

    def test_shadow_cache_reorder_and_crop(self):
        model = get_peft_model(make_llama_causal(), ShadowConfig(task_type="CAUSAL_LM"))
        model.eval()
        ids = torch.randint(0, 128, (2, 4))
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True)
        cache = out.past_key_values
        assert isinstance(cache, ShadowCache)
        cache.reorder_cache(torch.tensor([1, 0]))
        assert cache.get_seq_length() == 4
        cache.crop(-1)
        assert cache.get_seq_length() == 3
        assert cache.shadow.get_seq_length() == 3


class TestShadowSequenceClassification:
    def test_classifier_head_trainable_by_default(self):
        model = get_peft_model(make_llama_seqcls(num_labels=3), ShadowConfig(task_type="SEQ_CLS"))
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        assert any("shadow_head" in n for n in trainable)

    def test_classification_pooling_respects_padding_side(self, monkeypatch):
        # Regression test for https://github.com/huggingface/peft/issues/3620.
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_labels=3,
        )
        base = AutoModelForSequenceClassification.from_config(config)
        model = get_peft_model(
            base,
            ShadowConfig(task_type="SEQ_CLS", shadow_num_hidden_layers=1),
        )
        tuner = model.base_model
        tuner.shadow_head["default"] = torch.nn.Identity()

        hidden = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        labels = torch.tensor([0, 2])
        detached = tuner.unload_shadow()
        detached.shadow_hidden_projection = torch.nn.Identity()
        detached.head = torch.nn.Identity()
        monkeypatch.setattr(
            detached.backbone,
            "forward",
            lambda *args, **kwargs: BaseModelOutput(last_hidden_state=hidden),
        )

        cases = (
            (torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]]), torch.tensor([3, 3])),
            (torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]]), torch.tensor([1, 2])),
            (torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]]), torch.tensor([0, 3])),
            (None, torch.tensor([3, 3])),
        )
        batch_idx = torch.arange(hidden.shape[0])
        for attention_mask, expected_indices in cases:
            expected_logits = hidden[batch_idx, expected_indices]

            tuner._seed_shadow_state = hidden
            actual_loss = tuner.shadow_auxiliary_loss(labels, attention_mask=attention_mask)
            expected_loss = torch.nn.functional.cross_entropy(expected_logits, labels)
            torch.testing.assert_close(actual_loss, expected_loss)

            output = detached(
                inputs_embeds=torch.zeros(2, 4, config.hidden_size),
                attention_mask=attention_mask,
            )
            torch.testing.assert_close(output.logits, expected_logits)

    def test_unload_shadow_is_a_classifier(self):
        # For SEQ_CLS the standalone shadow model pools the last token and returns per-example class logits (not
        # per-position), so the shadow path's classification performance can be evaluated on its own.

        model = get_peft_model(make_llama_seqcls(num_labels=3), ShadowConfig(task_type="SEQ_CLS"))
        detached = model.base_model.unload_shadow()
        assert isinstance(detached, DetachedShadowModel)
        assert detached.is_classification
        assert not detached.can_generate()
        ids = torch.randint(1, 128, (4, 6))
        am = torch.ones_like(ids)
        with torch.no_grad():
            out = detached(input_ids=ids, attention_mask=am)
        assert isinstance(out, SequenceClassifierOutput)
        assert out.logits.shape == (4, 3)


class TestShadowBackboneVariants:
    def test_smaller_shadow_hidden_size_inserts_projection(self):
        base = make_llama_causal()
        shadow_hidden = base.config.hidden_size // 2
        model = get_peft_model(
            base,
            ShadowConfig(
                task_type="CAUSAL_LM",
                shadow_num_hidden_layers=2,
                shadow_hidden_size=shadow_hidden,
            ),
        )
        projection = model.base_model.shadow_projection["default"]
        assert isinstance(projection, torch.nn.Linear)
        assert (projection.in_features, projection.out_features) == (shadow_hidden, base.config.hidden_size)
        ids = torch.randint(0, 128, (2, 6))
        out = model(input_ids=ids, labels=ids.clone())
        out.loss.backward()

    def test_matching_shadow_hidden_size_uses_identity(self):
        model = get_peft_model(
            make_llama_causal(),
            ShadowConfig(task_type="CAUSAL_LM", shadow_num_hidden_layers=2),
        )
        assert isinstance(model.base_model.shadow_projection["default"], torch.nn.Identity)

    def test_pretrained_projected_shadow_checkpoint(self, tmp_path):
        # A "projected" shadow checkpoint (model_type == causal_lm_with_hidden_projection, e.g.
        # shadow-llm/Qwen3-0.6B-H8B) bundles a small backbone + a trained shadow_hidden -> base_hidden projection.
        # ShadowPEFT must load the pretrained backbone and reuse the trained projection.

        base = make_llama_causal()
        base_hidden = base.config.hidden_size
        shadow_hidden = base_hidden // 2
        vocab = base.config.vocab_size
        inner_cfg = LlamaConfig(
            vocab_size=vocab,
            hidden_size=shadow_hidden,
            intermediate_size=2 * shadow_hidden,
            num_hidden_layers=2,
            num_attention_heads=base.config.num_attention_heads,
            num_key_value_heads=getattr(base.config, "num_key_value_heads", base.config.num_attention_heads),
            max_position_embeddings=base.config.max_position_embeddings,
        )
        shadow_backbone = LlamaModel(inner_cfg)
        projection = torch.nn.Linear(shadow_hidden, base_hidden, bias=False)

        state = {f"shadow_model.{k}": v for k, v in shadow_backbone.state_dict().items()}
        state["shadow_hidden_projection.weight"] = projection.weight.data.clone()
        save_file(state, str(tmp_path / "model.safetensors"))
        raw_config = {
            "model_type": "causal_lm_with_hidden_projection",
            "base_hidden_size": base_hidden,
            "hidden_size": base_hidden,
            "shadow_model_class": "transformers.models.llama.modeling_llama:LlamaModel",
            "shadow_model_config": inner_cfg.to_dict(),
        }
        (tmp_path / "config.json").write_text(json.dumps(raw_config))

        model = get_peft_model(base, ShadowConfig(task_type="CAUSAL_LM", shadow_model=str(tmp_path)))

        proj = model.base_model.shadow_projection["default"]
        assert isinstance(proj, torch.nn.Linear)
        assert (proj.in_features, proj.out_features) == (shadow_hidden, base_hidden)
        assert torch.allclose(proj.weight.float(), projection.weight.float(), atol=1e-5)
        loaded_backbone = model.base_model.shadow_backbone["default"]
        assert torch.allclose(
            loaded_backbone.layers[0].self_attn.q_proj.weight.float(),
            shadow_backbone.layers[0].self_attn.q_proj.weight.float(),
            atol=1e-5,
        )
        # A pretrained shadow backbone keeps its (large) embedding table frozen -- it must not inflate the trainable
        # parameter count or receive gradients.
        embed = loaded_backbone.get_input_embeddings()
        assert not embed.weight.requires_grad
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        assert not any("shadow_backbone" in n and "embed_tokens" in n for n in trainable)
        ids = torch.randint(0, 128, (2, 6))
        model(input_ids=ids, labels=ids.clone()).loss.backward()
        assert embed.weight.grad is None

    def test_unload_shadow_returns_standalone_generatable_model(self):
        # unload_shadow returns the standalone shadow network (backbone + projection + head) as a causal LM. This is how
        # the shadow path's own performance is evaluated, independent of the base model.
        model = get_peft_model(
            make_llama_causal(),
            ShadowConfig(task_type="CAUSAL_LM", shadow_num_hidden_layers=2),
        )
        detached = model.base_model.unload_shadow()
        assert isinstance(detached, DetachedShadowModel)
        assert detached.can_generate()
        # Default is copy=False: share modules with the PEFT model (like merge_and_unload, no extra memory).
        assert detached.backbone is model.base_model.shadow_backbone["default"]
        ids = torch.randint(0, 128, (2, 5))
        with torch.no_grad():
            out = detached(input_ids=ids)
        # head(projection(backbone(x))) -> vocab logits (CausalLMOutputWithPast)
        assert out.logits.shape == (2, 5, model.config.vocab_size)
        # It behaves like a normal causal LM, so it can generate with KV caching.
        gen = detached.generate(input_ids=ids[:, :3], max_new_tokens=3, do_sample=False)
        assert gen.shape[1] == 6

    def test_unload_shadow_copy_returns_independent_modules(self):
        model = get_peft_model(
            make_llama_causal(),
            ShadowConfig(task_type="CAUSAL_LM", shadow_num_hidden_layers=2),
        )
        detached = model.base_model.unload_shadow(copy=True)
        assert detached.backbone is not model.base_model.shadow_backbone["default"]
        assert detached.shadow_hidden_projection is not model.base_model.shadow_projection["default"]
        ids = torch.randint(0, 128, (2, 5))
        with torch.no_grad():
            out = detached(input_ids=ids)
        assert out.logits.shape == (2, 5, model.config.vocab_size)

    def test_unload_shadow_copy_is_required_for_a_complete_checkpoint(self):
        # A mirrored backbone shares the frozen base input embeddings through a reference that is not a submodule, so
        # the shared table is missing from the state dict. copy=True re-attaches a private copy and saves everything.
        model = get_peft_model(
            make_llama_causal(),
            ShadowConfig(task_type="CAUSAL_LM", shadow_num_hidden_layers=2),
        )
        assert model.base_model._shadow_share_embeddings["default"] is True

        shared = model.base_model.unload_shadow()
        assert "backbone.embed_tokens.weight" not in shared.state_dict()

        copied = model.base_model.unload_shadow(copy=True)
        assert "backbone.embed_tokens.weight" in copied.state_dict()

    def test_unload_shadow_applies_projection(self):
        # With a smaller shadow hidden size the standalone model must apply the trained projection so the head receives
        # the correct (base) hidden width.
        base = make_llama_causal()
        model = get_peft_model(
            base,
            ShadowConfig(
                task_type="CAUSAL_LM",
                shadow_num_hidden_layers=2,
                shadow_hidden_size=base.config.hidden_size // 2,
            ),
        )
        detached = model.base_model.unload_shadow()
        ids = torch.randint(0, 128, (2, 5))
        with torch.no_grad():
            out = detached(input_ids=ids)
        assert out.logits.shape == (2, 5, model.config.vocab_size)


class _TinyDiTConfig:
    model_type = "tiny_dit"
    in_channels = 8
    num_attention_heads = 2
    attention_head_dim = 4

    @property
    def hidden_dim(self):
        return self.num_attention_heads * self.attention_head_dim

    def to_dict(self):
        return {
            "model_type": self.model_type,
            "num_attention_heads": self.num_attention_heads,
            "attention_head_dim": self.attention_head_dim,
        }


class _TinyDiTBlock(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states):
        return hidden_states + self.linear(self.norm(hidden_states))


class _TinyDiT(torch.nn.Module):
    """A diffusers-style transformer: `single_transformer_blocks`, no token ids, no `inputs_embeds`.

    Stands in for Flux: the top-level input is a latent that is embedded inside `forward`, so the shadow state cannot
    be seeded from the model's inputs and has to be built from the first wrapped block's hidden states instead.
    """

    def __init__(self, num_layers=3):
        super().__init__()
        self.config = _TinyDiTConfig()
        hidden_dim = self.config.hidden_dim
        self.x_embedder = torch.nn.Linear(hidden_dim, hidden_dim)
        self.single_transformer_blocks = torch.nn.ModuleList([_TinyDiTBlock(hidden_dim) for _ in range(num_layers)])
        self.proj_out = torch.nn.Linear(hidden_dim, hidden_dim)
        self.gradient_checkpointing = False

    def forward(self, hidden_states):
        hidden_states = self.x_embedder(hidden_states)
        for block in self.single_transformer_blocks:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(block.__call__, hidden_states, use_reentrant=False)
            else:
                hidden_states = block(hidden_states)
        return self.proj_out(hidden_states)


class TestShadowDiffusionTransformer:
    """ShadowPEFT on a diffusers-style transformer, where `s^(0)` is seeded from the first wrapped block."""

    @staticmethod
    def _make_peft_model():
        torch.manual_seed(0)
        return get_peft_model(
            _TinyDiT(),
            ShadowConfig(
                target_modules=r"single_transformer_blocks\.\d+$",
                share_embeddings=False,
                init_weights=False,
            ),
        )

    def test_seeds_the_shadow_state_from_the_first_wrapped_block(self):
        model = self._make_peft_model()
        assert len([m for m in model.modules() if isinstance(m, ShadowLayer)]) == 3
        latents = torch.randn(2, 5, model.base_model.model.config.hidden_dim)
        out = model(hidden_states=latents)
        out.pow(2).mean().backward()
        backbone = model.base_model.shadow_backbone["default"]
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in backbone.parameters())

    def test_unload_shadow_raises(self):
        model = self._make_peft_model()
        with pytest.raises(NotImplementedError, match="not supported for Diffusers models"):
            model.base_model.unload_shadow(copy=True)

    def test_non_flux_diffusers_model_uses_mlp_fallback(self):
        diffusers = pytest.importorskip("diffusers")
        transformer_cls = getattr(diffusers, "SanaTransformer2DModel", None)
        if transformer_cls is None:
            pytest.skip("The installed Diffusers version does not provide SanaTransformer2DModel.")

        base = transformer_cls(
            in_channels=4,
            out_channels=4,
            num_attention_heads=2,
            attention_head_dim=4,
            num_layers=2,
            num_cross_attention_heads=2,
            cross_attention_head_dim=4,
            cross_attention_dim=8,
            caption_channels=8,
            mlp_ratio=2.0,
            sample_size=4,
            patch_size=1,
        )
        model = get_peft_model(
            base,
            ShadowConfig(
                target_modules=r"transformer_blocks\.\d+$",
                share_embeddings=False,
                shadow_hidden_size=4,
                shadow_intermediate_size=6,
                shadow_num_hidden_layers=2,
                init_weights=False,
            ),
        )
        backbone = model.base_model.shadow_backbone["default"]
        assert backbone.__class__.__name__ == "_TokenShadowBackbone"
        assert len(backbone.blocks) == 2
        assert backbone.blocks[0][1].out_features == 6

        inputs = {
            "hidden_states": torch.randn(1, 4, 4, 4),
            "encoder_hidden_states": torch.randn(1, 3, 8),
            "timestep": torch.tensor([1]),
            "return_dict": False,
        }
        output = model(**inputs)[0]
        assert output.shape == inputs["hidden_states"].shape
        output.square().mean().backward()
        assert any(parameter.grad is not None for parameter in backbone.parameters())

        with pytest.raises(NotImplementedError, match="not supported for Diffusers models"):
            model.base_model.unload_shadow()

    def test_flux2_uses_architecture_backend(self):
        diffusers = pytest.importorskip("diffusers")
        transformer_cls = getattr(diffusers, "Flux2Transformer2DModel", None)
        if transformer_cls is None:
            pytest.skip("The installed Diffusers version does not provide Flux2Transformer2DModel.")

        base = transformer_cls(
            in_channels=4,
            out_channels=4,
            num_layers=1,
            num_single_layers=2,
            attention_head_dim=8,
            num_attention_heads=1,
            joint_attention_dim=8,
            timestep_guidance_channels=8,
            mlp_ratio=2.0,
            axes_dims_rope=(2, 2, 2, 2),
            guidance_embeds=False,
        )
        base.enable_gradient_checkpointing()
        model = get_peft_model(
            base,
            ShadowConfig(
                target_modules=r"single_transformer_blocks\.\d+$",
                share_embeddings=True,
                shadow_num_hidden_layers=1,
            ),
        )
        inputs = {
            "hidden_states": torch.randn(1, 4, 4),
            "encoder_hidden_states": torch.randn(1, 3, 8),
            "timestep": torch.tensor([0.5]),
            "img_ids": torch.zeros(1, 4, 4),
            "txt_ids": torch.zeros(1, 3, 4),
            "guidance": None,
            "return_dict": False,
        }
        with torch.no_grad():
            attached_output = model(**inputs)[0]
        assert attached_output.shape == inputs["hidden_states"].shape

        shadow_backbone = model.base_model.shadow_backbone["default"]
        assert isinstance(shadow_backbone, DetachedFluxShadowModel)
        assert shadow_backbone.model.x_embedder.shared_module is model.base_model.model.x_embedder
        assert not any("x_embedder.weight" in key for key in shadow_backbone.state_dict())
        assert shadow_backbone.model.config.num_layers == 1
        assert shadow_backbone.model.config.num_single_layers == 2

        model.zero_grad()
        attached_output = model(**inputs)[0]
        attached_output.square().mean().backward()
        assert next(shadow_backbone.model.single_transformer_blocks.parameters()).grad is not None
        assert shadow_backbone.model.x_embedder.weight.grad is None

        with pytest.raises(NotImplementedError, match="not supported for Diffusers models"):
            model.base_model.unload_shadow()

    def test_explicit_flux2_shadow_checkpoint_attached_and_roundtrip(self, tmp_path):
        diffusers = pytest.importorskip("diffusers")
        transformer_cls = getattr(diffusers, "Flux2Transformer2DModel", None)
        if transformer_cls is None:
            pytest.skip("The installed Diffusers version does not provide Flux2Transformer2DModel.")

        common = {
            "in_channels": 4,
            "out_channels": 4,
            "attention_head_dim": 8,
            "joint_attention_dim": 8,
            "timestep_guidance_channels": 8,
            "axes_dims_rope": (2, 2, 2, 2),
            "guidance_embeds": False,
        }
        base = transformer_cls(num_layers=2, num_single_layers=2, num_attention_heads=2, mlp_ratio=2.0, **common)
        shadow = transformer_cls(num_layers=1, num_single_layers=1, num_attention_heads=1, mlp_ratio=2.0, **common)
        shadow_path = tmp_path / "shadow"
        shadow.save_pretrained(shadow_path)
        model = get_peft_model(
            base,
            ShadowConfig(
                target_modules=r"single_transformer_blocks\.\d+$",
                shadow_model=str(shadow_path),
                share_embeddings=False,
            ),
        )
        backbone = model.base_model.shadow_backbone["default"]
        projection = model.base_model.shadow_projection["default"]
        assert isinstance(backbone, DetachedFluxShadowModel)
        assert (projection.in_features, projection.out_features) == (8, 16)
        assert torch.equal(projection.weight[:8], torch.eye(8))
        assert not backbone.model.x_embedder.weight.requires_grad
        assert next(backbone.model.single_transformer_blocks.parameters()).requires_grad
        assert next(backbone.model.norm_out.parameters()).requires_grad

        inputs = {
            "hidden_states": torch.randn(1, 4, 4),
            "encoder_hidden_states": torch.randn(1, 3, 8),
            "timestep": torch.tensor([0.5]),
            "img_ids": torch.zeros(1, 4, 4),
            "txt_ids": torch.zeros(1, 3, 4),
            "guidance": None,
            "return_dict": False,
        }
        output = model(**inputs)[0]
        output.square().mean().backward()
        assert next(backbone.model.single_transformer_blocks.parameters()).grad is not None
        with pytest.raises(NotImplementedError, match="not supported for Diffusers models"):
            model.base_model.unload_shadow()

        adapter_path = tmp_path / "adapter"
        model.save_pretrained(adapter_path)
        reloaded = PeftModel.from_pretrained(
            transformer_cls(num_layers=2, num_single_layers=2, num_attention_heads=2, mlp_ratio=2.0, **common),
            adapter_path,
            is_trainable=True,
        )
        assert isinstance(reloaded.base_model.shadow_backbone["default"], DetachedFluxShadowModel)
        assert reloaded(**inputs)[0].shape == inputs["hidden_states"].shape

    def test_explicit_flux2_shadow_rejects_incompatible_contract(self, tmp_path):
        diffusers = pytest.importorskip("diffusers")
        transformer_cls = getattr(diffusers, "Flux2Transformer2DModel", None)
        if transformer_cls is None:
            pytest.skip("The installed Diffusers version does not provide Flux2Transformer2DModel.")
        base = transformer_cls(
            in_channels=4,
            out_channels=4,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=8,
            num_attention_heads=1,
            joint_attention_dim=8,
            timestep_guidance_channels=8,
            mlp_ratio=2.0,
            axes_dims_rope=(2, 2, 2, 2),
            guidance_embeds=False,
        )
        incompatible = transformer_cls.from_config({**dict(base.config), "out_channels": 8})
        incompatible.save_pretrained(tmp_path)
        with pytest.raises(ValueError, match="incompatible `out_channels`"):
            get_peft_model(
                base,
                ShadowConfig(
                    target_modules=r"single_transformer_blocks\.\d+$",
                    shadow_model=str(tmp_path),
                ),
            )
        with pytest.raises(ValueError, match="mirror-only overrides"):
            ShadowConfig(shadow_model=str(tmp_path), shadow_hidden_size=8)

    def test_compact_flux2_shadow_uses_width_overrides_and_structured_pretrained_weights(self):
        diffusers = pytest.importorskip("diffusers")
        transformer_cls = getattr(diffusers, "Flux2Transformer2DModel", None)
        if transformer_cls is None:
            pytest.skip("The installed Diffusers version does not provide Flux2Transformer2DModel.")

        base = transformer_cls(
            in_channels=4,
            out_channels=4,
            num_layers=2,
            num_single_layers=2,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=8,
            timestep_guidance_channels=8,
            mlp_ratio=2.0,
            axes_dims_rope=(2, 2, 2, 2),
            guidance_embeds=False,
        )
        source_single = base.single_transformer_blocks[-1]
        with torch.no_grad():
            source_single.attn.to_qkv_mlp_proj.weight.copy_(
                torch.arange(source_single.attn.to_qkv_mlp_proj.weight.numel()).reshape_as(
                    source_single.attn.to_qkv_mlp_proj.weight
                )
            )
            source_single.attn.to_out.weight.copy_(
                torch.arange(source_single.attn.to_out.weight.numel()).reshape_as(source_single.attn.to_out.weight)
            )

        base_param_count = sum(parameter.numel() for parameter in base.parameters())
        model = get_peft_model(
            base,
            ShadowConfig(
                target_modules=r"single_transformer_blocks\.\d+$",
                share_embeddings=True,
                shadow_num_hidden_layers=1,
                shadow_hidden_size=8,
                shadow_num_attention_heads=1,
                shadow_intermediate_size=8,
            ),
        )
        backbone = model.base_model.shadow_backbone["default"]
        projection = model.base_model.shadow_projection["default"]
        reduced = backbone.model

        assert reduced.config.num_attention_heads == 1
        assert reduced.config.attention_head_dim == 8
        assert reduced.config.mlp_ratio == 1.0
        assert not hasattr(reduced.x_embedder, "shared_module")
        assert (projection.in_features, projection.out_features) == (8, 16)
        assert torch.equal(projection.weight[:8], torch.eye(8))
        assert torch.count_nonzero(projection.weight[8:]) == 0
        assert sum(parameter.numel() for parameter in reduced.parameters()) < base_param_count / 4

        source_qkv_mlp = source_single.attn.to_qkv_mlp_proj.weight
        expected_qkv_mlp = torch.cat(
            (
                source_qkv_mlp[0:8, :8],
                source_qkv_mlp[16:24, :8],
                source_qkv_mlp[32:40, :8],
                source_qkv_mlp[48:56, :8],
                source_qkv_mlp[80:88, :8],
            )
        )
        assert torch.equal(reduced.single_transformer_blocks[-1].attn.to_qkv_mlp_proj.weight, expected_qkv_mlp)
        expected_out = torch.cat(
            (source_single.attn.to_out.weight[:8, :8], source_single.attn.to_out.weight[:8, 16:24]), dim=1
        )
        assert torch.equal(reduced.single_transformer_blocks[-1].attn.to_out.weight, expected_out)

        inputs = {
            "hidden_states": torch.randn(1, 4, 4),
            "encoder_hidden_states": torch.randn(1, 3, 8),
            "timestep": torch.tensor([0.5]),
            "img_ids": torch.zeros(1, 4, 4),
            "txt_ids": torch.zeros(1, 3, 4),
            "guidance": None,
            "return_dict": False,
        }
        attached_output = model(**inputs)[0]
        assert attached_output.shape == inputs["hidden_states"].shape
        attached_output.square().mean().backward()
        assert next(backbone.model.single_transformer_blocks.parameters()).grad is not None
        assert model.base_model.model.x_embedder.weight.grad is None

        with pytest.raises(NotImplementedError, match="not supported for Diffusers models"):
            model.base_model.unload_shadow(copy=True)

    @pytest.mark.parametrize(
        ("hidden_size", "num_heads", "match"),
        [
            (10, 1, "preserve the base attention head dimension"),
            (None, 3, "must be between 1"),
        ],
    )
    def test_compact_flux2_shadow_rejects_invalid_width(self, hidden_size, num_heads, match):
        diffusers = pytest.importorskip("diffusers")
        transformer_cls = getattr(diffusers, "Flux2Transformer2DModel", None)
        if transformer_cls is None:
            pytest.skip("The installed Diffusers version does not provide Flux2Transformer2DModel.")

        base = transformer_cls(
            in_channels=4,
            out_channels=4,
            num_layers=1,
            num_single_layers=2,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=8,
            timestep_guidance_channels=8,
            mlp_ratio=2.0,
            axes_dims_rope=(2, 2, 2, 2),
            guidance_embeds=False,
        )
        with pytest.raises(ValueError, match=match):
            get_peft_model(
                base,
                ShadowConfig(
                    target_modules=r"single_transformer_blocks\.\d+$",
                    shadow_hidden_size=hidden_size,
                    shadow_num_attention_heads=num_heads,
                ),
            )

    def test_gradient_checkpointing_matches_the_uncheckpointed_run(self):
        # Gradient checkpointing runs the wrapped blocks a second time during recomputation, so the deferred seeding
        # must re-run the shadow backbone as well -- otherwise the recomputed graph saves fewer tensors than the
        # original one and torch raises. Results must be identical either way.
        latents = torch.randn(2, 5, _TinyDiTConfig().hidden_dim)

        def run(use_gradient_checkpointing):
            model = self._make_peft_model()
            model.train()
            model.base_model.model.gradient_checkpointing = use_gradient_checkpointing
            out = model(hidden_states=latents)
            out.pow(2).mean().backward()
            grads = {n: p.grad for n, p in model.named_parameters() if "shadow_" in n and p.grad is not None}
            return out, grads

        expected, expected_grads = run(use_gradient_checkpointing=False)
        actual, actual_grads = run(use_gradient_checkpointing=True)

        assert torch.allclose(actual, expected, atol=1e-6)
        assert actual_grads.keys() == expected_grads.keys()
        assert any("shadow_backbone" in name for name in actual_grads)
        for name, grad in actual_grads.items():
            assert torch.allclose(grad, expected_grads[name], atol=1e-6), name
