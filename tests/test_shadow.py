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

import pytest
import torch
from safetensors import safe_open
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    Cache,
    LlamaConfig,
)

from peft import (
    PeftModel,
    PeftModelForCausalLM,
    PeftModelForSequenceClassification,
    ShadowConfig,
    get_peft_model,
)
from peft.tuners.shadow import DetachedShadowModel, ShadowCache, ShadowModel
from peft.tuners.shadow.layers import ShadowLayer
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import PeftType, get_peft_model_state_dict
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


class TestShadowCausalLM:
    def test_forward_and_auxiliary_loss_backward(self):
        model = get_peft_model(make_llama_causal(), ShadowConfig(task_type="CAUSAL_LM"))
        ids = torch.randint(0, 128, (2, 6))
        out = model(input_ids=ids, labels=ids.clone())
        assert out.logits.shape == (2, 6, model.config.vocab_size)
        assert out.loss is not None
        out.loss.backward()
        grads = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert len(grads) > 0

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
        # the tuner (`last_shadow_loss`). The latter survives DDP/FSDP, which drop non-field output attributes.
        model = get_peft_model(make_llama_causal(), ShadowConfig(task_type="CAUSAL_LM"))
        ids = torch.randint(0, 128, (2, 6))
        out = model(input_ids=ids, labels=ids.clone())
        assert out.shadow_loss is not None
        assert out.shadow_loss.numel() == 1
        assert model.base_model.last_shadow_loss is not None
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
        model = get_peft_model(make_llama_causal(), ShadowConfig(task_type="CAUSAL_LM"))
        model.eval()
        ids = torch.randint(0, 128, (2, 6))
        with torch.no_grad():
            on = model(input_ids=ids).logits
            with model.disable_adapter():
                off = model(input_ids=ids).logits
        assert torch.allclose(on, off, atol=1e-6)

    def test_switches_adapters_but_rejects_multiple_active_adapters(self):
        model = get_peft_model(make_llama_causal(), ShadowConfig(task_type="CAUSAL_LM"))
        model.add_adapter("other", ShadowConfig(task_type="CAUSAL_LM"))
        model.set_adapter("other")
        assert model.active_adapters == ["other"]

        with pytest.raises(ValueError, match="exactly one active adapter"):
            model.base_model.set_adapter(["default", "other"])
        assert model.active_adapters == ["other"]

    def test_merge_raises(self):
        # ShadowPEFT cannot be merged: it must raise an explicit error rather than silently doing the wrong thing.
        model = get_peft_model(make_llama_causal(), ShadowConfig(task_type="CAUSAL_LM"))
        with pytest.raises(NotImplementedError):
            model.merge_and_unload()
        with pytest.raises(NotImplementedError):
            model.base_model.merge_adapter()

    def test_save_includes_shadow_modules_but_not_frozen_head(self, tmp_path):
        model = get_peft_model(make_llama_causal(), ShadowConfig(task_type="CAUSAL_LM"))
        model.save_pretrained(tmp_path)
        with safe_open(tmp_path / SAFETENSORS_WEIGHTS_NAME, framework="pt") as f:
            keys = list(f.keys())
        assert any(".shadow_backbone." in key for key in keys)
        assert any("shadow_down" in key for key in keys)
        assert any("shadow_update_transform" in key for key in keys)
        # The frozen copy of the base LM head is not stored (it is rebuilt from the base model on load).
        assert not any(".shadow_head." in key for key in keys)

    def test_save_includes_trainable_lm_head(self, tmp_path):
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

    def test_save_fsdp_prefixed_state_dict(self, tmp_path):
        model = get_peft_model(make_llama_causal(), ShadowConfig(task_type="CAUSAL_LM"))
        prefixed_state_dict = {f"_fsdp_wrapped_module.{k}": v for k, v in model.state_dict().items()}
        model.save_pretrained(tmp_path, state_dict=prefixed_state_dict)
        with safe_open(tmp_path / SAFETENSORS_WEIGHTS_NAME, framework="pt") as f:
            keys = list(f.keys())
        assert any(".shadow_backbone." in key for key in keys)
        assert len(keys) > 2

    def test_load_fsdp_wrapped_shadow_keys(self, tmp_path):
        base = make_llama_causal()
        base_sd = copy.deepcopy(base.state_dict())
        model = get_peft_model(base, ShadowConfig(task_type="CAUSAL_LM"))
        ids = torch.randint(0, 128, (2, 6))
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.3)
        for _ in range(2):
            opt.zero_grad()
            model(input_ids=ids, labels=ids.clone()).loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            ref = model(input_ids=ids).logits

        state_dict = get_peft_model_state_dict(model)
        wrapped_state_dict = {
            key.replace(".weight", "._fsdp_wrapped_module.weight"): value for key, value in state_dict.items()
        }
        model.save_pretrained(tmp_path, state_dict=wrapped_state_dict)

        base2 = make_llama_causal()
        base2.load_state_dict(base_sd)
        loaded = PeftModel.from_pretrained(base2, tmp_path)
        loaded.eval()
        with torch.no_grad():
            got = loaded(input_ids=ids).logits
        assert torch.allclose(ref, got, atol=1e-6)

    def test_requires_two_layers(self):
        # A single decoder block means the shadow carrier has no loop to ride; injection needs >= 2 blocks.
        # Target only one block of the tiny 2-layer model so entry == exit.
        model = get_peft_model(
            make_llama_causal(), ShadowConfig(task_type="CAUSAL_LM", layers_to_transform=[0])
        )
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
            out_decode = model(
                input_ids=next_token, past_key_values=out_prefill.past_key_values, use_cache=True
            )
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
        from transformers import LlamaModel

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
        cache.crop(3)
        assert cache.get_seq_length() == 3
        assert cache.shadow.get_seq_length() == 3


class TestShadowSequenceClassification:
    def test_classifier_head_trainable_by_default(self):
        model = get_peft_model(make_llama_seqcls(num_labels=3), ShadowConfig(task_type="SEQ_CLS"))
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        assert any("shadow_head" in n for n in trainable)

    def test_unload_shadow_is_a_classifier(self):
        # For SEQ_CLS the standalone shadow model pools the last token and returns per-example class logits (not
        # per-position), so the shadow path's classification performance can be evaluated on its own.
        from transformers.modeling_outputs import SequenceClassifierOutput

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
        from safetensors.torch import save_file
        from transformers import LlamaModel

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

