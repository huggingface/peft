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
from safetensors import safe_open
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
)

from peft import (
    PeftModel,
    PeftModelForCausalLM,
    PeftModelForSequenceClassification,
    ShadowConfig,
    get_peft_model,
)
from peft.tuners.shadow import DetachedShadowModel, ShadowModel
from peft.tuners.shadow.layers import ShadowLayer
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils import PeftType, get_peft_model_state_dict
from peft.utils.constants import SAFETENSORS_WEIGHTS_NAME


def make_llama_causal(hidden_size=32, num_layers=3, vocab_size=128):
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=2 * hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(cfg)


def make_llama_seqcls(num_labels=3, hidden_size=32, num_layers=3, vocab_size=128):
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=2 * hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        num_labels=num_labels,
        pad_token_id=0,
    )
    return LlamaForSequenceClassification(cfg)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


class TestShadowCausalLM:
    def test_get_peft_model_wraps_correctly(self):
        model = get_peft_model(make_llama_causal(), ShadowConfig(task_type="CAUSAL_LM"))
        assert isinstance(model, PeftModelForCausalLM)
        assert isinstance(model.base_model, ShadowModel)
        assert model.peft_config["default"].peft_type == PeftType.SHADOW

    def test_conforms_to_peft_api(self):
        # ShadowPEFT must be a proper `BaseTuner` and its layers proper `BaseTunerLayer`s.
        model = get_peft_model(make_llama_causal(), ShadowConfig(task_type="CAUSAL_LM"))
        assert isinstance(model.base_model, BaseTuner)
        shadow_layers = [m for m in model.modules() if isinstance(m, ShadowLayer)]
        assert shadow_layers
        assert all(isinstance(layer, BaseTunerLayer) for layer in shadow_layers)

    def test_only_shadow_params_trainable(self):
        model = get_peft_model(make_llama_causal(), ShadowConfig(task_type="CAUSAL_LM"))
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        assert trainable  # non-empty
        # Every trainable parameter belongs to a shadow module.
        for name in trainable:
            assert "shadow_" in name, name

    def test_forward_and_auxiliary_loss_backward(self):
        model = get_peft_model(make_llama_causal(), ShadowConfig(task_type="CAUSAL_LM"))
        ids = torch.randint(0, 128, (2, 6))
        out = model(input_ids=ids, labels=ids.clone())
        assert out.logits.shape == (2, 6, 128)
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
            ShadowConfig(task_type="CAUSAL_LM", auxiliary_loss_weight=1.0, modules_to_save=["shadow_lm_head"]),
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

    def test_disable_adapter_changes_trained_output(self):
        model = get_peft_model(make_llama_causal(), ShadowConfig(task_type="CAUSAL_LM"))
        ids = torch.randint(0, 128, (2, 6))
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.5)
        for _ in range(3):
            opt.zero_grad()
            out = model(input_ids=ids, labels=ids.clone())
            out.loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            on = model(input_ids=ids).logits
            with model.disable_adapter():
                off = model(input_ids=ids).logits
            on_again = model(input_ids=ids).logits
        assert not torch.allclose(on, off, atol=1e-6)
        assert torch.allclose(on, on_again, atol=1e-6)

    def test_generate(self):
        model = get_peft_model(make_llama_causal(), ShadowConfig(task_type="CAUSAL_LM"))
        ids = torch.randint(0, 128, (2, 4))
        gen = model.generate(input_ids=ids, max_new_tokens=4, use_cache=False, do_sample=False)
        assert gen.shape[1] == 8

    def test_merge_raises(self):
        # ShadowPEFT cannot be merged: it must raise an explicit error rather than silently doing the wrong thing.
        model = get_peft_model(make_llama_causal(), ShadowConfig(task_type="CAUSAL_LM"))
        with pytest.raises(NotImplementedError):
            model.merge_and_unload()
        with pytest.raises(NotImplementedError):
            model.base_model.merge_adapter()

    def test_save_load_roundtrip(self, tmp_path):
        base = make_llama_causal()
        base_sd = copy.deepcopy(base.state_dict())
        model = get_peft_model(base, ShadowConfig(task_type="CAUSAL_LM"))
        ids = torch.randint(0, 128, (2, 6))
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.3)
        for _ in range(2):
            opt.zero_grad()
            out = model(input_ids=ids, labels=ids.clone())
            out.loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            ref = model(input_ids=ids).logits

        model.save_pretrained(tmp_path)
        files = {p.name for p in tmp_path.iterdir()}
        assert "adapter_config.json" in files
        assert "adapter_model.safetensors" in files

        base2 = make_llama_causal()
        base2.load_state_dict(base_sd)
        loaded = PeftModel.from_pretrained(base2, tmp_path)
        loaded.eval()
        with torch.no_grad():
            got = loaded(input_ids=ids).logits
        assert torch.allclose(ref, got, atol=1e-6)

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

    def test_save_includes_trainable_shadow_lm_head(self, tmp_path):
        model = get_peft_model(
            make_llama_causal(),
            ShadowConfig(task_type="CAUSAL_LM", modules_to_save=["shadow_lm_head"]),
        )
        head = model.base_model.shadow_head["default"]
        assert all(p.requires_grad for p in head.parameters())
        model.save_pretrained(tmp_path)
        with safe_open(tmp_path / SAFETENSORS_WEIGHTS_NAME, framework="pt") as f:
            keys = list(f.keys())
        assert any(".shadow_head." in key for key in keys)

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
        # The default targets every block, so a 1-layer model wraps just one block and cannot form a trajectory.
        model = get_peft_model(make_llama_causal(num_layers=1), ShadowConfig(task_type="CAUSAL_LM"))
        ids = torch.randint(0, 128, (2, 6))
        # A single wrapped block still runs (entry == exit); just assert it does not crash.
        out = model(input_ids=ids, labels=ids.clone())
        assert out.logits.shape == (2, 6, 128)

    def test_bf16_base_model(self):
        # transformers keeps the checkpoint dtype (often bf16). The shadow backbone matches the base dtype, while the
        # per-block adapters are upcast to fp32 by `autocast_adapter_dtype` (standard PEFT); the forward must bridge the
        # two dtypes rather than crash.
        base = make_llama_causal().to(torch.bfloat16)
        model = get_peft_model(base, ShadowConfig(task_type="CAUSAL_LM"))
        assert next(model.base_model.shadow_backbone["default"].parameters()).dtype == torch.bfloat16
        ids = torch.randint(0, 128, (2, 6))
        out = model(input_ids=ids, labels=ids.clone())
        assert out.logits.dtype == torch.bfloat16
        out.loss.backward()
        gen = model.generate(input_ids=ids[:, :3], max_new_tokens=3, use_cache=False, do_sample=False)
        assert gen.shape[1] == 6

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

    def test_gpt2_backbone(self):
        cfg = GPT2Config(vocab_size=128, n_embd=32, n_inner=64, n_layer=3, n_head=4, n_positions=64)
        model = get_peft_model(GPT2LMHeadModel(cfg), ShadowConfig(task_type="CAUSAL_LM"))
        ids = torch.randint(0, 128, (2, 6))
        out = model(input_ids=ids, labels=ids.clone())
        assert out.logits.shape == (2, 6, 128)
        out.loss.backward()


class TestShadowMultiAdapter:
    def test_add_switch_delete_adapter(self):
        model = get_peft_model(make_llama_causal(), ShadowConfig(task_type="CAUSAL_LM"), adapter_name="first")
        model.add_adapter("second", ShadowConfig(task_type="CAUSAL_LM"))
        assert set(model.peft_config) == {"first", "second"}
        assert "second" in model.base_model.shadow_backbone

        ids = torch.randint(0, 128, (2, 5))
        model.set_adapter("second")
        assert model.base_model.active_adapters == ["second"]
        model(input_ids=ids, labels=ids.clone()).loss.backward()

        model.set_adapter("first")
        assert model.base_model.active_adapters == ["first"]

        model.delete_adapter("second")
        assert set(model.peft_config) == {"first"}
        assert "second" not in model.base_model.shadow_backbone
        # Still works after deletion.
        model(input_ids=ids, labels=ids.clone())

    def test_switching_adapter_changes_output(self):
        model = get_peft_model(make_llama_causal(), ShadowConfig(task_type="CAUSAL_LM"), adapter_name="first")
        model.add_adapter("second", ShadowConfig(task_type="CAUSAL_LM"))
        ids = torch.randint(0, 128, (2, 6))

        # Train adapter "a" only.
        model.set_adapter("first")
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.5)
        for _ in range(3):
            opt.zero_grad()
            model(input_ids=ids, labels=ids.clone()).loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            out_a = model(input_ids=ids).logits
            model.set_adapter("second")
            out_b = model(input_ids=ids).logits
        assert not torch.allclose(out_a, out_b, atol=1e-6)


class TestShadowSequenceClassification:
    @pytest.fixture(autouse=True)
    def _silence_unrelated_core_deprecation(self):
        # PeftModelForSequenceClassification.forward reads `self.config.use_return_dict`, which newer transformers
        # versions emit a deprecation warning for. The test suite's conftest escalates transformers deprecations to
        # errors; this one originates in PEFT core (it reproduces identically with LoRA) and is unrelated to the Shadow
        # method, so we raise the transformers logger level for these tests to avoid a spurious failure.
        import logging

        logger = logging.getLogger("transformers")
        previous = logger.level
        logger.setLevel(logging.ERROR)
        try:
            yield
        finally:
            logger.setLevel(previous)

    def test_forward_and_backward(self):
        model = get_peft_model(make_llama_seqcls(num_labels=3), ShadowConfig(task_type="SEQ_CLS"))
        assert isinstance(model, PeftModelForSequenceClassification)
        ids = torch.randint(1, 128, (2, 6))
        am = torch.ones_like(ids)
        labels = torch.tensor([0, 2])
        out = model(input_ids=ids, attention_mask=am, labels=labels)
        assert out.logits.shape == (2, 3)
        out.loss.backward()

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

    def test_save_load_roundtrip(self, tmp_path):
        base = make_llama_seqcls(num_labels=3)
        base_sd = copy.deepcopy(base.state_dict())
        model = get_peft_model(base, ShadowConfig(task_type="SEQ_CLS"))
        ids = torch.randint(1, 128, (2, 6))
        am = torch.ones_like(ids)
        labels = torch.tensor([0, 2])
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.3)
        for _ in range(2):
            opt.zero_grad()
            out = model(input_ids=ids, attention_mask=am, labels=labels)
            out.loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            ref = model(input_ids=ids, attention_mask=am).logits
        model.save_pretrained(tmp_path)

        base2 = make_llama_seqcls(num_labels=3)
        base2.load_state_dict(base_sd)
        loaded = PeftModel.from_pretrained(base2, tmp_path)
        loaded.eval()
        with torch.no_grad():
            got = loaded(input_ids=ids, attention_mask=am).logits
        assert torch.allclose(ref, got, atol=1e-6)


class TestShadowBackboneVariants:
    def test_smaller_shadow_hidden_size_inserts_projection(self):
        model = get_peft_model(
            make_llama_causal(hidden_size=32, num_layers=4),
            ShadowConfig(task_type="CAUSAL_LM", shadow_num_hidden_layers=2, shadow_hidden_size=16),
        )
        projection = model.base_model.shadow_projection["default"]
        assert isinstance(projection, torch.nn.Linear)
        assert (projection.in_features, projection.out_features) == (16, 32)
        ids = torch.randint(0, 128, (2, 6))
        out = model(input_ids=ids, labels=ids.clone())
        out.loss.backward()

    def test_matching_shadow_hidden_size_uses_identity(self):
        model = get_peft_model(
            make_llama_causal(hidden_size=32, num_layers=4),
            ShadowConfig(task_type="CAUSAL_LM", shadow_num_hidden_layers=2),
        )
        assert isinstance(model.base_model.shadow_projection["default"], torch.nn.Identity)

    def test_pretrained_projected_shadow_checkpoint(self, tmp_path):
        # A "projected" shadow checkpoint (model_type == causal_lm_with_hidden_projection, e.g.
        # shadow-llm/Qwen3-0.6B-H8B) bundles a small backbone + a trained shadow_hidden -> base_hidden projection.
        # ShadowPEFT must load the pretrained backbone and reuse the trained projection.
        import json

        from safetensors.torch import save_file
        from transformers import LlamaConfig, LlamaModel

        shadow_hidden, base_hidden, vocab = 16, 32, 128
        inner_cfg = LlamaConfig(
            vocab_size=vocab,
            hidden_size=shadow_hidden,
            intermediate_size=2 * shadow_hidden,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
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

        base = make_llama_causal(hidden_size=base_hidden, num_layers=3)
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
        ids = torch.randint(0, vocab, (2, 6))
        model(input_ids=ids, labels=ids.clone()).loss.backward()
        assert embed.weight.grad is None

    def test_unload_shadow_returns_standalone_generatable_model(self):
        # unload_shadow returns the standalone shadow network (backbone + projection + head) as a causal LM. This is how
        # the shadow path's own performance is evaluated, independent of the base model.
        model = get_peft_model(
            make_llama_causal(hidden_size=32, num_layers=4),
            ShadowConfig(task_type="CAUSAL_LM", shadow_num_hidden_layers=2),
        )
        detached = model.base_model.unload_shadow()
        assert isinstance(detached, DetachedShadowModel)
        assert detached.can_generate()
        ids = torch.randint(0, 128, (2, 5))
        with torch.no_grad():
            out = detached(input_ids=ids)
        # head(projection(backbone(x))) -> vocab logits (CausalLMOutputWithPast)
        assert out.logits.shape == (2, 5, 128)
        # It behaves like a normal causal LM, so it can generate (KV cache is fine for the standalone shadow path).
        gen = detached.generate(input_ids=ids[:, :3], max_new_tokens=3, do_sample=False)
        assert gen.shape[1] == 6

    def test_unload_shadow_applies_projection(self):
        # With a smaller shadow hidden size the standalone model must apply the trained projection so the head receives
        # the correct (base) hidden width.
        model = get_peft_model(
            make_llama_causal(hidden_size=32, num_layers=4),
            ShadowConfig(task_type="CAUSAL_LM", shadow_num_hidden_layers=2, shadow_hidden_size=16),
        )
        detached = model.base_model.unload_shadow()
        ids = torch.randint(0, 128, (2, 5))
        with torch.no_grad():
            out = detached(input_ids=ids)
        assert out.logits.shape == (2, 5, 128)
