# Copyright 2024-present the HuggingFace Inc. team.
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
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
)

from peft import (
    AutoModelForCausalLMWithHiddenProjection,
    PeftModel,
    PeftModelForCausalLM,
    PeftModelForSequenceClassification,
    ShadowConfig,
    get_peft_model,
)
from peft.tuners.shadow import ShadowModel
from peft.utils import PeftType, get_peft_model_state_dict
from peft.utils.constants import SAFETENSORS_WEIGHTS_NAME
from safetensors import safe_open


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
        model = get_peft_model(make_llama_causal(), ShadowConfig(num_shadow_layers=1, task_type="CAUSAL_LM"))
        assert isinstance(model, PeftModelForCausalLM)
        assert isinstance(model.base_model, ShadowModel)
        assert model.peft_config["default"].peft_type == PeftType.SHADOW

    def test_only_shadow_params_trainable(self):
        model = get_peft_model(make_llama_causal(), ShadowConfig(num_shadow_layers=1, task_type="CAUSAL_LM"))
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        assert trainable  # non-empty
        # Every trainable parameter belongs to a shadow module; the frozen base stays under base_model.model.
        for name in trainable:
            assert ".shadow_" in name, name

    def test_forward_returns_base_and_shadow_logits(self):
        model = get_peft_model(make_llama_causal(), ShadowConfig(num_shadow_layers=1, task_type="CAUSAL_LM"))
        ids = torch.randint(0, 128, (2, 6))
        out = model(input_ids=ids, labels=ids.clone())
        assert out.logits.shape == (2, 6, 128)
        assert out.shadow_logits.shape == (2, 6, 128)
        assert out.loss is not None
        out.loss.backward()
        grads = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert len(grads) > 0

    def test_untrained_adapter_is_noop(self):
        # injection_ups are zero-initialized, so an untrained shadow adapter must not change the base output.
        model = get_peft_model(make_llama_causal(), ShadowConfig(num_shadow_layers=1, task_type="CAUSAL_LM"))
        model.eval()
        ids = torch.randint(0, 128, (2, 6))
        with torch.no_grad():
            on = model(input_ids=ids).logits
            with model.disable_adapter():
                off = model(input_ids=ids).logits
        assert torch.allclose(on, off, atol=1e-6)

    def test_disable_adapter_changes_trained_output(self):
        model = get_peft_model(make_llama_causal(), ShadowConfig(num_shadow_layers=1, task_type="CAUSAL_LM"))
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
        model = get_peft_model(make_llama_causal(), ShadowConfig(num_shadow_layers=1, task_type="CAUSAL_LM"))
        ids = torch.randint(0, 128, (2, 4))
        gen = model.generate(input_ids=ids, max_new_tokens=4, use_cache=False, do_sample=False)
        assert gen.shape[1] == 8

    def test_shadow_only_inference(self):
        cfg = ShadowConfig(num_shadow_layers=1, task_type="CAUSAL_LM", shadow_inference_mode="shadow_only")
        model = get_peft_model(make_llama_causal(), cfg)
        ids = torch.randint(0, 128, (2, 6))
        out = model(input_ids=ids)
        assert torch.equal(out.logits, out.shadow_logits)
        # can be switched back to base_shadow at runtime
        model.base_model.set_inference_mode("base_shadow")
        out2 = model(input_ids=ids)
        assert out2.logits.shape == (2, 6, 128)

    def test_save_load_roundtrip(self, tmp_path):
        base = make_llama_causal()
        base_sd = copy.deepcopy(base.state_dict())
        model = get_peft_model(base, ShadowConfig(num_shadow_layers=1, task_type="CAUSAL_LM"))
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
            ref_shadow = model(input_ids=ids).shadow_logits

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
            got_shadow = loaded(input_ids=ids).shadow_logits
        assert torch.allclose(ref, got, atol=1e-6)
        assert torch.allclose(ref_shadow, got_shadow, atol=1e-6)

    def test_save_includes_shadow_backbone(self, tmp_path):
        model = get_peft_model(make_llama_causal(), ShadowConfig(num_shadow_layers=1, task_type="CAUSAL_LM"))
        model.save_pretrained(tmp_path)
        with safe_open(tmp_path / SAFETENSORS_WEIGHTS_NAME, framework="pt") as f:
            keys = list(f.keys())
        assert any(".shadow_model." in key for key in keys)
        assert any("shadow_injection_model" in key for key in keys)
        assert any("shadow_update_model" in key for key in keys)
        assert not any(".shadow_lm_head." in key for key in keys)

    def test_save_includes_modified_shadow_lm_head(self, tmp_path):
        model = get_peft_model(
            make_llama_causal(),
            ShadowConfig(num_shadow_layers=1, task_type="CAUSAL_LM", modules_to_save=["shadow_lm_head"]),
        )
        with torch.no_grad():
            model.base_model.shadow_lm_head.weight.add_(1.0)

        model.save_pretrained(tmp_path)

        with safe_open(tmp_path / SAFETENSORS_WEIGHTS_NAME, framework="pt") as f:
            keys = list(f.keys())
        assert any(".shadow_lm_head." in key for key in keys)

    def test_save_fsdp_prefixed_state_dict(self, tmp_path):
        model = get_peft_model(make_llama_causal(), ShadowConfig(num_shadow_layers=1, task_type="CAUSAL_LM"))
        prefixed_state_dict = {f"_fsdp_wrapped_module.{k}": v for k, v in model.state_dict().items()}
        model.save_pretrained(tmp_path, state_dict=prefixed_state_dict)
        with safe_open(tmp_path / SAFETENSORS_WEIGHTS_NAME, framework="pt") as f:
            keys = list(f.keys())
        assert any(".shadow_model." in key for key in keys)
        assert len(keys) > 2

    def test_load_fsdp_wrapped_shadow_keys(self, tmp_path):
        base = make_llama_causal()
        base_sd = copy.deepcopy(base.state_dict())
        model = get_peft_model(base, ShadowConfig(num_shadow_layers=1, task_type="CAUSAL_LM"))
        ids = torch.randint(0, 128, (2, 6))
        with torch.no_grad():
            ref = model(input_ids=ids).shadow_logits

        state_dict = get_peft_model_state_dict(model)
        wrapped_state_dict = {
            key.replace(".weight", "._fsdp_wrapped_module.weight"): value for key, value in state_dict.items()
        }
        model.save_pretrained(tmp_path, state_dict=wrapped_state_dict)

        base2 = make_llama_causal()
        base2.load_state_dict(base_sd)
        loaded = PeftModel.from_pretrained(base2, tmp_path)
        with torch.no_grad():
            got = loaded(input_ids=ids).shadow_logits
        assert torch.allclose(ref, got, atol=1e-6)

    def test_explicit_shadow_config_path_recorded(self, tmp_path):
        base = make_llama_causal(hidden_size=32, num_layers=4)
        shadow = make_llama_causal(hidden_size=16, num_layers=2)
        shadow.name_or_path = "/tmp/explicit-shadow-init"
        model = get_peft_model(base, ShadowConfig(task_type="CAUSAL_LM"), shadow_model=shadow)
        model.save_pretrained(tmp_path)
        saved_cfg = ShadowConfig.from_pretrained(tmp_path)
        assert saved_cfg.explicit_shadow_model_name_or_path == "/tmp/explicit-shadow-init"
        shadow_keys = [
            key
            for key in get_peft_model_state_dict(model).keys()
            if ".shadow_model." in key
        ]
        assert shadow_keys

    def test_requires_two_layers(self):
        with pytest.raises(ValueError, match="at least 2 decoder layers"):
            get_peft_model(make_llama_causal(num_layers=1), ShadowConfig(task_type="CAUSAL_LM"))

    def test_bf16_base_model(self):
        # transformers' from_pretrained keeps the checkpoint dtype (often bf16); the shadow modules must match the base
        # dtype, otherwise injected hidden states are upcast and mismatch the base model's bf16 parameters.
        base = make_llama_causal().to(torch.bfloat16)
        model = get_peft_model(base, ShadowConfig(num_shadow_layers=1, task_type="CAUSAL_LM"))
        assert model.base_model.shadow_injection_model.injection_downs.dtype == torch.bfloat16
        assert next(model.base_model.shadow_model.parameters()).dtype == torch.bfloat16
        ids = torch.randint(0, 128, (2, 6))
        out = model(input_ids=ids, labels=ids.clone())
        assert out.logits.dtype == torch.bfloat16
        assert out.shadow_logits.dtype == torch.bfloat16
        out.loss.backward()
        gen = model.generate(input_ids=ids[:, :3], max_new_tokens=3, use_cache=False, do_sample=False)
        assert gen.shape[1] == 6

    def test_gpt2_backbone(self):
        cfg = GPT2Config(vocab_size=128, n_embd=32, n_inner=64, n_layer=3, n_head=4, n_positions=64)
        model = get_peft_model(GPT2LMHeadModel(cfg), ShadowConfig(num_shadow_layers=1, task_type="CAUSAL_LM"))
        ids = torch.randint(0, 128, (2, 6))
        out = model(input_ids=ids, labels=ids.clone())
        assert out.shadow_logits.shape == (2, 6, 128)
        out.loss.backward()


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
        model = get_peft_model(make_llama_seqcls(num_labels=3), ShadowConfig(num_shadow_layers=1, task_type="SEQ_CLS"))
        assert isinstance(model, PeftModelForSequenceClassification)
        ids = torch.randint(1, 128, (2, 6))
        am = torch.ones_like(ids)
        labels = torch.tensor([0, 2])
        out = model(input_ids=ids, attention_mask=am, labels=labels)
        assert out.logits.shape == (2, 3)
        assert out.shadow_logits.shape == (2, 3)
        out.loss.backward()

    def test_classifier_heads_trainable_by_default(self):
        model = get_peft_model(make_llama_seqcls(num_labels=3), ShadowConfig(num_shadow_layers=1, task_type="SEQ_CLS"))
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        assert any("shadow_classifier_head" in n for n in trainable)

    def test_save_load_roundtrip(self, tmp_path):
        base = make_llama_seqcls(num_labels=3)
        base_sd = copy.deepcopy(base.state_dict())
        model = get_peft_model(base, ShadowConfig(num_shadow_layers=1, task_type="SEQ_CLS"))
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
            ref = model(input_ids=ids, attention_mask=am).shadow_logits
        model.save_pretrained(tmp_path)

        base2 = make_llama_seqcls(num_labels=3)
        base2.load_state_dict(base_sd)
        loaded = PeftModel.from_pretrained(base2, tmp_path)
        loaded.eval()
        with torch.no_grad():
            got = loaded(input_ids=ids, attention_mask=am).shadow_logits
        assert torch.allclose(ref, got, atol=1e-6)


class TestShadowExplicitModel:
    def test_explicit_same_hidden_size(self):
        base = make_llama_causal(hidden_size=32, num_layers=4)
        shadow = make_llama_causal(hidden_size=32, num_layers=2)
        model = get_peft_model(base, ShadowConfig(task_type="CAUSAL_LM"), shadow_model=shadow)
        assert isinstance(model.base_model.shadow_hidden_projection, nn.Identity)
        ids = torch.randint(0, 128, (2, 6))
        out = model(input_ids=ids, labels=ids.clone())
        out.loss.backward()

    def test_explicit_projection_inserted(self):
        base = make_llama_causal(hidden_size=32, num_layers=4)
        shadow = make_llama_causal(hidden_size=16, num_layers=2)
        model = get_peft_model(base, ShadowConfig(task_type="CAUSAL_LM"), shadow_model=shadow)
        proj = model.base_model.shadow_hidden_projection
        assert isinstance(proj, nn.Linear)
        assert (proj.in_features, proj.out_features) == (16, 32)
        ids = torch.randint(0, 128, (2, 6))
        out = model(input_ids=ids, labels=ids.clone())
        assert out.shadow_logits.shape == (2, 6, 128)
        out.loss.backward()

    def test_explicit_frozen_embeddings_stay_frozen(self):
        base = make_llama_causal(hidden_size=32, num_layers=4)
        shadow = make_llama_causal(hidden_size=16, num_layers=2)
        for param in shadow.get_input_embeddings().parameters():
            param.requires_grad = False

        model = get_peft_model(base, ShadowConfig(task_type="CAUSAL_LM"), shadow_model=shadow)

        shadow_embed = model.base_model.shadow_model.get_input_embeddings()
        assert not any(param.requires_grad for param in shadow_embed.parameters())
        assert any(
            param.requires_grad
            for name, param in model.base_model.shadow_model.named_parameters()
            if "embed_tokens" not in name
        )

    def test_explicit_frozen_backbone_layer_stays_frozen(self):
        base = make_llama_causal(hidden_size=32, num_layers=4)
        shadow = make_llama_causal(hidden_size=16, num_layers=2)
        for param in shadow.model.layers[0].parameters():
            param.requires_grad = False

        model = get_peft_model(base, ShadowConfig(task_type="CAUSAL_LM"), shadow_model=shadow)

        assert not any(param.requires_grad for param in model.base_model.shadow_model.layers[0].parameters())
        assert any(param.requires_grad for param in model.base_model.shadow_model.layers[1].parameters())

    def test_projected_explicit_frozen_lm_head_stays_frozen(self):
        base = make_llama_causal(hidden_size=32, num_layers=4)
        shadow = make_llama_causal(hidden_size=16, num_layers=2)
        projected = AutoModelForCausalLMWithHiddenProjection.wrap(
            shadow_model=shadow,
            shadow_hidden_projection=nn.Linear(16, 32, bias=False),
            lm_head=base.lm_head,
            init_optimal_projection=False,
        )
        for param in projected.lm_head.parameters():
            param.requires_grad = False

        model = get_peft_model(
            base,
            ShadowConfig(task_type="CAUSAL_LM", modules_to_save=["shadow_lm_head"]),
            shadow_model=projected,
        )

        assert not any(param.requires_grad for param in model.base_model.shadow_lm_head.parameters())

    def test_projected_explicit_frozen_projection_stays_frozen(self):
        base = make_llama_causal(hidden_size=32, num_layers=4)
        shadow = make_llama_causal(hidden_size=16, num_layers=2)
        projected = AutoModelForCausalLMWithHiddenProjection.wrap(
            shadow_model=shadow,
            shadow_hidden_projection=nn.Linear(16, 32, bias=False),
            lm_head=base.lm_head,
            init_optimal_projection=False,
        )
        for param in projected.shadow_hidden_projection.parameters():
            param.requires_grad = False

        model = get_peft_model(base, ShadowConfig(task_type="CAUSAL_LM"), shadow_model=projected)

        assert not any(param.requires_grad for param in model.base_model.shadow_hidden_projection.parameters())

    def test_explicit_roundtrip(self, tmp_path):
        base = make_llama_causal(hidden_size=32, num_layers=4)
        base_sd = copy.deepcopy(base.state_dict())
        shadow = make_llama_causal(hidden_size=16, num_layers=2)
        shadow_sd = copy.deepcopy(shadow.state_dict())
        model = get_peft_model(base, ShadowConfig(task_type="CAUSAL_LM"), shadow_model=shadow)
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

        base2 = make_llama_causal(hidden_size=32, num_layers=4)
        base2.load_state_dict(base_sd)
        shadow2 = make_llama_causal(hidden_size=16, num_layers=2)
        shadow2.load_state_dict(shadow_sd)
        loaded = PeftModel.from_pretrained(base2, tmp_path, shadow_model=shadow2)
        loaded.eval()
        with torch.no_grad():
            got = loaded(input_ids=ids).logits
        assert torch.allclose(ref, got, atol=1e-6)

    def test_shadow_model_requires_shadow_config(self):
        from peft import LoraConfig

        base = make_llama_causal()
        shadow = make_llama_causal(num_layers=2)
        with pytest.raises(ValueError, match="only supported with a `ShadowConfig`"):
            get_peft_model(base, LoraConfig(task_type="CAUSAL_LM"), shadow_model=shadow)

    def test_export_shadow_projected(self):
        base = make_llama_causal(hidden_size=32, num_layers=4)
        shadow = make_llama_causal(hidden_size=16, num_layers=2)
        model = get_peft_model(base, ShadowConfig(task_type="CAUSAL_LM"), shadow_model=shadow)
        exported = model.base_model.export_shadow()
        assert isinstance(exported, AutoModelForCausalLMWithHiddenProjection)
        ids = torch.randint(0, 128, (2, 5))
        with torch.no_grad():
            out = exported(input_ids=ids)
        assert out.logits.shape == (2, 5, 128)


class TestProjectedCausalLM:
    def test_wrap_and_roundtrip(self, tmp_path):
        small = make_llama_causal(hidden_size=16, num_layers=2)
        large = make_llama_causal(hidden_size=32, num_layers=4)
        wrapped = AutoModelForCausalLMWithHiddenProjection.wrap(
            shadow_model=small,
            shadow_hidden_projection=nn.Linear(16, 32, bias=False),
            lm_head=large.lm_head,
            init_optimal_projection=True,
            reference_lm_head=small.lm_head,
        )
        ids = torch.randint(0, 128, (2, 5))
        with torch.no_grad():
            ref = wrapped(input_ids=ids).logits
        wrapped.save_pretrained(tmp_path)
        reloaded = AutoModelForCausalLM.from_pretrained(tmp_path)
        assert isinstance(reloaded, AutoModelForCausalLMWithHiddenProjection)
        with torch.no_grad():
            got = reloaded(input_ids=ids).logits
        assert torch.allclose(ref, got, atol=1e-5)
