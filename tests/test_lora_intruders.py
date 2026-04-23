from functools import partial
from io import StringIO

import pytest
import torch
from transformers import AutoModelForCausalLM

from peft import LoraConfig, MissConfig, get_peft_model
from peft.tuners.lora.intruders import reduce_intruder_dimension

from .testing_utils import hub_online_once


class TestLoraIntruders:
    @pytest.fixture
    def model_lin(self):
        model_id = "trl-internal-testing/tiny-random-LlamaForCausalLM"
        with hub_online_once(model_id):
            base_model = AutoModelForCausalLM.from_pretrained(model_id)

        cfg = LoraConfig(target_modules=["q_proj"])
        peft_model = get_peft_model(base_model, cfg)

        return peft_model

    @pytest.fixture
    def model_emb(self):
        model_id = "trl-internal-testing/tiny-random-LlamaForCausalLM"
        with hub_online_once(model_id):
            base_model = AutoModelForCausalLM.from_pretrained(model_id)

        cfg = LoraConfig(target_modules=["embed_tokens"])
        peft_model = get_peft_model(base_model, cfg)

        return peft_model

    @pytest.fixture
    def model_lin_non_lora(self):
        model_id = "trl-internal-testing/tiny-random-LlamaForCausalLM"
        with hub_online_once(model_id):
            base_model = AutoModelForCausalLM.from_pretrained(model_id)

        cfg = MissConfig(target_modules=["q_proj"])
        peft_model = get_peft_model(base_model, cfg)

        return peft_model

    def test_lora_intruders_linear(self, model_lin):
        original_weights = {}
        for name, module in model_lin.named_modules():
            if "q_proj" in name and hasattr(module, "lora_B"):
                original_weights[name] = module.lora_B["default"].weight.detach().clone()

        buffer = StringIO()

        # use a high epsilon to make sure that we get a match to see whether layers get modified
        reduce_intruder_dimension(model_lin, threshold_epsilon=999, logging_sink=partial(print, file=buffer))

        # the old adapter should not be active anymore, just the new one. but the old one should still exist.
        assert model_lin.active_adapters == ["intruder_reduced"]
        assert set(model_lin.peft_config.keys()) == {"default", "intruder_reduced"}

        buffer.seek(0)
        lines = buffer.readlines()

        assert len(lines) > 0
        assert any("q_proj" in line for line in lines)

        for name, module in model_lin.named_modules():
            if name in original_weights:
                # Make sure that the original adapter was not modified
                assert torch.equal(module.lora_B["default"].weight.detach(), original_weights[name])

                # Since the epsilon is really low, we should modify every layer so the weights should differ
                new_weight = module.lora_B["intruder_reduced"].weight.detach()
                assert not torch.equal(new_weight, original_weights[name])

    def test_lora_intruders_embedding(self, model_emb):
        original_weights = {}
        for name, module in model_emb.named_modules():
            if "embed_tokens" in name and hasattr(module, "lora_B"):
                original_weights[name] = module.lora_embedding_B["default"].detach().clone()

        buffer = StringIO()

        # use a high epsilon to make sure that we get a match to see whether layers get modified
        reduce_intruder_dimension(model_emb, threshold_epsilon=999, logging_sink=partial(print, file=buffer))

        # the old adapter should not be active anymore, just the new one. but the old one should still exist.
        assert model_emb.active_adapters == ["intruder_reduced"]
        assert set(model_emb.peft_config.keys()) == {"default", "intruder_reduced"}

        buffer.seek(0)
        lines = buffer.readlines()

        assert len(lines) > 0
        assert any("embed_tokens" in line for line in lines)

        for name, module in model_emb.named_modules():
            if name in original_weights:
                # Make sure that the original adapter was not modified
                assert torch.equal(module.lora_embedding_B["default"].detach(), original_weights[name])

                # Since the epsilon is really low, we should modify every layer so the weights should differ
                new_weight = module.lora_embedding_B["intruder_reduced"].detach()
                assert not torch.equal(new_weight, original_weights[name])

    def test_non_lora_intruders_linear_raises(self, model_lin_non_lora):
        with pytest.raises(ValueError) as e:
            reduce_intruder_dimension(model_lin_non_lora, threshold_epsilon=999)
            assert "The provided model is not using LoRA" in str(e)
