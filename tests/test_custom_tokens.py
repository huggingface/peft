import copy

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import AutoPeftModel, TrainableTokensConfig, LoraConfig, get_peft_model


class TestTrainableTokens:
    @pytest.fixture
    def model_id(self):
        return "trl-internal-testing/tiny-random-LlamaForCausalLM"

    @pytest.fixture
    def model(self, model_id):
        return AutoModelForCausalLM.from_pretrained(model_id)

    @pytest.fixture
    def tokenizer(self, model_id):
        return AutoTokenizer.from_pretrained(model_id)

    def test_stand_alone_usage(self, model, tokenizer, tmp_path):
        original_model = copy.deepcopy(model)
        peft_config = TrainableTokensConfig(target_modules=["embed_tokens"], token_indices=[0, 1, 2])
        peft_model = get_peft_model(model, peft_config)
        save_path = tmp_path / "stand_alone_usage"

        # simulate normal use but take care to use the tokens that we expect to be modified
        # (+1 that we don't expect to be modified)
        X = {
            "input_ids": torch.tensor([[0, 1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }
        output_trn = peft_model.forward(output_hidden_states=True, **X)

        peft_model.save_pretrained(save_path)
        peft_model_org = peft_model

        # check whether the token indices differ from the base model after loading the model
        # from the checkpoint.
        peft_model = AutoPeftModel.from_pretrained(save_path)
        output_mod = peft_model.forward(output_hidden_states=True, **X)
        output_org = original_model.forward(output_hidden_states=True, **X)

        # on the way, make sure that the embedding matrix itself was not modified
        assert torch.allclose(
            peft_model.model.model.embed_tokens.weight,
            peft_model_org.model.model.embed_tokens.weight,
        )

        W_mod = output_mod.hidden_states[0]
        W_org = output_org.hidden_states[0]
        W_trn = output_trn.hidden_states[0]

        # all PEFT model embed outputs must equal the outputs during 'training' to make sure
        # that saving/loading works properly.
        assert torch.allclose(W_mod, W_trn)

        assert not torch.allclose(W_mod[:, :3], W_org[:, :3])
        assert torch.allclose(W_mod[:, 3:], W_org[:, 3:])

    def test_combined_with_lora_usage(self, model, tokenizer, tmp_path):
        original_model = copy.deepcopy(model)
        peft_config = LoraConfig(
            target_modules="all-linear",
            trainable_token_indices={"embed_tokens": [0, 1, 2]},
        )
        peft_model = get_peft_model(model, peft_config)
        save_path = tmp_path / "stand_alone_usage"

        # simulate normal use but take care to use the tokens that we expect to be modified
        # (+1 that we don't expect to be modified)
        X = {
            "input_ids": torch.tensor([[0, 1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }
        output_trn = peft_model.forward(output_hidden_states=True, **X)

        peft_model.save_pretrained(save_path)
        peft_model_org = peft_model

        # check whether the token indices differ from the base model
        peft_model = AutoPeftModel.from_pretrained(save_path)
        output_mod = peft_model.forward(output_hidden_states=True, **X)
        output_org = original_model.forward(output_hidden_states=True, **X)

        W_mod = output_mod.hidden_states[0]
        W_org = output_org.hidden_states[0]
        W_trn = output_trn.hidden_states[0]

        # all PEFT model embed outputs must equal the outputs during 'training' to make sure
        # that saving/loading works properly.
        assert torch.allclose(W_mod, W_trn)

        assert not torch.allclose(W_mod[:, :3], W_org[:, :3])
        assert torch.allclose(W_mod[:, 3:], W_org[:, 3:])
