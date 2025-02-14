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

from __future__ import annotations

import copy

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import AutoPeftModel, LoraConfig, TrainableTokensConfig, get_peft_model


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

        peft_config = TrainableTokensConfig(target_modules=["embed_tokens"], token_indices=[0, 1, 3])
        peft_model = get_peft_model(model, peft_config)
        save_path = tmp_path / "stand_alone_usage"

        # simulate normal use but take care to use the tokens that we expect to be modified
        # (+1 that we don't expect to be modified)
        X = {
            "input_ids": torch.tensor([[0, 1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }

        idcs_to_modify = peft_config.token_indices
        idcs_to_keep = [i for i in X["input_ids"][0].tolist() if i not in idcs_to_modify]

        output_train = peft_model.forward(output_hidden_states=True, **X)

        peft_model.save_pretrained(save_path)
        peft_model_org = peft_model

        # check whether the token indices differ from the base model after loading the model
        # from the checkpoint.
        peft_model = AutoPeftModel.from_pretrained(save_path)
        output_load = peft_model.forward(output_hidden_states=True, **X)
        output_orig = original_model.forward(output_hidden_states=True, **X)

        # on the way, make sure that the embedding matrix itself was not modified
        assert torch.allclose(
            peft_model.model.model.embed_tokens.weight,
            peft_model_org.model.model.embed_tokens.weight,
        )

        W_load = output_load.hidden_states[0]
        W_orig = output_orig.hidden_states[0]
        W_train = output_train.hidden_states[0]

        # all PEFT model embed outputs must equal the outputs during 'training' to make sure
        # that saving/loading works properly.
        assert torch.allclose(W_load, W_train)

        assert not torch.allclose(W_load[:, idcs_to_modify], W_orig[:, idcs_to_modify])
        assert torch.allclose(W_load[:, idcs_to_keep], W_orig[:, idcs_to_keep])

    @pytest.mark.parametrize(
        "peft_config",
        [
            LoraConfig(
                target_modules="all-linear",
                trainable_token_indices={"embed_tokens": [0, 1, 3]},
            ),
        ],
    )
    def test_combined_with_peft_method_usage(self, model, tokenizer, peft_config, tmp_path):
        original_model = copy.deepcopy(model)
        peft_model = get_peft_model(model, peft_config)
        save_path = tmp_path / "combined_usage"

        # simulate normal use but take care to use the tokens that we expect to be modified
        # (+2 that we don't expect to be modified)
        X = {
            "input_ids": torch.tensor([[0, 1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }

        idcs_to_modify = peft_config.trainable_token_indices["embed_tokens"]
        idcs_to_keep = [i for i in X["input_ids"][0].tolist() if i not in idcs_to_modify]

        output_train = peft_model.forward(output_hidden_states=True, **X)

        peft_model.save_pretrained(save_path)
        peft_model_org = peft_model

        # check whether the token indices differ from the base model
        peft_model = AutoPeftModel.from_pretrained(save_path)
        output_load = peft_model.forward(output_hidden_states=True, **X)
        output_orig = original_model.forward(output_hidden_states=True, **X)

        W_load = output_load.hidden_states[0]
        W_orig = output_orig.hidden_states[0]
        W_train = output_train.hidden_states[0]

        # all PEFT model embed outputs must equal the outputs during 'training' to make sure
        # that saving/loading works properly.
        assert torch.allclose(W_load, W_train)

        assert not torch.allclose(W_load[:, idcs_to_modify], W_orig[:, idcs_to_modify])
        assert torch.allclose(W_load[:, idcs_to_keep], W_orig[:, idcs_to_keep])
