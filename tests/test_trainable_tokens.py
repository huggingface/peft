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

from peft import AutoPeftModel, LoraConfig, PeftModel, TrainableTokensConfig, get_peft_model


class TestTrainableTokens:
    @pytest.fixture
    def model_id(self):
        return "trl-internal-testing/tiny-random-LlamaForCausalLM"

    @pytest.fixture
    def model_multi_embedding(self):
        class MultiEmbeddingMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb_text = torch.nn.Embedding(10, 5)
                self.emb_image = torch.nn.Embedding(8, 5)
                self.lin0 = torch.nn.Linear(5, 10)
                self.lin1 = torch.nn.Linear(10, 20)

            def forward(self, x_text, x_image):
                x_text = self.emb_text(x_text)
                x_image = self.emb_image(x_image)
                y = self.lin0(torch.concat([x_text, x_image], dim=1).view(-1, 5))
                y = self.lin1(y)
                return y, (x_text, x_image)

        return MultiEmbeddingMLP()

    @pytest.fixture
    def model(self, model_id):
        return AutoModelForCausalLM.from_pretrained(model_id)

    @pytest.fixture
    def tokenizer(self, model_id):
        return AutoTokenizer.from_pretrained(model_id)

    def simulate_training(self, trainable_tokens_layer, adapter_name="default"):
        """Simulates training of trainable_tokens adapter layer by assigning random
        values to the delta tokens.
        """
        trainable_tokens_layer.trainable_tokens_delta[adapter_name].data = torch.rand_like(
            trainable_tokens_layer.trainable_tokens_delta[adapter_name].data
        )

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

        self.simulate_training(peft_model.model.model.embed_tokens)
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

        self.simulate_training(peft_model.model.model.embed_tokens.token_adapter)
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

    def test_basic_training(self, model, tokenizer):
        # ensure that the model can be trained and backpropagation works
        config = TrainableTokensConfig(
            target_modules=["embed_tokens"],
            token_indices=[0, 10],
        )

        model = get_peft_model(model, config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1)

        X = {
            "input_ids": torch.tensor([[0, 1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }

        for step in range(3):
            optimizer.zero_grad()
            y_pred = model(**X)
            loss = y_pred.logits.mean()
            loss.backward()
            optimizer.step()

    @pytest.mark.parametrize(
        "peft_config",
        [
            LoraConfig(
                target_modules="all-linear",
                trainable_token_indices={"embed_tokens": [0, 1, 3]},
            ),
        ],
    )
    def test_disable_adapters_with_merging(self, model, tokenizer, peft_config):
        X = {
            "input_ids": torch.tensor([[0, 1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }

        model = get_peft_model(model, peft_config)
        model.eval()

        outputs_before = model(**X).logits

        model.train()
        lr = 0.01
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # train at least 3 steps for all parameters to be updated (probably this is required because of symmetry
        # breaking of some LoRA layers that are initialized with constants)
        for _ in range(3):
            optimizer.zero_grad()
            y_pred = model(**X)
            loss = y_pred.logits.mean()
            loss.backward()
            optimizer.step()

        model.eval()
        outputs_unmerged = model(**X).logits
        model.merge_adapter()
        outputs_after = model(**X).logits

        with model.disable_adapter():
            outputs_disabled = model(**X).logits

        # check that after leaving the disable_adapter context, everything is enabled again
        outputs_enabled_after_disable = model(**X).logits

        atol, rtol = 1e-5, 1e-5  # tolerances higher than defaults since merging introduces some numerical instability

        # check that there is a difference in results after training
        assert not torch.allclose(outputs_before, outputs_after, atol=atol, rtol=rtol)

        # unmerged or merged should make no difference
        assert torch.allclose(outputs_after, outputs_unmerged, atol=atol, rtol=rtol)

        # check that disabling adapters gives the same results as before training
        assert torch.allclose(outputs_before, outputs_disabled, atol=atol, rtol=rtol)

        # check that enabling + disabling adapters does not change the results
        assert torch.allclose(outputs_after, outputs_enabled_after_disable, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        "peft_config",
        [
            LoraConfig(
                target_modules="all-linear",
                trainable_token_indices={"embed_tokens": [0, 1, 3]},
            ),
        ],
    )
    def test_safe_merge_with_adapter(self, model, tokenizer, peft_config):
        X = {
            "input_ids": torch.tensor([[0, 1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }

        model = model.eval()
        logits_base = model(**X).logits

        model = get_peft_model(model, peft_config).eval()
        logits_peft = model(**X).logits

        atol, rtol = 1e-6, 1e-6  # default

        model_unloaded = model.merge_and_unload(safe_merge=True)
        logits_unloaded = model_unloaded(**X).logits

        # check that the logits are the same after unloading
        assert torch.allclose(logits_peft, logits_unloaded, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        "peft_config",
        [
            LoraConfig(
                target_modules="all-linear",
                trainable_token_indices={"embed_tokens": [0, 1, 3]},
            ),
        ],
    )
    def test_load_multiple_adapters(self, model, peft_config, tmp_path):
        # tests if having more than one adpater (even with just the same config) works
        original_model = copy.deepcopy(model)
        model = get_peft_model(model, peft_config)

        model.save_pretrained(tmp_path)
        del model

        model = original_model
        model = PeftModel.from_pretrained(model, tmp_path)
        load_result1 = model.load_adapter(tmp_path, adapter_name="other")
        load_result2 = model.load_adapter(tmp_path, adapter_name="yet-another")

        assert load_result1.missing_keys == []
        assert load_result2.missing_keys == []

    @pytest.mark.parametrize(
        "peft_config_factory",
        [
            lambda token_indices: LoraConfig(
                target_modules="all-linear",
                trainable_token_indices={"embed_tokens": token_indices},
            ),
        ],
    )
    def test_multiple_adapters_different_token_indices(self, model, peft_config_factory, tmp_path):
        # tests if multiple adapters with different token indices work
        original_model = copy.deepcopy(model)

        token_indices_1 = [0, 1, 2]
        token_indices_2 = [2, 3, 4]

        peft_config_1 = peft_config_factory(token_indices_1)
        peft_config_2 = peft_config_factory(token_indices_2)

        model = get_peft_model(model, peft_config_1, adapter_name="adapter_1")
        model.add_adapter("adapter_2", peft_config_2)

        # "train" adapter 1
        model.set_adapter("adapter_1")
        self.simulate_training(model.model.model.embed_tokens.token_adapter, "adapter_1")

        # "train" adapter 2
        model.set_adapter("adapter_2")
        self.simulate_training(model.model.model.embed_tokens.token_adapter, "adapter_2")

        # now we infer on adapter 1 and on adapter 2 and check if the requested indices are changed for
        # each adapter. e.g., for adapter 1, only token indices 1 should be changed.
        X = {
            "input_ids": torch.tensor([list(set(token_indices_1 + token_indices_2))]),
            "attention_mask": torch.tensor([[1] * (len(set(token_indices_1 + token_indices_2)))]),
        }

        original_output = original_model.forward(output_hidden_states=True, **X).hidden_states[0]

        # infer with adapter 1, embeddings for token indices 1 should be changed, no others.
        model.set_adapter("adapter_1")
        adapter_1_output = model.forward(output_hidden_states=True, **X).hidden_states[0]

        idcs_to_modify = token_indices_1
        idcs_to_keep = [i for i in X["input_ids"][0].tolist() if i not in idcs_to_modify]

        assert not torch.allclose(adapter_1_output[:, idcs_to_modify], original_output[:, idcs_to_modify])
        assert torch.allclose(adapter_1_output[:, idcs_to_keep], original_output[:, idcs_to_keep])

        # infer with adapter 2, embeddings for token indices 2 should be changed, no others.
        model.set_adapter("adapter_2")
        adapter_2_output = model.forward(output_hidden_states=True, **X).hidden_states[0]

        idcs_to_modify = token_indices_2
        idcs_to_keep = [i for i in X["input_ids"][0].tolist() if i not in idcs_to_modify]

        assert not torch.allclose(adapter_2_output[:, idcs_to_modify], original_output[:, idcs_to_modify])
        assert torch.allclose(adapter_2_output[:, idcs_to_keep], original_output[:, idcs_to_keep])

    @pytest.mark.parametrize(
        "peft_config_factory",
        [
            lambda token_indices: LoraConfig(
                target_modules="all-linear",
                trainable_token_indices={"embed_tokens": token_indices},
            ),
        ],
    )
    def test_multiple_adapters_overlapping_token_indices_merging(self, model, peft_config_factory, tmp_path):
        # tests that merging multiple adapters that have overlapping indices is not defined at the moment
        # and would yield undefined behavior. note that merging a single adapter is fine.
        original_model = copy.deepcopy(model)

        token_indices_1 = [0, 1, 2]
        token_indices_2 = [2, 3, 4]

        peft_config_1 = peft_config_factory(token_indices_1)
        peft_config_2 = peft_config_factory(token_indices_2)

        model = get_peft_model(model, peft_config_1, adapter_name="adapter_1")
        model.add_adapter("adapter_2", peft_config_2)

        with pytest.raises(ValueError) as e:
            model.merge_and_unload(adapter_names=["adapter_1", "adapter_2"])
        assert "are already defined and would result in undefined merging behavior" in str(e)

    @pytest.mark.parametrize(
        "peft_config_factory",
        [
            lambda targets, token_indices: LoraConfig(
                target_modules=targets,
                trainable_token_indices={"embed_tokens": token_indices},
            ),
        ],
    )
    def test_multiple_adapters_mixed_forward(self, model, peft_config_factory, tmp_path):
        # tests if multiple adapters with different token indices work
        original_model = copy.deepcopy(model)

        token_indices_1 = [0, 1, 2]
        token_indices_2 = [2, 3, 4]

        peft_config_1 = peft_config_factory(".*q_proj", token_indices_1)
        peft_config_2 = peft_config_factory(".*o_proj", token_indices_2)

        model = get_peft_model(model, peft_config_1, adapter_name="adapter_1")
        model.add_adapter("adapter_2", peft_config_2)

        # "train" adapter 1
        model.set_adapter("adapter_1")
        self.simulate_training(model.model.model.embed_tokens.token_adapter, "adapter_1")

        # "train" adapter 2
        model.set_adapter("adapter_2")
        self.simulate_training(model.model.model.embed_tokens.token_adapter, "adapter_2")

        # forward(adapter_names=...) is not available in train mode
        model.eval()

        # Build a batch of 2 items, each the same input sequence but each sequence will be passed to a different
        # adapter via mixed batch forward.
        input_sequence = list(set(token_indices_1 + token_indices_2))
        X = {
            "input_ids": torch.tensor([input_sequence, input_sequence]),
            "attention_mask": torch.tensor([[1] * len(input_sequence), [1] * len(input_sequence)]),
        }
        batch_adapter_names = ["adapter_1", "adapter_2"]

        original_output = original_model.forward(output_hidden_states=True, **X)
        mixed_output = model.forward(output_hidden_states=True, adapter_names=batch_adapter_names, **X)

        # check that the active adapter is still the last activated adapter, adapter_2
        assert model.model.model.embed_tokens.token_adapter.active_adapter == ["adapter_2"]

        adapter_1_output = mixed_output.hidden_states[0][0:1]
        original_output_1 = original_output.hidden_states[0][0:1]
        adapter_2_output = mixed_output.hidden_states[0][1:2]
        original_output_2 = original_output.hidden_states[0][1:2]

        idcs_to_modify = token_indices_1
        idcs_to_keep = [i for i in X["input_ids"][0].tolist() if i not in idcs_to_modify]

        assert not torch.allclose(adapter_1_output[:, idcs_to_modify], original_output_1[:, idcs_to_modify])
        assert torch.allclose(adapter_1_output[:, idcs_to_keep], original_output_1[:, idcs_to_keep])

        idcs_to_modify = token_indices_2
        idcs_to_keep = [i for i in X["input_ids"][0].tolist() if i not in idcs_to_modify]

        assert not torch.allclose(adapter_2_output[:, idcs_to_modify], original_output_2[:, idcs_to_modify])
        assert torch.allclose(adapter_2_output[:, idcs_to_keep], original_output_2[:, idcs_to_keep])

    def test_stand_alone_raises_target_layer_not_found(self, model):
        config = TrainableTokensConfig(target_modules=["doesnt_exist"], token_indices=[0, 1, 3])
        with pytest.raises(ValueError) as e:
            model = get_peft_model(model, config)
        assert "Target modules ['doesnt_exist'] not found in the base model." in str(e)

    @pytest.mark.parametrize(
        "peft_config, target_layer_name",
        [
            (LoraConfig(trainable_token_indices=[0, 1, 2]), "embedding"),  # default layer 'embedding'
            (LoraConfig(trainable_token_indices={"does-not-exist": [0, 1, 2]}), "does-not-exist"),
        ],
    )
    def test_combined_with_peft_raises_target_layer_not_found(self, model, peft_config, target_layer_name):
        # same as test_stand_alone_raises_target_layer_not_found but tests the peft method integration
        with pytest.raises(ValueError) as e:
            model = get_peft_model(model, peft_config)
        assert f"Target modules {{{repr(target_layer_name)}}} not found in the base model." in str(e)

    def test_multiple_targets(self, model_multi_embedding):
        # tests the ability of targeting two modules with the same token indices
        original_model = copy.deepcopy(model_multi_embedding)
        config = TrainableTokensConfig(target_modules=["emb_text", "emb_image"], token_indices=[0, 1])
        peft_model = get_peft_model(model_multi_embedding, config)

        self.simulate_training(peft_model.model.emb_text)
        self.simulate_training(peft_model.model.emb_image)

        X = {
            "x_text": torch.tensor([[0, 1, 2]]),
            "x_image": torch.tensor([[0, 1, 2]]),
        }

        _, (emb_text_orig, emb_image_orig) = original_model.forward(**X)
        _, (emb_text_peft, emb_image_peft) = peft_model.forward(**X)

        assert not torch.allclose(emb_text_orig[:, [0, 1]], emb_text_peft[:, [0, 1]])
        assert torch.allclose(emb_text_orig[:, [2]], emb_text_peft[:, [2]])
        assert not torch.allclose(emb_image_orig[:, [0, 1]], emb_image_peft[:, [0, 1]])
        assert torch.allclose(emb_image_orig[:, [2]], emb_image_peft[:, [2]])
