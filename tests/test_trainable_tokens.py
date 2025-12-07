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
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file as safe_load_file
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from peft import AutoPeftModel, LoraConfig, PeftModel, TrainableTokensConfig, get_peft_model
from peft.tuners.trainable_tokens.layer import TrainableTokensLayer
from peft.utils import TrainableTokensWrapper, get_peft_model_state_dict

from .testing_utils import hub_online_once


class ModelEmb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(100, 10)
        self.lin0 = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.lin0(self.emb(x))

    def get_input_embeddings(self):
        return self.emb


class ModelEmbedIn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_in = torch.nn.Embedding(100, 10)
        self.lin0 = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.lin0(self.embed_in(x))

    def get_input_embeddings(self):
        return self.embed_in


class ModelEmbedMultiple(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_in = torch.nn.Embedding(100, 10)
        self.embed_in_2 = torch.nn.Embedding(100, 10)
        self.lin0 = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.lin0(self.embed_in(x) + self.embed_in_2(x))

    def get_input_embeddings(self):
        return self.embed_in


class ModelEmbedInNoGet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_in = torch.nn.Embedding(100, 10)
        self.lin0 = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.lin0(self.embed_in(x))


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
        with hub_online_once(model_id):
            # This must not be a yield fixture so that we don't carry the hub_online_once
            # behavior over to the rest of the test that uses this fixture
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
        output_train = peft_model(output_hidden_states=True, **X)

        peft_model.save_pretrained(save_path)
        peft_model_org = peft_model

        # check whether the token indices differ from the base model after loading the model
        # from the checkpoint.
        peft_model = AutoPeftModel.from_pretrained(save_path)
        output_load = peft_model(output_hidden_states=True, **X)
        output_orig = original_model(output_hidden_states=True, **X)

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
        output_train = peft_model(output_hidden_states=True, **X)

        peft_model.save_pretrained(save_path)
        peft_model_org = peft_model

        # check whether the token indices differ from the base model
        peft_model = AutoPeftModel.from_pretrained(save_path)
        output_load = peft_model(output_hidden_states=True, **X)
        output_orig = original_model(output_hidden_states=True, **X)

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

        initial_delta = model.model.model.embed_tokens.trainable_tokens_delta.default.clone()
        initial_originals = model.model.model.embed_tokens.trainable_tokens_original.default.clone()

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

        assert torch.allclose(
            model.model.model.embed_tokens.trainable_tokens_original.default,
            initial_originals,
        )
        assert not torch.allclose(
            model.model.model.embed_tokens.trainable_tokens_delta.default,
            initial_delta,
        )

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

        original_output = original_model(output_hidden_states=True, **X).hidden_states[0]

        # infer with adapter 1, embeddings for token indices 1 should be changed, no others.
        model.set_adapter("adapter_1")
        adapter_1_output = model(output_hidden_states=True, **X).hidden_states[0]

        idcs_to_modify = token_indices_1
        idcs_to_keep = [i for i in X["input_ids"][0].tolist() if i not in idcs_to_modify]

        assert not torch.allclose(adapter_1_output[:, idcs_to_modify], original_output[:, idcs_to_modify])
        assert torch.allclose(adapter_1_output[:, idcs_to_keep], original_output[:, idcs_to_keep])

        # infer with adapter 2, embeddings for token indices 2 should be changed, no others.
        model.set_adapter("adapter_2")
        adapter_2_output = model(output_hidden_states=True, **X).hidden_states[0]

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

        original_output = original_model(output_hidden_states=True, **X)
        mixed_output = model(output_hidden_states=True, adapter_names=batch_adapter_names, **X)

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

        _, (emb_text_orig, emb_image_orig) = original_model(**X)
        _, (emb_text_peft, emb_image_peft) = peft_model(**X)

        assert not torch.allclose(emb_text_orig[:, [0, 1]], emb_text_peft[:, [0, 1]])
        assert torch.allclose(emb_text_orig[:, [2]], emb_text_peft[:, [2]])
        assert not torch.allclose(emb_image_orig[:, [0, 1]], emb_image_peft[:, [0, 1]])
        assert torch.allclose(emb_image_orig[:, [2]], emb_image_peft[:, [2]])

    @pytest.mark.parametrize(
        "peft_config",
        [
            LoraConfig(
                target_modules="all-linear",
                trainable_token_indices={"embed_tokens": [0, 1, 3]},
            ),
        ],
    )
    def test_no_embeddings_in_save_with_combined_usage(self, model, tokenizer, peft_config, tmp_path):
        # make sure that in combined use the only state dict key is that of the token deltas and nothing more

        peft_model = get_peft_model(model, peft_config)
        state_dict = get_peft_model_state_dict(
            model=peft_model,
            state_dict=None,
            adapter_name="default",
        )

        embedding_keys = [n for n in state_dict.keys() if "embed_tokens" in n]
        assert embedding_keys == ["base_model.model.model.embed_tokens.token_adapter.trainable_tokens_delta"]

    @pytest.fixture()
    def model_weight_untied(self, model):
        return model

    @pytest.fixture()
    def model_id_weight_tied(self):
        return "peft-internal-testing/opt-125m"

    @pytest.fixture()
    def model_weight_tied(self, request, model_id_weight_tied):
        model_weight_tied = AutoModelForCausalLM.from_pretrained(model_id_weight_tied)
        tied_keys = model_weight_tied._tied_weights_keys

        # TODO remove when transformers <5 is not supported anymore
        if not hasattr(request, "param") or request.param == "list":
            if isinstance(tied_keys, list):
                # transformers <5, list is already the default
                yield model_weight_tied
            else:
                # simulate transformers <5 for backward compatibility testing
                with patch.object(model_weight_tied, "_tied_weights_keys", list(tied_keys.keys())):
                    yield model_weight_tied

        elif request.param == "mapping":
            if isinstance(tied_keys, dict):
                # transformers >=5, mapping is already the default
                yield model_weight_tied
            else:
                # simulate transformers >=5
                mapping = {"lm_head.weight": "model.decoder.embed_tokens.weight"}

                with patch.object(model_weight_tied, "_tied_weights_keys", mapping):
                    yield model_weight_tied

        else:
            raise RuntimeError("Invalid request")

    @pytest.mark.parametrize(
        "peft_config",
        [
            LoraConfig(
                target_modules="all-linear",
                trainable_token_indices={"embed_tokens": [0, 1, 3]},
            ),
        ],
    )
    def test_weight_tying_noop_when_model_is_untied(self, model_weight_untied, peft_config, tmp_path):
        # test if the weight tying is affected as well when we modified the embedding.
        assert model_weight_untied._tied_weights_keys
        assert not model_weight_untied.config.tie_word_embeddings

        peft_model = get_peft_model(model_weight_untied, peft_config)
        assert hasattr(peft_model.model.model.embed_tokens, "token_adapter")
        assert not hasattr(peft_model.model.lm_head, "token_adapter")

    @pytest.mark.parametrize(
        "peft_config, model_weight_tied",
        [
            (
                LoraConfig(
                    target_modules="all-linear",
                    trainable_token_indices={"embed_tokens": [0, 1, 3]},
                ),
                "list",
            ),
            (
                LoraConfig(
                    target_modules="all-linear",
                    trainable_token_indices={"embed_tokens": [0, 1, 3]},
                ),
                "mapping",
            ),
        ],
        indirect=["model_weight_tied"],
    )
    def test_weight_tying_applied_when_model_is_tied(self, model_weight_tied, peft_config, tmp_path):
        # test if the weight tying is affected as well when we modified the embedding.
        assert model_weight_tied._tied_weights_keys
        assert model_weight_tied.config.tie_word_embeddings

        peft_model = get_peft_model(model_weight_tied, peft_config)

        # make it so that the input embeddings diverge. when the weights are tied this should
        # reflect in the output embeddings as well.
        self.simulate_training(peft_model.model.model.decoder.embed_tokens.token_adapter)

        # we have to find out if the input embedding tying is doing its job during forward.
        # for this we can leverage the fact that  emb_out(1/emb_in(x))  is  embed_dim  on the
        # diagonal iff emb_in.weight == emb_out.weight.
        token_indices = [0, 1, 2, 3]
        emb_dim = 768
        emb_in = peft_model.model.model.decoder.embed_tokens(torch.tensor([token_indices]))
        emb_out = peft_model.model.lm_head(1 / emb_in)

        assert torch.allclose(torch.diag(emb_out[0]), torch.tensor([emb_dim] * len(token_indices)).float())

        # make sure that the state dict does not include weight-tied weights.
        state_dict = get_peft_model_state_dict(peft_model)
        assert not [key for key in state_dict if any(tied_key in key for tied_key in peft_model._tied_weights_keys)]

        # make sure that merging and unloading restores the weight-tying.
        merged_model = peft_model.merge_and_unload()

        assert merged_model.model.decoder.embed_tokens.weight.data_ptr() == merged_model.lm_head.weight.data_ptr()

    @pytest.mark.parametrize("model_weight_tied", ["list", "mapping"], indirect=["model_weight_tied"])
    def test_weight_tying_applied_when_model_is_tied_standalone(self, model_weight_tied):
        # since weight tying is currently not supported make sure that an error is raised when attempting
        # to use a model that has tied input/output embeddings
        assert model_weight_tied._tied_weights_keys
        assert model_weight_tied.config.tie_word_embeddings

        peft_config = TrainableTokensConfig(
            target_modules=["embed_tokens"],
            token_indices=[0, 1, 3],
        )

        peft_model = get_peft_model(model_weight_tied, peft_config)

        # make it so that the input embeddings diverge. when the weights are tied this should
        # reflect in the output embeddings as well.
        self.simulate_training(peft_model.model.model.decoder.embed_tokens)

        # we have to find out if the input embedding tying is doing its job during forward.
        # for this we can leverage the fact that  emb_out(1/emb_in(x))  is  embed_dim  on the
        # diagonal iff  emb_in.weight == emb_out.weight.
        token_indices = [0, 1, 2, 3]
        emb_dim = 768
        emb_in = peft_model.model.model.decoder.embed_tokens(torch.tensor([token_indices]))
        emb_out = peft_model.model.lm_head(1 / emb_in)

        assert torch.allclose(torch.diag(emb_out[0]), torch.tensor([emb_dim] * len(token_indices)).float())

        # make sure that the state dict does not include weight-tied weights.
        state_dict = get_peft_model_state_dict(peft_model)
        assert not [key for key in state_dict if any(tied_key in key for tied_key in peft_model._tied_weights_keys)]

        # make sure that merging and unloading restores the weight-tying.
        merged_model = peft_model.merge_and_unload()

        assert merged_model.model.decoder.embed_tokens.weight.data_ptr() == merged_model.lm_head.weight.data_ptr()

    def test_weight_tying_normally_issues_warning(self, model_weight_tied, recwarn):
        # When using models with weight tying and targeting the embedding or the tied layer should raise a warning.
        peft_config = LoraConfig(target_modules=["embed_tokens"])
        peft_model = get_peft_model(model_weight_tied, peft_config)

        warnings = [w.message.args[0] for w in recwarn]
        warnings = [msg for msg in warnings if "Model with `tie_word_embeddings=True` and the" in msg]
        assert warnings

    def test_weight_tying_state_dict_ignores_tied_weights(self, model_weight_tied):
        # since weight tying is currently not supported make sure that an error is raised when attempting
        # to use a model that has tied input/output embeddings
        assert model_weight_tied._tied_weights_keys
        assert model_weight_tied.config.tie_word_embeddings

        peft_config = TrainableTokensConfig(
            target_modules=["embed_tokens"],
            token_indices=[0, 1, 3],
        )

        peft_model = get_peft_model(model_weight_tied, peft_config)

        state_dict = peft_model.state_dict()
        peft_state_dict = get_peft_model_state_dict(peft_model)

        # the state dict or the peft model state dict must not include tied adapter weights
        state_dict_keys = [n for n, _ in state_dict.items() if "tied_adapter." in n]
        peft_state_dict_keys = [n for n, _ in peft_state_dict.items() if "tied_adapter." in n]

        assert not state_dict_keys
        assert not peft_state_dict_keys

    @pytest.mark.parametrize(
        "peft_config",
        [
            LoraConfig(
                target_modules="all-linear",
                trainable_token_indices={"shared": [0, 1, 3]},
            ),
        ],
    )
    def test_weight_tying_applied_when_model_is_tied_encoder_decoder(self, peft_config):
        model_id = "peft-internal-testing/tiny-random-t5"
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

        peft_model = get_peft_model(base_model, peft_config)

        # make it so that the input embeddings diverge. when the weights are tied this should
        # reflect in the output embeddings as well.
        self.simulate_training(peft_model.model.shared.token_adapter)

        # we have to find out if the input embedding tying is doing its job during forward.
        # for this we can leverage the fact that  emb_out(1/emb_in(x))  is  embed_dim  on the
        # diagonal iff  emb_in.weight == emb_out.weight.
        token_indices = [0, 1, 2, 3]
        emb_dim = base_model.config.d_model
        emb_in = peft_model.model.encoder.embed_tokens(torch.tensor([token_indices]))
        emb_out = peft_model.model.lm_head(1 / emb_in)

        assert torch.allclose(torch.diag(emb_out[0]), torch.tensor([emb_dim] * len(token_indices)).float())

        # T5 has a decoder embedding layer, we can simply check if it's forward is equal to the encoder
        # embedding forward.
        emb_out = peft_model.model.decoder.embed_tokens(torch.tensor([token_indices]))

        assert torch.allclose(emb_in, emb_out)

        # make sure that the state dict does not include weight-tied weights.
        state_dict = get_peft_model_state_dict(peft_model)
        assert not [key for key in state_dict if any(tied_key in key for tied_key in peft_model._tied_weights_keys)]

        # make sure that merging and unloading restores the weight-tying.
        merged_model = peft_model.merge_and_unload()

        assert merged_model.encoder.embed_tokens.weight.data_ptr() == merged_model.lm_head.weight.data_ptr()
        assert (
            merged_model.encoder.embed_tokens.weight.data_ptr() == merged_model.decoder.embed_tokens.weight.data_ptr()
        )

    @pytest.mark.parametrize(
        "peft_config",
        [
            LoraConfig(
                target_modules="all-linear",
                trainable_token_indices={"embed_tokens": [0, 1, 3]},
                modules_to_save=["embed_tokens"],
            ),
        ],
    )
    def test_modules_to_save_excludes_trainable_tokens(self, model, peft_config):
        with pytest.raises(ValueError) as e:
            get_peft_model(model, peft_config)
        assert "The embedding layer is already marked to be trained fully" in str(e)

    def test_merge_and_unload_standalone(self, model):
        # test basic functionality of merge_and_unload for standalone TrainableTokens
        token_indices = [0, 1, 3]

        peft_config = TrainableTokensConfig(
            target_modules=["embed_tokens"],
            token_indices=token_indices,
        )

        peft_model = get_peft_model(model, peft_config)

        self.simulate_training(peft_model.model.model.embed_tokens)
        expected_changed_weights = peft_model.model.model.embed_tokens.trainable_tokens_delta.default.data.clone()

        # make sure no TrainableTokensLayer is in the module
        merged_model = peft_model.merge_and_unload()
        for _, module in merged_model.named_modules():
            assert not isinstance(module, TrainableTokensLayer)

        # make sure that deltas are applied to the embedding matrix
        assert torch.allclose(merged_model.model.embed_tokens.weight.data[token_indices], expected_changed_weights)

    def test_original_module_not_in_state_dict(self, model):
        # Every AuxiliaryTrainingWrapper has an original_module attribute. Since the TrainableTokensWrapper is wrapping
        # a TrainableTokensLayer and it already has a base layer which serves as the original module, we don't need that
        # and so it should not come up in the state dict to save memory.

        peft_config = LoraConfig(
            target_modules="all-linear",
            trainable_token_indices={"embed_tokens": [0, 1, 3]},
        )

        peft_model = get_peft_model(model, peft_config)

        # make sure that the original module is present and accessible even though
        # we want to exclude it from the state dict.
        assert peft_model.model.model.embed_tokens.original_module

        state_dict = get_peft_model_state_dict(peft_model)

        assert not [k for k in state_dict if ".original_module.weight" in k]

        state_dict = peft_model.state_dict()
        assert not [k for k in state_dict if ".original_module.weight" in k]

    @pytest.fixture
    def model_emb(self):
        return ModelEmb()

    @pytest.fixture
    def model_embed_in(self):
        return ModelEmbedIn()

    @pytest.fixture
    def model_embed_in_no_get(self):
        return ModelEmbedInNoGet()

    @pytest.fixture
    def model_embed_multiple(self):
        return ModelEmbedMultiple()

    @pytest.mark.parametrize(
        "model_fixture_name, getter",
        [
            ("model_emb", lambda model: model.emb),
            ("model_embed_in", lambda model: model.embed_in),
            ("model", lambda model: model.model.model.embed_tokens),
        ],
    )
    def test_default_embedding_name_is_inferred_standalone(self, model_fixture_name, getter, request):
        # make sure that the auto targeting works when `target_module=None`
        base_model = request.getfixturevalue(model_fixture_name)

        peft_config = TrainableTokensConfig(target_modules=None, token_indices=[0, 1, 3])
        peft_model = get_peft_model(base_model, peft_config)

        assert isinstance(getter(peft_model), TrainableTokensLayer)

    @pytest.mark.parametrize(
        "model_fixture_name, getter",
        [
            ("model_emb", lambda model: model.emb),
            ("model_embed_in", lambda model: model.embed_in),
            ("model", lambda model: model.model.model.embed_tokens),
        ],
    )
    def test_default_embedding_name_is_inferred_combined(self, model_fixture_name, getter, request):
        # make sure that the auto targeting works when `target_module=None`
        base_model = request.getfixturevalue(model_fixture_name)

        peft_config = LoraConfig(target_modules="all-linear", trainable_token_indices=[0, 1, 3])
        peft_model = get_peft_model(base_model, peft_config)

        assert isinstance(getter(peft_model), TrainableTokensWrapper)

    def test_default_embedding_name_cannot_be_inferred(self, model_embed_in_no_get):
        # should default to default value `embed_tokens` which is not present in this model
        base_model = model_embed_in_no_get

        peft_config = TrainableTokensConfig(target_modules=None, token_indices=[0, 1, 3])

        with pytest.raises(ValueError) as e:
            peft_model = get_peft_model(base_model, peft_config)

        assert "Target modules embed_tokens not found in the base model." in str(e)

    def test_embedding_name_is_used_when_given_standalone(self, model_embed_multiple):
        peft_config = TrainableTokensConfig(target_modules="embed_in_2", token_indices=[0, 1, 3])
        peft_model = get_peft_model(model_embed_multiple, peft_config)

        assert isinstance(peft_model.model.embed_in_2, TrainableTokensLayer)
        assert not isinstance(peft_model.model.embed_in, TrainableTokensLayer)

    def test_embedding_name_is_used_when_given_combined(self, model_embed_multiple):
        peft_config = LoraConfig(target_modules="all-linear", trainable_token_indices={"embed_in_2": [0, 1, 3]})
        peft_model = get_peft_model(model_embed_multiple, peft_config)

        assert isinstance(peft_model.model.embed_in_2, TrainableTokensWrapper)
        assert not isinstance(peft_model.model.embed_in, TrainableTokensWrapper)

    @pytest.mark.parametrize("resize_embedding", [True, False])
    @pytest.mark.parametrize(
        "peft_config",
        [
            LoraConfig(target_modules="all-linear", trainable_token_indices=[1, 2, 3]),
            TrainableTokensConfig(target_modules=None, token_indices=[1, 2, 3]),
        ],
    )
    def test_save_pretrained_auto(self, model, resize_embedding, peft_config, tmp_path):
        # make sure that embeddings are saved alongside trainable token weights but only when
        # the we detect the embedding to be resized (as detected by save_embedding_layers="auto")
        if resize_embedding:
            model.resize_token_embeddings(model.config.vocab_size + 2)
        peft_model = get_peft_model(model, peft_config)

        peft_model.save_pretrained(tmp_path, save_embedding_layers="auto")
        state_dict = safe_load_file(tmp_path / "adapter_model.safetensors")

        if isinstance(peft_config, TrainableTokensConfig):
            contains_embedding = "base_model.model.model.embed_tokens.base_layer.weight" in state_dict
        else:
            contains_embedding = "base_model.model.model.embed_tokens.token_adapter.base_layer.weight" in state_dict

        if resize_embedding:
            assert contains_embedding
        else:
            assert not contains_embedding

    def test_embed_scale_is_applied(self):
        """Test that TrainableTokens correctly handles embeddings with scaling (e.g., Gemma3)."""
        model_id = "hf-internal-testing/tiny-random-Gemma3ForCausalLM"
        with hub_online_once(model_id):
            base_model = AutoModelForCausalLM.from_pretrained(model_id)
            orig_embedding = base_model.get_input_embeddings()

            peft_config = TrainableTokensConfig(target_modules=["embed_tokens"], token_indices=[0, 1, 3])
            peft_model = get_peft_model(base_model, peft_config)

            # sanity check: with the default embed_scale, the embedding output should be reasonably sized
            peft_embedding = peft_model.base_model.model.get_input_embeddings()
            max_embedding_output = peft_embedding(torch.arange(10)).abs().max(0)[0]
            assert (max_embedding_output < 100.0).all()

            # set embed_scale to an absurdly high value, then check that the embedding output is also scaled to a high
            # value
            orig_embedding.embed_scale.fill_(10000.0)
            max_embedding_output = peft_embedding(torch.arange(10)).abs().max(0)[0]
            assert (max_embedding_output > 100.0).all()

            # set embed_scale to zero, then check that the embedding output is also zero
            orig_embedding.embed_scale.fill_(0)
            embedding_output = peft_embedding(torch.arange(10))
            assert (embedding_output == 0.0).all()

    def test_scaled_embedding_with_lora(self):
        """
        Test that TrainableTokens works with LoRA on scaled embeddings when both are active simultaneously.
        """
        model_id = "hf-internal-testing/tiny-random-Gemma3ForCausalLM"
        with hub_online_once(model_id):
            base_model = AutoModelForCausalLM.from_pretrained(model_id)
            orig_embedding = base_model.get_input_embeddings()

            # Apply both TrainableTokens and LoRA to the same model
            peft_config = LoraConfig(target_modules=["q_proj"], trainable_token_indices={"embed_tokens": [0, 1, 3]})
            peft_model = get_peft_model(base_model, peft_config)

            x = torch.arange(10)
            peft_embedding = peft_model.base_model.model.get_input_embeddings()
            embedding_output = peft_embedding(x)
            max_embedding_output = embedding_output.abs().max(0)[0]
            assert (max_embedding_output < 100.0).all()
            peft_model.merge_adapter()
            embedding_merged = peft_embedding(x)
            assert torch.allclose(embedding_output, embedding_merged)
            peft_model.unmerge_adapter()

            # set embed_scale to an absurdly high value, then check that the embedding output is also scaled to a high
            # value
            orig_embedding.embed_scale.fill_(10000.0)
            max_embedding_output = peft_embedding(x).abs().max(0)[0]
            assert (max_embedding_output > 100.0).all()

            # set embed_scale to zero, then check that the embedding output is also zero
            orig_embedding.embed_scale.fill_(0)
            embedding_output = peft_embedding(x)
            assert (embedding_output == 0.0).all()
