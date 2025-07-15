#!/usr/bin/env python3

# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
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
import pytest
import torch
from transformers import AutoModelForCausalLM

from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, inject_adapter_in_model
from peft.utils import ModulesToSaveWrapper

from .testing_common import hub_online_once


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 10)
        self.linear = torch.nn.Linear(10, 10)
        self.linear2 = torch.nn.Linear(10, 10, bias=True)
        self.lm_head = torch.nn.Linear(10, 10)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.linear(x)
        x = self.lm_head(x)
        return x


@pytest.fixture
def dummy_peft_model():
    """
    Creates a DummyModel and injects a LoRA adapter into it. This fixture is used by tests that need a pre-configured
    PEFT model.
    """
    model = DummyModel()
    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        target_modules=["linear"],
    )
    return inject_adapter_in_model(lora_config, model)


def test_inject_adapter_in_model(dummy_peft_model):
    dummy_inputs = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]])
    _ = dummy_peft_model(dummy_inputs)

    for name, module in dummy_peft_model.named_modules():
        if name == "linear":
            assert hasattr(module, "lora_A")
            assert hasattr(module, "lora_B")


def test_modules_to_save():
    model = DummyModel()

    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        target_modules=["linear"],
        modules_to_save=["embedding", "linear2"],
    )

    model = inject_adapter_in_model(lora_config, model)

    # Check for LoRA injection and module wrapping
    for name, module in model.named_modules():
        if name == "linear":
            assert hasattr(module, "lora_A")
            assert hasattr(module, "lora_B")
        elif name in ["embedding", "linear2"]:
            assert isinstance(module, ModulesToSaveWrapper)

    # Check that the state dict includes the saved modules
    state_dict = get_peft_model_state_dict(model)
    assert "embedding.weight" in state_dict
    assert "linear2.weight" in state_dict
    assert "linear2.bias" in state_dict

    # Check that original attributes are still accessible
    assert hasattr(model.embedding, "weight")
    assert hasattr(model.linear2, "weight")
    assert hasattr(model.linear2, "bias")


class TestGetStateDict:
    def test_get_peft_model_state_dict(self, dummy_peft_model):
        peft_state_dict = get_peft_model_state_dict(dummy_peft_model)

        # Ensure all keys in the state dict are for LoRA parameters
        for key in peft_state_dict.keys():
            assert "lora" in key

    # Testing save_embedding_layer="auto" needs to check the following logic:
    #
    # - when vocab size was NOT changed, embeddings should be saved only when targeted
    # however
    # - when vocab size was changed, embeddings should be saved automatically
    # but not when
    # - using PeftType.TRAINABLE_TOKENS
    # - LoRA using trainable_token_indices (because trainable tokens's benefit is that we only store the diff)
    @pytest.mark.parametrize(
        "peft_config, embedding_changed, expect_embedding",
        [
            # embeddings are not changed, no auto-saving
            (LoraConfig(target_modules="all-linear"), False, False),
            (LoraConfig(target_modules=["q_proj", "embed_tokens"]), False, True),
            (LoraConfig(target_modules=r".*\.embed_tokens"), False, True),
            # embeddings are changed, auto-saving since we cannot know if the embedding modification
            # was done using model.resize_token_embeddings or some other way and the adapter depends on it
            (LoraConfig(target_modules="all-linear"), True, True),
            (LoraConfig(target_modules=["q_proj", "embed_tokens"]), True, True),
            (LoraConfig(target_modules=r".*\.embed_tokens"), True, True),
            # embeddings are changed, trainable tokens is used -> no auto-saving since we expect trainable tokens
            # to cover the diff.
            (LoraConfig(target_modules="all-linear", trainable_token_indices=[1, 2, 3]), True, False),
            (LoraConfig(target_modules="all-linear", trainable_token_indices=[1, 2, 3]), False, False),
        ],
    )
    def test_save_embeddings_auto(self, peft_config, embedding_changed, expect_embedding):
        model_id = "trl-internal-testing/tiny-random-LlamaForCausalLM"
        with hub_online_once(model_id):
            base_model = AutoModelForCausalLM.from_pretrained(model_id)

            if embedding_changed:
                # Make sure to modify the embeddings so that auto saving is activated. We need to do that beforehand
                # since resizing a targeted layer doesn't work
                base_model.resize_token_embeddings(base_model.config.vocab_size + 2)

            peft_model = get_peft_model(base_model, peft_config)

        # important not to cache this call with `hub_online_once` as it attempts to fetch a config in some cases and the
        # caching is not aware of when that is, it will just cache once the call is completed.
        state_dict = get_peft_model_state_dict(peft_model, save_embedding_layers="auto")

        contains_embedding = (
            # not adapted, only resized -> 'normal' module path
            "base_model.model.model.embed_tokens.weight" in state_dict
            or
            # adapted (and possibly resized) -> base layer module path
            "base_model.model.model.embed_tokens.base_layer.weight" in state_dict
        )

        if expect_embedding:
            assert contains_embedding
        else:
            assert not contains_embedding
