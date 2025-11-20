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
import copy

import pytest
import torch
from huggingface_hub import ModelCard
from transformers import AutoModelForCausalLM

from peft import AutoPeftModelForCausalLM, BoneConfig, LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model

from .testing_utils import hub_online_once


PEFT_MODELS_TO_TEST = [("peft-internal-testing/test-lora-subfolder", "test")]


class PeftHubFeaturesTester:
    # TODO remove when/if Hub is more stable
    @pytest.mark.xfail(reason="Test is flaky on CI", raises=ValueError)
    def test_subfolder(self):
        r"""
        Test if subfolder argument works as expected
        """
        for model_id, subfolder in PEFT_MODELS_TO_TEST:
            config = PeftConfig.from_pretrained(model_id, subfolder=subfolder)

            model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
            )
            model = PeftModel.from_pretrained(model, model_id, subfolder=subfolder)

            assert isinstance(model, PeftModel)


class TestLocalModel:
    def test_local_model_saving_no_warning(self, recwarn, tmp_path):
        # When the model is saved, the library checks for vocab changes by
        # examining `config.json` in the model path.
        # However, previously, those checks only covered huggingface hub models.
        # This test makes sure that the local `config.json` is checked as well.
        # If `save_pretrained` could not find the file, it will issue a warning.
        model_id = "peft-internal-testing/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        local_dir = tmp_path / model_id
        model.save_pretrained(local_dir)
        del model

        base_model = AutoModelForCausalLM.from_pretrained(local_dir)
        peft_config = LoraConfig()
        peft_model = get_peft_model(base_model, peft_config)
        peft_model.save_pretrained(local_dir)

        for warning in recwarn.list:
            assert "Could not find a config file" not in warning.message.args[0]


class TestBaseModelRevision:
    def test_save_and_load_base_model_revision(self, tmp_path):
        r"""
        Test saving a PeftModel with a base model revision and loading with AutoPeftModel to recover the same base
        model
        """
        lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0)
        test_inputs = torch.arange(10).reshape(-1, 1)

        base_model_id = "peft-internal-testing/tiny-random-BertModel"
        revision = "v2.0.0"

        base_model_revision = AutoModelForCausalLM.from_pretrained(base_model_id, revision=revision).eval()
        peft_model_revision = get_peft_model(base_model_revision, lora_config, revision=revision)
        output_revision = peft_model_revision(test_inputs).logits

        # sanity check: the model without revision should be different
        base_model_no_revision = AutoModelForCausalLM.from_pretrained(base_model_id, revision="main").eval()
        # we need a copy of the config because otherwise, we are changing in-place the `revision` of the previous config and model
        lora_config_no_revision = copy.deepcopy(lora_config)
        lora_config_no_revision.revision = "main"
        peft_model_no_revision = get_peft_model(base_model_no_revision, lora_config_no_revision, revision="main")
        output_no_revision = peft_model_no_revision(test_inputs).logits
        assert not torch.allclose(output_no_revision, output_revision)

        # check that if we save and load the model, the output corresponds to the one with revision
        peft_model_revision.save_pretrained(tmp_path / "peft_model_revision")
        peft_model_revision_loaded = AutoPeftModelForCausalLM.from_pretrained(tmp_path / "peft_model_revision").eval()

        assert peft_model_revision_loaded.peft_config["default"].revision == revision

        output_revision_loaded = peft_model_revision_loaded(test_inputs).logits
        assert torch.allclose(output_revision, output_revision_loaded)

    def test_load_different_peft_and_base_model_revision(self, tmp_path):
        r"""
        Test loading an AutoPeftModel from the hub where the base model revision and peft revision differ
        """
        base_model_id = "hf-internal-testing/tiny-random-BertModel"
        base_model_revision = None
        peft_model_id = "peft-internal-testing/tiny-random-BertModel-lora"
        peft_model_revision = "v1.2.3"

        peft_model = AutoPeftModelForCausalLM.from_pretrained(peft_model_id, revision=peft_model_revision).eval()

        assert peft_model.peft_config["default"].base_model_name_or_path == base_model_id
        assert peft_model.peft_config["default"].revision == base_model_revision


class TestModelCard:
    @pytest.mark.parametrize(
        "model_id, peft_config, tags, excluded_tags, pipeline_tag",
        [
            (
                "hf-internal-testing/tiny-random-Gemma3ForCausalLM",
                LoraConfig(),
                ["transformers", "base_model:adapter:hf-internal-testing/tiny-random-Gemma3ForCausalLM", "lora"],
                [],
                None,
            ),
            (
                "hf-internal-testing/tiny-random-Gemma3ForCausalLM",
                BoneConfig(),
                ["transformers", "base_model:adapter:hf-internal-testing/tiny-random-Gemma3ForCausalLM"],
                ["lora"],
                None,
            ),
            (
                "peft-internal-testing/tiny-random-BartForConditionalGeneration",
                LoraConfig(),
                [
                    "transformers",
                    "base_model:adapter:peft-internal-testing/tiny-random-BartForConditionalGeneration",
                    "lora",
                ],
                [],
                None,
            ),
            (
                "hf-internal-testing/tiny-random-Gemma3ForCausalLM",
                LoraConfig(task_type=TaskType.CAUSAL_LM),
                ["transformers", "base_model:adapter:hf-internal-testing/tiny-random-Gemma3ForCausalLM", "lora"],
                [],
                "text-generation",
            ),
        ],
    )
    @pytest.mark.parametrize(
        "pre_tags",
        [
            ["tag1", "tag2"],
            [],
        ],
    )
    def test_model_card_has_expected_tags(
        self, model_id, peft_config, tags, excluded_tags, pipeline_tag, pre_tags, tmp_path
    ):
        """Make sure that PEFT sets the tags in the model card automatically and correctly.
        This is important so that a) the models are searchable on the Hub and also 2) some features depend on it to
        decide how to deal with them (e.g., inference).

        Makes sure that the base model tags are still present (if there are any).
        """
        with hub_online_once(model_id):
            base_model = AutoModelForCausalLM.from_pretrained(model_id)

            if pre_tags:
                base_model.add_model_tags(pre_tags)

            peft_model = get_peft_model(base_model, peft_config)
            save_path = tmp_path / "adapter"

            peft_model.save_pretrained(save_path)

            model_card = ModelCard.load(save_path / "README.md")
            assert set(tags).issubset(set(model_card.data.tags))

            if excluded_tags:
                assert set(excluded_tags).isdisjoint(set(model_card.data.tags))

            if pre_tags:
                assert set(pre_tags).issubset(set(model_card.data.tags))

            if pipeline_tag:
                assert model_card.data.pipeline_tag == pipeline_tag

    @pytest.fixture
    def custom_model_cls(self):
        class MyNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = torch.nn.Linear(10, 20)
                self.l2 = torch.nn.Linear(20, 1)

            def forward(self, X):
                return self.l2(self.l1(X))

        return MyNet

    def test_custom_models_dont_have_transformers_tag(self, custom_model_cls, tmp_path):
        base_model = custom_model_cls()
        peft_config = LoraConfig(target_modules="all-linear")
        peft_model = get_peft_model(base_model, peft_config)

        peft_model.save_pretrained(tmp_path)

        model_card = ModelCard.load(tmp_path / "README.md")

        assert model_card.data.tags is not None
        assert "transformers" not in model_card.data.tags

    def test_custom_peft_type_does_not_raise(self, tmp_path):
        # Passing a string value as peft_type value in the config is valid, so it should work.
        # See https://github.com/huggingface/peft/issues/2634
        model_id = "hf-internal-testing/tiny-random-Gemma3ForCausalLM"
        with hub_online_once(model_id):
            base_model = AutoModelForCausalLM.from_pretrained(model_id)
            peft_config = LoraConfig()

            # We simulate a custom PEFT type by using a string value of an existing method.  This skips the need for
            # registering a new method but tests the case where we pass a string value instead of an enum.
            peft_type = "LORA"
            peft_config.peft_type = peft_type

            peft_model = get_peft_model(base_model, peft_config)
            peft_model.save_pretrained(tmp_path)
