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
import unittest

import torch
from transformers import AutoModelForCausalLM

from peft import AutoPeftModelForCausalLM, LoraConfig, PeftConfig, PeftModel, get_peft_model


PEFT_MODELS_TO_TEST = [("peft-internal-testing/test-lora-subfolder", "test")]


class PeftHubFeaturesTester(unittest.TestCase):
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
        model_id = "facebook/opt-125m"
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
