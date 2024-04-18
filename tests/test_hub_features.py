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
import unittest

import torch
from transformers import AutoModelForCausalLM

from peft import AutoPeftModelForCausalLM, LoraConfig, PeftConfig, PeftModel, get_peft_model


PEFT_MODELS_TO_TEST = [("peft-internal-testing/test-lora-subfolder", "test")]

BASE_REVISION_MODELS_TO_TEST = [("peft-internal-testing/tiny-random-BertModel", "v2.0.0")]


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


class TestBaseModelRevision:
    def test_save_and_load_base_model_revision(self, tmp_path):
        r"""
        Test if subfolder argument works as expected
        """
        lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, init_lora_weights=False)
        test_inputs = torch.arange(10).reshape(-1, 1)

        for model_id, revision in BASE_REVISION_MODELS_TO_TEST:
            original_base_model = AutoModelForCausalLM.from_pretrained(model_id, revision="main").eval()
            original_peft_model = get_peft_model(original_base_model, lora_config)
            original_peft_sum = original_peft_model(test_inputs).logits.sum()

            revised_base_model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision).eval()
            revised_peft_model = get_peft_model(revised_base_model, lora_config)
            revised_peft_sum = revised_peft_model(test_inputs).logits.sum()

            assert not torch.eq(
                original_peft_sum, revised_peft_sum
            ), f"revision 'main' and {revision} of base model {model_id} must differ"

            revised_peft_model.save_pretrained(tmp_path / f"base_{revision}_model")

            reload_revised_peft_model = AutoPeftModelForCausalLM.from_pretrained(
                tmp_path / f"base_{revision}_model"
            ).eval()
            reload_revised_sum = reload_revised_peft_model(test_inputs).logits.sum()

            assert torch.eq(reload_revised_sum, reload_revised_sum)
