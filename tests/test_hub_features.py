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

from transformers import AutoModelForCausalLM

from peft import PeftConfig, PeftModel


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
