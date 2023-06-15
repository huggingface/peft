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
import os
import tempfile
import unittest

from peft import (
    AdaptionPromptConfig,
    LoraConfig,
    PeftConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
)


PEFT_MODELS_TO_TEST = [("lewtun/tiny-random-OPTForCausalLM-delta", "v1")]


class PeftConfigTestMixin:
    all_config_classes = (
        LoraConfig,
        PromptEncoderConfig,
        PrefixTuningConfig,
        PromptTuningConfig,
        AdaptionPromptConfig,
    )


class PeftConfigTester(unittest.TestCase, PeftConfigTestMixin):
    def test_methods(self):
        r"""
        Test if all configs have the expected methods. Here we test
        - to_dict
        - save_pretrained
        - from_pretrained
        - from_json_file
        """
        # test if all configs have the expected methods
        for config_class in self.all_config_classes:
            config = config_class()
            self.assertTrue(hasattr(config, "to_dict"))
            self.assertTrue(hasattr(config, "save_pretrained"))
            self.assertTrue(hasattr(config, "from_pretrained"))
            self.assertTrue(hasattr(config, "from_json_file"))

    def test_task_type(self):
        for config_class in self.all_config_classes:
            # assert this will not fail
            _ = config_class(task_type="test")

    def test_from_pretrained(self):
        r"""
        Test if the config is correctly loaded using:
        - from_pretrained
        """
        for config_class in self.all_config_classes:
            for model_name, revision in PEFT_MODELS_TO_TEST:
                # Test we can load config from delta
                _ = config_class.from_pretrained(model_name, revision=revision)

    def test_save_pretrained(self):
        r"""
        Test if the config is correctly saved and loaded using
        - save_pretrained
        """
        for config_class in self.all_config_classes:
            config = config_class()
            with tempfile.TemporaryDirectory() as tmp_dirname:
                config.save_pretrained(tmp_dirname)

                config_from_pretrained = config_class.from_pretrained(tmp_dirname)
                self.assertEqual(config.to_dict(), config_from_pretrained.to_dict())

    def test_from_json_file(self):
        for config_class in self.all_config_classes:
            config = config_class()
            with tempfile.TemporaryDirectory() as tmp_dirname:
                config.save_pretrained(tmp_dirname)

                config_from_json = config_class.from_json_file(os.path.join(tmp_dirname, "adapter_config.json"))
                self.assertEqual(config.to_dict(), config_from_json)

    def test_to_dict(self):
        r"""
        Test if the config can be correctly converted to a dict using:
        - to_dict
        - __dict__
        """
        for config_class in self.all_config_classes:
            config = config_class()
            self.assertEqual(config.to_dict(), config.__dict__)
            self.assertTrue(isinstance(config.to_dict(), dict))

    def test_from_pretrained_cache_dir(self):
        r"""
        Test if the config is correctly loaded with extra kwargs
        """
        with tempfile.TemporaryDirectory() as tmp_dirname:
            for config_class in self.all_config_classes:
                for model_name, revision in PEFT_MODELS_TO_TEST:
                    # Test we can load config from delta
                    _ = config_class.from_pretrained(model_name, revision=revision, cache_dir=tmp_dirname)

    def test_from_pretrained_cache_dir_remote(self):
        r"""
        Test if the config is correctly loaded with a checkpoint from the hub
        """
        with tempfile.TemporaryDirectory() as tmp_dirname:
            _ = PeftConfig.from_pretrained("ybelkada/test-st-lora", cache_dir=tmp_dirname)
            self.assertTrue("models--ybelkada--test-st-lora" in os.listdir(tmp_dirname))

    def test_set_attributes(self):
        # manually set attributes and check if they are correctly written
        for config_class in self.all_config_classes:
            config = config_class(peft_type="test")

            # save pretrained
            with tempfile.TemporaryDirectory() as tmp_dirname:
                config.save_pretrained(tmp_dirname)

                config_from_pretrained = config_class.from_pretrained(tmp_dirname)
                self.assertEqual(config.to_dict(), config_from_pretrained.to_dict())
