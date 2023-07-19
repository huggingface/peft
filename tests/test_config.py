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
import copy
import os
import pickle
import tempfile
import unittest
import warnings

import pytest

from peft import (
    AdaptionPromptConfig,
    IA3Config,
    LoraConfig,
    PeftConfig,
    PrefixTuningConfig,
    PromptEncoder,
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
        IA3Config,
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
        """
        for config_class in self.all_config_classes:
            config = config_class()
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

    def test_config_copy(self):
        # see https://github.com/huggingface/peft/issues/424
        for config_class in self.all_config_classes:
            config = config_class()
            copied = copy.copy(config)
            self.assertEqual(config.to_dict(), copied.to_dict())

    def test_config_deepcopy(self):
        # see https://github.com/huggingface/peft/issues/424
        for config_class in self.all_config_classes:
            config = config_class()
            copied = copy.deepcopy(config)
            self.assertEqual(config.to_dict(), copied.to_dict())

    def test_config_pickle_roundtrip(self):
        # see https://github.com/huggingface/peft/issues/424
        for config_class in self.all_config_classes:
            config = config_class()
            copied = pickle.loads(pickle.dumps(config))
            self.assertEqual(config.to_dict(), copied.to_dict())

    def test_prompt_encoder_warning_num_layers(self):
        # This test checks that if a prompt encoder config is created with an argument that is ignored, there should be
        # warning. However, there should be no warning if the default value is used.
        kwargs = {
            "num_virtual_tokens": 20,
            "num_transformer_submodules": 1,
            "token_dim": 768,
            "encoder_hidden_size": 768,
        }

        # there should be no warning with just default argument for encoder_num_layer
        config = PromptEncoderConfig(**kwargs)
        with warnings.catch_warnings():
            PromptEncoder(config)

        # when changing encoder_num_layer, there should be a warning for MLP since that value is not used
        config = PromptEncoderConfig(encoder_num_layers=123, **kwargs)
        with pytest.warns(UserWarning) as record:
            PromptEncoder(config)
        expected_msg = "for MLP, the argument `encoder_num_layers` is ignored. Exactly 2 MLP layers are used."
        assert str(record.list[0].message) == expected_msg
