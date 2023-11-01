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
from parameterized import parameterized

from peft import (
    AdaLoraConfig,
    AdaptionPromptConfig,
    IA3Config,
    LoHaConfig,
    LoraConfig,
    MultitaskPromptTuningConfig,
    PeftConfig,
    PrefixTuningConfig,
    PromptEncoder,
    PromptEncoderConfig,
    PromptTuningConfig,
)


PEFT_MODELS_TO_TEST = [("lewtun/tiny-random-OPTForCausalLM-delta", "v1")]

ALL_CONFIG_CLASSES = (
    AdaptionPromptConfig,
    AdaLoraConfig,
    IA3Config,
    LoHaConfig,
    LoraConfig,
    MultitaskPromptTuningConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
)


class PeftConfigTester(unittest.TestCase):
    @parameterized.expand(ALL_CONFIG_CLASSES)
    def test_methods(self, config_class):
        r"""
        Test if all configs have the expected methods. Here we test
        - to_dict
        - save_pretrained
        - from_pretrained
        - from_json_file
        """
        # test if all configs have the expected methods
        config = config_class()
        self.assertTrue(hasattr(config, "to_dict"))
        self.assertTrue(hasattr(config, "save_pretrained"))
        self.assertTrue(hasattr(config, "from_pretrained"))
        self.assertTrue(hasattr(config, "from_json_file"))

    @parameterized.expand(ALL_CONFIG_CLASSES)
    def test_task_type(self, config_class):
        config_class(task_type="test")

    @parameterized.expand(ALL_CONFIG_CLASSES)
    def test_from_pretrained(self, config_class):
        r"""
        Test if the config is correctly loaded using:
        - from_pretrained
        """
        for model_name, revision in PEFT_MODELS_TO_TEST:
            # Test we can load config from delta
            config_class.from_pretrained(model_name, revision=revision)

    @parameterized.expand(ALL_CONFIG_CLASSES)
    def test_save_pretrained(self, config_class):
        r"""
        Test if the config is correctly saved and loaded using
        - save_pretrained
        """
        config = config_class()
        with tempfile.TemporaryDirectory() as tmp_dirname:
            config.save_pretrained(tmp_dirname)

            config_from_pretrained = config_class.from_pretrained(tmp_dirname)
            self.assertEqual(config.to_dict(), config_from_pretrained.to_dict())

    @parameterized.expand(ALL_CONFIG_CLASSES)
    def test_from_json_file(self, config_class):
        config = config_class()
        with tempfile.TemporaryDirectory() as tmp_dirname:
            config.save_pretrained(tmp_dirname)

            config_from_json = config_class.from_json_file(os.path.join(tmp_dirname, "adapter_config.json"))
            self.assertEqual(config.to_dict(), config_from_json)

    @parameterized.expand(ALL_CONFIG_CLASSES)
    def test_to_dict(self, config_class):
        r"""
        Test if the config can be correctly converted to a dict using:
        - to_dict
        """
        config = config_class()
        self.assertTrue(isinstance(config.to_dict(), dict))

    @parameterized.expand(ALL_CONFIG_CLASSES)
    def test_from_pretrained_cache_dir(self, config_class):
        r"""
        Test if the config is correctly loaded with extra kwargs
        """
        with tempfile.TemporaryDirectory() as tmp_dirname:
            for model_name, revision in PEFT_MODELS_TO_TEST:
                # Test we can load config from delta
                config_class.from_pretrained(model_name, revision=revision, cache_dir=tmp_dirname)

    def test_from_pretrained_cache_dir_remote(self):
        r"""
        Test if the config is correctly loaded with a checkpoint from the hub
        """
        with tempfile.TemporaryDirectory() as tmp_dirname:
            PeftConfig.from_pretrained("ybelkada/test-st-lora", cache_dir=tmp_dirname)
            self.assertTrue("models--ybelkada--test-st-lora" in os.listdir(tmp_dirname))

    @parameterized.expand(ALL_CONFIG_CLASSES)
    def test_set_attributes(self, config_class):
        # manually set attributes and check if they are correctly written
        config = config_class(peft_type="test")

        # save pretrained
        with tempfile.TemporaryDirectory() as tmp_dirname:
            config.save_pretrained(tmp_dirname)

            config_from_pretrained = config_class.from_pretrained(tmp_dirname)
            self.assertEqual(config.to_dict(), config_from_pretrained.to_dict())

    @parameterized.expand(ALL_CONFIG_CLASSES)
    def test_config_copy(self, config_class):
        # see https://github.com/huggingface/peft/issues/424
        config = config_class()
        copied = copy.copy(config)
        self.assertEqual(config.to_dict(), copied.to_dict())

    @parameterized.expand(ALL_CONFIG_CLASSES)
    def test_config_deepcopy(self, config_class):
        # see https://github.com/huggingface/peft/issues/424
        config = config_class()
        copied = copy.deepcopy(config)
        self.assertEqual(config.to_dict(), copied.to_dict())

    @parameterized.expand(ALL_CONFIG_CLASSES)
    def test_config_pickle_roundtrip(self, config_class):
        # see https://github.com/huggingface/peft/issues/424
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

    @parameterized.expand([LoHaConfig, LoraConfig, IA3Config])
    def test_save_pretrained_with_target_modules(self, config_class):
        # See #1041, #1045
        config = config_class(target_modules=["a", "list"])
        with tempfile.TemporaryDirectory() as tmp_dirname:
            config.save_pretrained(tmp_dirname)

            config_from_pretrained = config_class.from_pretrained(tmp_dirname)
            self.assertEqual(config.to_dict(), config_from_pretrained.to_dict())
            # explicit test that target_modules should be converted to set
            self.assertTrue(isinstance(config_from_pretrained.target_modules, set))

    def test_regex_with_layer_indexing_lora(self):
        # This test checks that an error is raised if `target_modules` is a regex expression and `layers_to_transform` or
        # `layers_pattern` are not None

        invalid_config1 = {"target_modules": ".*foo", "layers_to_transform": [0]}
        invalid_config2 = {"target_modules": ".*foo", "layers_pattern": ["bar"]}

        valid_config = {"target_modules": ["foo"], "layers_pattern": ["bar"], "layers_to_transform": [0]}

        with self.assertRaisesRegex(
            ValueError,
            expected_regex="`layers_to_transform` cannot be used when `target_modules` is a str.",
        ):
            LoraConfig(**invalid_config1)

        with self.assertRaisesRegex(
            ValueError, expected_regex="`layers_pattern` cannot be used when `target_modules` is a str."
        ):
            LoraConfig(**invalid_config2)

        # should run without errors
        LoraConfig(**valid_config)
