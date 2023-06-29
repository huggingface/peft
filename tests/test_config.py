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

import pytest

from peft import (
    AdaptionPromptConfig,
    LoraConfig,
    PeftConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
)


# list of (model_id, revision)
ALL_MODELS = [("lewtun/tiny-random-OPTForCausalLM-delta", "v1")]
ALL_CONFIG_CLASSES = (
    LoraConfig,
    PromptEncoderConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    AdaptionPromptConfig,
)


class TestPeftConfig:
    @pytest.mark.parametrize("config_class", ALL_CONFIG_CLASSES)
    def test_methods_exist(self, config_class):
        r""" Test if all configs have the expected methods. Here we test """
        config = config_class()
        methods_expected = ["to_dict", "save_pretrained", "from_pretrained", "from_json_file"]
        for method in methods_expected:
            assert hasattr(config, method)

    @pytest.mark.parametrize("config_class", ALL_CONFIG_CLASSES)
    def test_task_type(self, config_class):
        # assert this will not fail
        config_class(task_type="test")

    @pytest.mark.parametrize("config_class", ALL_CONFIG_CLASSES)
    @pytest.mark.parametrize("peft_model", ALL_MODELS)
    def test_from_pretrained(self, config_class, peft_model):
        r"""
        Test if the config is correctly loaded using:
        - from_pretrained
        """
        model_name, revision = peft_model
        # Test we can load config from delta
        config_class.from_pretrained(model_name, revision=revision)

    @pytest.mark.parametrize("config_class", ALL_CONFIG_CLASSES)
    def test_save_pretrained(self, config_class, tmp_path):
        r"""
        Test if the config is correctly saved and loaded using
        - save_pretrained
        """
        config = config_class()
        config.save_pretrained(tmp_path)

        config_from_pretrained = config_class.from_pretrained(tmp_path)
        assert config.to_dict() == config_from_pretrained.to_dict()

    @pytest.mark.parametrize("config_class", ALL_CONFIG_CLASSES)
    def test_from_json_file(self, config_class, tmp_path):
        config = config_class()
        config.save_pretrained(tmp_path)

        config_from_json = config_class.from_json_file(tmp_path / "adapter_config.json")
        assert config.to_dict() == config_from_json

    @pytest.mark.parametrize("config_class", ALL_CONFIG_CLASSES)
    def test_to_dict(self, config_class):
        r"""
        Test if the config can be correctly converted to a dict using:
        - to_dict
        - __dict__
        """
        config = config_class()
        assert config.to_dict() == config.__dict__
        assert isinstance(config.to_dict(), dict)

    @pytest.mark.parametrize("config_class", ALL_CONFIG_CLASSES)
    @pytest.mark.parametrize("peft_model", ALL_MODELS)
    def test_from_pretrained_cache_dir(self, config_class, peft_model, tmp_path):
        r"""
        Test if the config is correctly loaded with extra kwargs
        """
        model_name, revision = peft_model
        # Test we can load config from delta
        config_class.from_pretrained(model_name, revision=revision, cache_dir=tmp_path)

    def test_from_pretrained_cache_dir_remote(self, tmp_path):
        r"""
        Test if the config is correctly loaded with a checkpoint from the hub
        """
        PeftConfig.from_pretrained("ybelkada/test-st-lora", cache_dir=tmp_path)
        assert "models--ybelkada--test-st-lora" in os.listdir(tmp_path)

    @pytest.mark.parametrize("config_class", ALL_CONFIG_CLASSES)
    def test_set_attributes(self, config_class, tmp_path):
        # manually set attributes and check if they are correctly written
        config = config_class(peft_type="test")

        # save pretrained
        config.save_pretrained(tmp_path)

        config_from_pretrained = config_class.from_pretrained(tmp_path)
        assert config.to_dict() == config_from_pretrained.to_dict()
