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
from dataclasses import asdict, replace

import numpy as np
import pytest
from diffusers import StableDiffusionPipeline

from peft import (
    BOFTConfig,
    HRAConfig,
    LoHaConfig,
    LoKrConfig,
    LoraConfig,
    OFTConfig,
    get_peft_model,
    get_peft_model_state_dict,
    inject_adapter_in_model,
    set_peft_model_state_dict,
)
from peft.tuners.tuners_utils import BaseTunerLayer

from .testing_common import PeftCommonTester
from .testing_utils import set_init_weights_false, temp_seed


PEFT_DIFFUSERS_SD_MODELS_TO_TEST = ["hf-internal-testing/tiny-sd-pipe"]
DIFFUSERS_CONFIGS = [
    (
        LoraConfig,
        {
            "text_encoder": {
                "r": 8,
                "lora_alpha": 32,
                "target_modules": ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
                "lora_dropout": 0.0,
                "bias": "none",
                "init_lora_weights": False,
            },
            "unet": {
                "r": 8,
                "lora_alpha": 32,
                "target_modules": [
                    "proj_in",
                    "proj_out",
                    "to_k",
                    "to_q",
                    "to_v",
                    "to_out.0",
                    "ff.net.0.proj",
                    "ff.net.2",
                ],
                "lora_dropout": 0.0,
                "bias": "none",
                "init_lora_weights": False,
            },
        },
    ),
    (
        LoHaConfig,
        {
            "text_encoder": {
                "r": 8,
                "alpha": 32,
                "target_modules": ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
                "rank_dropout": 0.0,
                "module_dropout": 0.0,
                "init_weights": False,
            },
            "unet": {
                "r": 8,
                "alpha": 32,
                "target_modules": [
                    "proj_in",
                    "proj_out",
                    "to_k",
                    "to_q",
                    "to_v",
                    "to_out.0",
                    "ff.net.0.proj",
                    "ff.net.2",
                ],
                "rank_dropout": 0.0,
                "module_dropout": 0.0,
                "init_weights": False,
            },
        },
    ),
    (
        LoKrConfig,
        {
            "text_encoder": {
                "r": 8,
                "alpha": 32,
                "target_modules": ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
                "rank_dropout": 0.0,
                "module_dropout": 0.0,
                "init_weights": False,
            },
            "unet": {
                "r": 8,
                "alpha": 32,
                "target_modules": [
                    "proj_in",
                    "proj_out",
                    "to_k",
                    "to_q",
                    "to_v",
                    "to_out.0",
                    "ff.net.0.proj",
                    "ff.net.2",
                ],
                "rank_dropout": 0.0,
                "module_dropout": 0.0,
                "init_weights": False,
            },
        },
    ),
    (
        OFTConfig,
        {
            "text_encoder": {
                "r": 1,
                "target_modules": ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
                "module_dropout": 0.0,
                "init_weights": False,
            },
            "unet": {
                "r": 1,
                "target_modules": [
                    "proj_in",
                    "proj_out",
                    "to_k",
                    "to_q",
                    "to_v",
                    "to_out.0",
                    "ff.net.0.proj",
                    "ff.net.2",
                ],
                "module_dropout": 0.0,
                "init_weights": False,
            },
        },
    ),
    (
        BOFTConfig,
        {
            "text_encoder": {
                "boft_block_num": 1,
                "boft_block_size": 0,
                "target_modules": ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
                "boft_dropout": 0.0,
                "init_weights": False,
            },
            "unet": {
                "boft_block_num": 1,
                "boft_block_size": 0,
                "target_modules": [
                    "proj_in",
                    "proj_out",
                    "to_k",
                    "to_q",
                    "to_v",
                    "to_out.0",
                    "ff.net.0.proj",
                    "ff.net.2",
                ],
                "boft_dropout": 0.0,
                "init_weights": False,
            },
        },
    ),
    (
        HRAConfig,
        {
            "text_encoder": {
                "r": 8,
                "target_modules": ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
                "init_weights": False,
            },
            "unet": {
                "r": 8,
                "target_modules": [
                    "proj_in",
                    "proj_out",
                    "to_k",
                    "to_q",
                    "to_v",
                    "to_out.0",
                    "ff.net.0.proj",
                    "ff.net.2",
                ],
                "init_weights": False,
            },
        },
    ),
]


def skip_if_not_lora(config_cls):
    if config_cls != LoraConfig:
        pytest.skip("Skipping test because it is only applicable to LoraConfig")


class TestStableDiffusionModel(PeftCommonTester):
    r"""
    Tests that diffusers StableDiffusion model works with PEFT as expected.
    """

    transformers_class = StableDiffusionPipeline
    sd_model = StableDiffusionPipeline.from_pretrained("hf-internal-testing/tiny-sd-pipe")

    def instantiate_sd_peft(self, model_id, config_cls, config_kwargs):
        # Instantiate StableDiffusionPipeline
        if model_id == "hf-internal-testing/tiny-sd-pipe":
            # in CI, this model often times out on the hub, let's cache it
            model = copy.deepcopy(self.sd_model)
        else:
            model = self.transformers_class.from_pretrained(model_id)

        config_kwargs = config_kwargs.copy()
        text_encoder_kwargs = config_kwargs.pop("text_encoder")
        unet_kwargs = config_kwargs.pop("unet")
        # the remaining config kwargs should be applied to both configs
        for key, val in config_kwargs.items():
            text_encoder_kwargs[key] = val
            unet_kwargs[key] = val

        # Instantiate text_encoder adapter
        config_text_encoder = config_cls(**text_encoder_kwargs)
        model.text_encoder = get_peft_model(model.text_encoder, config_text_encoder)

        # Instantiate unet adapter
        config_unet = config_cls(**unet_kwargs)
        model.unet = get_peft_model(model.unet, config_unet)

        # Move model to device
        model = model.to(self.torch_device)

        return model

    def prepare_inputs_for_testing(self):
        return {
            "prompt": "a high quality digital photo of a cute corgi",
            "num_inference_steps": 3,
        }

    @pytest.mark.parametrize("model_id", PEFT_DIFFUSERS_SD_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", DIFFUSERS_CONFIGS)
    def test_merge_layers(self, model_id, config_cls, config_kwargs):
        if (config_cls == LoKrConfig) and (self.torch_device not in ["cuda", "xpu"]):
            pytest.skip("Merging test with LoKr fails without GPU")

        # Instantiate model & adapters
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        model = self.instantiate_sd_peft(model_id, config_cls, config_kwargs)

        # Generate output for peft modified StableDiffusion
        dummy_input = self.prepare_inputs_for_testing()
        with temp_seed(seed=42):
            peft_output = np.array(model(**dummy_input).images[0]).astype(np.float32)

        # Merge adapter and model
        if config_cls not in [LoHaConfig, OFTConfig, HRAConfig]:
            # TODO: Merging the text_encoder is leading to issues on CPU with PyTorch 2.1
            model.text_encoder = model.text_encoder.merge_and_unload()
        model.unet = model.unet.merge_and_unload()

        # Generate output for peft merged StableDiffusion
        with temp_seed(seed=42):
            merged_output = np.array(model(**dummy_input).images[0]).astype(np.float32)

        # Images are in uint8 drange, so use large atol
        assert np.allclose(peft_output, merged_output, atol=1.0)

    @pytest.mark.parametrize("model_id", PEFT_DIFFUSERS_SD_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", DIFFUSERS_CONFIGS)
    def test_merge_layers_safe_merge(self, model_id, config_cls, config_kwargs):
        if (config_cls == LoKrConfig) and (self.torch_device not in ["cuda", "xpu"]):
            pytest.skip("Merging test with LoKr fails without GPU")

        # Instantiate model & adapters
        model = self.instantiate_sd_peft(model_id, config_cls, config_kwargs)

        # Generate output for peft modified StableDiffusion
        dummy_input = self.prepare_inputs_for_testing()
        with temp_seed(seed=42):
            peft_output = np.array(model(**dummy_input).images[0]).astype(np.float32)

        # Merge adapter and model
        if config_cls not in [LoHaConfig, OFTConfig, HRAConfig]:
            # TODO: Merging the text_encoder is leading to issues on CPU with PyTorch 2.1
            model.text_encoder = model.text_encoder.merge_and_unload(safe_merge=True)
        model.unet = model.unet.merge_and_unload(safe_merge=True)

        # Generate output for peft merged StableDiffusion
        with temp_seed(seed=42):
            merged_output = np.array(model(**dummy_input).images[0]).astype(np.float32)

        # Images are in uint8 drange, so use large atol
        assert np.allclose(peft_output, merged_output, atol=1.0)

    @pytest.mark.parametrize("model_id", PEFT_DIFFUSERS_SD_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", DIFFUSERS_CONFIGS)
    def test_add_weighted_adapter_base_unchanged(self, model_id, config_cls, config_kwargs):
        skip_if_not_lora(config_cls)
        # Instantiate model & adapters
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        model = self.instantiate_sd_peft(model_id, config_cls, config_kwargs)

        # Get current available adapter config
        text_encoder_adapter_name = next(iter(model.text_encoder.peft_config.keys()))
        unet_adapter_name = next(iter(model.unet.peft_config.keys()))
        text_encoder_adapter_config = replace(model.text_encoder.peft_config[text_encoder_adapter_name])
        unet_adapter_config = replace(model.unet.peft_config[unet_adapter_name])

        # Create weighted adapters
        model.text_encoder.add_weighted_adapter([unet_adapter_name], [0.5], "weighted_adapter_test")
        model.unet.add_weighted_adapter([unet_adapter_name], [0.5], "weighted_adapter_test")

        # Assert that base adapters config did not change
        assert asdict(text_encoder_adapter_config) == asdict(model.text_encoder.peft_config[text_encoder_adapter_name])
        assert asdict(unet_adapter_config) == asdict(model.unet.peft_config[unet_adapter_name])

    @pytest.mark.parametrize("model_id", PEFT_DIFFUSERS_SD_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", DIFFUSERS_CONFIGS)
    def test_disable_adapter(self, model_id, config_cls, config_kwargs):
        config_kwargs = set_init_weights_false(config_cls, config_kwargs)
        self._test_disable_adapter(model_id, config_cls, config_kwargs)

    @pytest.mark.parametrize("model_id", PEFT_DIFFUSERS_SD_MODELS_TO_TEST)
    @pytest.mark.parametrize("config_cls,config_kwargs", DIFFUSERS_CONFIGS)
    def test_load_model_low_cpu_mem_usage(self, model_id, config_cls, config_kwargs):
        # Instantiate model & adapters
        pipe = self.instantiate_sd_peft(model_id, config_cls, config_kwargs)

        te_state_dict = get_peft_model_state_dict(pipe.text_encoder)
        unet_state_dict = get_peft_model_state_dict(pipe.unet)

        del pipe
        pipe = self.instantiate_sd_peft(model_id, config_cls, config_kwargs)

        config_kwargs = config_kwargs.copy()
        text_encoder_kwargs = config_kwargs.pop("text_encoder")
        unet_kwargs = config_kwargs.pop("unet")
        # the remaining config kwargs should be applied to both configs
        for key, val in config_kwargs.items():
            text_encoder_kwargs[key] = val
            unet_kwargs[key] = val

        config_text_encoder = config_cls(**text_encoder_kwargs)
        config_unet = config_cls(**unet_kwargs)

        # check text encoder
        inject_adapter_in_model(config_text_encoder, pipe.text_encoder, low_cpu_mem_usage=True)
        # sanity check that the adapter was applied:
        assert any(isinstance(module, BaseTunerLayer) for module in pipe.text_encoder.modules())

        assert "meta" in {p.device.type for p in pipe.text_encoder.parameters()}
        set_peft_model_state_dict(pipe.text_encoder, te_state_dict, low_cpu_mem_usage=True)
        assert "meta" not in {p.device.type for p in pipe.text_encoder.parameters()}

        # check unet
        inject_adapter_in_model(config_unet, pipe.unet, low_cpu_mem_usage=True)
        # sanity check that the adapter was applied:
        assert any(isinstance(module, BaseTunerLayer) for module in pipe.unet.modules())

        assert "meta" in {p.device.type for p in pipe.unet.parameters()}
        set_peft_model_state_dict(pipe.unet, unet_state_dict, low_cpu_mem_usage=True)
        assert "meta" not in {p.device.type for p in pipe.unet.parameters()}
