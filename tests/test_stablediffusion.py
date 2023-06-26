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

from unittest import TestCase

import numpy as np
from diffusers import StableDiffusionPipeline
from parameterized import parameterized

from peft import LoraConfig, get_peft_model

from .testing_common import ClassInstantier, PeftCommonTester
from .testing_utils import temp_seed


PEFT_DIFFUSERS_SD_MODELS_TO_TEST = ["hf-internal-testing/tiny-stable-diffusion-torch"]
CONFIG_TESTING_KWARGS = (
    {
        "text_encoder": {
            "r": 8,
            "lora_alpha": 8,
            "target_modules": ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
            "lora_dropout": 0.0,
            "bias": "none",
        },
        "unet": {
            "r": 8,
            "lora_alpha": 8,
            "target_modules": ["proj_in", "proj_out", "to_k", "to_q", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
            "lora_dropout": 0.0,
            "bias": "none",
        },
    },
)
CLASSES_MAPPING = {
    "lora": (LoraConfig, CONFIG_TESTING_KWARGS[0]),
}


PeftStableDiffusionTestConfigManager = ClassInstantier(CLASSES_MAPPING)


class StableDiffusionModelTester(TestCase, PeftCommonTester):
    r"""
    Tests that diffusers StableDiffusion model works with PEFT as expected.

    """
    transformers_class = StableDiffusionPipeline

    def prepare_inputs_for_testing(self):
        return {
            "prompt": "a high quality digital photo of a cute corgi",
            "num_inference_steps": 20,
        }

    @parameterized.expand(
        PeftStableDiffusionTestConfigManager.get_grid_parameters(
            {
                "model_ids": PEFT_DIFFUSERS_SD_MODELS_TO_TEST,
                "lora_kwargs": {"init_lora_weights": [False]},
            },
        )
    )
    def test_merge_layers(self, test_name, model_id, config_cls, config_kwargs):
        # Instantiate StableDiffusionPipeline
        model = self.transformers_class.from_pretrained(model_id)

        # Instantiate text_encoder adapter
        config_text_encoder = config_cls(
            **config_kwargs["text_encoder"],
        )
        model.text_encoder = get_peft_model(model.text_encoder, config_text_encoder)

        # Instantiate unet adapter
        config_unet = config_cls(
            **config_kwargs["unet"],
        )
        model.unet = get_peft_model(model.unet, config_unet)

        # Move model to device
        model = model.to(self.torch_device)

        # Generate output for peft modified StableDiffusion
        dummy_input = self.prepare_inputs_for_testing()
        with temp_seed(seed=42):
            peft_output = np.array(model(**dummy_input).images[0]).astype(np.float32)

        # Merge adapter and model
        model.text_encoder = model.text_encoder.merge_and_unload()
        model.unet = model.unet.merge_and_unload()

        # Generate output for peft merged StableDiffusion
        with temp_seed(seed=42):
            merged_output = np.array(model(**dummy_input).images[0]).astype(np.float32)

        # Images are in uint8 drange, so use large atol
        self.assertTrue(np.allclose(peft_output, merged_output, atol=1.0))
