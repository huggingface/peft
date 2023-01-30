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

import torch
from transformers import AutoModelForCausalLM

from peft import (
    LoraConfig,
    PeftModel,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    get_peft_model,
    get_peft_model_state_dict,
)


class PeftTestMixin:
    checkpoints_to_test = [
        "hf-internal-testing/tiny-random-OPTForCausalLM",
    ]
    config_classes = (
        LoraConfig,
        PrefixTuningConfig,
        PromptEncoderConfig,
        PromptTuningConfig,
    )
    config_kwargs = (
        dict(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
        dict(
            num_virtual_tokens=10,
            task_type="CAUSAL_LM",
        ),
        dict(
            num_virtual_tokens=10,
            encoder_hidden_size=32,
            task_type="CAUSAL_LM",
        ),
        dict(
            num_virtual_tokens=10,
            task_type="CAUSAL_LM",
        ),
    )


class PeftModelTester(unittest.TestCase, PeftTestMixin):
    r"""
    Test if the PeftModel behaves as expected. This includes:
    - test if the model has the expected methods
    """

    def test_attributes_model(self):
        for model_id in self.checkpoints_to_test:
            for i, config_cls in enumerate(self.config_classes):
                model = AutoModelForCausalLM.from_pretrained(model_id)
                config = config_cls(
                    base_model_name_or_path=model_id,
                    **self.config_kwargs[i],
                )
                model = get_peft_model(model, config)

                self.assertTrue(hasattr(model, "save_pretrained"))
                self.assertTrue(hasattr(model, "from_pretrained"))
                self.assertTrue(hasattr(model, "push_to_hub"))

    def test_save_pretrained(self):
        r"""
        A test to check if `save_pretrained` behaves as expected. This function should only save the state dict of the
        adapter model and not the state dict of the base model. Hence inside each saved directory you should have:

        - README.md (that contains an entry `base_model`)
        - adapter_config.json
        - adapter_model.bin

        """
        for model_id in self.checkpoints_to_test:
            for i, config_cls in enumerate(self.config_classes):
                model = AutoModelForCausalLM.from_pretrained(model_id)
                config = config_cls(
                    base_model_name_or_path=model_id,
                    **self.config_kwargs[i],
                )
                model = get_peft_model(model, config)
                model.to(model.device)

                with tempfile.TemporaryDirectory() as tmp_dirname:
                    model.save_pretrained(tmp_dirname)

                    model_from_pretrained = AutoModelForCausalLM.from_pretrained(model_id)
                    model_from_pretrained = PeftModel.from_pretrained(model_from_pretrained, tmp_dirname)
                    model_from_pretrained.to(model.device)

                    # check if the state dicts are equal
                    state_dict = get_peft_model_state_dict(model)
                    state_dict_from_pretrained = get_peft_model_state_dict(model_from_pretrained)

                    # check if same keys
                    self.assertEqual(state_dict.keys(), state_dict_from_pretrained.keys())

                    # check if tensors equal
                    for key in state_dict.keys():
                        self.assertTrue(torch.allclose(state_dict[key], state_dict_from_pretrained[key]))

                    # check if `adapter_model.bin` is present
                    self.assertTrue(os.path.exists(os.path.join(tmp_dirname, "adapter_model.bin")))

                    # check if `adapter_config.json` is present
                    self.assertTrue(os.path.exists(os.path.join(tmp_dirname, "adapter_config.json")))

                    # check if `pytorch_model.bin` is not present
                    self.assertFalse(os.path.exists(os.path.join(tmp_dirname, "pytorch_model.bin")))

                    # check if `config.json` is not present
                    self.assertFalse(os.path.exists(os.path.join(tmp_dirname, "config.json")))
