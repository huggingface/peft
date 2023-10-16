#!/usr/bin/env python3

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
import unittest

from parameterized import parameterized
from transformers import AutoModel

from peft import AdaLoraConfig, IA3Config, LoraConfig, TaskType, get_peft_model
from peft.tuners.adalora import SVDLinear as AdaLoraLinear
from peft.tuners.ia3 import Linear as IA3Linear
from peft.tuners.lora import Linear as LoraLinear

from .testing_common import PeftCommonTester


TEST_CASES = [
    # LoRA
    (
        "T5 LoRA regex",
        TaskType.SEQ_2_SEQ_LM,
        "ybelkada/tiny-random-T5ForConditionalGeneration-calibrated",
        LoraConfig,
        {"target_modules": ".*.(SelfAttention|EncDecAttention|DenseReluDense).(k|q|v|wo|wi)$"},
        {
            "expected_target_modules": [
                "k",
                "q",
                "v",
                "wo",
                "wi",
            ]
        },
    ),
    (
        "T5 LoRA list",
        TaskType.SEQ_2_SEQ_LM,
        "ybelkada/tiny-random-T5ForConditionalGeneration-calibrated",
        LoraConfig,
        {"target_modules": ["k", "q", "v", "wo", "wi"]},
        {
            "expected_target_modules": [
                "k",
                "q",
                "v",
                "wo",
                "wi",
            ]
        },
    ),
    (
        "GPT2 LoRA regex",
        TaskType.CAUSAL_LM,
        "hf-internal-testing/tiny-random-gpt2",
        LoraConfig,
        {"target_modules": ".*(attn|mlp).(c_attn|c_proj|c_fc)$"},
        {"expected_target_modules": ["attn.c_attn", "attn.c_proj", "mlp.c_proj", "mlp.c_fc"]},
    ),
    # AdaLoRA
    (
        "T5 AdaLoRA regex",
        TaskType.SEQ_2_SEQ_LM,
        "ybelkada/tiny-random-T5ForConditionalGeneration-calibrated",
        AdaLoraConfig,
        {"target_modules": ".*.(SelfAttention|EncDecAttention|DenseReluDense).(k|q|v|wo|wi)$"},
        {
            "expected_target_modules": [
                "k",
                "q",
                "v",
                "wo",
                "wi",
            ]
        },
    ),
    (
        "T5 AdaLoRA list",
        TaskType.SEQ_2_SEQ_LM,
        "ybelkada/tiny-random-T5ForConditionalGeneration-calibrated",
        AdaLoraConfig,
        {"target_modules": ["k", "q", "v", "wo", "wi"]},
        {
            "expected_target_modules": [
                "k",
                "q",
                "v",
                "wo",
                "wi",
            ]
        },
    ),
    (
        "GPT2 AdaLoRA regex",
        TaskType.CAUSAL_LM,
        "hf-internal-testing/tiny-random-gpt2",
        AdaLoraConfig,
        {"target_modules": ".*(attn|mlp).(c_attn|c_proj|c_fc)$"},
        {"expected_target_modules": ["attn.c_attn", "attn.c_proj", "mlp.c_proj", "mlp.c_fc"]},
    ),
    # IA3
    (
        "T5 IA3 regex",
        TaskType.SEQ_2_SEQ_LM,
        "ybelkada/tiny-random-T5ForConditionalGeneration-calibrated",
        IA3Config,
        {
            "target_modules": ".*.(SelfAttention|EncDecAttention|DenseReluDense).(k|q|v|wo|wi)$",
            "feedforward_modules": ".*.DenseReluDense.(wo|wi)$",
        },
        {
            "expected_target_modules": [
                "k",
                "q",
                "v",
                "wo",
                "wi",
            ],
            "expected_feedforward_modules": ["wo", "wi"],
        },
    ),
    (
        "T5 IA3 list",
        TaskType.SEQ_2_SEQ_LM,
        "ybelkada/tiny-random-T5ForConditionalGeneration-calibrated",
        IA3Config,
        {"target_modules": ["k", "q", "v", "wo", "wi"], "feedforward_modules": ["wo", "wi"]},
        {
            "expected_target_modules": [
                "k",
                "q",
                "v",
                "wo",
                "wi",
            ],
            "expected_feedforward_modules": ["wo", "wi"],
        },
    ),
    (
        "GPT2 IA3 regex",
        TaskType.CAUSAL_LM,
        "hf-internal-testing/tiny-random-gpt2",
        IA3Config,
        {"target_modules": ".*(attn|mlp).(c_attn|c_proj|c_fc)$", "feedforward_modules": ".*.mlp.(c_proj|c_fc)$"},
        {
            "expected_target_modules": ["attn.c_attn", "attn.c_proj", "mlp.c_proj", "mlp.c_fc"],
            "expected_feedforward_modules": ["mlp.c_proj", "mlp.c_fc"],
        },
    ),
]

MODULE_MAPPING = {
    LoraConfig: LoraLinear,
    AdaLoraConfig: AdaLoraLinear,
    IA3Config: IA3Linear,
}


class PeftCustomKwargsTester(unittest.TestCase, PeftCommonTester):
    r"""
    Test if the PeftModel is instantiated with correct behaviour for custom kwargs. This includes:
    - test if adapters like LoRA, IA3, AdaLoRA replace the right layers
    - test if adapters handle custom kwargs the right way e.g. IA3 for `feedforward_modules`

    """

    transformers_class = AutoModel

    @parameterized.expand(TEST_CASES)
    def test_custom_kwargs(self, test_name, model_type, model_id, config_cls, config_kwargs, expected_mappings):
        model = self.transformers_class.from_pretrained(model_id)
        config = config_cls(task_type=model_type, **config_kwargs)
        model = get_peft_model(model, config)
        # check if the right layers are replaced, by going over child modules and looking at expected mappings
        for key, module in model.named_modules():
            # if a target module, then make sure it has been replaced
            replaced = False
            for target_key in expected_mappings["expected_target_modules"]:
                if key.endswith(f".{target_key}"):
                    replaced = True
                    self.assertTrue(isinstance(module, MODULE_MAPPING[config_cls]))
            if not replaced:  # other modules should be untouched
                self.assertFalse(isinstance(module, MODULE_MAPPING[config_cls]))

            if issubclass(config_cls, IA3Config):
                # if a feedforward module, make sure the flag is set
                is_feedforward = False
                for feedforward_key in expected_mappings["expected_feedforward_modules"]:
                    if key.endswith(f".{feedforward_key}"):
                        is_feedforward = True
                        self.assertTrue(module.is_feedforward)
                if replaced and not is_feedforward:  # other IA3 modules should not be marked as feedforward
                    self.assertFalse(module.is_feedforward)
