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

import torch

from peft import LoraConfig, get_peft_model_state_dict, inject_adapter_in_model


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 10)
        self.linear = torch.nn.Linear(10, 10)
        self.lm_head = torch.nn.Linear(10, 10)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.linear(x)
        x = self.lm_head(x)
        return x


class TestPeft(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()

        lora_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            target_modules=["linear"],
        )

        self.model = inject_adapter_in_model(lora_config, self.model)

    def test_inject_adapter_in_model(self):
        dummy_inputs = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]])
        _ = self.model(dummy_inputs)

        for name, module in self.model.named_modules():
            if name == "linear":
                self.assertTrue(hasattr(module, "lora_A"))
                self.assertTrue(hasattr(module, "lora_B"))

    def test_get_peft_model_state_dict(self):
        peft_state_dict = get_peft_model_state_dict(self.model)

        for key in peft_state_dict.keys():
            self.assertTrue("lora" in key)
