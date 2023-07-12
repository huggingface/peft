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
from transformers import AutoModelForCausalLM

from peft import LoraConfig, get_peft_model
from peft.utils import _is_matching_module_name


class TestMatchingModuleName(unittest.TestCase):
    """For tuners like LoRA, test that module name matching works as expected.

    There is a certain amount of logic for matching module names, including the option to indicate indices. This test
    tries to cover all of these cases.

    """

    test_cases = [
        # tuple of
        # 1. key
        # 2. target_modules
        # 3. layers_to_transform
        # 4. layers_pattern
        # 5. expected result
        # some basic examples
        ("", [], None, [], False),
        ("", ["foo"], None, [], False),
        ("foo", [], None, [], False),
        ("foo", ["foo"], None, [], True),
        ("foo", ["bar"], None, [], False),
        ("foo", ["foo", "bar"], None, [], True),
        # with layers_to_transform
        ("foo.bar.1.baz", ["baz"], [1], ["bar"], True),
        ("foo.bar.1.baz", ["baz"], [0], ["bar"], False),
        ("foo.bar.1.baz", ["baz"], [2], ["bar"], False),
        ("foo.bar.10.baz", ["baz"], [0], ["bar"], False),
        ("foo.bar.10.baz", ["baz"], [1], ["bar"], False),
        ("foo.bar.1.baz", ["baz"], [0, 1, 2], ["bar"], True),
        ("foo.bar.1.baz", ["baz", "spam"], [1], ["bar"], True),
        ("foo.bar.1.baz", ["baz", "spam"], [0, 1, 2], ["bar"], True),
        ("foo.bar.1.baz", ["baz"], [1], [], False),
        ("foo.bar.1.baz", ["baz"], [1], ["ar"], True),
        # some realistic examples: transformers model
        ("transformer.h.1.attn.attention.q_proj", [], None, [], False),
        ("transformer.h.1.attn.attention.q_proj", ["q_proj"], None, [], True),
        ("transformer.h.1.attn.attention.q_proj", ["q_proj", "v_proj"], None, [], True),
        ("transformer.h.1.attn.attention.resid_dropout", ["q_proj", "v_proj"], None, [], False),
        ("transformer.h.1.attn.attention.q_proj", ["q_proj"], [1], ["h"], True),
        ("transformer.h.1.attn.attention.q_proj", ["q_proj"], [0], ["h"], False),
        ("transformer.h.1.attn.attention.q_proj", ["q_proj"], [2], ["h"], False),
        ("transformer.h.1.attn.attention.q_proj", ["q_proj"], [0, 1, 2], ["h"], True),
        ("transformer.h.1.attn.attention.q_proj", ["q_proj", "v_proj"], [0, 1, 2], ["h"], True),
        # TODO: UNCLEAR if the output is actually  what we would expect
        ("transformer.h.1.attn.attention.q_proj.foo", ["q_proj"], None, [], False),
        ("transformer.hi.1.attn.attention.q_proj.foo", ["q_proj"], None, [], False),
        ("foo.barq_proj", ["q_proj"], None, [], True),
        ("foo.bar.1.baz", ["baz"], [1], ["foo"], False),
        ("foo.bar.1.baz", ["baz"], [1], ["baz"], False),
        ("bar.1.baz", ["baz"], [1], ["bar"], False),
        ("foo.bar.001.baz", ["baz"], [1], ["bar"], True),
        ("foo.bar.1.spam.2.baz", ["baz"], [1], ["bar"], True),
        ("foo.bar.2.spam.1.baz", ["baz"], [1], ["bar"], False),
        # TODO: UNCLEAR if the output is actually  what we would expect
        # some realistic examples: module using nn.Sequential
        ("blocks.1.weight", ["weight"], [1], ["blocks"], False),
        ("blocks.1.bias", ["weight"], [1], ["blocks"], False),
        ("mlp.blocks.1.weight", ["weight"], [1], ["blocks"], True),
        ("mlp.blocks.1.bias", ["weight"], [1], ["blocks"], False),
    ]

    @parameterized.expand(test_cases)
    def test_is_matching_module_name(self, key, target_modules, layers_to_transform, layers_pattern, expected):
        output = _is_matching_module_name(
            key, target_modules=target_modules, layers_to_transform=layers_to_transform, layers_pattern=layers_pattern
        )
        self.assertEqual(output, expected)

    def test_lora_inspect_matching_modules(self):
        # peft models that have a module matching capacity provide a method to inspect the matching modules to allow
        # users to easily debug their configuration. Here we only test a single case, not all possible combinations of
        # configs that could exist. This is okay because this method uses _is_matching_module_name under the hood, which
        # is extensively tested above.
        model_id = "hf-internal-testing/tiny-random-BloomForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        # by default, this model matches query_key_value
        config = LoraConfig()
        peft_model = get_peft_model(model, config)

        output = peft_model.inspect_matching_modules()
        matching = output["matching"]
        expected = [
            "transformer.h.0.self_attention.query_key_value",
            "transformer.h.1.self_attention.query_key_value",
            "transformer.h.2.self_attention.query_key_value",
            "transformer.h.3.self_attention.query_key_value",
            "transformer.h.4.self_attention.query_key_value",
        ]
        self.assertEqual(matching, expected)

        not_matching = output["not_matching"]
        for key in expected:
            self.assertFalse(key in not_matching)
