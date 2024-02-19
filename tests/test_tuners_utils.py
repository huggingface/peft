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
from copy import deepcopy

import pytest
from diffusers import StableDiffusionPipeline
from parameterized import parameterized
from torch import nn
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from peft import IA3Config, LoHaConfig, LoraConfig, get_peft_model
from peft.tuners.tuners_utils import (
    _maybe_include_all_linear_layers,
    check_target_module_exists,
    inspect_matched_modules,
)
from peft.utils import INCLUDE_LINEAR_LAYERS_SHORTHAND

from .testing_utils import require_bitsandbytes, require_torch_gpu


# Implements tests for regex matching logic common for all BaseTuner subclasses, and
# tests for correct behaviour with different config kwargs for BaseTuners (Ex: feedforward for IA3, etc) and
# tests for utility function to include all linear layers

REGEX_TEST_CASES = [
    # tuple of
    # 1. key
    # 2. target_modules
    # 3. layers_to_transform
    # 4. layers_pattern
    # 5. expected result
    # some basic examples
    ("", [], None, None, False),
    ("", ["foo"], None, None, False),
    ("foo", [], None, None, False),
    ("foo", ["foo"], None, None, True),
    ("foo", ["bar"], None, None, False),
    ("foo", ["foo", "bar"], None, None, True),
    # with regex
    ("foo", "foo", None, None, True),
    ("foo", ".*oo", None, None, True),
    ("foo", "fo.*", None, None, True),
    ("foo", ".*bar.*", None, None, False),
    ("foobar", ".*oba.*", None, None, True),
    # with layers_to_transform
    ("foo.bar.1.baz", ["baz"], [1], ["bar"], True),
    ("foo.bar.1.baz", ["baz"], [0], ["bar"], False),
    ("foo.bar.1.baz", ["baz"], [2], ["bar"], False),
    ("foo.bar.10.baz", ["baz"], [0], ["bar"], False),
    ("foo.bar.10.baz", ["baz"], [1], ["bar"], False),
    ("foo.bar.1.baz", ["baz"], [0, 1, 2], ["bar"], True),
    ("foo.bar.1.baz", ["baz", "spam"], [1], ["bar"], True),
    ("foo.bar.1.baz", ["baz", "spam"], [0, 1, 2], ["bar"], True),
    # empty layers_to_transform
    ("foo.bar.7.baz", ["baz"], [], ["bar"], True),
    ("foo.bar.7.baz", ["baz"], None, ["bar"], True),
    # empty layers_pattern
    ("foo.whatever.1.baz", ["baz"], [1], [], True),
    ("foo.whatever.1.baz", ["baz"], [0], [], False),
    ("foo.whatever.1.baz", ["baz"], [1], "", True),
    ("foo.whatever.1.baz", ["baz"], [0], "", False),
    ("foo.whatever.1.baz", ["baz"], [1], None, True),
    ("foo.whatever.1.baz", ["baz"], [0], None, False),
    # some realistic examples: transformers model
    ("transformer.h.1.attn.attention.q_proj.foo", ["q_proj"], None, [], False),
    ("transformer.h.1.attn.attention.q_proj", [], None, [], False),
    ("transformer.h.1.attn.attention.q_proj", ["q_proj"], None, [], True),
    ("transformer.h.1.attn.attention.q_proj", ["q_proj", "v_proj"], None, [], True),
    ("transformer.h.1.attn.attention.resid_dropout", ["q_proj", "v_proj"], None, [], False),
    ("transformer.h.1.attn.attention.q_proj", ["q_proj"], [1], ["h"], True),
    ("transformer.h.1.attn.attention.q_proj", ["q_proj"], [0], ["h"], False),
    ("transformer.h.1.attn.attention.q_proj", ["q_proj"], [2], ["h"], False),
    ("transformer.h.1.attn.attention.q_proj", ["q_proj"], [0, 1, 2], ["h"], True),
    ("transformer.h.1.attn.attention.q_proj", ["q_proj", "v_proj"], [0, 1, 2], ["h"], True),
    ("foo.bar.q_proj", ["q_proj"], None, [], True),
    ("foo.bar.1.baz", ["baz"], [1], ["foo"], False),
    # other corner cases. For ex, below is a case where layers_pattern
    # is one of the target nn.modules
    ("foo.bar.1.baz", ["baz"], [1], ["baz"], False),
    # here, layers_pattern is 'bar', but only keys that contain '.bar' are valid.
    ("bar.1.baz", ["baz"], [1], ["bar"], False),
    ("foo.bar.001.baz", ["baz"], [1], ["bar"], True),
    ("foo.bar.1.spam.2.baz", ["baz"], [1], ["bar"], True),
    ("foo.bar.2.spam.1.baz", ["baz"], [1], ["bar"], False),
    # some realistic examples: module using nn.Sequential
    # for the below test case, key should contain '.blocks' to be valid, because of how layers_pattern is matched
    ("blocks.1.weight", ["weight"], [1], ["blocks"], False),
    ("blocks.1.bias", ["weight"], [1], ["blocks"], False),
    ("mlp.blocks.1.weight", ["weight"], [1], ["blocks"], True),
    ("mlp.blocks.1.bias", ["weight"], [1], ["blocks"], False),
]

MAYBE_INCLUDE_ALL_LINEAR_LAYERS_TEST_CASES = [
    # model_name, model_type, initial_target_modules, expected_target_modules
    # test for a causal Llama model
    (
        "HuggingFaceH4/tiny-random-LlamaForCausalLM",
        "causal",
        INCLUDE_LINEAR_LAYERS_SHORTHAND,
        ["k_proj", "v_proj", "q_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
    ),
    # test for a Llama model without the LM head
    (
        "HuggingFaceH4/tiny-random-LlamaForCausalLM",
        "base",
        INCLUDE_LINEAR_LAYERS_SHORTHAND,
        ["k_proj", "v_proj", "q_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
    ),
    # test for gpt2 with Conv1D layers
    ("hf-internal-testing/tiny-random-gpt2", "causal", INCLUDE_LINEAR_LAYERS_SHORTHAND, ["c_attn", "c_proj", "c_fc"]),
    # test for T5 model
    (
        "hf-internal-testing/tiny-random-t5",
        "seq2seq",
        INCLUDE_LINEAR_LAYERS_SHORTHAND,
        ["k", "q", "v", "o", "wi", "wo"],
    ),
    # test for GPTNeoX. output module list should exclude classification head - which is named as "embed_out" instead of the usual "lm_head" for GPTNeoX
    (
        "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
        "causal",
        INCLUDE_LINEAR_LAYERS_SHORTHAND,
        ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    ),
]

# tests for a few args that should remain unchanged
MAYBE_INCLUDE_ALL_LINEAR_LAYERS_TEST_INTERNALS = [
    # initial_target_modules, expected_target_modules
    (["k_proj"], ["k_proj"]),
    # test with target_modules as None
    (None, None),
    # test with target_modules as a regex expression
    (".*(q_proj|v_proj)$", ".*(q_proj|v_proj)$"),
]

BNB_QUANTIZATIONS = [("4bit",), ("8bit",)]
BNB_TEST_CASES = [(x + y) for x in MAYBE_INCLUDE_ALL_LINEAR_LAYERS_TEST_CASES for y in BNB_QUANTIZATIONS]


class PeftCustomKwargsTester(unittest.TestCase):
    r"""
    Test if the PeftModel is instantiated with correct behaviour for custom kwargs. This includes:
    - test if regex matching works correctly
    - test if adapters handle custom kwargs the right way e.g. IA3 for `feedforward_modules`

    """

    transformers_class_map = {"causal": AutoModelForCausalLM, "seq2seq": AutoModelForSeq2SeqLM, "base": AutoModel}

    @parameterized.expand(REGEX_TEST_CASES)
    def test_regex_matching_valid(self, key, target_modules, layers_to_transform, layers_pattern, expected_result):
        # We use a LoRA Config for testing, but the regex matching function is common for all BaseTuner subclasses.
        # example model_id for config initialization. key is matched only against the target_modules given, so this can be any model
        model_id = "peft-internal-testing/tiny-OPTForCausalLM-lora"
        config = LoraConfig(
            base_model_name_or_path=model_id,
            target_modules=target_modules,
            layers_pattern=layers_pattern,
            layers_to_transform=layers_to_transform,
        )
        actual_result = bool(check_target_module_exists(config, key))
        assert actual_result == expected_result

    def test_module_matching_lora(self):
        # peft models that have a module matching method to inspect the matching modules to allow
        # users to easily debug their configuration. Here we only test a single case, not all possible combinations of
        # configs that could exist. This is okay as the method calls `check_target_module_exists` internally, which
        # has been extensively tested above.
        model_id = "hf-internal-testing/tiny-random-BloomForCausalLM"
        model = AutoModel.from_pretrained(model_id)
        # by default, this model matches query_key_value
        config = LoraConfig()
        peft_model = get_peft_model(model, config)

        output = inspect_matched_modules(peft_model)  # inspects default adapter for peft_model
        matched = output["matched"]
        expected = [
            "h.0.self_attention.query_key_value",
            "h.1.self_attention.query_key_value",
            "h.2.self_attention.query_key_value",
            "h.3.self_attention.query_key_value",
            "h.4.self_attention.query_key_value",
        ]
        assert matched == expected  # module lists should match exactly

        # no overlap with matched modules
        unmatched = output["unmatched"]
        for key in expected:
            assert key not in unmatched

    def test_feedforward_matching_ia3(self):
        model_id = "hf-internal-testing/tiny-random-T5ForConditionalGeneration"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        # simple example for just one t5 block for testing
        config_kwargs = {
            "target_modules": ".*encoder.*block.0.*(SelfAttention|EncDecAttention|DenseReluDense).(k|q|v|wo|wi)$",
            "feedforward_modules": ["wo", "wi"],
        }
        config = IA3Config(base_model_name_or_path=model_id, **config_kwargs)
        peft_model = get_peft_model(model, config)
        output = inspect_matched_modules(peft_model)  # inspects default adapter for peft_model
        matched = output["matched"]
        expected = [
            "encoder.block.0.layer.0.SelfAttention.q",
            "encoder.block.0.layer.0.SelfAttention.k",
            "encoder.block.0.layer.0.SelfAttention.v",
            "encoder.block.0.layer.1.DenseReluDense.wi",
            "encoder.block.0.layer.1.DenseReluDense.wo",
        ]
        expected_feedforward = [
            "encoder.block.0.layer.1.DenseReluDense.wi",
            "encoder.block.0.layer.1.DenseReluDense.wo",
        ]
        assert matched == expected  # not required since we do similar checks above, but just to be sure
        module_dict = dict(model.named_modules())
        for key in matched:
            module = module_dict[key]
            if key in expected_feedforward:
                assert module.is_feedforward
            else:  # other IA3 modules should not be marked as feedforward
                assert not module.is_feedforward

    @parameterized.expand(MAYBE_INCLUDE_ALL_LINEAR_LAYERS_TEST_CASES)
    def test_maybe_include_all_linear_layers_lora(
        self, model_id, model_type, initial_target_modules, expected_target_modules
    ):
        model = self.transformers_class_map[model_type].from_pretrained(model_id)
        config_cls = LoraConfig
        self._check_match_with_expected_target_modules(
            model_id, model, config_cls, initial_target_modules, expected_target_modules
        )

    @parameterized.expand(BNB_TEST_CASES)
    @require_torch_gpu
    @require_bitsandbytes
    def test_maybe_include_all_linear_layers_lora_bnb(
        self, model_id, model_type, initial_target_modules, expected_target_modules, quantization
    ):
        if quantization == "4bit":
            config_kwargs = {"load_in_4bit": True}
        elif quantization == "8bit":
            config_kwargs = {"load_in_8bit": True}
        model = self.transformers_class_map[model_type].from_pretrained(model_id, device_map="auto", **config_kwargs)
        config_cls = LoraConfig
        self._check_match_with_expected_target_modules(
            model_id, model, config_cls, initial_target_modules, expected_target_modules
        )

    def _check_match_with_expected_target_modules(
        self, model_id, model, config_cls, initial_target_modules, expected_target_modules
    ):
        """
        Helper function for the test for `_maybe_include_all_linear_layers`
        """
        actual_config = config_cls(base_model_name_or_path=model_id, target_modules=initial_target_modules)
        expected_config = config_cls(base_model_name_or_path=model_id, target_modules=expected_target_modules)
        model_copy = deepcopy(model)
        actual_model = get_peft_model(model, peft_config=actual_config)
        expected_model = get_peft_model(model_copy, peft_config=expected_config)
        expected_model_module_dict = dict(expected_model.named_modules())
        # compare the two models and assert that all layers are of the same type
        for name, actual_module in actual_model.named_modules():
            expected_module = expected_model_module_dict[name]
            assert type(actual_module) == type(expected_module)

    def test_maybe_include_all_linear_layers_ia3_loha(self):
        model_id, initial_target_modules, expected_target_modules = (
            "HuggingFaceH4/tiny-random-LlamaForCausalLM",
            INCLUDE_LINEAR_LAYERS_SHORTHAND,
            ["k_proj", "v_proj", "q_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
        )
        model_ia3 = AutoModelForCausalLM.from_pretrained(model_id)
        model_loha = deepcopy(model_ia3)
        config_classes = [IA3Config, LoHaConfig]
        models = [model_ia3, model_loha]
        for config_cls, model in zip(config_classes, models):
            self._check_match_with_expected_target_modules(
                model_id, model, config_cls, initial_target_modules, expected_target_modules
            )

    @parameterized.expand(MAYBE_INCLUDE_ALL_LINEAR_LAYERS_TEST_INTERNALS)
    def test_maybe_include_all_linear_layers_internals(self, initial_target_modules, expected_target_modules):
        model_id = "HuggingFaceH4/tiny-random-LlamaForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        config = LoraConfig(base_model_name_or_path=model_id, target_modules=initial_target_modules)
        new_config = _maybe_include_all_linear_layers(config, model)
        if isinstance(expected_target_modules, list):
            # assert that expected and actual target_modules have the same items
            assert set(new_config.target_modules) == set(expected_target_modules)
        else:
            assert new_config.target_modules == expected_target_modules

    def test_maybe_include_all_linear_layers_diffusion(self):
        model_id = "hf-internal-testing/tiny-stable-diffusion-torch"
        model = StableDiffusionPipeline.from_pretrained(model_id)
        config = LoraConfig(base_model_name_or_path=model_id, target_modules="all-linear")
        with pytest.raises(
            ValueError,
            match="Only instances of PreTrainedModel support `target_modules='all-linear'`",
        ):
            model.unet = get_peft_model(model.unet, config)


class MLP(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.lin0 = nn.Linear(10, 20, bias=bias)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.lin1 = nn.Linear(20, 2, bias=bias)
        self.sm = nn.LogSoftmax(dim=-1)


class TestTargetedModuleNames(unittest.TestCase):
    """Check that the attribute targeted_module_names is correctly set.

    This checks LoRA and IA³, but this should be sufficient, testing all other tuners is not necessary.
    """

    def test_one_targeted_module_regex(self):
        model = MLP()
        model = get_peft_model(model, LoraConfig(target_modules="lin0"))
        assert model.targeted_module_names == ["lin0"]

    def test_two_targeted_module_regex(self):
        model = MLP()
        model = get_peft_model(model, LoraConfig(target_modules="lin.*"))
        assert model.targeted_module_names == ["lin0", "lin1"]

    def test_one_targeted_module_list(self):
        model = MLP()
        model = get_peft_model(model, LoraConfig(target_modules=["lin0"]))
        assert model.targeted_module_names == ["lin0"]

    def test_two_targeted_module_list(self):
        model = MLP()
        model = get_peft_model(model, LoraConfig(target_modules=["lin0", "lin1"]))
        assert model.targeted_module_names == ["lin0", "lin1"]

    def test_ia3_targeted_module_regex(self):
        model = MLP()
        model = get_peft_model(model, IA3Config(target_modules=".*lin.*", feedforward_modules=".*lin.*"))
        assert model.targeted_module_names == ["lin0", "lin1"]

    def test_ia3_targeted_module_list(self):
        model = MLP()
        model = get_peft_model(model, IA3Config(target_modules=["lin0", "lin1"], feedforward_modules=["lin0", "lin1"]))
        assert model.targeted_module_names == ["lin0", "lin1"]

    def test_realistic_example(self):
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-BloomForCausalLM")
        config = LoraConfig(task_type="CAUSAL_LM")
        model = get_peft_model(model, config)
        expected = [
            f"transformer.h.{i}.self_attention.query_key_value" for i in range(len(model.base_model.transformer.h))
        ]
        assert model.targeted_module_names == expected
