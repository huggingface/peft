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
import re
import unittest
from copy import deepcopy

import pytest
import torch
from diffusers import StableDiffusionPipeline
from parameterized import parameterized
from torch import nn
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from transformers.pytorch_utils import Conv1D

from peft import (
    AdaptionPromptConfig,
    IA3Config,
    LoHaConfig,
    LoraConfig,
    PeftModel,
    PromptTuningConfig,
    VeraConfig,
    get_layer_status,
    get_model_status,
    get_peft_model,
)
from peft.tuners.lora.layer import LoraLayer
from peft.tuners.tuners_utils import (
    BaseTuner,
    BaseTunerLayer,
    _maybe_include_all_linear_layers,
    check_target_module_exists,
    inspect_matched_modules,
)
from peft.tuners.tuners_utils import (
    _find_minimal_target_modules as find_minimal_target_modules,
)
from peft.utils import INCLUDE_LINEAR_LAYERS_SHORTHAND, ModulesToSaveWrapper, infer_device
from peft.utils.constants import DUMMY_MODEL_CONFIG, MIN_TARGET_MODULES_FOR_OPTIMIZATION

from .testing_common import hub_online_once
from .testing_utils import require_bitsandbytes, require_non_cpu


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
        with hub_online_once(model_id):
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
        with hub_online_once(model_id):
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
        with hub_online_once(model_id):
            model = self.transformers_class_map[model_type].from_pretrained(model_id)
        config_cls = LoraConfig
        self._check_match_with_expected_target_modules(
            model_id, model, config_cls, initial_target_modules, expected_target_modules
        )

    @parameterized.expand(BNB_TEST_CASES)
    @require_non_cpu
    @require_bitsandbytes
    def test_maybe_include_all_linear_layers_lora_bnb(
        self, model_id, model_type, initial_target_modules, expected_target_modules, quantization
    ):
        if quantization == "4bit":
            config_kwargs = {"quantization_config": BitsAndBytesConfig(load_in_4bit=True)}
        elif quantization == "8bit":
            config_kwargs = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}

        with hub_online_once(model_id):
            model = self.transformers_class_map[model_type].from_pretrained(
                model_id, device_map="auto", **config_kwargs
            )
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
            assert type(actual_module) is type(expected_module)

    def test_maybe_include_all_linear_layers_ia3_loha(self):
        model_id, initial_target_modules, expected_target_modules = (
            "HuggingFaceH4/tiny-random-LlamaForCausalLM",
            INCLUDE_LINEAR_LAYERS_SHORTHAND,
            ["k_proj", "v_proj", "q_proj", "o_proj", "down_proj", "up_proj", "gate_proj"],
        )
        with hub_online_once(model_id):
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
        with hub_online_once(model_id):
            model = AutoModelForCausalLM.from_pretrained(model_id)
        config = LoraConfig(base_model_name_or_path=model_id, target_modules=initial_target_modules)
        new_config = _maybe_include_all_linear_layers(config, model)
        if isinstance(expected_target_modules, list):
            # assert that expected and actual target_modules have the same items
            assert set(new_config.target_modules) == set(expected_target_modules)
        else:
            assert new_config.target_modules == expected_target_modules

    def test_maybe_include_all_linear_layers_diffusion(self):
        model_id = "hf-internal-testing/tiny-sd-pipe"
        with hub_online_once(model_id):
            model = StableDiffusionPipeline.from_pretrained(model_id)
        config = LoraConfig(base_model_name_or_path=model_id, target_modules="all-linear")

        # all linear layers should be converted
        num_linear = sum(isinstance(module, (nn.Linear, Conv1D)) for module in model.unet.modules())
        model.unet = get_peft_model(model.unet, config)
        num_lora = sum(isinstance(module, LoraLayer) for module in model.unet.modules())
        assert num_lora == num_linear

    def test_maybe_include_all_linear_does_not_target_classifier_head(self):
        # See issue 2027
        # Ensure that if a SEQ_CLS model is being used with target_modules="all-linear", the classification head is not
        # targeted by the adapter layer.
        model_id = "HuggingFaceH4/tiny-random-LlamaForCausalLM"
        with hub_online_once(model_id):
            model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=10)
        # sanity check
        assert isinstance(model.score, nn.Linear)

        num_linear = sum(isinstance(module, (nn.Linear, Conv1D)) for module in model.modules())

        config = LoraConfig(task_type="SEQ_CLS", target_modules="all-linear")
        model = get_peft_model(model, config)
        assert isinstance(model.base_model.score, ModulesToSaveWrapper)

        # the bug was that these were lora.Linear instances
        assert isinstance(model.base_model.score.original_module, nn.Linear)
        assert isinstance(model.base_model.score.modules_to_save["default"], nn.Linear)

        # ensure that all but one linear layer was targeted by LoRA
        num_lora = sum(isinstance(module, LoraLayer) for module in model.modules())
        assert num_lora == num_linear - 1

    @parameterized.expand(MAYBE_INCLUDE_ALL_LINEAR_LAYERS_TEST_CASES)
    def test_all_linear_nested_targets_correct_layers(
        self, model_id, model_type, initial_target_modules, expected_target_modules
    ):
        # See 2390
        # Ensure that if adapter layers are already applied, we don't get nested adapter layers (e.g. LoRA targeting the
        # lora_A, lora_B layers)
        with hub_online_once(model_id):
            model = self.transformers_class_map[model_type].from_pretrained(model_id)
        config_cls = LoraConfig
        self._check_match_with_expected_target_modules(
            model_id, model, config_cls, initial_target_modules, expected_target_modules
        )
        # re-use the same model, i.e. the adapter is already applied
        self._check_match_with_expected_target_modules(
            model_id, model, config_cls, initial_target_modules, expected_target_modules
        )

    def test_add_second_adapter_with_all_linear_works(self):
        # See 2390 Similar test to test_all_linear_nested_targets_correct_layers above, but using add_adapter instead of
        # calling get_peft_model in an already adapted model
        model_id = "HuggingFaceH4/tiny-random-LlamaForCausalLM"
        with hub_online_once(model_id):
            model = AutoModelForCausalLM.from_pretrained(model_id)

        # important: don't reuse the first config, since config.target_modules will be overwritten, which would make the
        # test pass trivially.
        config0 = LoraConfig(target_modules=INCLUDE_LINEAR_LAYERS_SHORTHAND)
        config1 = LoraConfig(target_modules=INCLUDE_LINEAR_LAYERS_SHORTHAND)

        model = get_peft_model(model, config0)
        model.add_adapter(adapter_name="other", peft_config=config1)

        # both configs should result in the same target modules being chosen (remember that config.target_modules will
        # be replaced by the actual set of target_modules)
        assert config0.target_modules == config1.target_modules

        for layer in model.base_model.model.model.layers:
            projs = (
                layer.self_attn.q_proj,
                layer.self_attn.v_proj,
                layer.self_attn.k_proj,
                layer.mlp.gate_proj,
                layer.mlp.up_proj,
                layer.mlp.down_proj,
            )
            for proj in projs:
                # the targted layer itself, which in the base model was the nn.Linear layer, is now a LoraLayer
                assert isinstance(proj, LoraLayer)
                # all children of that layer are still normal nn.Linear layers
                assert isinstance(proj.base_layer, nn.Linear)
                assert isinstance(proj.lora_A["default"], nn.Linear)
                assert isinstance(proj.lora_B["default"], nn.Linear)
                assert isinstance(proj.lora_A["other"], nn.Linear)
                assert isinstance(proj.lora_B["other"], nn.Linear)


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
        model_id = "hf-internal-testing/tiny-random-BloomForCausalLM"
        with hub_online_once(model_id):
            model = AutoModelForCausalLM.from_pretrained(model_id)
        config = LoraConfig(task_type="CAUSAL_LM")
        model = get_peft_model(model, config)
        expected = [
            f"transformer.h.{i}.self_attention.query_key_value" for i in range(len(model.base_model.transformer.h))
        ]
        assert model.targeted_module_names == expected


class TestTargetedParameterNames(unittest.TestCase):
    """Check that the attribute targeted_parameter_names (via target_parameters) is correctly set.

    This is only implemented for LoRA. Regex matching is currently not implemented.
    """

    def test_one_targeted_parameters_list(self):
        model = MLP()
        model = get_peft_model(model, LoraConfig(target_parameters=["lin0.weight"]))
        assert model.targeted_parameter_names == ["lin0.weight"]

    def test_two_targeted_parameters_list(self):
        model = MLP()
        model = get_peft_model(model, LoraConfig(target_parameters=["lin0.weight", "lin1.weight"]))
        assert model.targeted_parameter_names == ["lin0.weight", "lin1.weight"]

    def test_realistic_example(self):
        model_id = "trl-internal-testing/tiny-random-LlamaForCausalLM"
        with hub_online_once(model_id):
            model = AutoModelForCausalLM.from_pretrained(model_id)
        config = LoraConfig(target_modules=[], task_type="CAUSAL_LM", target_parameters=["v_proj.weight"])
        model = get_peft_model(model, config)
        expected = [
            f"model.layers.{i}.self_attn.v_proj.weight" for i in range(len(model.base_model.model.model.layers))
        ]
        assert model.targeted_parameter_names == expected


class TestExcludedModuleNames(unittest.TestCase):
    """Check that the attribute exclude_module is correctly set.

    This checks LoRA and IA³, but this should be sufficient, testing all other tuners is not necessary.
    """

    def test_two_excluded_module_regex(self):
        model = MLP()
        model = get_peft_model(model, LoraConfig(target_modules=("lin.*"), exclude_modules="lin0"))
        assert model.targeted_module_names == ["lin1"]

    def test_two_excluded_module_list(self):
        model = MLP()
        model = get_peft_model(model, LoraConfig(target_modules=["lin0", "lin1"], exclude_modules="lin0"))
        assert model.targeted_module_names == ["lin1"]

    def test_multiple_excluded_modules_list(self):
        model = MLP()
        model = get_peft_model(model, LoraConfig(target_modules=["lin0", "lin1"], exclude_modules=["lin0"]))
        assert model.targeted_module_names == ["lin1"]

    def test_ia3_two_excluded_module_regex(self):
        model = MLP()
        model = get_peft_model(
            model, IA3Config(target_modules=".*lin.*", feedforward_modules=".*lin.*", exclude_modules="lin0")
        )
        assert model.targeted_module_names == ["lin1"]

    def test_ia3_multiple_excluded_modules_list(self):
        model = MLP()
        model = get_peft_model(
            model, IA3Config(target_modules=["lin0", "lin1"], feedforward_modules=".*lin.*", exclude_modules=["lin1"])
        )
        assert model.targeted_module_names == ["lin0"]

    def test_all_modules_excluded(self):
        model = MLP()
        with pytest.raises(ValueError, match="All modules were excluded"):
            get_peft_model(
                model,
                LoraConfig(
                    target_modules=["lin0", "lin1", "relu", "drop", "sm"],
                    exclude_modules=["lin0", "lin1", "relu", "drop", "sm"],
                ),
            )

    def test_no_modules_matched(self):
        model = MLP()
        with pytest.raises(ValueError, match="Target modules .* not found in the base model"):
            get_peft_model(model, LoraConfig(target_modules=["non_existent_module"]))

    def test_some_modules_excluded_some_unmatched(self):
        model = MLP()
        with pytest.raises(ValueError, match="No modules were targeted for adaptation"):
            get_peft_model(model, LoraConfig(target_modules=["lin0", "non_existent_module"], exclude_modules=["lin0"]))

    def test_exclude_modules_not_used(self):
        model = MLP()
        with pytest.warns(UserWarning, match="You have passed exclude_modules=.* but no modules were excluded"):
            get_peft_model(model, LoraConfig(target_modules=["lin1"], exclude_modules=["non_existent_module"]))

    def test_realistic_example(self):
        model_id = "hf-internal-testing/tiny-random-BloomForCausalLM"
        with hub_online_once(model_id):
            model = AutoModelForCausalLM.from_pretrained(model_id)
        config = LoraConfig(task_type="CAUSAL_LM", exclude_modules="transformer.h.2.self_attention.query_key_value")
        model = get_peft_model(model, config)
        expected = [
            f"transformer.h.{i}.self_attention.query_key_value"
            for i in range(len(model.base_model.transformer.h))
            if i != 2
        ]
        assert model.targeted_module_names == expected


class TestModelAndLayerStatus:
    """Check the methods `get_layer_status` and `get_model_status`.`

    Note that we only test LoRA here but the same logic should work for other tuner types (if they support the
    corresponding features like merging).

    """

    torch_device = infer_device()

    @pytest.fixture
    def small_model(self):
        class SmallModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin0 = nn.Linear(10, 10)
                self.lin1 = nn.Linear(10, 10)

        config = LoraConfig(target_modules="lin0")
        return get_peft_model(SmallModel(), config)

    @pytest.fixture
    def large_model(self):
        class LargeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin0 = nn.Linear(10, 10)
                self.conv0 = nn.Conv2d(3, 10, 3)
                self.emb0 = nn.Embedding(10, 10)
                self.lin1 = nn.Linear(10, 10)
                self.conv1 = nn.Conv2d(3, 10, 3)
                self.emb1 = nn.Embedding(10, 10)

        config0 = LoraConfig(target_modules=["lin0", "conv1", "emb0"])
        config1 = LoraConfig(target_modules=["lin0", "lin1"], r=16)
        model = get_peft_model(LargeModel(), config0)
        model.add_adapter("other", config1)
        return model

    ################
    # layer status #
    ################

    def test_layer_names_small(self, small_model):
        layer_status = small_model.get_layer_status()
        expected = ["model.lin0"]
        assert [status.name for status in layer_status] == expected

    def test_layer_names_large(self, large_model):
        layer_status = large_model.get_layer_status()
        result = sorted([status.name for status in layer_status])
        expected = ["model.conv1", "model.emb0", "model.lin0", "model.lin1"]
        assert result == expected

    def test_module_type_small(self, small_model):
        layer_status = small_model.get_layer_status()
        assert [status.module_type for status in layer_status] == ["lora.Linear"]

    def test_module_type_large(self, large_model):
        layer_status = large_model.get_layer_status()
        result = sorted([status.module_type for status in layer_status])
        expected = ["lora.Conv2d", "lora.Embedding", "lora.Linear", "lora.Linear"]
        assert result == expected

    def test_enabled_small(self, small_model):
        layer_status = small_model.get_layer_status()
        assert [status.enabled for status in layer_status] == [True]

    def test_enabled_large(self, large_model):
        layer_status = large_model.get_layer_status()
        result = [status.enabled for status in layer_status]
        expected = [True, True, True, True]
        assert result == expected

    def test_enabled_irregular(self, large_model):
        # this is an invalid state, but we should still test it
        # disable a single layer
        for module in large_model.modules():
            if isinstance(module, BaseTunerLayer):
                module.enable_adapters(False)
                break

        layer_status = large_model.get_layer_status()
        result = [status.enabled for status in layer_status]
        expected = [False, True, True, True]
        assert result == expected

    def test_active_adapters_small(self, small_model):
        layer_status = small_model.get_layer_status()
        assert [status.active_adapters for status in layer_status] == [["default"]]

    def test_active_adapters_large(self, large_model):
        layer_status = large_model.get_layer_status()
        result = [status.active_adapters for status in layer_status]
        # note: as currently implemented, the active adapter can be an adapter that does not exist on this specific
        # layer, for instance, layer 3 (i.e. index 2) only has the "other" adapter but "default" is still shown as the
        # active adapter
        expected = [["default"], ["default"], ["default"], ["default"]]
        assert result == expected

        # switch to "other"
        large_model.set_adapter("other")
        layer_status = large_model.get_layer_status()
        result = [status.active_adapters for status in layer_status]
        expected = [["other"], ["other"], ["other"], ["other"]]

    def test_merge_adapters_small(self, small_model):
        layer_status = small_model.get_layer_status()
        assert [status.merged_adapters for status in layer_status] == [[]]
        assert [status.available_adapters for status in layer_status] == [["default"]]

        # now merge "default"
        small_model.merge_adapter(["default"])
        layer_status = small_model.get_layer_status()
        assert [status.merged_adapters for status in layer_status] == [["default"]]
        assert [status.available_adapters for status in layer_status] == [["default"]]

    def test_merge_adapters_large(self, large_model):
        layer_status = large_model.get_layer_status()
        result = [status.merged_adapters for status in layer_status]
        assert result == [[], [], [], []]

        # now merge "default"
        large_model.merge_adapter(["default"])
        layer_status = large_model.get_layer_status()
        result = [status.merged_adapters for status in layer_status]
        # default is on layer 0, 1, and 3
        assert result == [["default"], ["default"], [], ["default"]]

        # now merge "other"
        large_model.unmerge_adapter()
        large_model.merge_adapter(["other"])
        layer_status = large_model.get_layer_status()
        result = [status.merged_adapters for status in layer_status]
        # other is on layer 0 and 2
        assert result == [["other"], [], ["other"], []]

        # now merge both
        large_model.merge_adapter(["default", "other"])
        layer_status = large_model.get_layer_status()
        result = [status.merged_adapters for status in layer_status]
        # default is on layer 0, 1, and 3, other is on layer 0 and 2
        assert result == [["other", "default"], ["default"], ["other"], ["default"]]

    def test_requires_grad_small(self, small_model):
        layer_status = small_model.get_layer_status()
        assert [status.requires_grad for status in layer_status] == [{"default": True}]

    def test_requires_grad_large(self, large_model):
        layer_status = large_model.get_layer_status()
        result = [status.requires_grad for status in layer_status]
        # default is on layer 0, 1, and 3, other is on layer 0 and 2
        expected = [{"default": True, "other": False}, {"default": True}, {"other": False}, {"default": True}]
        assert result == expected

        # now activate "other"
        large_model.set_adapter("other")
        layer_status = large_model.get_layer_status()
        result = [status.requires_grad for status in layer_status]
        expected = [{"default": False, "other": True}, {"default": False}, {"other": True}, {"default": False}]
        assert result == expected

    def test_requires_grad_irregular(self, large_model):
        # inject an embedding layer with requires_grad=False
        # this is an invalid state, but we should still test it
        lora_embedding_A = nn.Parameter(torch.zeros(10, 10))
        lora_embedding_B = nn.Parameter(torch.zeros(10, 10))
        lora_embedding_A.requires_grad = False
        lora_embedding_B.requires_grad = False
        large_model.base_model.model.lin0.lora_embedding_A["default"] = lora_embedding_A
        large_model.base_model.model.lin0.lora_embedding_B["default"] = lora_embedding_B

        layer_status = large_model.get_layer_status()
        result = [status.requires_grad for status in layer_status]
        expected = [{"default": "irregular", "other": False}, {"default": True}, {"other": False}, {"default": True}]
        assert result == expected

    def test_available_adapters_small(self, small_model):
        layer_status = small_model.get_layer_status()
        result = [status.available_adapters for status in layer_status]
        expected = [["default"]]
        assert result == expected

    def test_available_adapters_large(self, large_model):
        layer_status = large_model.get_layer_status()
        result = [status.available_adapters for status in layer_status]
        expected = [["default", "other"], ["default"], ["other"], ["default"]]
        assert result == expected

    def test_devices_all_cpu_small(self, small_model):
        layer_status = small_model.get_layer_status()
        result = [status.devices for status in layer_status]
        expected = [{"default": ["cpu"]}]
        assert result == expected

    def test_devices_all_cpu_large(self, large_model):
        layer_status = large_model.get_layer_status()
        result = [status.devices for status in layer_status]
        expected = [
            {"default": ["cpu"], "other": ["cpu"]},
            {"default": ["cpu"]},
            {"other": ["cpu"]},
            {"default": ["cpu"]},
        ]
        assert result == expected

    @require_non_cpu
    def test_devices_all_gpu_large(self, large_model):
        large_model.to(self.torch_device)
        layer_status = large_model.get_layer_status()
        result = [status.devices for status in layer_status]
        expected = [
            {"default": [self.torch_device], "other": [self.torch_device]},
            {"default": [self.torch_device]},
            {"other": [self.torch_device]},
            {"default": [self.torch_device]},
        ]
        assert result == expected

    @require_non_cpu
    def test_devices_cpu_and_gpu_large(self, large_model):
        # move the embedding layer to GPU
        large_model.model.lin0.lora_A["default"] = large_model.model.lin0.lora_A["default"].to(self.torch_device)
        layer_status = large_model.get_layer_status()
        result = [status.devices for status in layer_status]
        expected = [
            {"default": ["cpu", self.torch_device], "other": ["cpu"]},
            {"default": ["cpu"]},
            {"other": ["cpu"]},
            {"default": ["cpu"]},
        ]
        assert result == expected

    def test_target_parameters(self, large_model):
        # don't check each attribute, just the relevant ones
        # first remove the normal LoRA layers
        large_model = large_model.merge_and_unload()
        config = LoraConfig(target_parameters=["lin0.weight", "lin1.weight"])
        large_model = get_peft_model(large_model, config)
        layer_status = large_model.get_layer_status()
        assert [status.name for status in layer_status] == ["model.lin0", "model.lin1"]
        assert [status.module_type for status in layer_status] == ["lora.ParamWrapper"] * 2

    def test_target_parameters_and_target_modules(self, large_model):
        # don't check each attribute, just the relevant ones
        # first remove the normal LoRA layers
        large_model = large_model.merge_and_unload()
        config = LoraConfig(target_parameters=["lin0.weight"], target_modules=["lin1"])
        large_model = get_peft_model(large_model, config)
        layer_status = large_model.get_layer_status()
        assert [status.name for status in layer_status] == ["model.lin0", "model.lin1"]
        assert [status.module_type for status in layer_status] == ["lora.ParamWrapper", "lora.Linear"]

    ################
    # model status #
    ################

    def test_base_model_type_small(self, small_model):
        model_status = small_model.get_model_status()
        assert model_status.base_model_type == "SmallModel"

    def test_base_model_type_large(self, large_model):
        model_status = large_model.get_model_status()
        assert model_status.base_model_type == "LargeModel"

    def test_base_model_type_transformers_automodel(self):
        # ensure that this also works with transformers AutoModels
        model_id = "google/flan-t5-small"
        with hub_online_once(model_id):
            model = AutoModel.from_pretrained(model_id)
        model = get_peft_model(model, LoraConfig())
        model_status = model.get_model_status()
        assert model_status.base_model_type == "T5Model"

    def test_adapter_model_type_small(self, small_model):
        model_status = small_model.get_model_status()
        assert model_status.adapter_model_type == "LoraModel"

    def test_adapter_model_type_large(self, large_model):
        model_status = large_model.get_model_status()
        assert model_status.adapter_model_type == "LoraModel"

    def test_peft_types_small(self, small_model):
        model_status = small_model.get_model_status()
        assert model_status.peft_types == {"default": "LORA"}

    def test_peft_types_large(self, large_model):
        model_status = large_model.get_model_status()
        assert model_status.peft_types == {"default": "LORA", "other": "LORA"}

    def test_nb_params_small(self, small_model):
        model_status = small_model.get_model_status()
        assert model_status.trainable_params == 160
        assert model_status.total_params == 380

    def test_nb_params_large(self, large_model):
        model_status = large_model.get_model_status()
        assert model_status.trainable_params == 616
        assert model_status.total_params == 2236

    def test_num_adapter_layers_small(self, small_model):
        model_status = small_model.get_model_status()
        assert model_status.num_adapter_layers == 1

    def test_num_adapter_layers_large(self, large_model):
        model_status = large_model.get_model_status()
        assert model_status.num_adapter_layers == 4

    def test_model_enabled_small(self, small_model):
        model_status = small_model.get_model_status()
        assert model_status.enabled is True

    def test_model_enabled_large(self, large_model):
        model_status = large_model.get_model_status()
        assert model_status.enabled is True

    def test_model_disabled_small(self, small_model):
        small_model.disable_adapter_layers()
        model_status = small_model.get_model_status()
        assert model_status.enabled is False

    def test_model_disabled_large(self, large_model):
        large_model.disable_adapter_layers()
        model_status = large_model.get_model_status()
        assert model_status.enabled is False

    def test_model_enabled_irregular(self, large_model):
        # this is an invalid state, but we should still test it
        # disable a single layer
        for module in large_model.modules():
            if isinstance(module, BaseTunerLayer):
                module.enable_adapters(False)
                break

        model_status = large_model.get_model_status()
        assert model_status.enabled == "irregular"

    def test_model_active_adapters_small(self, small_model):
        model_status = small_model.get_model_status()
        assert model_status.active_adapters == ["default"]

    def test_model_active_adapters_large(self, large_model):
        model_status = large_model.get_model_status()
        assert model_status.active_adapters == ["default"]

        large_model.set_adapter("other")
        model_status = large_model.get_model_status()
        assert model_status.active_adapters == ["other"]

    def test_model_active_adapters_irregular(self, large_model):
        # this is an invalid state, but we should still test it
        # disable a single layer
        for module in large_model.modules():
            if isinstance(module, BaseTunerLayer):
                # switch a single layer's active adapter from default to other
                if module.active_adapters == ["default"]:
                    module._active_adapter = "other"
                    assert module.active_adapters == ["other"]
                    break

        model_status = large_model.get_model_status()
        assert model_status.active_adapters == "irregular"

    def test_model_merged_adapters_small(self, small_model):
        model_status = small_model.get_model_status()
        assert model_status.merged_adapters == []

        small_model.merge_adapter()
        model_status = small_model.get_model_status()
        assert model_status.merged_adapters == ["default"]

        small_model.unmerge_adapter()
        model_status = small_model.get_model_status()
        assert model_status.merged_adapters == []

    def test_model_merged_adapters_large(self, large_model):
        model_status = large_model.get_model_status()
        assert model_status.merged_adapters == []

        large_model.merge_adapter(["default"])
        model_status = large_model.get_model_status()
        assert model_status.merged_adapters == ["default"]

        large_model.unmerge_adapter()
        large_model.merge_adapter(["other"])
        model_status = large_model.get_model_status()
        assert model_status.merged_adapters == ["other"]

        large_model.unmerge_adapter()
        large_model.merge_adapter(["default", "other"])
        model_status = large_model.get_model_status()
        assert model_status.merged_adapters == ["default", "other"]

    def test_model_merged_adapters_irregular(self, large_model):
        # this is an invalid state, but we should still test it
        # by merging only lin0 of "default", we end up in a irregular state, because not all "default" layers are merged
        large_model.base_model.lin0.merge(["default"])

        model_status = large_model.get_model_status()
        assert model_status.merged_adapters == "irregular"

    def test_model_requires_grad_model_small(self, small_model):
        model_status = small_model.get_model_status()
        assert model_status.requires_grad == {"default": True}

    def test_model_requires_grad_model_large(self, large_model):
        model_status = large_model.get_model_status()
        assert model_status.requires_grad == {"default": True, "other": False}

        large_model.set_adapter("other")
        model_status = large_model.get_model_status()
        assert model_status.requires_grad == {"default": False, "other": True}

    def test_model_requires_grad_model_irregular(self, large_model):
        # inject an embedding layer with requires_grad=False
        # this is an invalid state, but we should still test it
        lora_embedding_A = nn.Parameter(torch.zeros(10, 10))
        lora_embedding_B = nn.Parameter(torch.zeros(10, 10))
        lora_embedding_A.requires_grad = False
        lora_embedding_B.requires_grad = False
        large_model.base_model.model.lin0.lora_embedding_A["default"] = lora_embedding_A
        large_model.base_model.model.lin0.lora_embedding_B["default"] = lora_embedding_B

        model_status = large_model.get_model_status()
        assert model_status.requires_grad == {"default": "irregular", "other": False}

    def test_model_available_adapters_small(self, small_model):
        model_status = small_model.get_model_status()
        assert model_status.available_adapters == ["default"]

    def test_model_available_adapters_large(self, large_model):
        model_status = large_model.get_model_status()
        assert model_status.available_adapters == ["default", "other"]

    def test_model_devices_all_cpu_small(self, small_model):
        model_status = small_model.get_model_status()
        assert model_status.devices == {"default": ["cpu"]}

    def test_model_devices_all_cpu_large(self, large_model):
        model_status = large_model.get_model_status()
        assert model_status.devices == {"default": ["cpu"], "other": ["cpu"]}

    @require_non_cpu
    def test_model_devices_all_gpu_large(self, large_model):
        large_model.to(self.torch_device)
        model_status = large_model.get_model_status()
        assert model_status.devices == {"default": [self.torch_device], "other": [self.torch_device]}

    @require_non_cpu
    def test_model_devices_cpu_and_gpu_large(self, large_model):
        # move the embedding layer to GPU
        large_model.model.lin0.lora_A["default"] = large_model.model.lin0.lora_A["default"].to(self.torch_device)
        model_status = large_model.get_model_status()
        assert model_status.devices == {"default": ["cpu", self.torch_device], "other": ["cpu"]}

    def test_model_target_parameters(self, large_model):
        # don't check each attribute, just the relevant ones
        # first remove the normal LoRA layers
        large_model = large_model.merge_and_unload()
        config = LoraConfig(target_parameters=["lin0.weight", "lin1.weight"])
        large_model = get_peft_model(large_model, config)
        model_status = large_model.get_model_status()
        model_status = large_model.get_model_status()
        assert model_status.adapter_model_type == "LoraModel"
        assert model_status.peft_types == {"default": "LORA", "other": "LORA"}
        assert model_status.num_adapter_layers == 2
        assert model_status.trainable_params == 2 * (8 * 10 + 10 * 8)

    def test_model_target_parameters_and_target_modules(self, large_model):
        # don't check each attribute, just the relevant ones
        # first remove the normal LoRA layers
        large_model = large_model.merge_and_unload()
        config = LoraConfig(target_parameters=["lin0.weight"], target_modules=["lin1"])
        large_model = get_peft_model(large_model, config)
        model_status = large_model.get_model_status()
        assert model_status.adapter_model_type == "LoraModel"
        assert model_status.peft_types == {"default": "LORA", "other": "LORA"}
        assert model_status.num_adapter_layers == 2
        assert model_status.trainable_params == 2 * (8 * 10 + 10 * 8)

    def test_loha_model(self):
        # ensure that this also works with non-LoRA, it's not necessary to test all tuners
        class SmallModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin0 = nn.Linear(10, 10)
                self.lin1 = nn.Linear(10, 10)

        base_model = SmallModel()
        config = LoHaConfig(target_modules=["lin0", "lin1"], init_weights=False)
        model = get_peft_model(base_model, config)

        model_status = model.get_model_status()
        layer_status = model.get_layer_status()

        assert model_status.base_model_type == "SmallModel"
        assert model_status.adapter_model_type == "LoHaModel"
        assert model_status.peft_types == {"default": "LOHA"}
        assert model_status.trainable_params == 640
        assert model_status.total_params == 860
        assert model_status.num_adapter_layers == 2
        assert model_status.enabled is True
        assert model_status.active_adapters == ["default"]
        assert model_status.merged_adapters == []
        assert model_status.requires_grad == {"default": True}
        assert model_status.available_adapters == ["default"]
        assert model_status.devices == {"default": ["cpu"]}

        layer_status0 = layer_status[0]
        assert len(layer_status) == 2
        assert layer_status0.name == "model.lin0"
        assert layer_status0.module_type == "loha.Linear"
        assert layer_status0.enabled is True
        assert layer_status0.active_adapters == ["default"]
        assert layer_status0.merged_adapters == []
        assert layer_status0.requires_grad == {"default": True}
        assert layer_status0.available_adapters == ["default"]
        assert layer_status0.devices == {"default": ["cpu"]}

    @require_non_cpu
    def test_vera_model(self):
        # let's also test VeRA because it uses BufferDict
        class SmallModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin0 = nn.Linear(10, 10)
                self.lin1 = nn.Linear(10, 10)

        base_model = SmallModel()
        config = VeraConfig(target_modules=["lin0", "lin1"], init_weights=False)
        model = get_peft_model(base_model, config)

        # move the buffer dict to GPU
        model.lin0.vera_A["default"] = model.lin0.vera_A["default"].to(self.torch_device)

        model_status = model.get_model_status()
        layer_status = model.get_layer_status()

        assert model_status.base_model_type == "SmallModel"
        assert model_status.adapter_model_type == "VeraModel"
        assert model_status.peft_types == {"default": "VERA"}
        assert model_status.trainable_params == 532
        assert model_status.total_params == 752
        assert model_status.num_adapter_layers == 2
        assert model_status.enabled is True
        assert model_status.active_adapters == ["default"]
        assert model_status.merged_adapters == []
        assert model_status.requires_grad == {"default": True}
        assert model_status.available_adapters == ["default"]
        assert model_status.devices == {"default": ["cpu", self.torch_device]}

        layer_status0 = layer_status[0]
        assert len(layer_status) == 2
        assert layer_status0.name == "model.lin0"
        assert layer_status0.module_type == "vera.Linear"
        assert layer_status0.enabled is True
        assert layer_status0.active_adapters == ["default"]
        assert layer_status0.merged_adapters == []
        assert layer_status0.requires_grad == {"default": True}
        assert layer_status0.available_adapters == ["default"]
        assert layer_status0.devices == {"default": ["cpu", self.torch_device]}

    ###################
    # non-PEFT models #
    ###################

    def test_transformers_model(self):
        model_id = "peft-internal-testing/gpt2-lora-random"
        # note that loading through AutoModelForCausalLM.from_pretrained does not enable training mode, hence
        # requires_grad=False
        with hub_online_once(model_id):
            model = AutoModelForCausalLM.from_pretrained(model_id)
        model_status = get_model_status(model)
        layer_status = get_layer_status(model)

        assert model_status.base_model_type == "GPT2LMHeadModel"
        assert model_status.adapter_model_type == "None"
        assert model_status.peft_types == {}
        assert model_status.trainable_params == 0
        assert model_status.total_params == 124734720
        assert model_status.num_adapter_layers == 12
        assert model_status.enabled is True
        assert model_status.active_adapters == ["default"]
        assert model_status.merged_adapters == []
        assert model_status.requires_grad == {"default": False}
        assert model_status.available_adapters == ["default"]
        assert model_status.devices == {"default": ["cpu"]}

        layer_status0 = layer_status[0]
        assert len(layer_status) == 12
        assert layer_status0.name == "transformer.h.0.attn.c_attn"
        assert layer_status0.module_type == "lora.Linear"
        assert layer_status0.enabled is True
        assert layer_status0.active_adapters == ["default"]
        assert layer_status0.merged_adapters == []
        assert layer_status0.requires_grad == {"default": False}
        assert layer_status0.available_adapters == ["default"]
        assert layer_status0.devices == {"default": ["cpu"]}

    def test_model_with_injected_layers(self, large_model):
        model = large_model.base_model.model
        model_status = get_model_status(model)
        layer_status = get_layer_status(model)

        assert model_status.base_model_type == "other"
        assert model_status.adapter_model_type == "None"
        assert model_status.peft_types == {}
        assert model_status.trainable_params == 616
        assert model_status.total_params == 2236
        assert model_status.num_adapter_layers == 4
        assert model_status.enabled is True
        assert model_status.active_adapters == ["default"]
        assert model_status.merged_adapters == []
        assert model_status.requires_grad == {"default": True, "other": False}
        assert model_status.available_adapters == ["default", "other"]
        assert model_status.devices == {"default": ["cpu"], "other": ["cpu"]}

        layer_status1 = layer_status[1]
        assert len(layer_status) == 4
        assert layer_status1.name == "emb0"
        assert layer_status1.module_type == "lora.Embedding"
        assert layer_status1.enabled is True
        assert layer_status1.active_adapters == ["default"]
        assert layer_status1.merged_adapters == []
        assert layer_status1.requires_grad == {"default": True}
        assert layer_status1.available_adapters == ["default"]
        assert layer_status1.devices == {"default": ["cpu"]}

    ###############
    # error cases #
    ###############

    def test_vanilla_model_raises(self):
        model = nn.Linear(10, 10)
        # note: full error message is longer
        with pytest.raises(ValueError, match="No adapter layers found in the model"):
            get_layer_status(model)

        with pytest.raises(ValueError, match="No adapter layers found in the model"):
            get_model_status(model)

    def test_transformer_model_without_adapter_raises(self):
        model_id = "gpt2"
        with hub_online_once(model_id):
            model = AutoModelForCausalLM.from_pretrained(model_id)
        # note: full error message is longer
        with pytest.raises(ValueError, match="No adapter layers found in the model"):
            get_layer_status(model)

        with pytest.raises(ValueError, match="No adapter layers found in the model"):
            get_model_status(model)

    def test_prefix_tuning(self):
        model_id = "hf-internal-testing/tiny-random-BartForConditionalGeneration"
        with hub_online_once(model_id):
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        config = PromptTuningConfig(task_type="SEQ_2_SEQ_LM", num_virtual_tokens=10)
        model = get_peft_model(model, config)

        # note: full error message is longer
        with pytest.raises(TypeError, match=re.escape("get_layer_status() got an invalid PeftModel instance")):
            model.get_layer_status()

        with pytest.raises(TypeError, match=re.escape("get_model_status() got an invalid PeftModel instance")):
            model.get_model_status()

    def test_adaption_prompt(self):
        model_id = "HuggingFaceH4/tiny-random-LlamaForCausalLM"
        with hub_online_once(model_id):
            model = AutoModelForCausalLM.from_pretrained(model_id)
        config = AdaptionPromptConfig(adapter_layers=1, adapter_len=4)
        model = get_peft_model(model, config)

        # note: full error message is longer
        with pytest.raises(TypeError, match=re.escape("get_layer_status() got an invalid PeftModel instance")):
            model.get_layer_status()

        with pytest.raises(TypeError, match=re.escape("get_model_status() got an invalid PeftModel instance")):
            model.get_model_status()

    def test_mixed_model_raises(self):
        class SimpleNet(nn.Module):
            def __init__(self, bias=True):
                super().__init__()
                # note: out_features must be > rank or else OFT will be an identity transform
                self.lin0 = nn.Linear(10, 20, bias=bias)
                self.relu = nn.ReLU()
                self.lin1 = nn.Linear(20, 16, bias=bias)

            def forward(self, X):
                X = X.float()
                X = self.lin0(X)
                X = self.relu(X)
                X = self.lin1(X)
                return X

        base_model = SimpleNet()
        config0 = LoraConfig(target_modules=["lin0"], init_lora_weights=False)
        config1 = LoHaConfig(target_modules=["lin0", "lin1"], init_weights=False)
        model = get_peft_model(base_model, config0, adapter_name="adapter0", mixed="mixed")
        model.add_adapter("adapter1", config1)

        # note: full error message is longer
        with pytest.raises(TypeError, match="get_layer_status is not supported for PeftMixedModel"):
            model.get_layer_status()

        with pytest.raises(TypeError, match="get_model_status is not supported for PeftMixedModel"):
            model.get_model_status()


# Tests for BaseTuner
class MockModelConfig:
    config = {"mock_key": "mock_value"}

    def to_dict(self):
        return self.config


class ModelWithConfig(nn.Module):
    def __init__(self):
        self.config = MockModelConfig()


class ModelWithDictConfig(nn.Module):
    def __init__(self):
        self.config = MockModelConfig.config


class ModelWithNoConfig(nn.Module):
    pass


class TestBaseTunerGetModelConfig(unittest.TestCase):
    def test_get_model_config_use_to_dict(self):
        config = BaseTuner.get_model_config(ModelWithConfig())
        assert config == MockModelConfig.config

    def test_get_model_config_as_dict(self):
        config = BaseTuner.get_model_config(ModelWithDictConfig())
        assert config == MockModelConfig.config

    def test_get_model_config_with_no_config(self):
        config = BaseTuner.get_model_config(ModelWithNoConfig())
        assert config == DUMMY_MODEL_CONFIG


class TestBaseTunerWarnForTiedEmbeddings:
    model_id = "HuggingFaceH4/tiny-random-LlamaForCausalLM"
    warn_end_inject = "huggingface/peft/issues/2018."
    warn_end_merge = (
        "# Now use the original model but in untied format\n"
        "model = AutoModelForCausalLM.from_pretrained(untied_model_dir)\n```\n"
    )

    def _get_peft_model(self, tie_word_embeddings, target_module):
        with hub_online_once(self.model_id):
            base_model = AutoModelForCausalLM.from_pretrained(self.model_id, tie_word_embeddings=tie_word_embeddings)
        model = get_peft_model(
            base_model,
            LoraConfig(target_modules=[target_module]),
        )
        return model

    def _is_warn_triggered(self, warning_list, endswith):
        return any(str(warning.message).endswith(endswith) for warning in warning_list)

    def test_warn_for_tied_embeddings_inject(self, recwarn):
        self._get_peft_model(tie_word_embeddings=True, target_module="lm_head")
        assert self._is_warn_triggered(recwarn.list, self.warn_end_inject)

    def test_warn_for_tied_embeddings_merge(self, recwarn):
        model = self._get_peft_model(tie_word_embeddings=True, target_module="lm_head")
        model.merge_and_unload()
        assert self._is_warn_triggered(recwarn.list, self.warn_end_merge)

    def test_no_warn_for_untied_embeddings_inject(self, recwarn):
        self._get_peft_model(tie_word_embeddings=False, target_module="lm_head")
        assert not self._is_warn_triggered(recwarn.list, self.warn_end_inject)

    def test_no_warn_for_untied_embeddings_merge(self, recwarn):
        model_not_tied = self._get_peft_model(tie_word_embeddings=False, target_module="lm_head")
        model_not_tied.merge_and_unload()
        assert not self._is_warn_triggered(recwarn.list, self.warn_end_merge)

    def test_no_warn_for_no_target_module_inject(self, recwarn):
        self._get_peft_model(tie_word_embeddings=True, target_module="q_proj")
        assert not self._is_warn_triggered(recwarn.list, self.warn_end_inject)

    def test_no_warn_for_no_target_module_merge(self, recwarn):
        model_no_target_module = self._get_peft_model(tie_word_embeddings=True, target_module="q_proj")
        model_no_target_module.merge_and_unload()
        assert not self._is_warn_triggered(recwarn.list, self.warn_end_merge)


class TestFindMinimalTargetModules:
    @pytest.mark.parametrize(
        "target_modules, other_module_names, expected",
        [
            (["bar"], [], {"bar"}),
            (["foo"], ["bar"], {"foo"}),
            (["1.foo", "2.foo"], ["3.foo", "4.foo"], {"1.foo", "2.foo"}),
            # Could also return "bar.baz" but we want the shorter one
            (["bar.baz"], ["foo.bar"], {"baz"}),
            (["1.foo", "2.foo", "bar.baz"], ["3.foo", "bar.bla"], {"1.foo", "2.foo", "baz"}),
            # Case with longer suffix chains and nested suffixes
            (["a.b.c", "d.e.f", "g.h.i"], ["j.k.l", "m.n.o"], {"c", "f", "i"}),
            (["a.b.c", "d.e.f", "g.h.i"], ["a.b.x", "d.x.f", "x.h.i"], {"c", "e.f", "g.h.i"}),
            # Case with multiple items that can be covered by a single suffix
            (["foo.bar.baz", "qux.bar.baz"], ["baz.bar.foo"], {"baz"}),
            # Realistic examples
            # Only match k_proj
            (
                ["model.decoder.layers.{i}.self_attn.k_proj" for i in range(12)],
                (
                    ["model.decoder.layers.{i}.self_attn" for i in range(12)]
                    + ["model.decoder.layers.{i}.self_attn.v_proj" for i in range(12)]
                    + ["model.decoder.layers.{i}.self_attn.q_proj" for i in range(12)]
                ),
                {"k_proj"},
            ),
            # Match all k_proj except the one in layer 5 => no common suffix
            (
                ["model.decoder.layers.{i}.self_attn.k_proj" for i in range(12) if i != 5],
                (
                    ["model.decoder.layers.5.self_attn.k_proj"]
                    + ["model.decoder.layers.{i}.self_attn" for i in range(12)]
                    + ["model.decoder.layers.{i}.self_attn.v_proj" for i in range(12)]
                    + ["model.decoder.layers.{i}.self_attn.q_proj" for i in range(12)]
                ),
                {"{i}.self_attn.k_proj" for i in range(12) if i != 5},
            ),
        ],
    )
    def test_find_minimal_target_modules(self, target_modules, other_module_names, expected):
        # check all possible combinations of list and set
        result = find_minimal_target_modules(target_modules, other_module_names)
        assert result == expected

        result = find_minimal_target_modules(set(target_modules), other_module_names)
        assert result == expected

        result = find_minimal_target_modules(target_modules, set(other_module_names))
        assert result == expected

        result = find_minimal_target_modules(set(target_modules), set(other_module_names))
        assert result == expected

    def test_find_minimal_target_modules_empty_raises(self):
        with pytest.raises(ValueError, match="target_modules should be a list or set of strings"):
            find_minimal_target_modules([], ["foo"])

        with pytest.raises(ValueError, match="target_modules should be a list or set of strings"):
            find_minimal_target_modules(set(), ["foo"])

    def test_find_minimal_target_modules_contains_empty_string_raises(self):
        target_modules = ["", "foo", "bar.baz"]
        other_module_names = ["bar"]
        with pytest.raises(ValueError, match="target_modules should not contain an empty string"):
            find_minimal_target_modules(target_modules, other_module_names)

    def test_find_minimal_target_modules_string_raises(self):
        target_modules = "foo"
        other_module_names = ["bar"]
        with pytest.raises(ValueError, match="target_modules should be a list or set of strings"):
            find_minimal_target_modules(target_modules, other_module_names)

    @pytest.mark.parametrize(
        "target_modules, other_module_names",
        [
            (["foo"], ["foo"]),
            (["foo.bar"], ["foo.bar"]),
            (["foo.bar", "spam", "eggs"], ["foo.bar"]),
            (["foo.bar", "spam"], ["foo.bar", "eggs"]),
            (["foo.bar"], ["foo.bar", "spam", "eggs"]),
        ],
    )
    def test_find_minimal_target_modules_not_disjoint_raises(self, target_modules, other_module_names):
        msg = (
            "target_modules and other_module_names contain common elements, this should not happen, please "
            "open a GitHub issue at https://github.com/huggingface/peft/issues with the code to reproduce this issue"
        )
        with pytest.raises(ValueError, match=msg):
            find_minimal_target_modules(target_modules, other_module_names)

    def test_get_peft_model_applies_find_target_modules(self):
        # Check that when calling get_peft_model, the target_module optimization is indeed applied if the length of
        # target_modules is big enough. The resulting model itself should be unaffected.
        torch.manual_seed(0)
        model_id = "facebook/opt-125m"  # must be big enough for optimization to trigger
        with hub_online_once(model_id):
            model = AutoModelForCausalLM.from_pretrained(model_id)

        # base case: specify target_modules in a minimal fashion
        config = LoraConfig(init_lora_weights=False, target_modules=["q_proj", "v_proj"])
        model = get_peft_model(model, config)

        # this list contains all targeted modules listed separately
        big_target_modules = [name for name, module in model.named_modules() if isinstance(module, LoraLayer)]
        # sanity check
        assert len(big_target_modules) > MIN_TARGET_MODULES_FOR_OPTIMIZATION

        # make a "checksum" of the model for comparison
        model_check_sum_before = sum(p.sum() for p in model.parameters())

        # strip prefix so that the names they can be used as new target_modules
        prefix_to_strip = "base_model.model.model."
        big_target_modules = [name[len(prefix_to_strip) :] for name in big_target_modules]

        del model

        torch.manual_seed(0)
        with hub_online_once(model_id):
            model = AutoModelForCausalLM.from_pretrained(model_id)
        # pass the big target_modules to config
        config = LoraConfig(init_lora_weights=False, target_modules=big_target_modules)
        model = get_peft_model(model, config)

        # check that target modules have been condensed
        assert model.peft_config["default"].target_modules == {"q_proj", "v_proj"}

        # check that the resulting model is still the same
        model_check_after = sum(p.sum() for p in model.parameters())
        assert model_check_sum_before == model_check_after

    def test_suffix_is_substring_of_other_suffix(self):
        # This test is based on a real world bug found in diffusers. The issue was that we needed the suffix
        # 'time_emb_proj' in the minimal target modules. However, if there already was the suffix 'proj' in the
        # required_suffixes, 'time_emb_proj' would not be added because the test was `endswith(suffix)` and
        # 'time_emb_proj' ends with 'proj'. The correct logic is to test if `endswith("." + suffix")`. The module names
        # chosen here are only a subset of the hundreds of actual module names but this subset is sufficient to
        # replicate the bug.
        target_modules = [
            "down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj",
            "mid_block.attentions.0.transformer_blocks.0.ff.net.0.proj",
            "up_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj",
            "mid_block.attentions.0.proj_out",
            "up_blocks.0.attentions.0.proj_out",
            "down_blocks.1.attentions.0.proj_out",
            "up_blocks.0.resnets.0.time_emb_proj",
            "down_blocks.0.resnets.0.time_emb_proj",
            "mid_block.resnets.0.time_emb_proj",
        ]
        other_module_names = [
            "conv_in",
            "time_proj",
            "time_embedding",
            "time_embedding.linear_1",
            "add_time_proj",
            "add_embedding",
            "add_embedding.linear_1",
            "add_embedding.linear_2",
            "down_blocks",
            "down_blocks.0",
            "down_blocks.0.resnets",
            "down_blocks.0.resnets.0",
            "up_blocks",
            "up_blocks.0",
            "up_blocks.0.attentions",
            "up_blocks.0.attentions.0",
            "up_blocks.0.attentions.0.norm",
            "up_blocks.0.attentions.0.transformer_blocks",
            "up_blocks.0.attentions.0.transformer_blocks.0",
            "up_blocks.0.attentions.0.transformer_blocks.0.norm1",
            "up_blocks.0.attentions.0.transformer_blocks.0.attn1",
        ]
        expected = {"time_emb_proj", "proj", "proj_out"}
        result = find_minimal_target_modules(target_modules, other_module_names)
        assert result == expected

    def test_get_peft_modules_module_name_is_suffix_of_another_module(self):
        # Solves the following bug:
        # https://github.com/huggingface/diffusers/pull/9622#issuecomment-2404789721

        # The cause for the bug is as follows: When we have, say, a module called "bar.0.query" that we want to target
        # and another module called "foo_bar.0.query" that we don't want to target, there was potential for an error.
        # This is not caused by _find_minimal_target_modules directly, but rather the bug was inside of
        # BaseTuner.inject_adapter and how the names_no_target were chosen. Those used to be chosen based on suffix. In
        # our example, however, "bar.0.query" is a suffix of "foo_bar.0.query", therefore "foo_bar.0.query" was *not*
        # added to names_no_target when it should have. As a consequence, during the optimization, it looks like "query"
        # is safe to use as target_modules because we don't see that it wrongly matches "foo_bar.0.query".

        # ensure that we have sufficiently many modules to trigger the optimization
        n_layers = MIN_TARGET_MODULES_FOR_OPTIMIZATION + 1

        class InnerModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.query = nn.Linear(10, 10)

        class OuterModule(nn.Module):
            def __init__(self):
                super().__init__()
                # note that "transformer_blocks" is a suffix of "single_transformer_blocks"
                self.transformer_blocks = nn.ModuleList([InnerModule() for _ in range(n_layers)])
                self.single_transformer_blocks = nn.ModuleList([InnerModule() for _ in range(n_layers)])

        # we want to match all "transformer_blocks" layers but not "single_transformer_blocks"
        target_modules = [f"transformer_blocks.{i}.query" for i in range(n_layers)]
        model = get_peft_model(OuterModule(), LoraConfig(target_modules=target_modules))

        # sanity check: we should have n_layers PEFT layers in model.transformer_blocks
        transformer_blocks = model.base_model.model.transformer_blocks
        assert sum(isinstance(module, BaseTunerLayer) for module in transformer_blocks.modules()) == n_layers

        # we should not have any PEFT layers in model.single_transformer_blocks
        single_transformer_blocks = model.base_model.model.single_transformer_blocks
        assert not any(isinstance(module, BaseTunerLayer) for module in single_transformer_blocks.modules())

        # target modules should *not* be simplified to "query" as that would match "single_transformers_blocks" too
        assert model.peft_config["default"].target_modules != {"query"}

    def test_find_minimal_target_modules_does_not_error_with_ia3(self, tmp_path):
        # See #2429
        # There is an issue with the compression of the target_modules attribute when using IA³. There, we additionally
        # have the feedforward_modules attribute, which must be subset of target_modules. When target_modules is shrunk,
        # the subset check will fail. This test ensures that this doesn't happen.
        n_layers = MIN_TARGET_MODULES_FOR_OPTIMIZATION + 1

        class InnerModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.query = nn.Linear(10, 10)

        class OuterModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([InnerModule() for _ in range(n_layers)])

        target_modules = [f"blocks.{i}.query" for i in range(n_layers)]
        feedforward_modules = [f"blocks.{i}.query" for i in range(n_layers)]
        # the subset check happens here
        config = IA3Config(target_modules=target_modules, feedforward_modules=feedforward_modules)
        # the optimization step happens here, after the subset check, so at first we're fine, but we will run into an
        # issue after a save/load roundtrip
        model = get_peft_model(OuterModule(), config)
        model.save_pretrained(tmp_path)
        del model

        # does not raise
        PeftModel.from_pretrained(OuterModule(), tmp_path)


class TestRankAndAlphaPattern:
    @pytest.fixture
    def model(self):
        # we always target the foo layers, the *bar* layers are used as a control group to ensure that they are not
        # accidentally targeted
        class Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = nn.Linear(1, 1)
                self.barfoo = nn.Linear(1, 1)

        class Middle(nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = nn.Linear(1, 1)
                self.foobar = nn.Linear(1, 1)
                self.module = Inner()

        class Outer(nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = nn.Linear(1, 1)
                self.bar = nn.Linear(1, 1)
                self.module = Middle()

        # resulting model for overview:
        # Outer(
        #   (foo): Linear(...)
        #   (bar): Linear(...)
        #   (module): Middle(
        #     (foo): Linear(...)
        #     (foobar): Linear(...)
        #     (module): Inner(
        #       (foo): Linear(...)
        #       (barfoo): Linear(...)
        #     )
        #   )
        # )

        return Outer()

    def test_no_rank_nor_alpha_pattern(self, model):
        # sanity check the default case, no rank or alpha pattern
        config = LoraConfig(target_modules="all-linear")
        model = get_peft_model(model, config).base_model.model
        # r is the default rank and alpha, thus scaling is 1.0
        assert model.foo.r["default"] == 8
        assert model.foo.scaling["default"] == 1.0
        assert model.bar.r["default"] == 8
        assert model.bar.scaling["default"] == 1.0
        assert model.module.foo.r["default"] == 8
        assert model.module.foo.scaling["default"] == 1.0
        assert model.module.foobar.r["default"] == 8
        assert model.module.foobar.scaling["default"] == 1.0
        assert model.module.module.foo.r["default"] == 8
        assert model.module.module.foo.scaling["default"] == 1.0
        assert model.module.module.barfoo.r["default"] == 8
        assert model.module.module.barfoo.scaling["default"] == 1.0

    def test_rank_and_alpha_pattern_no_matching_keys(self, model):
        # sanity check for non-matching keys, no rank or alpha pattern
        config = LoraConfig(target_modules="all-linear", rank_pattern={"bla": 4, "oof": 6}, alpha_pattern={"baz": 3})
        model = get_peft_model(model, config).base_model.model
        # r is the default rank and alpha, thus scaling is 1.0
        assert model.foo.r["default"] == 8
        assert model.foo.scaling["default"] == 1.0
        assert model.bar.r["default"] == 8
        assert model.bar.scaling["default"] == 1.0
        assert model.module.foo.r["default"] == 8
        assert model.module.foo.scaling["default"] == 1.0
        assert model.module.foobar.r["default"] == 8
        assert model.module.foobar.scaling["default"] == 1.0
        assert model.module.module.foo.r["default"] == 8
        assert model.module.module.foo.scaling["default"] == 1.0
        assert model.module.module.barfoo.r["default"] == 8
        assert model.module.module.barfoo.scaling["default"] == 1.0

    # below, we test all permutations for rank_pattern of targeting outer, middle, and inner foo layers:

    def test_rank_pattern_target_all(self, model):
        config = LoraConfig(target_modules="all-linear", rank_pattern={"foo": 16})
        model = get_peft_model(model, config).base_model.model
        assert model.foo.r["default"] == 16
        assert model.bar.r["default"] == 8
        assert model.module.foo.r["default"] == 16
        assert model.module.foobar.r["default"] == 8
        assert model.module.module.foo.r["default"] == 16
        assert model.module.module.barfoo.r["default"] == 8

    def test_rank_pattern_target_outer(self, model):
        config = LoraConfig(target_modules="all-linear", rank_pattern={"^foo": 16})
        model = get_peft_model(model, config).base_model.model
        assert model.foo.r["default"] == 16
        assert model.bar.r["default"] == 8
        assert model.module.foo.r["default"] == 8
        assert model.module.foobar.r["default"] == 8
        assert model.module.module.foo.r["default"] == 8
        assert model.module.module.barfoo.r["default"] == 8

    def test_rank_pattern_target_middle(self, model):
        config = LoraConfig(target_modules="all-linear", rank_pattern={"^module.foo": 16})
        model = get_peft_model(model, config).base_model.model
        assert model.foo.r["default"] == 8
        assert model.bar.r["default"] == 8
        assert model.module.foo.r["default"] == 16
        assert model.module.foobar.r["default"] == 8
        assert model.module.module.foo.r["default"] == 8
        assert model.module.module.barfoo.r["default"] == 8

    def test_rank_pattern_target_inner(self, model):
        config = LoraConfig(target_modules="all-linear", rank_pattern={"module.module.foo": 16})
        model = get_peft_model(model, config).base_model.model
        assert model.foo.r["default"] == 8
        assert model.bar.r["default"] == 8
        assert model.module.foo.r["default"] == 8
        assert model.module.foobar.r["default"] == 8
        assert model.module.module.foo.r["default"] == 16
        assert model.module.module.barfoo.r["default"] == 8

    def test_rank_pattern_target_inner_with_caret(self, model):
        # same as before, but using the caret in the regex should also work
        config = LoraConfig(target_modules="all-linear", rank_pattern={"^module.module.foo": 16})
        model = get_peft_model(model, config).base_model.model
        assert model.foo.r["default"] == 8
        assert model.bar.r["default"] == 8
        assert model.module.foo.r["default"] == 8
        assert model.module.foobar.r["default"] == 8
        assert model.module.module.foo.r["default"] == 16
        assert model.module.module.barfoo.r["default"] == 8

    def test_rank_pattern_target_middle_inner(self, model):
        config = LoraConfig(target_modules="all-linear", rank_pattern={"module.foo": 16})
        model = get_peft_model(model, config).base_model.model
        assert model.foo.r["default"] == 8
        assert model.bar.r["default"] == 8
        assert model.module.foo.r["default"] == 16
        assert model.module.foobar.r["default"] == 8
        assert model.module.module.foo.r["default"] == 16
        assert model.module.module.barfoo.r["default"] == 8

    def test_rank_pattern_target_middle_inner_different_ranks(self, model):
        # same layers targeted as in previous test, but with different ranks
        config = LoraConfig(target_modules="all-linear", rank_pattern={"^module.foo": 16, "^module.module.foo": 24})
        model = get_peft_model(model, config).base_model.model
        assert model.foo.r["default"] == 8
        assert model.bar.r["default"] == 8
        assert model.module.foo.r["default"] == 16
        assert model.module.foobar.r["default"] == 8
        assert model.module.module.foo.r["default"] == 24
        assert model.module.module.barfoo.r["default"] == 8

    def test_rank_pattern_target_outer_middle(self, model):
        config = LoraConfig(target_modules="all-linear", rank_pattern={"^foo": 16, "^module.foo": 24})
        model = get_peft_model(model, config).base_model.model
        assert model.foo.r["default"] == 16
        assert model.bar.r["default"] == 8
        assert model.module.foo.r["default"] == 24
        assert model.module.foobar.r["default"] == 8
        assert model.module.module.foo.r["default"] == 8
        assert model.module.module.barfoo.r["default"] == 8

    def test_rank_pattern_target_outer_inner(self, model):
        config = LoraConfig(target_modules="all-linear", rank_pattern={"^foo": 16, "module.module.foo": 24})
        model = get_peft_model(model, config).base_model.model
        assert model.foo.r["default"] == 16
        assert model.bar.r["default"] == 8
        assert model.module.foo.r["default"] == 8
        assert model.module.foobar.r["default"] == 8
        assert model.module.module.foo.r["default"] == 24
        assert model.module.module.barfoo.r["default"] == 8

    def test_rank_pattern_target_outer_inner_with_caret(self, model):
        # same as before, but using the caret in the regex should also work
        config = LoraConfig(target_modules="all-linear", rank_pattern={"^foo": 16, "^module.module.foo": 24})
        model = get_peft_model(model, config).base_model.model
        assert model.foo.r["default"] == 16
        assert model.bar.r["default"] == 8
        assert model.module.foo.r["default"] == 8
        assert model.module.foobar.r["default"] == 8
        assert model.module.module.foo.r["default"] == 24
        assert model.module.module.barfoo.r["default"] == 8

    def test_rank_pattern_target_outer_middle_inner_with_caret(self, model):
        # indicate each layer with a different rank and use the caret in the regex
        config = LoraConfig(
            target_modules="all-linear", rank_pattern={"^foo": 16, "^module.foo": 24, "^module.module.foo": 32}
        )
        model = get_peft_model(model, config).base_model.model
        assert model.foo.r["default"] == 16
        assert model.bar.r["default"] == 8
        assert model.module.foo.r["default"] == 24
        assert model.module.foobar.r["default"] == 8
        assert model.module.module.foo.r["default"] == 32
        assert model.module.module.barfoo.r["default"] == 8

    def test_rank_pattern_target_outer_middle_inner_with_caret_dict_order(self, model):
        # same as before, but change the order of the rank_pattern dict
        config = LoraConfig(
            target_modules="all-linear", rank_pattern={"^module.module.foo": 32, "^module.foo": 24, "^foo": 16}
        )
        model = get_peft_model(model, config).base_model.model
        assert model.foo.r["default"] == 16
        assert model.bar.r["default"] == 8
        assert model.module.foo.r["default"] == 24
        assert model.module.foobar.r["default"] == 8
        assert model.module.module.foo.r["default"] == 32
        assert model.module.module.barfoo.r["default"] == 8

    # below, we test all permutations for alpha_pattern of targeting outer, middle, and inner foo layers:
    # these tests are analogous to the rank_pattern tests above

    def test_alpha_pattern_target_all(self, model):
        config = LoraConfig(target_modules="all-linear", alpha_pattern={"foo": 4})
        model = get_peft_model(model, config).base_model.model
        assert model.foo.scaling["default"] == 0.5
        assert model.bar.scaling["default"] == 1.0
        assert model.module.foo.scaling["default"] == 0.5
        assert model.module.foobar.scaling["default"] == 1.0
        assert model.module.module.foo.scaling["default"] == 0.5
        assert model.module.module.barfoo.scaling["default"] == 1.0

    def test_alpha_pattern_target_outer(self, model):
        config = LoraConfig(target_modules="all-linear", alpha_pattern={"^foo": 4})
        model = get_peft_model(model, config).base_model.model
        assert model.foo.scaling["default"] == 0.5
        assert model.bar.scaling["default"] == 1.0
        assert model.module.foo.scaling["default"] == 1.0
        assert model.module.foobar.scaling["default"] == 1.0
        assert model.module.module.foo.scaling["default"] == 1.0
        assert model.module.module.barfoo.scaling["default"] == 1.0

    def test_alpha_pattern_target_middle(self, model):
        config = LoraConfig(target_modules="all-linear", alpha_pattern={"^module.foo": 4})
        model = get_peft_model(model, config).base_model.model
        assert model.foo.scaling["default"] == 1.0
        assert model.bar.scaling["default"] == 1.0
        assert model.module.foo.scaling["default"] == 0.5
        assert model.module.foobar.scaling["default"] == 1.0
        assert model.module.module.foo.scaling["default"] == 1.0
        assert model.module.module.barfoo.scaling["default"] == 1.0

    def test_alpha_pattern_target_inner(self, model):
        config = LoraConfig(target_modules="all-linear", alpha_pattern={"module.module.foo": 4})
        model = get_peft_model(model, config).base_model.model
        assert model.foo.scaling["default"] == 1.0
        assert model.bar.scaling["default"] == 1.0
        assert model.module.foo.scaling["default"] == 1.0
        assert model.module.foobar.scaling["default"] == 1.0
        assert model.module.module.foo.scaling["default"] == 0.5
        assert model.module.module.barfoo.scaling["default"] == 1.0

    def test_alpha_pattern_target_inner_with_caret(self, model):
        # same as before, but using the caret in the regex should also work
        config = LoraConfig(target_modules="all-linear", alpha_pattern={"^module.module.foo": 4})
        model = get_peft_model(model, config).base_model.model
        assert model.foo.scaling["default"] == 1.0
        assert model.bar.scaling["default"] == 1.0
        assert model.module.foo.scaling["default"] == 1.0
        assert model.module.foobar.scaling["default"] == 1.0
        assert model.module.module.foo.scaling["default"] == 0.5
        assert model.module.module.barfoo.scaling["default"] == 1.0

    def test_alpha_pattern_target_middle_inner(self, model):
        config = LoraConfig(target_modules="all-linear", alpha_pattern={"module.foo": 4})
        model = get_peft_model(model, config).base_model.model
        assert model.foo.scaling["default"] == 1.0
        assert model.bar.scaling["default"] == 1.0
        assert model.module.foo.scaling["default"] == 0.5
        assert model.module.foobar.scaling["default"] == 1.0
        assert model.module.module.foo.scaling["default"] == 0.5
        assert model.module.module.barfoo.scaling["default"] == 1.0

    def test_alpha_pattern_target_middle_inner_different_alphas(self, model):
        # same layers targeted as in previous test, but with different alphas
        config = LoraConfig(target_modules="all-linear", alpha_pattern={"^module.foo": 4, "^module.module.foo": 2})
        model = get_peft_model(model, config).base_model.model
        assert model.foo.scaling["default"] == 1.0
        assert model.bar.scaling["default"] == 1.0
        assert model.module.foo.scaling["default"] == 0.5
        assert model.module.foobar.scaling["default"] == 1.0
        assert model.module.module.foo.scaling["default"] == 0.25
        assert model.module.module.barfoo.scaling["default"] == 1.0

    def test_alpha_pattern_target_outer_middle(self, model):
        config = LoraConfig(target_modules="all-linear", alpha_pattern={"^foo": 4, "^module.foo": 2})
        model = get_peft_model(model, config).base_model.model
        assert model.foo.scaling["default"] == 0.5
        assert model.bar.scaling["default"] == 1.0
        assert model.module.foo.scaling["default"] == 0.25
        assert model.module.foobar.scaling["default"] == 1.0
        assert model.module.module.foo.scaling["default"] == 1.0
        assert model.module.module.barfoo.scaling["default"] == 1.0

    def test_alpha_pattern_target_outer_inner(self, model):
        config = LoraConfig(target_modules="all-linear", alpha_pattern={"^foo": 4, "module.module.foo": 2})
        model = get_peft_model(model, config).base_model.model
        assert model.foo.scaling["default"] == 0.5
        assert model.bar.scaling["default"] == 1.0
        assert model.module.foo.scaling["default"] == 1.0
        assert model.module.foobar.scaling["default"] == 1.0
        assert model.module.module.foo.scaling["default"] == 0.25
        assert model.module.module.barfoo.scaling["default"] == 1.0

    def test_alpha_pattern_target_outer_inner_with_caret(self, model):
        # same as before, but using the caret in the regex should also work
        config = LoraConfig(target_modules="all-linear", alpha_pattern={"^foo": 4, "^module.module.foo": 2})
        model = get_peft_model(model, config).base_model.model
        assert model.foo.scaling["default"] == 0.5
        assert model.bar.scaling["default"] == 1.0
        assert model.module.foo.scaling["default"] == 1.0
        assert model.module.foobar.scaling["default"] == 1.0
        assert model.module.module.foo.scaling["default"] == 0.25
        assert model.module.module.barfoo.scaling["default"] == 1.0

    def test_alpha_pattern_target_outer_middle_inner_with_caret(self, model):
        # indicate each layer with a different alpha and use the caret in the regex
        config = LoraConfig(
            target_modules="all-linear", alpha_pattern={"^foo": 4, "^module.foo": 2, "^module.module.foo": 1}
        )
        model = get_peft_model(model, config).base_model.model
        assert model.foo.scaling["default"] == 0.5
        assert model.bar.scaling["default"] == 1.0
        assert model.module.foo.scaling["default"] == 0.25
        assert model.module.foobar.scaling["default"] == 1.0
        assert model.module.module.foo.scaling["default"] == 0.125
        assert model.module.module.barfoo.scaling["default"] == 1.0

    def test_alpha_pattern_target_outer_middle_inner_with_caret_dict_order(self, model):
        # same as before, but change the order of the alpha_pattern dict
        config = LoraConfig(
            target_modules="all-linear", alpha_pattern={"^module.module.foo": 1, "^module.foo": 2, "^foo": 4}
        )
        model = get_peft_model(model, config).base_model.model
        assert model.foo.scaling["default"] == 0.5
        assert model.bar.scaling["default"] == 1.0
        assert model.module.foo.scaling["default"] == 0.25
        assert model.module.foobar.scaling["default"] == 1.0
        assert model.module.module.foo.scaling["default"] == 0.125
        assert model.module.module.barfoo.scaling["default"] == 1.0
