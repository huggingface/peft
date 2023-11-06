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
import copy
import os
import tempfile
import unittest

import torch
from parameterized import parameterized
from torch import nn
from transformers.pytorch_utils import Conv1D

from peft import AdaLoraConfig, IA3Config, LoHaConfig, LoKrConfig, LoraConfig, PeftModel, get_peft_model
from peft.tuners.tuners_utils import BaseTunerLayer

from .testing_common import PeftCommonTester
from .testing_utils import get_state_dict


# MLP is a vanilla FF network with only linear layers
# EmbConv1D has an embedding and a Conv1D layer
# Conv2D has a Conv2D layer
TEST_CASES = [
    ########
    # LoRA #
    ########
    ("Vanilla MLP 1 LoRA", "MLP", LoraConfig, {"target_modules": "lin0"}),
    ("Vanilla MLP 2 LoRA", "MLP", LoraConfig, {"target_modules": ["lin0"]}),
    ("Vanilla MLP 3 LoRA", "MLP", LoraConfig, {"target_modules": ["lin1"]}),
    ("Vanilla MLP 4 LoRA", "MLP", LoraConfig, {"target_modules": ["lin0", "lin1"]}),
    ("Vanilla MLP 5 LoRA", "MLP", LoraConfig, {"target_modules": ["lin0"], "modules_to_save": ["lin1"]}),
    (
        "Vanilla MLP 6 LoRA",
        "MLP",
        LoraConfig,
        {
            "target_modules": ["lin0"],
            "lora_alpha": 4,
            "lora_dropout": 0.1,
        },
    ),
    ("Embedding + transformers Conv1D 1 LoRA", "EmbConv1D", LoraConfig, {"target_modules": ["conv1d"]}),
    ("Embedding + transformers Conv1D 2 LoRA", "EmbConv1D", LoraConfig, {"target_modules": ["emb"]}),
    ("Embedding + transformers Conv1D 3 LoRA", "EmbConv1D", LoraConfig, {"target_modules": ["emb", "conv1d"]}),
    ("Conv2d 1 LoRA", "Conv2d", LoraConfig, {"target_modules": ["conv2d"]}),
    ("Conv2d 2 LoRA", "Conv2d", LoraConfig, {"target_modules": ["conv2d", "lin0"]}),
    #######
    # IA³ #
    #######
    ("Vanilla MLP 1 IA3", "MLP", IA3Config, {"target_modules": "lin0", "feedforward_modules": []}),
    ("Vanilla MLP 2 IA3", "MLP", IA3Config, {"target_modules": "lin0", "feedforward_modules": "lin0"}),
    ("Vanilla MLP 3 IA3", "MLP", IA3Config, {"target_modules": ["lin0"], "feedforward_modules": []}),
    ("Vanilla MLP 4 IA3", "MLP", IA3Config, {"target_modules": ["lin0"], "feedforward_modules": ["lin0"]}),
    ("Vanilla MLP 5 IA3", "MLP", IA3Config, {"target_modules": ["lin1"], "feedforward_modules": []}),
    ("Vanilla MLP 6 IA3", "MLP", IA3Config, {"target_modules": ["lin1"], "feedforward_modules": ["lin1"]}),
    (
        "Vanilla MLP 7 IA3",
        "MLP",
        IA3Config,
        {"target_modules": ["lin0", "lin1"], "feedforward_modules": []},
    ),
    (
        "Vanilla MLP 8 IA3",
        "MLP",
        IA3Config,
        {"target_modules": ["lin0", "lin1"], "feedforward_modules": ["lin0", "lin1"]},
    ),
    (
        "Vanilla MLP 9 IA3",
        "MLP",
        IA3Config,
        {"target_modules": ["lin0"], "modules_to_save": ["lin1"], "feedforward_modules": ["lin0"]},
    ),
    (
        "transformers Conv1D 1 IA3",
        "EmbConv1D",
        IA3Config,
        {"target_modules": ["conv1d"], "feedforward_modules": ["conv1d"]},
    ),
    (
        "transformers Conv1D 2 IA3",
        "EmbConv1D",
        IA3Config,
        {"target_modules": ["conv1d", "lin0"], "feedforward_modules": ["conv1d", "lin0"]},
    ),
    (
        "transformers Conv1D 1 IA3",
        "EmbConv1D",
        IA3Config,
        {"target_modules": ["conv1d"], "feedforward_modules": ["conv1d"], "modules_to_save": ["lin1"]},
    ),
    ("Conv2d 1 IA3", "Conv2d", IA3Config, {"target_modules": ["conv2d"], "feedforward_modules": []}),
    ("Conv2d 2 IA3", "Conv2d", IA3Config, {"target_modules": ["conv2d"], "feedforward_modules": ["conv2d"]}),
    (
        "Conv2d 3 IA3",
        "Conv2d",
        IA3Config,
        {"target_modules": ["conv2d", "lin0"], "feedforward_modules": []},
    ),
    (
        "Conv2d 4 IA3",
        "Conv2d",
        IA3Config,
        {"target_modules": ["conv2d", "lin0"], "feedforward_modules": ["conv2d"]},
    ),
    (
        "Conv2d 5 IA3",
        "Conv2d",
        IA3Config,
        {"target_modules": ["conv2d", "lin0"], "feedforward_modules": ["conv2d", "lin0"]},
    ),
    ########
    # LoHa #
    ########
    ("Vanilla MLP 1 LOHA", "MLP", LoHaConfig, {"target_modules": "lin0"}),
    ("Vanilla MLP 2 LOHA", "MLP", LoHaConfig, {"target_modules": ["lin0"]}),
    ("Vanilla MLP 3 LOHA", "MLP", LoHaConfig, {"target_modules": ["lin1"]}),
    ("Vanilla MLP 4 LOHA", "MLP", LoHaConfig, {"target_modules": ["lin0", "lin1"]}),
    ("Vanilla MLP 5 LOHA", "MLP", LoHaConfig, {"target_modules": ["lin0"], "modules_to_save": ["lin1"]}),
    (
        "Vanilla MLP 6 LOHA",
        "MLP",
        LoHaConfig,
        {
            "target_modules": ["lin0"],
            "alpha": 4,
            "module_dropout": 0.1,
        },
    ),
    ("Vanilla MLP 7 LOHA", "MLP", LoHaConfig, {"target_modules": "lin0", "rank_dropout": 0.5}),
    ("Conv2d 1 LOHA", "Conv2d", LoHaConfig, {"target_modules": ["conv2d"]}),
    ("Conv2d 2 LOHA", "Conv2d", LoHaConfig, {"target_modules": ["conv2d", "lin0"]}),
    ("Conv2d 3 LOHA", "Conv2d", LoHaConfig, {"target_modules": ["conv2d"], "use_effective_conv2d": True}),
    ("Conv2d 4 LOHA", "Conv2d", LoHaConfig, {"target_modules": ["conv2d", "lin0"], "use_effective_conv2d": True}),
    # LoKr
    ("Vanilla MLP 1 LOKR", "MLP", LoKrConfig, {"target_modules": "lin0"}),
    ("Vanilla MLP 2 LOKR", "MLP", LoKrConfig, {"target_modules": ["lin0"]}),
    ("Vanilla MLP 3 LOKR", "MLP", LoKrConfig, {"target_modules": ["lin1"]}),
    ("Vanilla MLP 4 LOKR", "MLP", LoKrConfig, {"target_modules": ["lin0", "lin1"]}),
    ("Vanilla MLP 5 LOKR", "MLP", LoKrConfig, {"target_modules": ["lin0"], "modules_to_save": ["lin1"]}),
    (
        "Vanilla MLP 6 LOKR",
        "MLP",
        LoKrConfig,
        {
            "target_modules": ["lin0"],
            "alpha": 4,
            "module_dropout": 0.1,
        },
    ),
    ("Vanilla MLP 7 LOKR", "MLP", LoKrConfig, {"target_modules": "lin0", "rank_dropout": 0.5}),
    ("Vanilla MLP 8 LOKR", "MLP", LoKrConfig, {"target_modules": "lin0", "decompose_both": True, "r": 1, "alpha": 1}),
    ("Conv2d 1 LOKR", "Conv2d", LoKrConfig, {"target_modules": ["conv2d"]}),
    ("Conv2d 2 LOKR", "Conv2d", LoKrConfig, {"target_modules": ["conv2d", "lin0"]}),
    ("Conv2d 3 LOKR", "Conv2d", LoKrConfig, {"target_modules": ["conv2d"], "use_effective_conv2d": True}),
    ("Conv2d 4 LOKR", "Conv2d", LoKrConfig, {"target_modules": ["conv2d", "lin0"], "use_effective_conv2d": True}),
    (
        "Conv2d 5 LOKR",
        "Conv2d",
        LoKrConfig,
        {"target_modules": ["conv2d", "lin0"], "use_effective_conv2d": True, "decompose_both": True},
    ),
    (
        "Conv2d 6 LOKR",
        "Conv2d",
        LoKrConfig,
        {"target_modules": ["conv2d", "lin0"], "use_effective_conv2d": True, "decompose_factor": 4},
    ),
    (
        "Conv2d 7 LOKR",
        "Conv2d",
        LoKrConfig,
        {
            "target_modules": ["conv2d", "lin0"],
            "use_effective_conv2d": True,
            "decompose_both": True,
            "decompose_factor": 4,
        },
    ),
]

MULTIPLE_ACTIVE_ADAPTERS_TEST_CASES = [
    (
        "LoRA Same",
        "lora",
        LoraConfig,
        {"target_modules": ["lin0"], "init_lora_weights": False},
        {"target_modules": ["lin0"], "init_lora_weights": False},
    ),
    (
        "LoRA Different",
        "lora",
        LoraConfig,
        {"target_modules": ["lin0"], "init_lora_weights": False},
        {"target_modules": ["lin1"], "init_lora_weights": False},
    ),
    (
        "IA3 Same",
        "ia3",
        IA3Config,
        {
            "target_modules": ["lin0"],
            "feedforward_modules": ["lin0"],
            "init_ia3_weights": False,
        },
        {
            "target_modules": ["lin0"],
            "feedforward_modules": ["lin0"],
            "init_ia3_weights": False,
        },
    ),
    (
        "IA3 Different",
        "ia3",
        IA3Config,
        {
            "target_modules": ["lin0"],
            "feedforward_modules": ["lin0"],
            "init_ia3_weights": False,
        },
        {
            "target_modules": ["lin1"],
            "feedforward_modules": ["lin1"],
            "init_ia3_weights": False,
        },
    ),
    (
        "AdaLora Same",
        "adalora",
        AdaLoraConfig,
        {"target_modules": ["lin0"], "init_lora_weights": False, "inference_mode": True},
        {"target_modules": ["lin0"], "init_lora_weights": False, "inference_mode": True},
    ),
    (
        "AdaLora Different",
        "adalora",
        AdaLoraConfig,
        {"target_modules": ["lin0"], "init_lora_weights": False, "inference_mode": True},
        {"target_modules": ["lin1"], "init_lora_weights": False, "inference_mode": True},
    ),
]
PREFIXES = {
    IA3Config: "ia3_",
    LoraConfig: "lora_",
    LoHaConfig: "hada_",
    LoKrConfig: "lokr_",
}


class MLP(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.lin0 = nn.Linear(10, 20, bias=bias)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.lin1 = nn.Linear(20, 2, bias=bias)
        self.sm = nn.LogSoftmax(dim=-1)

    def forward(self, X):
        X = X.float()
        X = self.lin0(X)
        X = self.relu(X)
        X = self.drop(X)
        X = self.lin1(X)
        X = self.sm(X)
        return X


class Block(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.lin0 = nn.Linear(10, 20, bias=bias)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.lin1 = nn.Linear(20, 10, bias=bias)

    def forward(self, X):
        X = X.float()
        X = self.lin0(X)
        X = self.relu(X)
        X = self.drop(X)
        X = self.lin1(X)
        return X


class DeepMLP(nn.Module):
    def __init__(self, bias=True, num_hidden_layers=12):
        super().__init__()
        self.layers = nn.ModuleList([Block(bias=bias) for _ in range(num_hidden_layers)])
        self.out = nn.Linear(10, 2, bias=bias)
        self.sm = nn.LogSoftmax(dim=-1)

    def forward(self, X):
        X = X.float(X)
        for layer in self.layers:
            X = layer(X)
        X = self.out(X)
        X = self.sm(X)
        return X


class ModelEmbConv1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(100, 5)
        self.conv1d = Conv1D(1, 5)
        self.relu = nn.ReLU()
        self.flat = nn.Flatten()
        self.lin0 = nn.Linear(10, 2)
        self.sm = nn.LogSoftmax(dim=-1)

    def forward(self, X):
        X = self.emb(X)
        X = self.conv1d(X)
        X = self.relu(X)
        X = self.flat(X)
        X = self.lin0(X)
        X = self.sm(X)
        return X


class ModelConv2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(5, 10, 3)
        self.relu = nn.ReLU()
        self.flat = nn.Flatten()
        self.lin0 = nn.Linear(10, 2)
        self.sm = nn.LogSoftmax(dim=-1)

    def forward(self, X):
        X = X.float().reshape(2, 5, 3, 3)
        X = self.conv2d(X)
        X = self.relu(X)
        X = self.flat(X)
        X = self.lin0(X)
        X = self.sm(X)
        return X


class MockTransformerWrapper:
    """Mock class to behave like a transformers model.

    This is needed because the tests initialize the model by calling transformers_class.from_pretrained.

    """

    @classmethod
    def from_pretrained(cls, model_id, torch_dtype=None):
        # set the seed so that from_pretrained always returns the same model
        torch.manual_seed(0)

        if torch_dtype is None:
            torch_dtype = torch.float32

        if model_id == "MLP":
            return MLP().to(torch_dtype)

        if model_id == "EmbConv1D":
            return ModelEmbConv1D().to(torch_dtype)

        if model_id == "Conv2d":
            return ModelConv2D().to(torch_dtype)

        raise ValueError(f"model_id {model_id} not implemented")


class PeftCustomModelTester(unittest.TestCase, PeftCommonTester):
    """TODO"""

    transformers_class = MockTransformerWrapper

    def prepare_inputs_for_testing(self):
        X = torch.arange(90).view(9, 10).to(self.torch_device)
        return {"X": X}

    @parameterized.expand(TEST_CASES)
    def test_attributes_parametrized(self, test_name, model_id, config_cls, config_kwargs):
        self._test_model_attr(model_id, config_cls, config_kwargs)

    @parameterized.expand(TEST_CASES)
    def test_adapter_name(self, test_name, model_id, config_cls, config_kwargs):
        self._test_adapter_name(model_id, config_cls, config_kwargs)

    @parameterized.expand(TEST_CASES)
    def test_prepare_for_training_parametrized(self, test_name, model_id, config_cls, config_kwargs):
        # This test does not work with custom models because it assumes that
        # there is always a method get_input_embeddings that returns a layer
        # which does not need updates. Instead, a new test is added below that
        # checks that LoRA works as expected.
        pass

    @parameterized.expand(TEST_CASES)
    def test_save_pretrained(self, test_name, model_id, config_cls, config_kwargs):
        self._test_save_pretrained(model_id, config_cls, config_kwargs)

    @parameterized.expand(TEST_CASES)
    def test_from_pretrained_config_construction(self, test_name, model_id, config_cls, config_kwargs):
        self._test_from_pretrained_config_construction(model_id, config_cls, config_kwargs)

    @parameterized.expand(TEST_CASES)
    def test_merge_layers(self, test_name, model_id, config_cls, config_kwargs):
        config_kwargs = config_kwargs.copy()
        if issubclass(config_cls, LoraConfig):
            config_kwargs["init_lora_weights"] = False
        elif issubclass(config_cls, IA3Config):
            config_kwargs["init_ia3_weights"] = False
        self._test_merge_layers(model_id, config_cls, config_kwargs)

    @parameterized.expand(TEST_CASES)
    def test_merge_layers_fp16(self, test_name, model_id, config_cls, config_kwargs):
        config_kwargs = config_kwargs.copy()
        if issubclass(config_cls, LoraConfig):
            config_kwargs["init_lora_weights"] = False
        elif issubclass(config_cls, IA3Config):
            config_kwargs["init_ia3_weights"] = False
        self._test_merge_layers_fp16(model_id, config_cls, config_kwargs)

    @parameterized.expand(TEST_CASES)
    def test_generate(self, test_name, model_id, config_cls, config_kwargs):
        # Custom models do not (necessarily) have a generate method, so this test is not performed
        pass

    @parameterized.expand(TEST_CASES)
    def test_generate_half_prec(self, test_name, model_id, config_cls, config_kwargs):
        # Custom models do not (necessarily) have a generate method, so this test is not performed
        pass

    @parameterized.expand(TEST_CASES)
    def test_training_custom_models(self, test_name, model_id, config_cls, config_kwargs):
        self._test_training(model_id, config_cls, config_kwargs)

    @parameterized.expand(TEST_CASES)
    def test_training_custom_models_layer_indexing(self, test_name, model_id, config_cls, config_kwargs):
        # At the moment, layer indexing only works when layer names conform to a specific pattern, which is not
        # guaranteed here. Therefore, this test is not performed.
        pass

    @parameterized.expand(TEST_CASES)
    def test_training_custom_models_gradient_checkpointing(self, test_name, model_id, config_cls, config_kwargs):
        self._test_training_gradient_checkpointing(model_id, config_cls, config_kwargs)

    @parameterized.expand(TEST_CASES)
    def test_inference_safetensors(self, test_name, model_id, config_cls, config_kwargs):
        self._test_inference_safetensors(model_id, config_cls, config_kwargs)

    @parameterized.expand(TEST_CASES)
    def test_peft_model_device_map(self, test_name, model_id, config_cls, config_kwargs):
        self._test_peft_model_device_map(model_id, config_cls, config_kwargs)

    @parameterized.expand(TEST_CASES)
    def test_only_params_are_updated(self, test_name, model_id, config_cls, config_kwargs):
        # An explicit test that when using LoRA on a custom model, only the LoRA parameters are updated during training
        X = self.prepare_inputs_for_testing()
        model = self.transformers_class.from_pretrained(model_id).to(self.torch_device)
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)
        model_before = copy.deepcopy(model)

        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

        # train at least 3 steps for all parameters to be updated (probably this is required because of symmetry
        # breaking of some LoRA layers that are initialized with constants)
        for _ in range(3):
            optimizer.zero_grad()
            y_pred = model(**X)
            loss = y_pred.sum()
            loss.backward()
            optimizer.step()

        tol = 1e-4
        params_before = dict(model_before.named_parameters())
        params_after = dict(model.named_parameters())
        self.assertEqual(params_before.keys(), params_after.keys())

        prefix = PREFIXES[config_cls]
        for name, param_before in params_before.items():
            param_after = params_after[name]
            if (prefix in name) or ("modules_to_save" in name):
                # target_modules and modules_to_save _are_ updated
                self.assertFalse(torch.allclose(param_before, param_after, atol=tol, rtol=tol))
            else:
                self.assertTrue(torch.allclose(param_before, param_after, atol=tol, rtol=tol))

    @parameterized.expand(TEST_CASES)
    def test_parameters_after_loading_model(self, test_name, model_id, config_cls, config_kwargs):
        # An explicit test that when loading a trained model, the parameters are loaded correctly
        # see issue #808
        X = self.prepare_inputs_for_testing()
        model = self.transformers_class.from_pretrained(model_id).to(self.torch_device)
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

        # train at least 3 steps for all parameters to be updated (probably this is required because of symmetry
        # breaking of some LoRA layers that are initialized with constants)
        for _ in range(3):
            optimizer.zero_grad()
            y_pred = model(**X)
            loss = y_pred.sum()
            loss.backward()
            optimizer.step()

        tol = 1e-4
        params_before = get_state_dict(model)
        # note: no need to sanity check if parameters were updated at all, this
        # is already covered in the previous test

        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname)
            model_from_pretrained = self.transformers_class.from_pretrained(model_id).to(self.torch_device)
            model_from_pretrained = PeftModel.from_pretrained(model_from_pretrained, tmp_dirname)
            params_after = get_state_dict(model_from_pretrained)

            self.assertEqual(params_before.keys(), params_after.keys())
            for name, param_before in params_before.items():
                param_after = params_after[name]
                self.assertTrue(torch.allclose(param_before, param_after, atol=tol, rtol=tol))

    @parameterized.expand(TEST_CASES)
    def test_disable_adapters(self, test_name, model_id, config_cls, config_kwargs):
        X = self.prepare_inputs_for_testing()
        model = self.transformers_class.from_pretrained(model_id).to(self.torch_device)
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)
        model.eval()
        outputs_before = model(**X)

        model.train()
        # EmbConv1D is slow to learn for some reason
        lr = 0.01 if model_id != "EmbConv1D" else 1.0
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        # train at least 3 steps for all parameters to be updated (probably this is required because of symmetry
        # breaking of some LoRA layers that are initialized with constants)
        for _ in range(3):
            optimizer.zero_grad()
            y_pred = model(**X)
            y = torch.arange(len(y_pred)).to(self.torch_device) % 2
            loss = nn.functional.nll_loss(y_pred, y)
            loss.backward()
            optimizer.step()

        model.eval()
        outputs_after = model(**X)

        with model.disable_adapter():
            outputs_disabled = model(**X)

        # check that after leaving the disable_adapter context, everything is enabled again
        outputs_enabled_after_disable = model(**X)

        self.assertFalse(torch.allclose(outputs_before, outputs_after))
        self.assertTrue(torch.allclose(outputs_before, outputs_disabled))
        self.assertTrue(torch.allclose(outputs_after, outputs_enabled_after_disable))

    @parameterized.expand(TEST_CASES)
    def test_disable_adapters_with_merging(self, test_name, model_id, config_cls, config_kwargs):
        # same as test_disable_adapters, but with merging
        X = self.prepare_inputs_for_testing()
        model = self.transformers_class.from_pretrained(model_id).to(self.torch_device)
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)
        model.eval()
        outputs_before = model(**X)

        model.train()
        lr = 0.01
        # Adam optimizer since SGD isn't great for small models with IA3 + Conv1D
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # train at least 3 steps for all parameters to be updated (probably this is required because of symmetry
        # breaking of some LoRA layers that are initialized with constants)
        for _ in range(3):
            optimizer.zero_grad()
            y_pred = model(**X)
            y = torch.arange(len(y_pred)).to(self.torch_device) % 2
            loss = nn.functional.nll_loss(y_pred, y)
            loss.backward()
            optimizer.step()

        model.eval()
        model.merge_adapter()
        outputs_after = model(**X)

        with model.disable_adapter():
            outputs_disabled = model(**X)

        # check that after leaving the disable_adapter context, everything is enabled again
        outputs_enabled_after_disable = model(**X)

        atol, rtol = 1e-5, 1e-5  # tolerances higher than defaults since merging introduces some numerical instability

        if issubclass(config_cls, IA3Config) and model_id == "Conv2d":  # more instability with Conv2d + IA3
            atol, rtol = 1e-3, 1e-3

        # check that there is a difference in results after training
        self.assertFalse(torch.allclose(outputs_before, outputs_after, atol=atol, rtol=rtol))

        # check that disabling adapters gives the same results as before training
        self.assertTrue(torch.allclose(outputs_before, outputs_disabled, atol=atol, rtol=rtol))

        # check that enabling + disabling adapters does not change the results
        self.assertTrue(torch.allclose(outputs_after, outputs_enabled_after_disable, atol=atol, rtol=rtol))

    @parameterized.expand(TEST_CASES)
    def test_disable_adapter_with_bias_warns(self, test_name, model_id, config_cls, config_kwargs):
        # When training biases in lora, disabling adapters does not reset the biases, so the output is not what users
        # might expect. Therefore, a warning should be given.

        # Note: We test only with custom models since they run really fast. There is really no point in testing the same
        # thing with decoder, encoder_decoder, etc.
        if config_cls != LoraConfig:
            # skip this test for other configs as bias is specific to Lora
            self.skipTest("Testing bias warnings only for LoraConfig")

        if not issubclass(config_cls, LoraConfig):
            self.skipTest("Bias argument is only supported for LoRA models")

        def run_with_disable(config_kwargs, bias):
            config_kwargs = config_kwargs.copy()
            config_kwargs["bias"] = bias
            model = self.transformers_class.from_pretrained(model_id).to(self.torch_device)
            config = config_cls(
                base_model_name_or_path=model_id,
                **config_kwargs,
            )
            peft_model = get_peft_model(model, config)
            with peft_model.disable_adapter():
                pass  # there is nothing to be done

        # check that bias=all and bias=lora_only give a warning with the correct message
        msg_start = "Careful, disabling adapter layers with bias configured to be"
        with self.assertWarns(UserWarning, msg=msg_start):
            run_with_disable(config_kwargs, bias="lora_only")
        with self.assertWarns(UserWarning, msg=msg_start):
            run_with_disable(config_kwargs, bias="all")

        # For bias=none, there is no warning. Unfortunately, AFAIK unittest has no option to assert that no warning is
        # given, therefore, we check that the unittest gives us an AssertionError if we check for a warning
        bias_warning_was_given = False
        try:
            with self.assertWarns(UserWarning) as cm:
                run_with_disable(config_kwargs, bias="none")
                # if we get here, it means there was no AssertionError, i.e. there are warnings -- let's check that they
                # are not related to the bias setting
                if any(warning.message.args[0].startswith(msg_start) for warning in cm.warnings):
                    bias_warning_was_given = True
        except AssertionError:
            # This is good, there was an AssertionError, i.e. there was no warning
            pass
        if bias_warning_was_given:
            # This is bad, there was a warning about the bias when there should not have been any.
            self.fail("There should be no warning when bias is set to 'none'")

    @parameterized.expand(TEST_CASES)
    def test_adding_multiple_adapters_with_bias_raises(self, test_name, model_id, config_cls, config_kwargs):
        self._test_adding_multiple_adapters_with_bias_raises(model_id, config_cls, config_kwargs)

    def test_existing_model_card(self):
        # ensure that if there is already a model card, it is not overwritten
        model = MLP()
        config = LoraConfig(target_modules=["lin0"])
        model = get_peft_model(model, config)

        with tempfile.TemporaryDirectory() as tmp_dirname:
            # create a model card
            text = "---\nmeta: hello\n---\nThis is a model card\n"
            with open(os.path.join(tmp_dirname, "README.md"), "w") as f:
                f.write(text)

            model.save_pretrained(tmp_dirname)
            with open(os.path.join(tmp_dirname, "README.md"), "r") as f:
                model_card = f.read()

        self.assertIn("library_name: peft", model_card)
        self.assertIn("meta: hello", model_card)
        self.assertIn("This is a model card", model_card)

    def test_non_existing_model_card(self):
        # ensure that if there is already a model card, it is not overwritten
        model = MLP()
        config = LoraConfig(target_modules=["lin0"])
        model = get_peft_model(model, config)

        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname)
            with open(os.path.join(tmp_dirname, "README.md"), "r") as f:
                model_card = f.read()

        self.assertIn("library_name: peft", model_card)
        # rough check that the model card is pre-filled
        self.assertGreater(len(model_card), 1000)


class TestMultiRankAdapter(unittest.TestCase):
    """Tests related to multirank LoRA adapters"""

    def test_multirank(self):
        config_1 = LoraConfig(
            r=8,
            lora_alpha=8,
            init_lora_weights=False,
            target_modules=["lin0", "lin1"],
        )
        config_2 = LoraConfig(
            r=8,
            lora_alpha=8,
            init_lora_weights=False,
            target_modules=["lin0", "lin1"],
            rank_pattern={"lin0": 4},
            alpha_pattern={"lin0": 4},
        )

        # Add first adapter
        model = get_peft_model(MLP(), config_1, adapter_name="first")

        # Add second adapter
        model.add_adapter("second", config_2)

        # Extract current and expected ranks
        rank_current = model.lin0.lora_A["second"].weight.shape[0]
        rank_expected = config_2.rank_pattern["lin0"]

        self.assertTrue(rank_current == rank_expected, f"Rank {rank_current} is not equal to expected {rank_expected}")

    def test_multirank_2(self):
        rank_pattern = {}
        alpha_pattern = {}
        r = 4
        lora_alpha = 8

        for i in range(10):
            rank = 64 // (i + 1)
            for j in range(2):
                rank_pattern[f"layers.{i}.lin{j}"] = rank
                alpha_pattern[f"layers.{i}.lin{j}"] = 2 * rank

        config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            init_lora_weights=False,
            target_modules=["lin0", "lin1"],
            rank_pattern=rank_pattern,
            alpha_pattern=alpha_pattern,
        )

        # Add first adapter
        model = get_peft_model(DeepMLP(), config, adapter_name="first")

        # Add second adapter
        model.add_adapter("second", config)

        for adapter in ["first", "second"]:
            for key, module in model.base_model.model.named_modules():
                if isinstance(module, BaseTunerLayer):
                    rank_expected = rank_pattern.get(key, r)
                    rank_current = module.lora_A[adapter].weight.shape[0]
                    self.assertTrue(
                        rank_current == rank_expected, f"Rank {rank_current} is not equal to expected {rank_expected}"
                    )


class TestRepr(unittest.TestCase):
    """Tests related to the repr of adapted models"""

    def test_repr_lora_linear(self):
        config = LoraConfig(target_modules=["lin0"])
        model = get_peft_model(MLP(), config)
        print_output = repr(model.model.lin0)
        self.assertTrue(print_output.startswith("Linear"))
        self.assertTrue("in_features=10, out_features=20" in print_output)
        self.assertTrue("lora_A" in print_output)
        self.assertTrue("lora_B" in print_output)
        self.assertTrue("default" in print_output)

    def test_repr_lora_embedding(self):
        config = LoraConfig(target_modules=["emb"])
        model = get_peft_model(ModelEmbConv1D(), config)
        print_output = repr(model.model.emb)
        self.assertTrue(print_output.startswith("Embedding"))
        self.assertTrue("100, 5" in print_output)
        self.assertTrue("lora_embedding_A" in print_output)
        self.assertTrue("lora_embedding_B" in print_output)
        self.assertTrue("default" in print_output)

    def test_repr_lora_conv1d(self):
        config = LoraConfig(target_modules=["conv1d"])
        model = get_peft_model(ModelEmbConv1D(), config)
        print_output = repr(model.model.conv1d)
        self.assertTrue(print_output.startswith("Linear"))
        self.assertTrue("in_features=5, out_features=1" in print_output)
        self.assertTrue("lora_A" in print_output)
        self.assertTrue("lora_B" in print_output)
        self.assertTrue("default" in print_output)

    def test_repr_lora_conv2d(self):
        config = LoraConfig(target_modules=["conv2d"])
        model = get_peft_model(ModelConv2D(), config)
        print_output = repr(model.model.conv2d)
        self.assertTrue(print_output.startswith("Conv2d"))
        self.assertTrue("5, 10" in print_output)
        self.assertTrue("kernel_size=(3, 3)" in print_output)
        self.assertTrue("stride=(1, 1)" in print_output)
        self.assertTrue("lora_A" in print_output)
        self.assertTrue("lora_B" in print_output)
        self.assertTrue("default" in print_output)


class MultipleActiveAdaptersTester(unittest.TestCase):
    """
    A test class to test the functionality of multiple active adapters.

    This is not specifically tied to custom models, it's just easy to test here and testing it on all types of models
    would be overkill.
    """

    def prepare_inputs_for_testing(self):
        X = torch.arange(90).view(9, 10)
        return {"X": X}

    def set_multiple_active_adapters(self, model, adapter_names):
        for module in model.modules():
            if isinstance(module, BaseTunerLayer):
                module.set_adapter(adapter_names)

    @parameterized.expand(MULTIPLE_ACTIVE_ADAPTERS_TEST_CASES)
    def test_multiple_active_adapters_forward(
        self, test_name, tuner_method, config_cls, config_kwargs_1, config_kwargs_2
    ):
        model = MLP(bias=tuner_method != "ia3")
        model.eval()
        X = self.prepare_inputs_for_testing()

        config_1 = config_cls(**config_kwargs_1)
        config_2 = config_cls(**config_kwargs_2)

        peft_model = get_peft_model(model, config_1, adapter_name="adapter_1")
        peft_model.add_adapter("adapter_2", config_2)

        # set adapter_1
        peft_model.set_adapter("adapter_1")
        adapter_1_output = peft_model(**X)

        # set adapter_2
        peft_model.set_adapter("adapter_2")
        adapter_2_output = peft_model(**X)

        # set ["adapter_1", "adapter_2"]
        self.set_multiple_active_adapters(peft_model, ["adapter_1", "adapter_2"])
        combined_output = peft_model(**X)

        self.assertFalse(torch.allclose(adapter_1_output, adapter_2_output, atol=1e-5))
        self.assertFalse(torch.allclose(adapter_1_output, combined_output, atol=1e-5))
        self.assertFalse(torch.allclose(adapter_2_output, combined_output, atol=1e-5))

        if tuner_method == "lora":
            # create a weighted adapter combining both adapters and check that
            # its output is same as setting multiple active adapters
            peft_model.add_weighted_adapter(
                ["adapter_1", "adapter_2"], [1.0, 1.0], "new_combined_adapter", combination_type="cat"
            )
            peft_model.set_adapter("new_combined_adapter")
            new_combined_output = peft_model(**X)
            self.assertTrue(torch.allclose(new_combined_output, combined_output, atol=1e-5))

    @parameterized.expand(MULTIPLE_ACTIVE_ADAPTERS_TEST_CASES)
    def test_multiple_active_adapters_merge_and_unmerge(
        self, test_name, tuner_method, config_cls, config_kwargs_1, config_kwargs_2
    ):
        model = MLP(bias=tuner_method != "ia3")
        model.eval()
        X = self.prepare_inputs_for_testing()
        base_output = model(**X)

        config_1 = config_cls(**config_kwargs_1)
        config_2 = config_cls(**config_kwargs_2)

        peft_model = get_peft_model(model, config_1, adapter_name="adapter_1")
        peft_model.add_adapter("adapter_2", config_2)

        # set ["adapter_1", "adapter_2"]
        self.set_multiple_active_adapters(peft_model, ["adapter_1", "adapter_2"])
        combined_output = peft_model(**X)

        peft_model.merge_adapter()
        merged_combined_output = peft_model(**X)
        self.assertTrue(torch.allclose(merged_combined_output, combined_output, atol=1e-5))

        peft_model.unmerge_adapter()

        with peft_model.disable_adapter():
            disabled_adapter_output = peft_model(**X)

        self.assertTrue(torch.allclose(disabled_adapter_output, base_output, atol=1e-4))


class RequiresGradTester(unittest.TestCase):
    """Test that requires_grad is set correctly in specific circumstances

    # See issue #899.

    This is not specifically tied to custom models, it's just easy to test here and testing it on all types of models
    would be overkill.

    """

    def test_requires_grad_modules_to_save_default(self):
        config = LoraConfig(target_modules=["lin0"], modules_to_save=["lin1"])
        peft_model = get_peft_model(MLP(), config)

        self.assertTrue(peft_model.model.lin1.modules_to_save.default.weight.requires_grad)
        self.assertTrue(peft_model.model.lin1.modules_to_save.default.bias.requires_grad)
        self.assertFalse(peft_model.model.lin1.original_module.weight.requires_grad)
        self.assertFalse(peft_model.model.lin1.original_module.bias.requires_grad)

    def test_requires_grad_modules_to_save_disabling(self):
        config = LoraConfig(target_modules=["lin0"], modules_to_save=["lin1"])
        peft_model = get_peft_model(MLP(), config)

        # when disabling the adapter, the original module's grad should be enabled and vice versa
        peft_model.disable_adapter_layers()
        self.assertFalse(peft_model.model.lin1.modules_to_save.default.weight.requires_grad)
        self.assertFalse(peft_model.model.lin1.modules_to_save.default.bias.requires_grad)
        self.assertTrue(peft_model.model.lin1.original_module.weight.requires_grad)
        self.assertTrue(peft_model.model.lin1.original_module.bias.requires_grad)

        # when re-enabling the adapter, the original module's grad should be disabled and vice versa
        peft_model.enable_adapter_layers()
        self.assertTrue(peft_model.model.lin1.modules_to_save.default.weight.requires_grad)
        self.assertTrue(peft_model.model.lin1.modules_to_save.default.bias.requires_grad)
        self.assertFalse(peft_model.model.lin1.original_module.weight.requires_grad)
        self.assertFalse(peft_model.model.lin1.original_module.bias.requires_grad)

        # when using the disable_adapter context, the original module's grad should be enabled and vice versa
        with peft_model.disable_adapter():
            self.assertFalse(peft_model.model.lin1.modules_to_save.default.weight.requires_grad)
            self.assertFalse(peft_model.model.lin1.modules_to_save.default.bias.requires_grad)
            self.assertTrue(peft_model.model.lin1.original_module.weight.requires_grad)
            self.assertTrue(peft_model.model.lin1.original_module.bias.requires_grad)

        # after context is exited, return to the previous state
        self.assertTrue(peft_model.model.lin1.modules_to_save.default.weight.requires_grad)
        self.assertTrue(peft_model.model.lin1.modules_to_save.default.bias.requires_grad)
        self.assertFalse(peft_model.model.lin1.original_module.weight.requires_grad)
        self.assertFalse(peft_model.model.lin1.original_module.bias.requires_grad)

    def test_requires_grad_modules_to_save_multiple_adapters(self):
        config0 = LoraConfig(target_modules=["lin0"], modules_to_save=["lin1"])
        peft_model = get_peft_model(MLP(), config0)

        config1 = LoraConfig(target_modules=["lin0"], modules_to_save=["lin1"])
        peft_model.add_adapter("adapter1", config1)

        # active adapter is still "default"
        self.assertTrue(peft_model.model.lin1.modules_to_save.default.weight.requires_grad)
        self.assertTrue(peft_model.model.lin1.modules_to_save.default.bias.requires_grad)
        self.assertFalse(peft_model.model.lin1.modules_to_save.adapter1.weight.requires_grad)
        self.assertFalse(peft_model.model.lin1.modules_to_save.adapter1.bias.requires_grad)

        # set config0 as active, should not change anything
        peft_model.set_adapter("default")
        self.assertTrue(peft_model.model.lin1.modules_to_save.default.weight.requires_grad)
        self.assertTrue(peft_model.model.lin1.modules_to_save.default.bias.requires_grad)
        self.assertFalse(peft_model.model.lin1.modules_to_save.adapter1.weight.requires_grad)
        self.assertFalse(peft_model.model.lin1.modules_to_save.adapter1.bias.requires_grad)

        # set config1 as active, should lead to adapter1 requiring grad
        peft_model.set_adapter("adapter1")
        self.assertFalse(peft_model.model.lin1.modules_to_save.default.weight.requires_grad)
        self.assertFalse(peft_model.model.lin1.modules_to_save.default.bias.requires_grad)
        self.assertTrue(peft_model.model.lin1.modules_to_save.adapter1.weight.requires_grad)
        self.assertTrue(peft_model.model.lin1.modules_to_save.adapter1.bias.requires_grad)

    def test_requires_grad_lora_different_targets(self):
        # test two different LoRA adapters that target different modules
        config0 = LoraConfig(target_modules=["lin0"])
        peft_model = get_peft_model(MLP(), config0)

        config1 = LoraConfig(target_modules=["lin1"])
        peft_model.add_adapter("adapter1", config1)

        # active adapter is still "default"
        self.assertTrue(peft_model.model.lin0.lora_A.default.weight.requires_grad)
        self.assertTrue(peft_model.model.lin0.lora_B.default.weight.requires_grad)
        self.assertFalse(peft_model.model.lin1.lora_A.adapter1.weight.requires_grad)
        self.assertFalse(peft_model.model.lin1.lora_B.adapter1.weight.requires_grad)

        # set config0 as active, should not change anything
        peft_model.set_adapter("default")
        self.assertTrue(peft_model.model.lin0.lora_A.default.weight.requires_grad)
        self.assertTrue(peft_model.model.lin0.lora_B.default.weight.requires_grad)
        self.assertFalse(peft_model.model.lin1.lora_A.adapter1.weight.requires_grad)
        self.assertFalse(peft_model.model.lin1.lora_B.adapter1.weight.requires_grad)

        # change activate adapter to adapter1
        peft_model.set_adapter("adapter1")
        self.assertFalse(peft_model.model.lin0.lora_A.default.weight.requires_grad)
        self.assertFalse(peft_model.model.lin0.lora_B.default.weight.requires_grad)
        self.assertTrue(peft_model.model.lin1.lora_A.adapter1.weight.requires_grad)
        self.assertTrue(peft_model.model.lin1.lora_B.adapter1.weight.requires_grad)

        # disable all adapters
        with peft_model.disable_adapter():
            self.assertFalse(peft_model.model.lin0.lora_A.default.weight.requires_grad)
            self.assertFalse(peft_model.model.lin0.lora_B.default.weight.requires_grad)
            self.assertFalse(peft_model.model.lin1.lora_A.adapter1.weight.requires_grad)
            self.assertFalse(peft_model.model.lin1.lora_B.adapter1.weight.requires_grad)

        # after context is exited, return to the previous state
        peft_model.set_adapter("adapter1")
        self.assertFalse(peft_model.model.lin0.lora_A.default.weight.requires_grad)
        self.assertFalse(peft_model.model.lin0.lora_B.default.weight.requires_grad)
        self.assertTrue(peft_model.model.lin1.lora_A.adapter1.weight.requires_grad)
        self.assertTrue(peft_model.model.lin1.lora_B.adapter1.weight.requires_grad)

    def test_requires_grad_lora_same_targets(self):
        # same as previous test, except that LoRA adapters target the same layer
        config0 = LoraConfig(target_modules=["lin0"])
        peft_model = get_peft_model(MLP(), config0)

        config1 = LoraConfig(target_modules=["lin0"])
        peft_model.add_adapter("adapter1", config1)

        # active adapter is still "default"
        self.assertTrue(peft_model.model.lin0.lora_A.default.weight.requires_grad)
        self.assertTrue(peft_model.model.lin0.lora_B.default.weight.requires_grad)
        self.assertFalse(peft_model.model.lin0.lora_A.adapter1.weight.requires_grad)
        self.assertFalse(peft_model.model.lin0.lora_B.adapter1.weight.requires_grad)

        # set config0 as active, should not change anything
        peft_model.set_adapter("default")
        self.assertTrue(peft_model.model.lin0.lora_A.default.weight.requires_grad)
        self.assertTrue(peft_model.model.lin0.lora_B.default.weight.requires_grad)
        self.assertFalse(peft_model.model.lin0.lora_A.adapter1.weight.requires_grad)
        self.assertFalse(peft_model.model.lin0.lora_B.adapter1.weight.requires_grad)

        # change activate adapter to adapter1
        peft_model.set_adapter("adapter1")
        self.assertFalse(peft_model.model.lin0.lora_A.default.weight.requires_grad)
        self.assertFalse(peft_model.model.lin0.lora_B.default.weight.requires_grad)
        self.assertTrue(peft_model.model.lin0.lora_A.adapter1.weight.requires_grad)
        self.assertTrue(peft_model.model.lin0.lora_B.adapter1.weight.requires_grad)

        # disable all adapters
        with peft_model.disable_adapter():
            self.assertFalse(peft_model.model.lin0.lora_A.default.weight.requires_grad)
            self.assertFalse(peft_model.model.lin0.lora_B.default.weight.requires_grad)
            self.assertFalse(peft_model.model.lin0.lora_A.adapter1.weight.requires_grad)
            self.assertFalse(peft_model.model.lin0.lora_B.adapter1.weight.requires_grad)

        # after context is exited, return to the previous state
        peft_model.set_adapter("adapter1")
        self.assertFalse(peft_model.model.lin0.lora_A.default.weight.requires_grad)
        self.assertFalse(peft_model.model.lin0.lora_B.default.weight.requires_grad)
        self.assertTrue(peft_model.model.lin0.lora_A.adapter1.weight.requires_grad)
        self.assertTrue(peft_model.model.lin0.lora_B.adapter1.weight.requires_grad)

    def test_requires_grad_ia3_different_targets(self):
        # test two different IA3 adapters that target different modules
        config0 = IA3Config(target_modules=["lin0"], feedforward_modules=["lin0"])
        peft_model = get_peft_model(MLP(), config0)

        config1 = IA3Config(target_modules=["lin1"], feedforward_modules=["lin1"])
        peft_model.add_adapter("adapter1", config1)

        # active adapter is still "default"
        self.assertTrue(peft_model.model.lin0.ia3_l.default.requires_grad)
        self.assertFalse(peft_model.model.lin1.ia3_l.adapter1.requires_grad)

        # set config0 as active, should not change anything
        peft_model.set_adapter("default")
        self.assertTrue(peft_model.model.lin0.ia3_l.default.requires_grad)
        self.assertFalse(peft_model.model.lin1.ia3_l.adapter1.requires_grad)

        # change activate adapter to adapter1
        peft_model.set_adapter("adapter1")
        self.assertFalse(peft_model.model.lin0.ia3_l.default.requires_grad)
        self.assertTrue(peft_model.model.lin1.ia3_l.adapter1.requires_grad)

        # disable all adapters
        with peft_model.disable_adapter():
            self.assertFalse(peft_model.model.lin0.ia3_l.default.requires_grad)
            self.assertFalse(peft_model.model.lin1.ia3_l.adapter1.requires_grad)

        # after context is exited, return to the previous state
        peft_model.set_adapter("adapter1")
        self.assertFalse(peft_model.model.lin0.ia3_l.default.requires_grad)
        self.assertTrue(peft_model.model.lin1.ia3_l.adapter1.requires_grad)

    def test_requires_grad_ia3_same_targets(self):
        # same as previous test, except that IA3 adapters target the same layer
        config0 = IA3Config(target_modules=["lin0"], feedforward_modules=["lin0"])
        peft_model = get_peft_model(MLP(), config0)

        config1 = IA3Config(target_modules=["lin0"], feedforward_modules=["lin0"])
        peft_model.add_adapter("adapter1", config1)

        # active adapter is still "default"
        self.assertTrue(peft_model.model.lin0.ia3_l.default.requires_grad)
        self.assertFalse(peft_model.model.lin0.ia3_l.adapter1.requires_grad)

        # set config0 as active, should not change anything
        peft_model.set_adapter("default")
        self.assertTrue(peft_model.model.lin0.ia3_l.default.requires_grad)
        self.assertFalse(peft_model.model.lin0.ia3_l.adapter1.requires_grad)

        # change activate adapter to adapter1
        peft_model.set_adapter("adapter1")
        self.assertFalse(peft_model.model.lin0.ia3_l.default.requires_grad)
        self.assertTrue(peft_model.model.lin0.ia3_l.adapter1.requires_grad)

        # disable all adapters
        with peft_model.disable_adapter():
            self.assertFalse(peft_model.model.lin0.ia3_l.default.requires_grad)
            self.assertFalse(peft_model.model.lin0.ia3_l.adapter1.requires_grad)

        # after context is exited, return to the previous state
        peft_model.set_adapter("adapter1")
        self.assertFalse(peft_model.model.lin0.ia3_l.default.requires_grad)
        self.assertTrue(peft_model.model.lin0.ia3_l.adapter1.requires_grad)

    def test_requires_grad_adalora_different_targets(self):
        # test two different AdaLora adapters that target different modules
        config0 = AdaLoraConfig(target_modules=["lin0"])
        peft_model = get_peft_model(MLP(), config0)

        config1 = AdaLoraConfig(target_modules=["lin1"], inference_mode=True)
        peft_model.add_adapter("adapter1", config1)

        # active adapter is still "default"
        self.assertTrue(peft_model.model.lin0.lora_A.default.requires_grad)
        self.assertTrue(peft_model.model.lin0.lora_B.default.requires_grad)
        self.assertTrue(peft_model.model.lin0.lora_E.default.requires_grad)
        self.assertFalse(peft_model.model.lin1.lora_A.adapter1.requires_grad)
        self.assertFalse(peft_model.model.lin1.lora_B.adapter1.requires_grad)
        self.assertFalse(peft_model.model.lin1.lora_E.adapter1.requires_grad)

        # set config0 as active, should not change anything
        peft_model.set_adapter("default")
        self.assertTrue(peft_model.model.lin0.lora_A.default.requires_grad)
        self.assertTrue(peft_model.model.lin0.lora_B.default.requires_grad)
        self.assertTrue(peft_model.model.lin0.lora_E.default.requires_grad)
        self.assertFalse(peft_model.model.lin1.lora_A.adapter1.requires_grad)
        self.assertFalse(peft_model.model.lin1.lora_B.adapter1.requires_grad)
        self.assertFalse(peft_model.model.lin1.lora_E.adapter1.requires_grad)

        # change activate adapter to adapter1
        peft_model.set_adapter("adapter1")
        self.assertFalse(peft_model.model.lin0.lora_A.default.requires_grad)
        self.assertFalse(peft_model.model.lin0.lora_B.default.requires_grad)
        self.assertFalse(peft_model.model.lin0.lora_E.default.requires_grad)
        self.assertTrue(peft_model.model.lin1.lora_A.adapter1.requires_grad)
        self.assertTrue(peft_model.model.lin1.lora_B.adapter1.requires_grad)
        self.assertTrue(peft_model.model.lin1.lora_E.adapter1.requires_grad)

        # disable all adapters
        with peft_model.disable_adapter():
            self.assertFalse(peft_model.model.lin0.lora_A.default.requires_grad)
            self.assertFalse(peft_model.model.lin0.lora_B.default.requires_grad)
            self.assertFalse(peft_model.model.lin0.lora_E.default.requires_grad)
            self.assertFalse(peft_model.model.lin1.lora_A.adapter1.requires_grad)
            self.assertFalse(peft_model.model.lin1.lora_B.adapter1.requires_grad)
            self.assertFalse(peft_model.model.lin1.lora_E.adapter1.requires_grad)

        # after context is exited, return to the previous state
        peft_model.set_adapter("adapter1")
        self.assertFalse(peft_model.model.lin0.lora_A.default.requires_grad)
        self.assertFalse(peft_model.model.lin0.lora_B.default.requires_grad)
        self.assertFalse(peft_model.model.lin0.lora_E.default.requires_grad)
        self.assertTrue(peft_model.model.lin1.lora_A.adapter1.requires_grad)
        self.assertTrue(peft_model.model.lin1.lora_B.adapter1.requires_grad)
        self.assertTrue(peft_model.model.lin1.lora_E.adapter1.requires_grad)

    def test_requires_grad_adalora_same_targets(self):
        # same as previous test, except that AdaLora adapters target the same layer
        config0 = AdaLoraConfig(target_modules=["lin0"])
        peft_model = get_peft_model(MLP(), config0)

        config1 = AdaLoraConfig(target_modules=["lin0"], inference_mode=True)
        peft_model.add_adapter("adapter1", config1)

        # active adapter is still "default"
        self.assertTrue(peft_model.model.lin0.lora_A.default.requires_grad)
        self.assertTrue(peft_model.model.lin0.lora_B.default.requires_grad)
        self.assertFalse(peft_model.model.lin0.lora_E.adapter1.requires_grad)
        self.assertFalse(peft_model.model.lin0.lora_A.adapter1.requires_grad)
        self.assertFalse(peft_model.model.lin0.lora_B.adapter1.requires_grad)
        self.assertFalse(peft_model.model.lin0.lora_E.adapter1.requires_grad)

        # set config0 as active, should not change anything
        peft_model.set_adapter("default")
        self.assertTrue(peft_model.model.lin0.lora_A.default.requires_grad)
        self.assertTrue(peft_model.model.lin0.lora_B.default.requires_grad)
        self.assertFalse(peft_model.model.lin0.lora_E.adapter1.requires_grad)
        self.assertFalse(peft_model.model.lin0.lora_A.adapter1.requires_grad)
        self.assertFalse(peft_model.model.lin0.lora_B.adapter1.requires_grad)
        self.assertFalse(peft_model.model.lin0.lora_E.adapter1.requires_grad)

        # change activate adapter to adapter1
        peft_model.set_adapter("adapter1")
        self.assertFalse(peft_model.model.lin0.lora_A.default.requires_grad)
        self.assertFalse(peft_model.model.lin0.lora_B.default.requires_grad)
        self.assertFalse(peft_model.model.lin0.lora_E.default.requires_grad)
        self.assertTrue(peft_model.model.lin0.lora_A.adapter1.requires_grad)
        self.assertTrue(peft_model.model.lin0.lora_B.adapter1.requires_grad)
        self.assertTrue(peft_model.model.lin0.lora_E.adapter1.requires_grad)

        # disable all adapters
        with peft_model.disable_adapter():
            self.assertFalse(peft_model.model.lin0.lora_A.default.requires_grad)
            self.assertFalse(peft_model.model.lin0.lora_B.default.requires_grad)
            self.assertFalse(peft_model.model.lin0.lora_E.default.requires_grad)
            self.assertFalse(peft_model.model.lin0.lora_A.adapter1.requires_grad)
            self.assertFalse(peft_model.model.lin0.lora_B.adapter1.requires_grad)
            self.assertFalse(peft_model.model.lin0.lora_E.adapter1.requires_grad)

        # after context is exited, return to the previous state
        peft_model.set_adapter("adapter1")
        self.assertFalse(peft_model.model.lin0.lora_A.default.requires_grad)
        self.assertFalse(peft_model.model.lin0.lora_B.default.requires_grad)
        self.assertFalse(peft_model.model.lin0.lora_E.default.requires_grad)
        self.assertTrue(peft_model.model.lin0.lora_A.adapter1.requires_grad)
        self.assertTrue(peft_model.model.lin0.lora_B.adapter1.requires_grad)
        self.assertTrue(peft_model.model.lin0.lora_E.adapter1.requires_grad)
