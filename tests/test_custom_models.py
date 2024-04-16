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
import re
import shutil
import tempfile
import time
import unittest
from contextlib import contextmanager
from functools import partial

import pytest
import torch
from parameterized import parameterized
from torch import nn
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers.pytorch_utils import Conv1D

from peft import (
    AdaLoraConfig,
    BOFTConfig,
    IA3Config,
    LoHaConfig,
    LoKrConfig,
    LoraConfig,
    OFTConfig,
    PeftModel,
    TaskType,
    get_peft_model,
)
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import ModulesToSaveWrapper, infer_device

from .testing_common import PeftCommonTester
from .testing_utils import get_state_dict, require_torch_gpu


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
    ("Vanilla MLP 7 LoRA with DoRA", "MLP", LoraConfig, {"target_modules": ["lin0"], "use_dora": True}),
    ("Vanilla MLP 8 LoRA with DoRA", "MLP", LoraConfig, {"target_modules": ["lin0", "lin1"], "use_dora": True}),
    (
        "Vanilla MLP 9 LoRA with DoRA",
        "MLP",
        LoraConfig,
        {"target_modules": "lin1", "use_dora": True, "lora_alpha": 32},
    ),
    ("Embedding + transformers Conv1D 1 LoRA", "EmbConv1D", LoraConfig, {"target_modules": ["conv1d"]}),
    ("Embedding + transformers Conv1D 2 LoRA", "EmbConv1D", LoraConfig, {"target_modules": ["emb"]}),
    ("Embedding + transformers Conv1D 3 LoRA", "EmbConv1D", LoraConfig, {"target_modules": ["emb", "conv1d"]}),
    ("Conv2d 1 LoRA", "Conv2d", LoraConfig, {"target_modules": ["conv2d"]}),
    ("Conv2d 2 LoRA", "Conv2d", LoraConfig, {"target_modules": ["conv2d", "lin0"]}),
    ("Conv2d 1 LoRA with DoRA", "Conv2d", LoraConfig, {"target_modules": ["conv2d"], "use_dora": True}),
    ("Conv2d 2 LoRA with DoRA", "Conv2d", LoraConfig, {"target_modules": ["conv2d", "lin0"], "use_dora": True}),
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
    ########
    # OFT #
    ########
    ("Vanilla MLP 1 OFT", "MLP", OFTConfig, {"target_modules": "lin0"}),
    ("Vanilla MLP 2 OFT", "MLP", OFTConfig, {"target_modules": ["lin0"]}),
    ("Vanilla MLP 5 OFT", "MLP", OFTConfig, {"target_modules": ["lin0"], "modules_to_save": ["lin1"]}),
    (
        "Vanilla MLP 6 OFT",
        "MLP",
        OFTConfig,
        {
            "target_modules": ["lin0"],
            "module_dropout": 0.1,
        },
    ),
    ("Vanilla MLP 7 OFT", "MLP", OFTConfig, {"target_modules": ["lin0"], "coft": True}),
    ("Vanilla MLP 8 OFT", "MLP", OFTConfig, {"target_modules": ["lin0"], "block_share": True}),
    ("Vanilla MLP 9 OFT", "MLP", OFTConfig, {"target_modules": ["lin0"], "coft": True, "block_share": True}),
    ("Conv2d 1 OFT", "Conv2d", OFTConfig, {"target_modules": ["conv2d"]}),
    ("Conv2d 3 OFT", "Conv2d", OFTConfig, {"target_modules": ["conv2d"], "coft": True}),
    ("Conv2d 4 OFT", "Conv2d", OFTConfig, {"target_modules": ["conv2d"], "block_share": True}),
    ("Conv2d 5 OFT", "Conv2d", OFTConfig, {"target_modules": ["conv2d"], "coft": True, "block_share": True}),
    ########
    # BOFT #
    ########
    ("Vanilla MLP 1 BOFT", "MLP", BOFTConfig, {"target_modules": ["lin1"], "boft_block_size": 2}),
    (
        "Vanilla MLP 2 BOFT",
        "MLP",
        BOFTConfig,
        {"target_modules": ["lin1"], "modules_to_save": ["lin0"], "boft_block_size": 2},
    ),
    (
        "Vanilla MLP 3 BOFT",
        "MLP",
        BOFTConfig,
        {
            "target_modules": ["lin1"],
            "boft_block_size": 2,
            "boft_dropout": 0.1,
        },
    ),
    (
        "Vanilla MLP 4 BOFT",
        "MLP",
        BOFTConfig,
        {"target_modules": ["lin1"], "boft_block_size": 2, "boft_block_num": 0, "boft_n_butterfly_factor": 1},
    ),
    (
        "Vanilla MLP 5 BOFT",
        "MLP",
        BOFTConfig,
        {"target_modules": ["lin1"], "boft_block_size": 0, "boft_block_num": 2, "boft_n_butterfly_factor": 1},
    ),
    (
        "Vanilla MLP 6 BOFT",
        "MLP",
        BOFTConfig,
        {"target_modules": ["lin1"], "boft_block_size": 10, "boft_block_num": 0, "boft_n_butterfly_factor": 2},
    ),
    (
        "Conv2d 1 BOFT",
        "Conv2d",
        BOFTConfig,
        {"target_modules": ["conv2d"], "boft_block_size": 45, "boft_block_num": 0, "boft_n_butterfly_factor": 1},
    ),
    (
        "Conv2d 2 BOFT",
        "Conv2d",
        BOFTConfig,
        {"target_modules": ["conv2d"], "boft_block_size": 0, "boft_block_num": 1, "boft_n_butterfly_factor": 1},
    ),
    (
        "MLP2 1 BOFT",
        "MLP2",
        BOFTConfig,
        {"target_modules": ["lin1"], "boft_block_size": 2, "boft_block_num": 0, "boft_n_butterfly_factor": 3},
    ),
    (
        "MLP2 2 BOFT",
        "MLP2",
        BOFTConfig,
        {"target_modules": ["lin1"], "boft_block_size": 0, "boft_block_num": 8, "boft_n_butterfly_factor": 3},
    ),
    (
        "Conv2d2 1 BOFT",
        "Conv2d2",
        BOFTConfig,
        {"target_modules": ["conv2d"], "boft_block_size": 2, "boft_block_num": 0, "boft_n_butterfly_factor": 2},
    ),
    (
        "Conv2d2 1 BOFT",
        "Conv2d2",
        BOFTConfig,
        {"target_modules": ["conv2d"], "boft_block_size": 2, "boft_block_num": 0, "boft_n_butterfly_factor": 3},
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
    OFTConfig: "oft_",
    BOFTConfig: "boft_",
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


class MLP2(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.lin0 = nn.Linear(10, 32, bias=bias)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.lin1 = nn.Linear(32, 2, bias=bias)
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
    def __init__(self, emb_size=100):
        super().__init__()
        self.emb = nn.Embedding(emb_size, 5)
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


class ModelEmbWithEmbeddingUtils(nn.Module):
    # Adds `get_input_embeddings` and `get_output_embeddings` methods to mimic 🤗 transformers models
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(100, 5)
        self.conv1d = Conv1D(1, 5)
        self.relu = nn.ReLU()
        self.flat = nn.Flatten()
        self.lin0 = nn.Linear(10, 2)
        self.sm = nn.LogSoftmax(dim=-1)

    def forward(self, X):
        X = self.embed_tokens(X)
        X = self.conv1d(X)
        X = self.relu(X)
        X = self.flat(X)
        X = self.lin0(X)
        X = self.sm(X)
        return X

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return None


class ModelConv2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(5, 10, 3)
        self.relu = nn.ReLU()
        self.flat = nn.Flatten()
        self.lin0 = nn.Linear(10, 2)
        self.sm = nn.LogSoftmax(dim=-1)

    def forward(self, X):
        X = X.float().reshape(-1, 5, 3, 3)
        X = self.conv2d(X)
        X = self.relu(X)
        X = self.flat(X)
        X = self.lin0(X)
        X = self.sm(X)
        return X


class ModelConv2D2(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin0 = nn.Linear(10, 40)
        self.conv2d = nn.Conv2d(8, 32, 3)
        self.relu = nn.ReLU()
        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(32, 2)
        self.sm = nn.LogSoftmax(dim=-1)

    def forward(self, X):
        X = X.float()
        X = self.lin0(X)
        X = self.relu(X)
        X = X.reshape(-1, 8, 3, 3)
        X = self.conv2d(X)
        X = self.relu(X)
        X = self.flat(X)
        X = self.lin1(X)
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

        if model_id == "MLP2":
            return MLP2().to(torch_dtype)

        if model_id == "Conv2d2":
            return ModelConv2D2().to(torch_dtype)

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
    def test_save_pretrained_pickle(self, test_name, model_id, config_cls, config_kwargs):
        self._test_save_pretrained(model_id, config_cls, config_kwargs, safe_serialization=False)

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
        else:
            config_kwargs["init_weights"] = False
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
    def test_merge_layers_is_idempotent(self, test_name, model_id, config_cls, config_kwargs):
        # calling merge twice with the same arguments should not change the output
        config_kwargs = config_kwargs.copy()
        if issubclass(config_cls, LoraConfig):
            config_kwargs["init_lora_weights"] = False
        elif issubclass(config_cls, IA3Config):
            config_kwargs["init_ia3_weights"] = False
        self._test_merge_layers_is_idempotent(model_id, config_cls, config_kwargs)

    @parameterized.expand(TEST_CASES)
    def test_safe_merge(self, test_name, model_id, config_cls, config_kwargs):
        # calling merge twice with the same arguments should not change the output
        config_kwargs = config_kwargs.copy()
        if issubclass(config_cls, LoraConfig):
            config_kwargs["init_lora_weights"] = False
        elif issubclass(config_cls, IA3Config):
            config_kwargs["init_ia3_weights"] = False
        else:
            config_kwargs["init_weights"] = False
        self._test_safe_merge(model_id, config_cls, config_kwargs)

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
    def test_forward_output_finite(self, test_name, model_id, config_cls, config_kwargs):
        X = self.prepare_inputs_for_testing()
        model = self.transformers_class.from_pretrained(model_id).to(self.torch_device)
        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)
        model.eval()
        with torch.no_grad():
            output = model(**X)
        assert torch.isfinite(output).all()

    @parameterized.expand(TEST_CASES)
    def test_only_params_are_updated(self, test_name, model_id, config_cls, config_kwargs):
        # An explicit test that when using an adapter on a custom model, only the adapter parameters are updated during
        # training
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
        assert params_before.keys() == params_after.keys()

        prefix = PREFIXES[config_cls]
        for name, param_before in params_before.items():
            param_after = params_after[name]
            if (prefix in name) or ("modules_to_save" in name):
                # target_modules and modules_to_save _are_ updated
                assert not torch.allclose(param_before, param_after, atol=tol, rtol=tol)
            else:
                assert torch.allclose(param_before, param_after, atol=tol, rtol=tol)

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
        lr = 0.5 if not config_kwargs.get("use_dora") else 0.1  # otherwise we get nan
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

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

            assert params_before.keys() == params_after.keys()
            for name, param_before in params_before.items():
                param_after = params_after[name]
                assert torch.allclose(param_before, param_after, atol=tol, rtol=tol)

    @parameterized.expand(TEST_CASES)
    def test_disable_adapters(self, test_name, model_id, config_cls, config_kwargs):
        X = self.prepare_inputs_for_testing()
        model = self.transformers_class.from_pretrained(model_id).to(self.torch_device).eval()

        outputs_base = model(**X)

        config = config_cls(
            base_model_name_or_path=model_id,
            **config_kwargs,
        )
        model = get_peft_model(model, config)
        model.eval()
        outputs_before = model(**X)

        assert torch.allclose(outputs_base, outputs_before)

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

        assert not torch.allclose(outputs_before, outputs_after)
        assert torch.allclose(outputs_before, outputs_disabled)
        assert torch.allclose(outputs_after, outputs_enabled_after_disable)

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
        outputs_unmerged = model(**X)
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
        assert not torch.allclose(outputs_before, outputs_after, atol=atol, rtol=rtol)

        # unmerged or merged should make no difference
        assert torch.allclose(outputs_after, outputs_unmerged, atol=atol, rtol=rtol)

        # check that disabling adapters gives the same results as before training
        assert torch.allclose(outputs_before, outputs_disabled, atol=atol, rtol=rtol)

        # check that enabling + disabling adapters does not change the results
        assert torch.allclose(outputs_after, outputs_enabled_after_disable, atol=atol, rtol=rtol)

    @parameterized.expand(TEST_CASES)
    def test_disable_adapter_with_bias_warns(self, test_name, model_id, config_cls, config_kwargs):
        # When training biases in lora, disabling adapters does not reset the biases, so the output is not what users
        # might expect. Therefore, a warning should be given.

        # Note: We test only with custom models since they run really fast. There is really no point in testing the same
        # thing with decoder, encoder_decoder, etc.
        if config_cls != LoraConfig or config_cls != BOFTConfig:
            # skip this test for other configs as bias is specific to Lora
            self.skipTest("Testing bias warnings only for LoraConfig or BOFTConfig")

        if not issubclass(config_cls, (LoraConfig, BOFTConfig)):
            self.skipTest("Bias argument is only supported for LoRA or BOFT models")

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

        if config_cls == LoraConfig:
            # check that bias=all and bias=lora_only give a warning with the correct message
            msg_start = "Careful, disabling adapter layers with bias configured to be"
            with pytest.warns(UserWarning, match=msg_start):
                run_with_disable(config_kwargs, bias="lora_only")
            with pytest.warns(UserWarning, match=msg_start):
                run_with_disable(config_kwargs, bias="all")

        if config_cls == BOFTConfig:
            # check that bias=all and bias=boft_only give a warning with the correct message
            msg_start = "Careful, disabling adapter layers with bias configured to be"
            with pytest.warns(UserWarning, match=msg_start):
                run_with_disable(config_kwargs, bias="boft_only")
            with pytest.warns(UserWarning, match=msg_start):
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
    def test_delete_adapter(self, test_name, model_id, config_cls, config_kwargs):
        self._test_delete_adapter(model_id, config_cls, config_kwargs)

    @parameterized.expand(TEST_CASES)
    def test_delete_inactive_adapter(self, test_name, model_id, config_cls, config_kwargs):
        self._test_delete_inactive_adapter(model_id, config_cls, config_kwargs)

    @parameterized.expand(TEST_CASES)
    def test_adding_multiple_adapters_with_bias_raises(self, test_name, model_id, config_cls, config_kwargs):
        self._test_adding_multiple_adapters_with_bias_raises(model_id, config_cls, config_kwargs)

    def test_weight_bias_attributes(self):
        model = MLP()
        config = LoraConfig(target_modules=["lin0"])
        model = get_peft_model(model, config)
        assert hasattr(model.base_model.model.lin0, "weight")
        assert hasattr(model.base_model.model.lin0, "bias")

    def test_multiple_adapters_automatic_modules_to_save(self):
        # See issue 1574
        # When we use certain task types, PeftModel.modules_to_save is automatically updated to include some extra
        # layers not specified in the PeftConfig. This attribute should be honored for all adapters, not just for
        # the default adapter.
        config0 = LoraConfig(task_type=TaskType.SEQ_CLS)
        config1 = LoraConfig(task_type=TaskType.SEQ_CLS)
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        model = get_peft_model(model, config0)
        # sanity check
        assert model.modules_to_save

        model.add_adapter("other", config1)
        assert "default" in model.base_model.classifier.modules_to_save
        assert "other" in model.base_model.classifier.modules_to_save

    @parameterized.expand([IA3Config, LoHaConfig, LoKrConfig, LoraConfig, OFTConfig])
    def test_multiple_adapters_mixed_modules_to_save(self, config_cls):
        # See issue 1574
        # Check that we can have a model where one adapter has modules_to_save and the other doesn't. It should be
        # possible to switch between those adapters and to use them.
        if hasattr(config_cls, "feedforward_modules"):  # IA³
            config_cls = partial(config_cls, feedforward_modules=["lin0"])

        config0 = config_cls(target_modules=["lin0"], modules_to_save=["lin1"])
        config1 = config_cls(target_modules=["lin0"])
        model = MLP()
        model = get_peft_model(model, config0).to(self.torch_device)
        model.add_adapter("other", config1)

        assert "default" in model.base_model.lin1.modules_to_save
        assert "other" not in model.base_model.lin1.modules_to_save

        # check that switching adapters and predicting does not raise
        inputs = self.prepare_inputs_for_testing()
        # "default" adapter is active
        model(**inputs)
        # switch to "other" adapter
        model.set_adapter("other")
        model(**inputs)

    @parameterized.expand([IA3Config, LoHaConfig, LoKrConfig, LoraConfig, OFTConfig])
    def test_multiple_adapters_mixed_modules_to_save_order_switched(self, config_cls):
        # See issue 1574
        # Same test as test_multiple_adapters_mixed_modules_to_save, but this time the 2nd adapter has modules_to_save.
        if hasattr(config_cls, "feedforward_modules"):  # IA³
            config_cls = partial(config_cls, feedforward_modules=["lin0"])

        config0 = config_cls(target_modules=["lin0"])
        config1 = config_cls(target_modules=["lin0"], modules_to_save=["lin1"])
        model = MLP()
        model = get_peft_model(model, config0).to(self.torch_device)
        model.add_adapter("other", config1)

        assert "default" not in model.base_model.lin1.modules_to_save
        assert "other" in model.base_model.lin1.modules_to_save

        # check that switching adapters and predicting does not raise
        inputs = self.prepare_inputs_for_testing()
        # "default" adapter is active
        model(**inputs)
        # switch to "other" adapter
        model.set_adapter("other")
        model(**inputs)

    def test_multiple_adapters_mixed_modules_to_save_merging_adapters(self):
        # See issue 1574
        # This test is similar to test_multiple_adapters_mixed_modules_to_save, but it also checks that merging adapter
        # weights works when one adapter has a modules_to_save and the other hasn't
        config0 = LoraConfig(target_modules=["lin0"], modules_to_save=["lin1"])
        config1 = LoraConfig(target_modules=["lin0"])
        model = MLP()
        model = get_peft_model(model, config0).to(self.torch_device)
        model.add_adapter("other", config1)

        # check that this does not raise
        model.add_weighted_adapter(["default", "other"], weights=[1.0, 1.0], adapter_name="merged")

        # since one of the adapters that was merged has a modules_to_save, that one should be used for the merged
        # adapter
        assert "default" in model.base_model.model.lin1.modules_to_save
        assert "other" not in model.base_model.model.lin1.modules_to_save
        assert "merged" in model.base_model.model.lin1.modules_to_save

        # check that using the merged adapter does not raise
        model.set_adapter("merged")
        inputs = self.prepare_inputs_for_testing()
        model(**inputs)

    def test_multiple_adapters_same_modules_to_save_merging_adapters_raises(self):
        # See issue 1574
        # This test is similar to test_multiple_adapters_mixed_modules_to_save_merging_adapters but here the two
        # adapters target the same module with modules_to_save. In this case, trying to merge the adapter weights
        # should raise an error.
        config0 = LoraConfig(target_modules=["lin0"], modules_to_save=["lin1"])
        config1 = LoraConfig(target_modules=["lin0"], modules_to_save=["lin1"])
        model = MLP()
        model = get_peft_model(model, config0).to(self.torch_device)
        model.add_adapter("other", config1)

        msg = re.escape(
            "Cannot add weighted adapters if they target the same module with modules_to_save, but found 1 such "
            "instance(s)."
        )
        with pytest.raises(ValueError, match=msg):
            model.add_weighted_adapter(["default", "other"], weights=[1.0, 1.0], adapter_name="merged")

    def test_multiple_adapters_seq_cls_mixed_modules_to_save_merging_adapters(self):
        # See issue 1574
        # This test is similar to test_multiple_adapters_mixed_modules_to_save_merging_adapters but uses a SEQ_CLS
        # model like in test_multiple_adapters_automatic_modules_to_save. This should raise an error because the same
        # module is implicitly targeted by modules_to_save twice.
        config0 = LoraConfig(task_type=TaskType.SEQ_CLS)
        config1 = LoraConfig(task_type=TaskType.SEQ_CLS)
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        model = get_peft_model(model, config0)
        model.add_adapter("other", config1)

        msg = re.escape(
            "Cannot add weighted adapters if they target the same module with modules_to_save, but found 1 such "
            "instance(s)."
        )
        with pytest.raises(ValueError, match=msg):
            model.add_weighted_adapter(["default", "other"], weights=[1.0, 1.0], adapter_name="merged")

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
            with open(os.path.join(tmp_dirname, "README.md")) as f:
                model_card = f.read()

        assert "library_name: peft" in model_card
        assert "meta: hello" in model_card
        assert "This is a model card" in model_card

    def test_non_existing_model_card(self):
        # ensure that if there is already a model card, it is not overwritten
        model = MLP()
        config = LoraConfig(target_modules=["lin0"])
        model = get_peft_model(model, config)

        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname)
            with open(os.path.join(tmp_dirname, "README.md")) as f:
                model_card = f.read()

        assert "library_name: peft" in model_card
        # rough check that the model card is pre-filled
        assert len(model_card) > 1000

    @parameterized.expand(["auto", True, False])
    def test_targeting_lora_to_embedding_layer(self, save_embedding_layers):
        model = ModelEmbWithEmbeddingUtils()
        config = LoraConfig(target_modules=["embed_tokens", "lin0"], init_lora_weights=False)
        model = get_peft_model(model, config)

        with tempfile.TemporaryDirectory() as tmp_dirname:
            if save_embedding_layers == "auto":
                # assert warning
                msg_start = "Setting `save_embedding_layers` to `True` as embedding layers found in `target_modules`."
                with pytest.warns(UserWarning, match=msg_start):
                    model.save_pretrained(tmp_dirname, save_embedding_layers=save_embedding_layers)
            else:
                model.save_pretrained(tmp_dirname, save_embedding_layers=save_embedding_layers)
            from safetensors.torch import load_file as safe_load_file

            state_dict = safe_load_file(os.path.join(tmp_dirname, "adapter_model.safetensors"))
            if save_embedding_layers in ["auto", True]:
                assert "base_model.model.embed_tokens.base_layer.weight" in state_dict
                assert torch.allclose(
                    model.base_model.model.embed_tokens.base_layer.weight,
                    state_dict["base_model.model.embed_tokens.base_layer.weight"],
                )
            else:
                assert "base_model.model.embed_tokens.base_layer.weight" not in state_dict
            del state_dict

    @parameterized.expand(["auto", True, False])
    def test_targeting_lora_to_embedding_layer_non_transformers(self, save_embedding_layers):
        model = ModelEmbConv1D()
        config = LoraConfig(target_modules=["emb", "lin0"], init_lora_weights=False)
        model = get_peft_model(model, config)

        with tempfile.TemporaryDirectory() as tmp_dirname:
            if save_embedding_layers is True:
                with pytest.warns(
                    UserWarning,
                    match=r"Could not identify embedding layer\(s\) because the model is not a 🤗 transformers model\.",
                ):
                    model.save_pretrained(tmp_dirname, save_embedding_layers=save_embedding_layers)
            else:
                model.save_pretrained(tmp_dirname, save_embedding_layers=save_embedding_layers)
            from safetensors.torch import load_file as safe_load_file

            state_dict = safe_load_file(os.path.join(tmp_dirname, "adapter_model.safetensors"))
            assert "base_model.model.emb.base_layer.weight" not in state_dict
            del state_dict

    def test_load_resized_embedding_ignore_mismatched_sizes(self):
        # issue #1605
        # Make it possible to load a LoRA layer that targets an embedding layer even if the sizes mismatch by passing
        # ignore_mismatched_sizes=True
        model = ModelEmbConv1D(emb_size=100)
        config = LoraConfig(target_modules=["emb", "lin0"], init_lora_weights=False)
        model = get_peft_model(model, config)

        # note: not using the context manager here because it fails on Windows CI for some reason
        tmp_dirname = tempfile.mkdtemp()
        try:
            model.save_pretrained(tmp_dirname)
            model = ModelEmbConv1D(emb_size=105)

            # first check that this raises
            with pytest.raises(RuntimeError) as exc:
                PeftModel.from_pretrained(model, tmp_dirname)
            msg = exc.value.args[0]
            assert "size mismatch" in msg and "100" in msg and "105" in msg

            # does not raise
            PeftModel.from_pretrained(model, tmp_dirname, ignore_mismatched_sizes=True)
        finally:
            try:
                shutil.rmtree(tmp_dirname)
            except PermissionError:
                # windows error
                pass

    @parameterized.expand(
        [
            LoraConfig(target_modules=["lin0"], init_lora_weights=False),
            LoKrConfig(target_modules=["lin0"], init_weights=False),
            LoHaConfig(target_modules=["lin0"], init_weights=False),
            AdaLoraConfig(target_modules=["lin0"], init_lora_weights=False),
            IA3Config(target_modules=["lin0"], feedforward_modules=["lin0"], init_ia3_weights=False),
            OFTConfig(target_modules=["lin0"], init_weights=False),
            BOFTConfig(target_modules=["lin0"], init_weights=False, boft_block_size=2),
        ]
    )
    def test_adapter_name_makes_no_difference(self, config0):
        # It should not matter whether we use the default adapter name or a custom one
        model_cls = MLP
        input = torch.arange(90).reshape(9, 10).to(self.torch_device)

        # base model
        torch.manual_seed(0)
        base_model = model_cls().eval().to(self.torch_device)
        output_base = base_model(input)

        # default name
        torch.manual_seed(0)
        base_model = model_cls().eval().to(self.torch_device)
        torch.manual_seed(0)
        peft_model_default = get_peft_model(base_model, config0, adapter_name="default").eval().to(self.torch_device)
        output_default = peft_model_default(input)
        sd_default = peft_model_default.state_dict()

        # custom name 1
        torch.manual_seed(0)
        base_model = model_cls().eval().to(self.torch_device)
        torch.manual_seed(0)
        peft_model_custom1 = get_peft_model(base_model, config0, adapter_name="adapter").eval().to(self.torch_device)
        output_custom1 = peft_model_custom1(input)
        sd_custom1 = peft_model_custom1.state_dict()

        # custom name 2
        torch.manual_seed(0)
        base_model = model_cls().eval().to(self.torch_device)
        torch.manual_seed(0)
        peft_model_custom2 = (
            get_peft_model(base_model, config0, adapter_name="other-name").eval().to(self.torch_device)
        )
        output_custom2 = peft_model_custom2(input)
        sd_custom2 = peft_model_custom2.state_dict()

        assert len(sd_default) == len(sd_custom1) == len(sd_custom2)
        for key in sd_default:
            key1 = key.replace("default", "adapter")
            key2 = key.replace("default", "other-name")
            assert key1 in sd_custom1
            assert key2 in sd_custom2
        for k0, k1, k2 in zip(sd_default, sd_custom1, sd_custom2):
            assert torch.allclose(sd_default[k0], sd_custom1[k1])
            assert torch.allclose(sd_default[k0], sd_custom2[k2])

        assert not torch.allclose(output_base, output_default)
        assert not torch.allclose(output_base, output_custom1)
        assert not torch.allclose(output_base, output_custom2)
        assert torch.allclose(output_custom1, output_custom2)
        assert torch.allclose(output_default, output_custom1)

    @parameterized.expand(["merge_and_unload", "unload"])
    def test_double_wrapping_merge_and_unload(self, method):
        # see issue #1485
        from transformers import AutoModelForTokenClassification

        model = AutoModelForTokenClassification.from_pretrained("hf-internal-testing/tiny-random-RobertaModel")
        config = LoraConfig(task_type="TOKEN_CLS", target_modules="all-linear")
        model = get_peft_model(model, config)

        # first check that double-wrapping happened
        # Note: this may get fixed in a future PR, in which case this test can be removed
        assert isinstance(model.base_model.model.classifier, ModulesToSaveWrapper)
        assert hasattr(model.base_model.model.classifier.original_module, "lora_A")
        assert hasattr(model.base_model.model.classifier.modules_to_save.default, "lora_A")

        # after unloading, despite double wrapping, the classifier module should be a normal nn.Linear layer
        if method == "merge_and_unload":
            unloaded = model.merge_and_unload()
        else:
            unloaded = model.unload()

        assert isinstance(unloaded.classifier, nn.Linear)

    def test_gpt2_dora_merge_and_unload(self):
        # see https://github.com/huggingface/peft/pull/1588#discussion_r1537914207
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        config = LoraConfig(task_type="CAUSAL_LM", use_dora=True)
        model = get_peft_model(model, config)
        # should not raise an error
        model.merge_and_unload()

    def test_gpt2_dora_merge_and_unload_safe_merge(self):
        # see https://github.com/huggingface/peft/pull/1588#discussion_r1537914207
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        config = LoraConfig(task_type="CAUSAL_LM", use_dora=True)
        model = get_peft_model(model, config)
        # should not raise an error
        model.merge_and_unload(safe_merge=True)


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

        assert rank_current == rank_expected, f"Rank {rank_current} is not equal to expected {rank_expected}"

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
                    assert (
                        rank_current == rank_expected
                    ), f"Rank {rank_current} is not equal to expected {rank_expected}"


class TestRepr(unittest.TestCase):
    """Tests related to the repr of adapted models"""

    def test_repr_lora_linear(self):
        config = LoraConfig(target_modules=["lin0"])
        model = get_peft_model(MLP(), config)
        print_output = repr(model.model.lin0)
        assert print_output.startswith("lora.Linear")
        assert "in_features=10" in print_output
        assert "out_features=20" in print_output
        assert "lora_A" in print_output
        assert "lora_B" in print_output
        assert "default" in print_output

    def test_repr_lora_embedding(self):
        config = LoraConfig(target_modules=["emb"])
        model = get_peft_model(ModelEmbConv1D(), config)
        print_output = repr(model.model.emb)
        assert print_output.startswith("lora.Embedding")
        assert "100, 5" in print_output
        assert "lora_embedding_A" in print_output
        assert "lora_embedding_B" in print_output
        assert "default" in print_output

    def test_repr_lora_conv1d(self):
        config = LoraConfig(target_modules=["conv1d"])
        model = get_peft_model(ModelEmbConv1D(), config)
        print_output = repr(model.model.conv1d)
        assert print_output.startswith("lora.Linear")
        assert "in_features=5" in print_output
        assert "out_features=1" in print_output
        assert "lora_A" in print_output
        assert "lora_B" in print_output
        assert "default" in print_output

    def test_repr_lora_conv2d(self):
        config = LoraConfig(target_modules=["conv2d"])
        model = get_peft_model(ModelConv2D(), config)
        print_output = repr(model.model.conv2d)
        assert print_output.startswith("lora.Conv2d")
        assert "5, 10" in print_output
        assert "kernel_size=(3, 3)" in print_output
        assert "stride=(1, 1)" in print_output
        assert "lora_A" in print_output
        assert "lora_B" in print_output
        assert "default" in print_output


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
        torch.manual_seed(0)
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

        assert not torch.allclose(adapter_1_output, adapter_2_output, atol=1e-5)
        assert not torch.allclose(adapter_1_output, combined_output, atol=1e-5)
        assert not torch.allclose(adapter_2_output, combined_output, atol=1e-5)

        if tuner_method == "lora":
            # create a weighted adapter combining both adapters and check that
            # its output is same as setting multiple active adapters
            peft_model.add_weighted_adapter(
                ["adapter_1", "adapter_2"], [1.0, 1.0], "new_combined_adapter", combination_type="cat"
            )
            peft_model.set_adapter("new_combined_adapter")
            new_combined_output = peft_model(**X)
            assert torch.allclose(new_combined_output, combined_output, atol=1e-5)

    @parameterized.expand(MULTIPLE_ACTIVE_ADAPTERS_TEST_CASES)
    def test_multiple_active_adapters_merge_and_unmerge(
        self, test_name, tuner_method, config_cls, config_kwargs_1, config_kwargs_2
    ):
        torch.manual_seed(0)
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
        assert torch.allclose(merged_combined_output, combined_output, atol=1e-5)

        peft_model.unmerge_adapter()

        with peft_model.disable_adapter():
            disabled_adapter_output = peft_model(**X)

        assert torch.allclose(disabled_adapter_output, base_output, atol=1e-4)

    @parameterized.expand(MULTIPLE_ACTIVE_ADAPTERS_TEST_CASES)
    def test_merge_layers_multi(self, test_name, tuner_method, config_cls, config_kwargs_1, config_kwargs_2):
        torch.manual_seed(0)
        model = MLP(bias=tuner_method != "ia3")
        model.eval()

        config_1 = config_cls(**config_kwargs_1)
        config_2 = config_cls(**config_kwargs_2)

        model = get_peft_model(model, config_1)

        dummy_input = self.prepare_inputs_for_testing()
        model.eval()

        with torch.inference_mode():
            logits_adapter_1 = model(**dummy_input)[0]

        model.add_adapter("adapter-2", config_2)
        model.set_adapter("adapter-2")
        model.eval()

        with torch.inference_mode():
            logits_adapter_2 = model(**dummy_input)[0]

        assert not torch.allclose(logits_adapter_1, logits_adapter_2, atol=1e-3, rtol=1e-3)

        model.set_adapter("default")

        with torch.inference_mode():
            logits_adapter_1_after_set = model(**dummy_input)[0]

        assert torch.allclose(logits_adapter_1_after_set, logits_adapter_1, atol=1e-3, rtol=1e-3)

        model_copy = copy.deepcopy(model)
        model_copy_2 = copy.deepcopy(model)
        model_merged_all = model.merge_and_unload(adapter_names=["adapter-2", "default"])

        with torch.inference_mode():
            logits_merged_all = model_merged_all(**dummy_input)[0]

        assert not torch.allclose(logits_merged_all, logits_adapter_2, atol=1e-3, rtol=1e-3)
        assert not torch.allclose(logits_merged_all, logits_adapter_1, atol=1e-3, rtol=1e-3)

        model_merged_adapter_2 = model_copy.merge_and_unload(adapter_names=["adapter-2"])

        with torch.inference_mode():
            logits_merged_adapter_2 = model_merged_adapter_2(**dummy_input)[0]

        assert torch.allclose(logits_merged_adapter_2, logits_adapter_2, atol=1e-3, rtol=1e-3)

        model_merged_adapter_default = model_copy_2.merge_and_unload(adapter_names=["default"])

        with torch.inference_mode():
            logits_merged_adapter_default = model_merged_adapter_default(**dummy_input)[0]

        assert torch.allclose(logits_merged_adapter_default, logits_adapter_1, atol=1e-3, rtol=1e-3)


class RequiresGradTester(unittest.TestCase):
    """Test that requires_grad is set correctly in specific circumstances

    # See issue #899.

    This is not specifically tied to custom models, it's just easy to test here and testing it on all types of models
    would be overkill.

    """

    def check_requires_grad(self, model, *params_expected: str):
        # Check that only the given parameters have requires_grad=True, and all others have requires_grad=False.
        # Calling without arguments besides the model means that all parameters should have requires_grad=False.
        params_with_requires_grad = [name for name, param in model.named_parameters() if param.requires_grad]
        diff = set(params_expected).symmetric_difference(set(params_with_requires_grad))
        msg = f"Expected {params_expected} to require gradients, got {params_with_requires_grad}"
        assert len(diff) == 0, msg

    def test_requires_grad_modules_to_save_default(self):
        config = LoraConfig(target_modules=["lin0"], modules_to_save=["lin1"])
        peft_model = get_peft_model(MLP(), config)

        self.check_requires_grad(
            peft_model,
            "base_model.model.lin1.modules_to_save.default.weight",
            "base_model.model.lin1.modules_to_save.default.bias",
            "base_model.model.lin0.lora_A.default.weight",
            "base_model.model.lin0.lora_B.default.weight",
        )

    def test_requires_grad_modules_to_save_disabling(self):
        config = LoraConfig(target_modules=["lin0"], modules_to_save=["lin1"])
        peft_model = get_peft_model(MLP(), config)

        # when disabling the adapter, the original module's grad should be enabled and vice versa
        peft_model.disable_adapter_layers()
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin1.original_module.weight",
            "base_model.model.lin1.original_module.bias",
        )

        # when re-enabling the adapter, the original module's grad should be disabled and vice versa
        peft_model.enable_adapter_layers()
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin1.modules_to_save.default.weight",
            "base_model.model.lin1.modules_to_save.default.bias",
            "base_model.model.lin0.lora_A.default.weight",
            "base_model.model.lin0.lora_B.default.weight",
        )

        # when using the disable_adapter context, the original module's grad should be enabled and vice versa
        with peft_model.disable_adapter():
            self.check_requires_grad(
                peft_model,
                "base_model.model.lin1.original_module.weight",
                "base_model.model.lin1.original_module.bias",
            )

        # after context is exited, return to the previous state
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin1.modules_to_save.default.weight",
            "base_model.model.lin1.modules_to_save.default.bias",
            "base_model.model.lin0.lora_A.default.weight",
            "base_model.model.lin0.lora_B.default.weight",
        )

    def test_requires_grad_modules_to_save_multiple_adapters(self):
        config0 = LoraConfig(target_modules=["lin0"], modules_to_save=["lin1"])
        peft_model = get_peft_model(MLP(), config0)

        config1 = LoraConfig(target_modules=["lin0"], modules_to_save=["lin1"])
        peft_model.add_adapter("adapter1", config1)

        # active adapter is still "default"
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin1.modules_to_save.default.weight",
            "base_model.model.lin1.modules_to_save.default.bias",
            "base_model.model.lin0.lora_A.default.weight",
            "base_model.model.lin0.lora_B.default.weight",
        )

        # set config0 as active, should not change anything
        peft_model.set_adapter("default")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin1.modules_to_save.default.weight",
            "base_model.model.lin1.modules_to_save.default.bias",
            "base_model.model.lin0.lora_A.default.weight",
            "base_model.model.lin0.lora_B.default.weight",
        )

        # set config1 as active, should lead to adapter1 requiring grad
        peft_model.set_adapter("adapter1")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin1.modules_to_save.adapter1.weight",
            "base_model.model.lin1.modules_to_save.adapter1.bias",
            "base_model.model.lin0.lora_A.adapter1.weight",
            "base_model.model.lin0.lora_B.adapter1.weight",
        )

    def test_requires_grad_lora_different_targets(self):
        # test two different LoRA adapters that target different modules
        config0 = LoraConfig(target_modules=["lin0"])
        peft_model = get_peft_model(MLP(), config0)

        config1 = LoraConfig(target_modules=["lin1"])
        peft_model.add_adapter("adapter1", config1)

        # active adapter is still "default"
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.lora_A.default.weight",
            "base_model.model.lin0.lora_B.default.weight",
        )

        # set config0 as active, should not change anything
        peft_model.set_adapter("default")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.lora_A.default.weight",
            "base_model.model.lin0.lora_B.default.weight",
        )

        # change activate adapter to adapter1
        peft_model.set_adapter("adapter1")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin1.lora_A.adapter1.weight",
            "base_model.model.lin1.lora_B.adapter1.weight",
        )

        # disable all adapters
        with peft_model.disable_adapter():
            self.check_requires_grad(peft_model)

        # after context is exited, return to the previous state
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin1.lora_A.adapter1.weight",
            "base_model.model.lin1.lora_B.adapter1.weight",
        )

    def test_requires_grad_lora_same_targets(self):
        # same as previous test, except that LoRA adapters target the same layer
        config0 = LoraConfig(target_modules=["lin0"])
        peft_model = get_peft_model(MLP(), config0)

        config1 = LoraConfig(target_modules=["lin0"])
        peft_model.add_adapter("adapter1", config1)

        # active adapter is still "default"
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.lora_A.default.weight",
            "base_model.model.lin0.lora_B.default.weight",
        )

        # set config0 as active, should not change anything
        peft_model.set_adapter("default")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.lora_A.default.weight",
            "base_model.model.lin0.lora_B.default.weight",
        )

        # change activate adapter to adapter1
        peft_model.set_adapter("adapter1")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.lora_A.adapter1.weight",
            "base_model.model.lin0.lora_B.adapter1.weight",
        )

        # disable all adapters
        with peft_model.disable_adapter():
            self.check_requires_grad(peft_model)

        # after context is exited, return to the previous state
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.lora_A.adapter1.weight",
            "base_model.model.lin0.lora_B.adapter1.weight",
        )

    def test_requires_grad_ia3_different_targets(self):
        # test two different IA3 adapters that target different modules
        config0 = IA3Config(target_modules=["lin0"], feedforward_modules=["lin0"])
        peft_model = get_peft_model(MLP(), config0)

        config1 = IA3Config(target_modules=["lin1"], feedforward_modules=["lin1"])
        peft_model.add_adapter("adapter1", config1)

        # active adapter is still "default"
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.ia3_l.default",
        )

        # set config0 as active, should not change anything
        peft_model.set_adapter("default")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.ia3_l.default",
        )

        # change activate adapter to adapter1
        peft_model.set_adapter("adapter1")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin1.ia3_l.adapter1",
        )

        # disable all adapters
        with peft_model.disable_adapter():
            self.check_requires_grad(peft_model)

        # after context is exited, return to the previous state
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin1.ia3_l.adapter1",
        )

    def test_requires_grad_ia3_same_targets(self):
        # same as previous test, except that IA3 adapters target the same layer
        config0 = IA3Config(target_modules=["lin0"], feedforward_modules=["lin0"])
        peft_model = get_peft_model(MLP(), config0)

        config1 = IA3Config(target_modules=["lin0"], feedforward_modules=["lin0"])
        peft_model.add_adapter("adapter1", config1)

        # active adapter is still "default"
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.ia3_l.default",
        )

        # set config0 as active, should not change anything
        peft_model.set_adapter("default")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.ia3_l.default",
        )

        # change activate adapter to adapter1
        peft_model.set_adapter("adapter1")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.ia3_l.adapter1",
        )

        # disable all adapters
        with peft_model.disable_adapter():
            self.check_requires_grad(peft_model)

        # after context is exited, return to the previous state
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.ia3_l.adapter1",
        )

    def test_requires_grad_adalora_different_targets(self):
        # test two different AdaLora adapters that target different modules
        config0 = AdaLoraConfig(target_modules=["lin0"])
        peft_model = get_peft_model(MLP(), config0)

        config1 = AdaLoraConfig(target_modules=["lin1"], inference_mode=True)
        peft_model.add_adapter("adapter1", config1)

        # active adapter is still "default"
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.lora_A.default",
            "base_model.model.lin0.lora_B.default",
            "base_model.model.lin0.lora_E.default",
        )

        # set config0 as active, should not change anything
        peft_model.set_adapter("default")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.lora_A.default",
            "base_model.model.lin0.lora_B.default",
            "base_model.model.lin0.lora_E.default",
        )

        # change activate adapter to adapter1
        peft_model.set_adapter("adapter1")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin1.lora_A.adapter1",
            "base_model.model.lin1.lora_B.adapter1",
            "base_model.model.lin1.lora_E.adapter1",
        )

        # disable all adapters
        with peft_model.disable_adapter():
            self.check_requires_grad(peft_model)

        # after context is exited, return to the previous state
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin1.lora_A.adapter1",
            "base_model.model.lin1.lora_B.adapter1",
            "base_model.model.lin1.lora_E.adapter1",
        )

    def test_requires_grad_adalora_same_targets(self):
        # same as previous test, except that AdaLora adapters target the same layer
        config0 = AdaLoraConfig(target_modules=["lin0"])
        peft_model = get_peft_model(MLP(), config0)

        config1 = AdaLoraConfig(target_modules=["lin0"], inference_mode=True)
        peft_model.add_adapter("adapter1", config1)

        # active adapter is still "default"
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.lora_A.default",
            "base_model.model.lin0.lora_B.default",
            "base_model.model.lin0.lora_E.default",
        )

        # set config0 as active, should not change anything
        peft_model.set_adapter("default")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.lora_A.default",
            "base_model.model.lin0.lora_B.default",
            "base_model.model.lin0.lora_E.default",
        )

        # change activate adapter to adapter1
        peft_model.set_adapter("adapter1")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.lora_A.adapter1",
            "base_model.model.lin0.lora_B.adapter1",
            "base_model.model.lin0.lora_E.adapter1",
        )

        # disable all adapters
        with peft_model.disable_adapter():
            self.check_requires_grad(peft_model)

        # after context is exited, return to the previous state
        peft_model.set_adapter("adapter1")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.lora_A.adapter1",
            "base_model.model.lin0.lora_B.adapter1",
            "base_model.model.lin0.lora_E.adapter1",
        )

    def test_requires_grad_lora_conv2d(self):
        # test two different LoRA adapters that target different modules
        config0 = LoraConfig(target_modules=["conv2d"])
        peft_model = get_peft_model(ModelConv2D(), config0)

        config1 = LoraConfig(target_modules=["lin0"])
        peft_model.add_adapter("adapter1", config1)

        # active adapter is still "default"
        self.check_requires_grad(
            peft_model,
            "base_model.model.conv2d.lora_A.default.weight",
            "base_model.model.conv2d.lora_B.default.weight",
        )

        # set config0 as active, should not change anything
        peft_model.set_adapter("default")
        self.check_requires_grad(
            peft_model,
            "base_model.model.conv2d.lora_A.default.weight",
            "base_model.model.conv2d.lora_B.default.weight",
        )

        # change activate adapter to adapter1
        peft_model.set_adapter("adapter1")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.lora_A.adapter1.weight",
            "base_model.model.lin0.lora_B.adapter1.weight",
        )

        # disable all adapters
        with peft_model.disable_adapter():
            self.check_requires_grad(peft_model)

        # after context is exited, return to the previous state
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.lora_A.adapter1.weight",
            "base_model.model.lin0.lora_B.adapter1.weight",
        )

    def test_requires_grad_lora_emb_conv1d(self):
        # test two different LoRA adapters that target different modules
        config0 = LoraConfig(target_modules=["conv1d"])
        peft_model = get_peft_model(ModelEmbConv1D(), config0)

        config1 = LoraConfig(target_modules=["emb"])
        peft_model.add_adapter("adapter1", config1)

        # active adapter is still "default"
        self.check_requires_grad(
            peft_model,
            "base_model.model.conv1d.lora_A.default.weight",
            "base_model.model.conv1d.lora_B.default.weight",
        )

        # set config0 as active, should not change anything
        peft_model.set_adapter("default")
        self.check_requires_grad(
            peft_model,
            "base_model.model.conv1d.lora_A.default.weight",
            "base_model.model.conv1d.lora_B.default.weight",
        )

        # change activate adapter to adapter1
        peft_model.set_adapter("adapter1")
        self.check_requires_grad(
            peft_model,
            "base_model.model.emb.lora_embedding_A.adapter1",
            "base_model.model.emb.lora_embedding_B.adapter1",
        )

        # disable all adapters
        with peft_model.disable_adapter():
            self.check_requires_grad(peft_model)

        # after context is exited, return to the previous state
        self.check_requires_grad(
            peft_model,
            "base_model.model.emb.lora_embedding_A.adapter1",
            "base_model.model.emb.lora_embedding_B.adapter1",
        )

    def test_requires_grad_ia3_conv1d(self):
        # test two different LoRA adapters that target different modules
        config0 = IA3Config(target_modules=["conv1d"], feedforward_modules=[])
        peft_model = get_peft_model(ModelEmbConv1D(), config0)

        config1 = IA3Config(target_modules=["lin0"], feedforward_modules=["lin0"])
        peft_model.add_adapter("adapter1", config1)

        # active adapter is still "default"
        self.check_requires_grad(
            peft_model,
            "base_model.model.conv1d.ia3_l.default",
        )

        # set config0 as active, should not change anything
        peft_model.set_adapter("default")
        self.check_requires_grad(
            peft_model,
            "base_model.model.conv1d.ia3_l.default",
        )

        # change activate adapter to adapter1
        peft_model.set_adapter("adapter1")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.ia3_l.adapter1",
        )

        # disable all adapters
        with peft_model.disable_adapter():
            self.check_requires_grad(peft_model)

        # after context is exited, return to the previous state
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.ia3_l.adapter1",
        )

    def test_requires_grad_ia3_conv2d(self):
        # test two different LoRA adapters that target different modules
        config0 = IA3Config(target_modules=["conv2d"], feedforward_modules=["conv2d"])
        peft_model = get_peft_model(ModelConv2D(), config0)

        config1 = IA3Config(target_modules=["lin0"], feedforward_modules=[])
        peft_model.add_adapter("adapter1", config1)

        # active adapter is still "default"
        self.check_requires_grad(
            peft_model,
            "base_model.model.conv2d.ia3_l.default",
        )

        # set config0 as active, should not change anything
        peft_model.set_adapter("default")
        self.check_requires_grad(
            peft_model,
            "base_model.model.conv2d.ia3_l.default",
        )

        # change activate adapter to adapter1
        peft_model.set_adapter("adapter1")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.ia3_l.adapter1",
        )

        # disable all adapters
        with peft_model.disable_adapter():
            self.check_requires_grad(peft_model)

        # after context is exited, return to the previous state
        peft_model.set_adapter("adapter1")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.ia3_l.adapter1",
        )

    def test_requires_grad_loha_different_targets(self):
        # test two different LoHa adapters that target different modules
        config0 = LoHaConfig(target_modules=["lin0"])
        peft_model = get_peft_model(MLP(), config0)

        config1 = LoHaConfig(target_modules=["lin1"], inference_mode=True)
        peft_model.add_adapter("adapter1", config1)

        # active pter is still "default"
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.hada_w1_a.default",
            "base_model.model.lin0.hada_w1_b.default",
            "base_model.model.lin0.hada_w2_a.default",
            "base_model.model.lin0.hada_w2_b.default",
        )

        # set config0 as active, should not change anything
        peft_model.set_adapter("default")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.hada_w1_a.default",
            "base_model.model.lin0.hada_w1_b.default",
            "base_model.model.lin0.hada_w2_a.default",
            "base_model.model.lin0.hada_w2_b.default",
        )

        # change activate pter to pter1
        peft_model.set_adapter("adapter1")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin1.hada_w1_a.adapter1",
            "base_model.model.lin1.hada_w1_b.adapter1",
            "base_model.model.lin1.hada_w2_a.adapter1",
            "base_model.model.lin1.hada_w2_b.adapter1",
        )

        # disable all pters
        with peft_model.disable_adapter():
            self.check_requires_grad(peft_model)

        # after context is exited, return to the previous state
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin1.hada_w1_a.adapter1",
            "base_model.model.lin1.hada_w1_b.adapter1",
            "base_model.model.lin1.hada_w2_a.adapter1",
            "base_model.model.lin1.hada_w2_b.adapter1",
        )

    def test_requires_grad_loha_same_targets(self):
        # same as previous test, except that LoHa adapters target the same layer
        config0 = LoHaConfig(target_modules=["lin0"])
        peft_model = get_peft_model(MLP(), config0)

        config1 = LoHaConfig(target_modules=["lin0"], inference_mode=True)
        peft_model.add_adapter("adapter1", config1)

        # active adapter is still "default"
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.hada_w1_a.default",
            "base_model.model.lin0.hada_w1_b.default",
            "base_model.model.lin0.hada_w2_a.default",
            "base_model.model.lin0.hada_w2_b.default",
        )

        # set config0 as active, should not change anything
        peft_model.set_adapter("default")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.hada_w1_a.default",
            "base_model.model.lin0.hada_w1_b.default",
            "base_model.model.lin0.hada_w2_a.default",
            "base_model.model.lin0.hada_w2_b.default",
        )

        # change activate adapter to adapter1
        peft_model.set_adapter("adapter1")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.hada_w1_a.adapter1",
            "base_model.model.lin0.hada_w1_b.adapter1",
            "base_model.model.lin0.hada_w2_a.adapter1",
            "base_model.model.lin0.hada_w2_b.adapter1",
        )

        # disable all adapters
        with peft_model.disable_adapter():
            self.check_requires_grad(peft_model)

        # after context is exited, return to the previous state
        peft_model.set_adapter("adapter1")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.hada_w1_a.adapter1",
            "base_model.model.lin0.hada_w1_b.adapter1",
            "base_model.model.lin0.hada_w2_a.adapter1",
            "base_model.model.lin0.hada_w2_b.adapter1",
        )

    def test_requires_grad_lokr_different_targets(self):
        # test two different LoKr adapters that target different modules
        config0 = LoKrConfig(target_modules=["lin0"])
        peft_model = get_peft_model(MLP(), config0)

        config1 = LoKrConfig(target_modules=["lin1"], inference_mode=True)
        peft_model.add_adapter("adapter1", config1)

        # active pter is still "default"
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.lokr_w1.default",
            "base_model.model.lin0.lokr_w2.default",
        )

        # set config0 as active, should not change anything
        peft_model.set_adapter("default")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.lokr_w1.default",
            "base_model.model.lin0.lokr_w2.default",
        )

        # change activate pter to pter1
        peft_model.set_adapter("adapter1")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin1.lokr_w1.adapter1",
            "base_model.model.lin1.lokr_w2.adapter1",
        )

        # disable all pters
        with peft_model.disable_adapter():
            self.check_requires_grad(peft_model)

        # after context is exited, return to the previous state
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin1.lokr_w1.adapter1",
            "base_model.model.lin1.lokr_w2.adapter1",
        )

    def test_requires_grad_lokr_same_targets(self):
        # same as previous test, except that LoKr adapters target the same layer
        config0 = LoKrConfig(target_modules=["lin0"])
        peft_model = get_peft_model(MLP(), config0)

        config1 = LoKrConfig(target_modules=["lin0"], inference_mode=True)
        peft_model.add_adapter("adapter1", config1)

        # active adapter is still "default"
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.lokr_w1.default",
            "base_model.model.lin0.lokr_w2.default",
        )

        # set config0 as active, should not change anything
        peft_model.set_adapter("default")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.lokr_w1.default",
            "base_model.model.lin0.lokr_w2.default",
        )

        # change activate adapter to adapter1
        peft_model.set_adapter("adapter1")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.lokr_w1.adapter1",
            "base_model.model.lin0.lokr_w2.adapter1",
        )

        # disable all adapters
        with peft_model.disable_adapter():
            self.check_requires_grad(peft_model)

        # after context is exited, return to the previous state
        peft_model.set_adapter("adapter1")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.lokr_w1.adapter1",
            "base_model.model.lin0.lokr_w2.adapter1",
        )

    def test_requires_grad_oft_different_targets(self):
        # test two different OFT adapters that target different modules
        config0 = OFTConfig(target_modules=["lin0"])
        peft_model = get_peft_model(MLP(), config0)

        config1 = OFTConfig(target_modules=["lin1"], inference_mode=True)
        peft_model.add_adapter("adapter1", config1)

        # active pter is still "default"
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.oft_r.default",
        )

        # set config0 as active, should not change anything
        peft_model.set_adapter("default")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.oft_r.default",
        )

        # change activate pter to pter1
        peft_model.set_adapter("adapter1")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin1.oft_r.adapter1",
        )

        # disable all pters
        with peft_model.disable_adapter():
            self.check_requires_grad(peft_model)

        # after context is exited, return to the previous state
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin1.oft_r.adapter1",
        )

    def test_requires_grad_oft_same_targets(self):
        # same as previous test, except that OFT adapters target the same layer
        config0 = OFTConfig(target_modules=["lin0"])
        peft_model = get_peft_model(MLP(), config0)

        config1 = OFTConfig(target_modules=["lin0"], inference_mode=True)
        peft_model.add_adapter("adapter1", config1)

        # active adapter is still "default"
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.oft_r.default",
        )

        # set config0 as active, should not change anything
        peft_model.set_adapter("default")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.oft_r.default",
        )

        # change activate adapter to adapter1
        peft_model.set_adapter("adapter1")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.oft_r.adapter1",
        )

        # disable all adapters
        with peft_model.disable_adapter():
            self.check_requires_grad(peft_model)

        # after context is exited, return to the previous state
        peft_model.set_adapter("adapter1")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.oft_r.adapter1",
        )

    def test_requires_grad_boft_different_targets(self):
        # test two different OFT adapters that target different modules
        config0 = BOFTConfig(target_modules=["lin0"], boft_block_size=2)
        peft_model = get_peft_model(MLP2(), config0)

        config1 = BOFTConfig(target_modules=["lin1"], boft_block_size=2, inference_mode=True)
        peft_model.add_adapter("adapter1", config1)

        # active pter is still "default"
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.boft_R.default",
            "base_model.model.lin0.boft_s.default",
        )

        # set config0 as active, should not change anything
        peft_model.set_adapter("default")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin0.boft_R.default",
            "base_model.model.lin0.boft_s.default",
        )

        # change activate pter to pter1
        peft_model.set_adapter("adapter1")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin1.boft_R.adapter1",
            "base_model.model.lin1.boft_s.adapter1",
        )

        # disable all pters
        with peft_model.disable_adapter():
            self.check_requires_grad(peft_model)

        # after context is exited, return to the previous state
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin1.boft_R.adapter1",
            "base_model.model.lin1.boft_s.adapter1",
        )

    def test_requires_grad_boft_same_targets(self):
        # same as previous test, except that BOFT adapters target the same layer
        config0 = BOFTConfig(target_modules=["lin1"], boft_block_size=2)
        peft_model = get_peft_model(MLP(), config0)

        config1 = BOFTConfig(target_modules=["lin1"], boft_block_size=2, inference_mode=True)
        peft_model.add_adapter("adapter1", config1)

        # active adapter is still "default"
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin1.boft_R.default",
            "base_model.model.lin1.boft_s.default",
        )

        # set config0 as active, should not change anything
        peft_model.set_adapter("default")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin1.boft_R.default",
            "base_model.model.lin1.boft_s.default",
        )

        # change activate adapter to adapter1
        peft_model.set_adapter("adapter1")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin1.boft_R.adapter1",
            "base_model.model.lin1.boft_s.adapter1",
        )

        # disable all adapters
        with peft_model.disable_adapter():
            self.check_requires_grad(peft_model)

        # after context is exited, return to the previous state
        peft_model.set_adapter("adapter1")
        self.check_requires_grad(
            peft_model,
            "base_model.model.lin1.boft_R.adapter1",
            "base_model.model.lin1.boft_s.adapter1",
        )


class TestMixedAdapterBatches:
    torch_device = infer_device()

    @pytest.fixture
    def mlp_lora(self):
        """A simple MLP with 2 LoRA adapters"""
        torch.manual_seed(0)

        base_model = MLP().to(self.torch_device).eval()
        config0 = LoraConfig(target_modules=["lin0"], init_lora_weights=False)
        config1 = LoraConfig(target_modules=["lin0"], r=16, init_lora_weights=False)
        peft_model = get_peft_model(base_model, config0, "adapter0").eval()
        peft_model.add_adapter("adapter1", config1)
        return peft_model

    def run_checks(self, model, inputs):
        # This checks that we can have mixed adapters in a single batch. The test works by creating the outputs for the
        # base model, adapter 0, and adapter 1 separately. Then, we create an output with mixed adapters, where the
        # sample [0, 3, 6] are for the base model, [1, 4, 7] for adapter 0, and [2, 5, 8] for adapter 1. Finally, we
        # check that the outputs of the mixed batch are correct for the corresponding indices.
        adapter_name0, adapter_name1 = model.peft_config.keys()

        with model.disable_adapter():
            output_base = model(**inputs)

        model.set_adapter(adapter_name0)
        output0 = model(**inputs)

        # sanity check, outputs are not the same
        assert not torch.allclose(output_base, output0)

        model.set_adapter(adapter_name1)
        output1 = model(**inputs)

        # sanity check, outputs have the right shape and are not the same
        assert len(output_base) >= 3
        assert len(output_base) == len(output0) == len(output1)
        assert not torch.allclose(output_base, output0)
        assert not torch.allclose(output_base, output1)

        # set adapter_indices so that it alternates between base, adapter 0, and adapter 1
        adapters = ["__base__", adapter_name0, adapter_name1]
        inputs["adapter_names"] = [adapters[i % 3] for i in (range(len(inputs["X"])))]
        output_mixed = model.forward(**inputs)

        assert torch.allclose(output_base[::3], output_mixed[::3])
        assert torch.allclose(output0[1::3], output_mixed[1::3])
        assert torch.allclose(output1[2::3], output_mixed[2::3])

    def test_mixed_adapter_batches_lora_mlp(self, mlp_lora):
        inputs = {"X": torch.arange(90).view(-1, 10).to(self.torch_device)}
        self.run_checks(mlp_lora, inputs)

    def test_mixed_adapter_batches_lora_different_target_layers(self, mlp_lora):
        base_model = MLP().to(self.torch_device).eval()
        # target different lora layers
        config0 = LoraConfig(target_modules=["lin0"], init_lora_weights=False)
        config1 = LoraConfig(target_modules=["lin1"], init_lora_weights=False)
        peft_model = get_peft_model(base_model, config0, "adapter0").eval()
        peft_model.add_adapter("adapter1", config1)

        inputs = {"X": torch.arange(90).view(-1, 10).to(self.torch_device)}
        self.run_checks(peft_model, inputs)

    def test_mixed_adapter_batches_lora_partly_overlapping_target_layers(self, mlp_lora):
        base_model = MLP().to(self.torch_device).eval()
        # target different lora layers
        config0 = LoraConfig(target_modules=["lin0"], init_lora_weights=False)
        config1 = LoraConfig(target_modules=["lin0", "lin1"], init_lora_weights=False)
        peft_model = get_peft_model(base_model, config0, "adapter0").eval()
        peft_model.add_adapter("adapter1", config1)

        inputs = {"X": torch.arange(90).view(-1, 10).to(self.torch_device)}
        self.run_checks(peft_model, inputs)

    def test_mixed_adapter_batches_lora_conv1d_emb(self):
        base_model = ModelEmbConv1D().to(self.torch_device).eval()
        config0 = LoraConfig(target_modules=["emb", "conv1d"], init_lora_weights=False)
        config1 = LoraConfig(target_modules=["emb", "conv1d"], r=16, init_lora_weights=False)
        peft_model = get_peft_model(base_model, config0, "adapter0").eval()
        peft_model.add_adapter("adapter1", config1)

        inputs = {"X": torch.arange(90).view(-1, 10).to(self.torch_device)}
        self.run_checks(peft_model, inputs)

    def test_mixed_adapter_batches_lora_conv2d(self):
        base_model = ModelConv2D().to(self.torch_device).eval()
        config0 = LoraConfig(target_modules=["conv2d"], init_lora_weights=False)
        config1 = LoraConfig(target_modules=["conv2d"], r=16, init_lora_weights=False)
        peft_model = get_peft_model(base_model, config0, "adapter0").eval()
        peft_model.add_adapter("adapter1", config1)

        inputs = {"X": torch.arange(270).view(6, 5, 3, 3).to(self.torch_device)}
        self.run_checks(peft_model, inputs)

    def test_mixed_adapter_batches_lora_length_mismatch_raises(self, mlp_lora):
        inputs = {
            "X": torch.arange(90).view(-1, 10).to(self.torch_device),
            "adapter_names": ["__base__"] * 5,  # wrong length!
        }
        msg = r"Length of `adapter_names` should be the same as the number of inputs, but got "
        with pytest.raises(ValueError, match=msg):
            mlp_lora.forward(**inputs)

    def test_mixed_adapter_batches_lora_training_mode_raises(self, mlp_lora):
        inputs = {
            "X": torch.arange(90).view(-1, 10).to(self.torch_device),
            "adapter_names": ["__base__"] * 9,
        }
        mlp_lora = mlp_lora.train()
        msg = r"Cannot pass `adapter_names` when the model is in training mode."
        with pytest.raises(ValueError, match=msg):
            mlp_lora.forward(**inputs)

    def test_mixed_adapter_batches_lora_disabled(self, mlp_lora):
        # Disabling adapters should have precedence over passing adapter names
        inputs = {"X": torch.arange(90).view(-1, 10).to(self.torch_device)}
        with mlp_lora.disable_adapter():
            output_disabled = mlp_lora(**inputs)

        adapters = ["__base__", "adapter0", "adapter1"]
        inputs["adapter_names"] = [adapters[i % 3] for i in (range(len(inputs["X"])))]
        with mlp_lora.disable_adapter():
            output_mixed = mlp_lora.forward(**inputs)

        assert torch.allclose(output_disabled, output_mixed)

    def test_mixed_adapter_batches_lora_merged_raises(self, mlp_lora):
        # When there are merged adapters, passing adapter names should raise an error
        inputs = {
            "X": torch.arange(90).view(-1, 10).to(self.torch_device),
            "adapter_names": ["default"] * 9,
        }
        mlp_lora.merge_adapter(["adapter0"])
        msg = r"Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
        with pytest.raises(ValueError, match=msg):
            mlp_lora.forward(**inputs)

    def test_mixed_adapter_batches_lora_with_dora_raises(self):
        # When there are Dora adapters, passing adapter names should raise an error
        torch.manual_seed(0)
        inputs = {
            "X": torch.arange(90).view(-1, 10).to(self.torch_device),
            "adapter_names": ["default"] * 9,
        }

        base_model = MLP().to(self.torch_device).eval()
        config = LoraConfig(target_modules=["lin0"], init_lora_weights=False, use_dora=True)
        peft_model = get_peft_model(base_model, config).eval()
        msg = r"Cannot pass `adapter_names` when DoRA is enabled."
        with pytest.raises(ValueError, match=msg):
            peft_model.forward(**inputs)

    @require_torch_gpu
    def test_mixed_adapter_batches_lora_opt_timing(self):
        # Use a more realistic model (opt-125m) and do a simple runtime check to ensure that mixed adapter batches
        # don't add too much overhead. These types of tests are inherently flaky, so we try to add in some robustness.
        logs = []  # store the time it takes to run each forward pass here

        @contextmanager
        def timed():
            tic = time.perf_counter()
            yield
            toc = time.perf_counter()
            logs.append(toc - tic)

        base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m").to(self.torch_device).eval()
        inputs = {"input_ids": torch.randint(0, 1000, (16, 64)).to(self.torch_device)}
        with timed():
            output_base = base_model(**inputs).logits

        config0 = LoraConfig(task_type="CAUSAL_LM", init_lora_weights=False)
        peft_model = get_peft_model(base_model, config0, "adapter1").eval()
        with timed():
            output0 = peft_model(**inputs).logits

        # sanity check, outputs are not the same
        assert not torch.allclose(output_base, output0)

        config1 = LoraConfig(task_type="CAUSAL_LM", r=16, init_lora_weights=False)
        peft_model.add_adapter("adapter2", config1)
        peft_model.set_adapter("adapter2")
        with timed():
            output1 = peft_model(**inputs).logits

        # sanity check, outputs are not the same
        assert not torch.allclose(output_base, output1)

        # set adapter_indices so that it alternates between 0 (base), lora 1, and lora 2
        adapters = ["__base__", "adapter1", "adapter2"]
        inputs["adapter_names"] = [adapters[i % 3] for i in (range(len(inputs["input_ids"])))]
        with timed():
            output_mixed = peft_model.forward(**inputs).logits

        atol, rtol = 1e-4, 1e-4
        assert torch.allclose(output_base[::3], output_mixed[::3], atol=atol, rtol=rtol)
        assert torch.allclose(output0[1::3], output_mixed[1::3], atol=atol, rtol=rtol)
        assert torch.allclose(output1[2::3], output_mixed[2::3], atol=atol, rtol=rtol)

        # Check that the overhead in time added by mixed batches is not too high.
        # To prevent flakiness, we measure mixed inference 3 times and take the lowest value, then compare it to the mean
        # of the non-mixed inference times. We also grant a generous margin of 2x the mean time.
        with timed():
            output_mixed = peft_model.forward(**inputs).logits
        with timed():
            output_mixed = peft_model.forward(**inputs).logits

        time_base, time0, time1, *time_mixed = logs
        time_non_mixed = (time_base + time0 + time1) / 3
        time_mixed = min(time_mixed)

        factor = 2.0
        assert time_mixed < factor * time_non_mixed

        # Measure timing of running base and adapter separately vs using a mixed batch. Note that on CPU, the
        # differences are quite small, so this test requires GPU to avoid flakiness.
        for _ in range(3):
            with timed():
                with peft_model.disable_adapter():
                    peft_model(**{k: v[::3] for k, v in inputs.items()})
                peft_model.set_adapter("adapter1")
                peft_model(**{k: v[1::3] for k, v in inputs.items()})
                peft_model.set_adapter("adapter2")
                peft_model(**{k: v[2::3] for k, v in inputs.items()})

        times_separate = logs[-3:]
        time_separate = sum(times_separate) / 3
        assert time_separate > time_mixed
