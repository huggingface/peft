# Copyright 2024-present the HuggingFace Inc. team.
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
import unittest

import pytest
import torch
from transformers import AutoModelForCausalLM

from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraGAConfig, preprocess_loraga


class TestLoraGAPreprocessing:
    """Test preprocess_loraga functionality."""

    def get_dummy_model(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5),
        )
        return model

    def test_preprocess_basic(self):
        model = self.get_dummy_model()
        model.train()

        # Define train_step callback
        def train_step():
            for _ in range(4):
                inputs = torch.randn(2, 10)
                labels = torch.randint(0, 5, (2,))
                model.zero_grad()
                outputs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                loss.backward()

        lora_ga_config = LoraGAConfig(direction="ArB2r", scale="stable")
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["0", "2"],
            init_lora_weights="lora_ga",
            lora_ga_config=lora_ga_config,
        )

        # Run preprocessing
        preprocess_loraga(model, lora_config, train_step)

        # Check that gradients were attached
        assert hasattr(model[0], "_peft_loraga_grad")
        assert hasattr(model[2], "_peft_loraga_grad")
        assert model[0]._peft_loraga_grad.shape == model[0].weight.shape
        assert model[2]._peft_loraga_grad.shape == model[2].weight.shape

    def test_preprocess_without_lora_ga_config_raises(self):
        model = self.get_dummy_model()

        def train_step():
            pass

        lora_config = LoraConfig(r=4, lora_alpha=8, target_modules=["0"])

        with pytest.raises(ValueError, match="If you want to use LoRA-GA"):
            preprocess_loraga(model, lora_config, train_step)


@pytest.fixture
def simple_model():
    """Fixture providing a simple sequential model for testing."""
    model = torch.nn.Sequential(torch.nn.Linear(10, 10))
    model.train()
    return model


@pytest.fixture
def simple_train_step(simple_model):
    """Fixture providing a train step function for the simple model."""
    def train_step():
        for _ in range(4):
            inputs = torch.randn(2, 10)
            outputs = simple_model(inputs)
            loss = outputs.sum()
            simple_model.zero_grad()
            loss.backward()
    return train_step


class TestLoraGASaveLoad:
    """Test save/load with mutated weights."""

    def test_save_pretrained_creates_file(self, tmp_path, simple_model, simple_train_step):
        model = simple_model
        train_step = simple_train_step

        lora_ga_config = LoraGAConfig()
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["0"],
            init_lora_weights="lora_ga",
            lora_ga_config=lora_ga_config,
        )

        preprocess_loraga(model, lora_config, train_step)
        peft_model = get_peft_model(model, lora_config)

        peft_model.save_pretrained(tmp_path)

        assert os.path.exists(os.path.join(tmp_path, "adapter_config.json"))
        assert os.path.exists(os.path.join(tmp_path, "adapter_model.safetensors"))


class TestLoraGAIntegration:
    """Integration tests."""

    def test_weights_are_modified(self):
        model = torch.nn.Sequential(torch.nn.Linear(20, 20))
        original_weight = model[0].weight.data.clone()
        model.train()

        def train_step():
            for _ in range(4):
                inputs = torch.randn(2, 20)
                outputs = model(inputs)
                loss = outputs.sum()
                model.zero_grad()
                loss.backward()

        lora_ga_config = LoraGAConfig()
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["0"],
            init_lora_weights="lora_ga",
            lora_ga_config=lora_ga_config,
        )

        preprocess_loraga(model, lora_config, train_step)
        peft_model = get_peft_model(model, lora_config)

        base_layer_weight = peft_model.base_model.model[0].weight.data

        # Base weights should be modified by LoRA-GA (even if slightly)
        # Use torch.equal instead of allclose to detect any modification
        assert not torch.equal(original_weight, base_layer_weight)

    def test_forward_pass_after_init(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(20, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 10),
        )
        model.train()

        def train_step():
            for _ in range(4):
                inputs = torch.randn(2, 20)
                labels = torch.randint(0, 10, (2,))
                outputs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                model.zero_grad()
                loss.backward()

        lora_ga_config = LoraGAConfig(direction="ArB2r", scale="stable")
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["0", "2"],
            init_lora_weights="lora_ga",
            lora_ga_config=lora_ga_config,
        )

        preprocess_loraga(model, lora_config, train_step)
        peft_model = get_peft_model(model, lora_config)

        # Test forward pass
        test_input = torch.randn(2, 20)
        with torch.no_grad():
            output = peft_model(test_input)
            assert output.shape == torch.Size([2, 10])

    def test_trainable_parameters(self):
        model = torch.nn.Sequential(torch.nn.Linear(20, 20))
        model.train()

        def train_step():
            for _ in range(4):
                inputs = torch.randn(2, 20)
                outputs = model(inputs)
                loss = outputs.sum()
                model.zero_grad()
                loss.backward()

        lora_ga_config = LoraGAConfig()
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["0"],
            init_lora_weights="lora_ga",
            lora_ga_config=lora_ga_config,
        )

        preprocess_loraga(model, lora_config, train_step)
        peft_model = get_peft_model(model, lora_config)

        # Check that only LoRA parameters are trainable
        trainable_params = [n for n, p in peft_model.named_parameters() if p.requires_grad]
        assert all("lora" in name.lower() for name in trainable_params)
