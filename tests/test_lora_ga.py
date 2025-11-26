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

import tempfile
import unittest
import warnings

import pytest
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import LoraGAConfig, get_peft_model
from peft.utils import LoraGAContext, estimate_gradient, save_loraga_model_final, save_loraga_model_init


class TestLoraGAConfig:
    """Test LoraGAConfig validation and defaults."""

    def test_default_config(self):
        config = LoraGAConfig(r=8, target_modules=["q_proj", "v_proj"])
        assert config.peft_type.value == "LORAGA"
        assert config.init_lora_weights == "lora_ga"
        assert config.direction == "ArB2r"
        assert config.scale == "stable"
        assert config.bsz == 2
        assert config.iters == 64
        assert config.stable_gamma == 16

    def test_custom_config(self):
        config = LoraGAConfig(
            r=16,
            target_modules=["q_proj"],
            direction="ArBr",
            scale="weight_svd",
            bsz=4,
            iters=32,
            stable_gamma=8,
        )
        assert config.direction == "ArBr"
        assert config.scale == "weight_svd"
        assert config.bsz == 4
        assert config.iters == 32
        assert config.stable_gamma == 8

    def test_invalid_direction(self):
        with pytest.raises(ValueError, match="direction must be one of"):
            LoraGAConfig(r=8, target_modules=["q_proj"], direction="invalid")

    def test_invalid_scale(self):
        with pytest.raises(ValueError, match="scale must be one of"):
            LoraGAConfig(r=8, target_modules=["q_proj"], scale="invalid")

    def test_invalid_stable_gamma(self):
        with pytest.raises(ValueError, match="stable_gamma must be positive"):
            LoraGAConfig(r=8, target_modules=["q_proj"], stable_gamma=-1)


class TestGradientEstimation:
    """Test gradient estimation functionality."""

    def get_dummy_model(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5),
        )
        return model

    def get_dummy_dataloader(self, num_samples=16, batch_size=2):
        x = torch.randn(num_samples, 10)
        y = torch.randint(0, 5, (num_samples,))
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=batch_size)

    def test_estimate_gradient_basic(self):
        model = self.get_dummy_model()

        class DummyBatch:
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def __iter__(self):
                return iter([self.x, self.y])

        class DummyDataLoader:
            def __init__(self):
                self.data = [{"x": torch.randn(2, 10), "y": torch.randint(0, 5, (2,))} for _ in range(8)]

            def __iter__(self):
                return iter(self.data)

            def __len__(self):
                return len(self.data)

        class DummyAccelerator:
            def backward(self, loss):
                loss.backward()

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)

            def forward(self, x, y):
                logits = self.linear(x)
                loss = torch.nn.functional.cross_entropy(logits, y)
                return type('obj', (object,), {'loss': loss})()

            def named_modules(self):
                yield 'linear', self.linear

            def train(self):
                pass

            def eval(self):
                pass

            def zero_grad(self):
                for p in self.linear.parameters():
                    if p.grad is not None:
                        p.grad.zero_()

            def modules(self):
                return [self.linear]

        model = DummyModel()
        dataloader = DummyDataLoader()
        accelerator = DummyAccelerator()

        named_grad = estimate_gradient(
            model, dataloader, accelerator, iters=5, quant_flag=False
        )

        assert len(named_grad) > 0
        assert all(isinstance(v, torch.Tensor) for v in named_grad.values())

    def test_gradient_shapes(self):
        model = torch.nn.Linear(10, 5)
        model.train()

        class DummyDataLoader:
            def __iter__(self):
                for _ in range(4):
                    yield {'input': torch.randn(2, 10), 'labels': torch.randint(0, 5, (2,))}

            def __len__(self):
                return 4

        class DummyAccelerator:
            def backward(self, loss):
                loss.backward()

        class DummyModelWrapper:
            def __init__(self, linear):
                self.linear = linear

            def __call__(self, input, labels):
                logits = self.linear(input)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                return type('obj', (object,), {'loss': loss})()

            def named_modules(self):
                yield '', self.linear

            def train(self):
                pass

            def eval(self):
                pass

            def zero_grad(self):
                self.linear.zero_grad()

            def modules(self):
                return [self.linear]

        wrapped_model = DummyModelWrapper(model)
        dataloader = DummyDataLoader()
        accelerator = DummyAccelerator()

        named_grad = estimate_gradient(wrapped_model, dataloader, accelerator, iters=4)

        for name, grad in named_grad.items():
            assert grad.shape == model.weight.shape


class TestLoraGAContext:
    """Test context manager for gradient attachment."""

    def test_context_attaches_gradients(self):
        model = torch.nn.Linear(10, 10)
        named_grad = {"layer": torch.randn(10, 10)}

        with LoraGAContext(model, named_grad):
            assert hasattr(model, "named_grad")
            assert model.named_grad == named_grad

        assert not hasattr(model, "named_grad")

    def test_context_cleanup_on_exception(self):
        model = torch.nn.Linear(10, 10)
        named_grad = {"layer": torch.randn(10, 10)}

        try:
            with LoraGAContext(model, named_grad):
                assert hasattr(model, "named_grad")
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert not hasattr(model, "named_grad")


class TestLoraGAInitialization:
    """Test LoRA-GA initialization methods."""

    @pytest.mark.parametrize("direction", ["ArBr", "A2rBr", "ArB2r", "random"])
    def test_all_directions(self, direction):
        # Create a model with named layers
        model = torch.nn.Sequential(torch.nn.Linear(20, 20))
        grad = torch.randn(20, 20)
        named_grad = {"0": grad}

        config = LoraGAConfig(
            r=8,
            lora_alpha=16,
            target_modules=["0"],
            direction=direction,
        )

        with LoraGAContext(model, named_grad):
            try:
                peft_model = get_peft_model(model, config)
                assert peft_model is not None
            except Exception as e:
                pytest.fail(f"Direction {direction} failed: {str(e)}")

    @pytest.mark.parametrize("scale", ["stable", "weight_svd", "gd_scale", "unit"])
    def test_all_scales(self, scale):
        # Create a model with named layers
        model = torch.nn.Sequential(torch.nn.Linear(20, 20))
        grad = torch.randn(20, 20)
        named_grad = {"0": grad}

        config = LoraGAConfig(
            r=8,
            lora_alpha=16,
            target_modules=["0"],
            scale=scale,
        )

        with LoraGAContext(model, named_grad):
            try:
                peft_model = get_peft_model(model, config)
                assert peft_model is not None
            except Exception as e:
                pytest.fail(f"Scale {scale} failed: {str(e)}")


class TestLoraGASaveLoad:
    """Test save/load with delta computation."""

    def test_save_init_creates_file(self):
        model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        config = LoraGAConfig(r=4, target_modules=["0"])

        grad = torch.randn(10, 10)
        named_grad = {"0": grad}

        with tempfile.TemporaryDirectory() as tmp_dir:
            with LoraGAContext(model, named_grad):
                peft_model = get_peft_model(model, config)

            save_loraga_model_init(peft_model, tmp_dir)

            import os
            assert os.path.exists(os.path.join(tmp_dir, "adapter_model_init.safetensors"))

    def test_save_final_without_init_warns(self):
        model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        config = LoraGAConfig(r=4, target_modules=["0"])

        grad = torch.randn(10, 10)
        named_grad = {"0": grad}

        with tempfile.TemporaryDirectory() as tmp_dir:
            with LoraGAContext(model, named_grad):
                peft_model = get_peft_model(model, config)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                save_loraga_model_final(peft_model, tmp_dir)
                assert len(w) > 0
                assert "Initial state not found" in str(w[0].message)


class TestLoraGAIntegration:
    """Integration tests."""

    def test_initialization_without_gradient_warns(self):
        model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        config = LoraGAConfig(r=4, target_modules=["0"])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            peft_model = get_peft_model(model, config)
            assert len(w) > 0

    def test_weights_are_modified(self):
        model = torch.nn.Sequential(torch.nn.Linear(20, 20))
        original_weight = model[0].weight.data.clone()

        grad = torch.randn(20, 20)
        named_grad = {"0": grad}

        config = LoraGAConfig(r=8, lora_alpha=16, target_modules=["0"])

        with LoraGAContext(model, named_grad):
            peft_model = get_peft_model(model, config)

        base_layer_weight = peft_model.base_model.model[0].weight.data

        assert not torch.allclose(original_weight, base_layer_weight)
