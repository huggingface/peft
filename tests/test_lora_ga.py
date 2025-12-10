# Copyright 2025-present the HuggingFace Inc. team.
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


import pytest
import torch

from peft import LoraConfig, PeftModel, get_peft_model
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

    def test_init_without_lora_ga_config_raises(self):
        model = self.get_dummy_model()
        model.train()

        def train_step():
            for _ in range(4):
                inputs = torch.randn(2, 10)
                labels = torch.randint(0, 5, (2,))
                model.zero_grad()
                outputs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                loss.backward()

        # Properly preprocess with lora_ga_config
        lora_ga_config = LoraGAConfig(direction="ArB2r", scale="stable")
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["0", "2"],
            init_lora_weights="lora_ga",
            lora_ga_config=lora_ga_config,
        )
        preprocess_loraga(model, lora_config, train_step)

        # Now try to create a config without lora_ga_config but with init_lora_weights="lora_ga"
        bad_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["0", "2"],
            init_lora_weights="lora_ga",
            lora_ga_config=None,  # Missing lora_ga_config!
        )

        # This should raise an error during get_peft_model
        with pytest.raises(ValueError, match="lora_ga_config must be provided"):
            get_peft_model(model, bad_config)


@pytest.fixture
def simple_model():
    """Fixture providing a fresh simple sequential model for each test."""

    def _make_model():
        model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        model.train()
        return model

    return _make_model


@pytest.fixture
def simple_train_step():
    """Fixture providing a train step function factory."""

    def _make_train_step(model):
        def train_step():
            for _ in range(4):
                inputs = torch.randn(2, 10)
                outputs = model(inputs)
                loss = outputs.sum()
                model.zero_grad()
                loss.backward()

        return train_step

    return _make_train_step


class TestLoraGAIntegration:
    """Integration tests for LoRA-GA."""

    @pytest.mark.parametrize("direction", ["ArBr", "A2rBr", "ArB2r", "random"])
    @pytest.mark.parametrize("scale", ["stable", "weight_svd", "gd_scale", "unit"])
    def test_save_load_inference(self, tmp_path, simple_model, simple_train_step, direction, scale):
        """Test that saved and loaded models produce the same output."""
        torch.manual_seed(42)

        model = simple_model()
        train_step = simple_train_step(model)

        lora_ga_config = LoraGAConfig(direction=direction, scale=scale)
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["0"],
            init_lora_weights="lora_ga",
            lora_ga_config=lora_ga_config,
        )

        preprocess_loraga(model, lora_config, train_step)
        peft_model = get_peft_model(model, lora_config)

        # Generate output before saving
        test_input = torch.randn(2, 10)
        with torch.no_grad():
            output_before = peft_model(test_input)

        # Save model
        peft_model.save_pretrained(str(tmp_path))

        # Load model - need to use the same base model that was modified by LoRA-GA
        # Create a fresh model and load the saved state
        loaded_model = PeftModel.from_pretrained(model, str(tmp_path))

        # Generate output after loading
        with torch.no_grad():
            output_after = loaded_model(test_input)

        # Outputs should be identical
        assert torch.allclose(output_before, output_after, atol=1e-5)

    @pytest.mark.parametrize("direction", ["ArBr", "A2rBr", "ArB2r", "random"])
    @pytest.mark.parametrize("scale", ["stable", "weight_svd", "gd_scale", "unit"])
    def test_save_load_with_weight_conversion(self, tmp_path, simple_model, simple_train_step, direction, scale):
        """Test save/load with path_initial_model_for_weight_conversion."""
        torch.manual_seed(42)

        model = simple_model()
        train_step = simple_train_step(model)

        # Save original base model weights (before LoRA-GA preprocessing)
        original_weights = {k: v.clone() for k, v in model.state_dict().items()}

        lora_ga_config = LoraGAConfig(direction=direction, scale=scale)
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["0"],
            init_lora_weights="lora_ga",
            lora_ga_config=lora_ga_config,
        )

        preprocess_loraga(model, lora_config, train_step)
        peft_model = get_peft_model(model, lora_config)

        # Save the initialized adapter (before training)
        init_adapter_path = tmp_path / "init_adapter"
        peft_model.peft_config["default"].init_lora_weights = True
        peft_model.save_pretrained(str(init_adapter_path))

        # Generate output before saving (simulating after training)
        test_input = torch.randn(2, 10)
        with torch.no_grad():
            output_before = peft_model(test_input)

        # Save with weight conversion
        adapter_path = tmp_path / "adapter"
        peft_model.save_pretrained(str(adapter_path), path_initial_model_for_weight_conversion=str(init_adapter_path))

        # Load with original base model
        base_model = simple_model()
        base_model.load_state_dict(original_weights)

        # Load converted adapter
        loaded_model = PeftModel.from_pretrained(base_model, str(adapter_path))

        # Generate output after loading
        with torch.no_grad():
            output_after = loaded_model(test_input)

        # Outputs should be identical
        assert torch.allclose(output_before, output_after, atol=1e-5)

    def test_cached_gradients(self, tmp_path, simple_model, simple_train_step):
        """Test that cached gradients produce the same results as fresh gradients."""
        torch.manual_seed(42)

        # First run: compute gradients and save to cache
        model1 = simple_model()
        train_step1 = simple_train_step(model1)

        lora_ga_config = LoraGAConfig(direction="ArB2r", scale="stable")
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["0"],
            init_lora_weights="lora_ga",
            lora_ga_config=lora_ga_config,
        )

        cache_file = tmp_path / "gradient_cache.pt"
        preprocess_loraga(model1, lora_config, train_step1, cache_file=str(cache_file))
        peft_model1 = get_peft_model(model1, lora_config)

        # Check that cache file was created
        assert cache_file.exists()
        assert cache_file.stat().st_size > 0

        # Generate output from first model
        test_input = torch.randn(2, 10)
        with torch.no_grad():
            output1 = peft_model1(test_input)

        # Second run: load gradients from cache
        torch.manual_seed(42)  # Reset seed to get same initial weights
        model2 = simple_model()
        train_step2 = simple_train_step(model2)

        # Use same config and cache file - should load from cache without running train_step
        preprocess_loraga(model2, lora_config, train_step2, cache_file=str(cache_file))
        peft_model2 = get_peft_model(model2, lora_config)

        # Generate output from second model
        with torch.no_grad():
            output2 = peft_model2(test_input)

        # Outputs should be identical since both used the same cached gradients
        assert torch.allclose(output1, output2, atol=1e-5)
