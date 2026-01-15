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

    def test_preprocess_basic(self, simple_model, simple_train_step):
        lora_ga_config = LoraGAConfig(direction="ArB2r", scale="stable")
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["0"],
            init_lora_weights="lora_ga",
            lora_ga_config=lora_ga_config,
        )

        # Run preprocessing
        preprocess_loraga(simple_model, lora_config, simple_train_step)

        # Check that gradients were attached
        assert hasattr(simple_model[0], "_peft_loraga_grad")
        assert simple_model[0]._peft_loraga_grad.shape == simple_model[0].weight.shape

    def test_preprocess_without_lora_ga_config_raises(self, simple_model):
        def train_step():
            pass

        lora_config = LoraConfig(r=4, lora_alpha=8, target_modules=["0"])

        with pytest.raises(ValueError, match="If you want to use LoRA-GA"):
            preprocess_loraga(simple_model, lora_config, train_step)

    def test_init_without_lora_ga_config_raises(self, simple_model, simple_train_step):
        # Properly preprocess with lora_ga_config
        lora_ga_config = LoraGAConfig(direction="ArB2r", scale="stable")
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["0"],
            init_lora_weights="lora_ga",
            lora_ga_config=lora_ga_config,
        )
        preprocess_loraga(simple_model, lora_config, simple_train_step)

        # Now try to create a config without lora_ga_config but with init_lora_weights="lora_ga"
        bad_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["0"],
            init_lora_weights="lora_ga",
            lora_ga_config=None,  # Missing lora_ga_config!
        )

        # This should raise an error during get_peft_model
        with pytest.raises(ValueError, match="lora_ga_config must be provided"):
            get_peft_model(simple_model, bad_config)


@pytest.fixture
def simple_model():
    """Fixture providing a fresh simple sequential model for each test."""
    model = torch.nn.Sequential(torch.nn.Linear(10, 10))
    model.train()
    return model


@pytest.fixture
def simple_train_step(simple_model):
    """Fixture providing a train step function for the model."""

    def train_step():
        for _ in range(4):
            inputs = torch.randn(2, 10)
            outputs = simple_model(inputs)
            loss = outputs.sum()
            loss.backward()

    return train_step


class TestLoraGAIntegration:
    """Integration tests for LoRA-GA."""

    @pytest.mark.parametrize("direction", ["ArBr", "A2rBr", "ArB2r", "random"])
    @pytest.mark.parametrize("scale", ["stable", "weight_svd", "gd_scale", "unit"])
    def test_save_load_inference(self, tmp_path, simple_model, simple_train_step, direction, scale):
        """Test that saved and loaded models produce the same output."""
        torch.manual_seed(42)

        lora_ga_config = LoraGAConfig(direction=direction, scale=scale)
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["0"],
            init_lora_weights="lora_ga",
            lora_ga_config=lora_ga_config,
        )

        preprocess_loraga(simple_model, lora_config, simple_train_step)
        peft_model = get_peft_model(simple_model, lora_config)

        # Generate output before saving
        test_input = torch.randn(2, 10)
        with torch.no_grad():
            output_before = peft_model(test_input)

        # Save model
        peft_model.save_pretrained(str(tmp_path))

        # Load model - need to use the same base model that was modified by LoRA-GA
        # Create a fresh model and load the saved state
        loaded_model = PeftModel.from_pretrained(simple_model, str(tmp_path))

        # Generate output after loading
        with torch.no_grad():
            output_after = loaded_model(test_input)

        # Outputs should be identical
        assert torch.allclose(output_before, output_after, atol=1e-5)

    @pytest.mark.parametrize("scale", ["stable", "weight_svd", "gd_scale", "unit"])
    @pytest.mark.parametrize("direction", ["ArBr", "A2rBr", "ArB2r", "random"])
    def test_save_load_with_weight_conversion(self, tmp_path, simple_model, simple_train_step, direction, scale):
        # Skip the random+weight_svd combination as it produces non-deterministic results
        if direction == "random" and scale == "weight_svd":
            pytest.skip("Skipping random+weight_svd combination due to non-deterministic behavior")
        """Test save/load with path_initial_model_for_weight_conversion."""
        torch.manual_seed(42)
        # Save RNG state for reproducing exact initialization later
        rng_state = torch.get_rng_state()

        # Save original base model weights (before LoRA-GA preprocessing)
        original_weights = {k: v.clone() for k, v in simple_model.state_dict().items()}

        lora_ga_config = LoraGAConfig(direction=direction, scale=scale)
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["0"],
            init_lora_weights="lora_ga",
            lora_ga_config=lora_ga_config,
        )

        preprocess_loraga(simple_model, lora_config, simple_train_step)
        peft_model = get_peft_model(simple_model, lora_config)

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

        # Load with original base model - need fresh model instance with same original weights
        # Restore RNG state to ensure random operations (like randperm for direction="random") are reproducible
        torch.set_rng_state(rng_state)
        base_model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        base_model.train()
        base_model.load_state_dict(original_weights)

        # Load converted adapter
        loaded_model = PeftModel.from_pretrained(base_model, str(adapter_path))

        # Generate output after loading
        with torch.no_grad():
            output_after = loaded_model(test_input)

        # Outputs should be identical
        assert torch.allclose(output_before, output_after, atol=1e-5)

    def test_cached_gradients(self, tmp_path):
        """Test that cached gradients produce the same results as fresh gradients."""
        torch.manual_seed(42)

        # First run: compute gradients and save to cache
        model1 = torch.nn.Sequential(torch.nn.Linear(10, 10))
        model1.train()

        def train_step1():
            for _ in range(4):
                inputs = torch.randn(2, 10)
                outputs = model1(inputs)
                loss = outputs.sum()
                model1.zero_grad()
                loss.backward()

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
        model2 = torch.nn.Sequential(torch.nn.Linear(10, 10))
        model2.train()

        def train_step2():
            for _ in range(4):
                inputs = torch.randn(2, 10)
                outputs = model2(inputs)
                loss = outputs.sum()
                model2.zero_grad()
                loss.backward()

        # Use same config and cache file - should load from cache without running train_step
        preprocess_loraga(model2, lora_config, train_step2, cache_file=str(cache_file))
        peft_model2 = get_peft_model(model2, lora_config)

        # Generate output from second model
        with torch.no_grad():
            output2 = peft_model2(test_input)

        # Outputs should be identical since both used the same cached gradients
        assert torch.allclose(output1, output2, atol=1e-5)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_lower_precision_dtype(self, tmp_path, dtype):
        """Test LoRA-GA works with lower precision dtypes (fp16/bf16)."""
        torch.manual_seed(42)

        # Create model in lower precision
        model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        model = model.to(dtype=dtype)
        model.train()

        def train_step():
            for _ in range(4):
                inputs = torch.randn(2, 10, dtype=dtype)
                outputs = model(inputs)
                loss = outputs.sum()
                model.zero_grad()
                loss.backward()

        lora_ga_config = LoraGAConfig(direction="ArB2r", scale="stable")
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["0"],
            init_lora_weights="lora_ga",
            lora_ga_config=lora_ga_config,
        )

        # Preprocess and create PEFT model with autocast_adapter_dtype=False
        # to ensure LoRA adapters are also in lower precision
        preprocess_loraga(model, lora_config, train_step)
        peft_model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)

        # Verify adapter dtype matches model dtype
        for name, module in peft_model.named_modules():
            if hasattr(module, "lora_A"):
                assert module.lora_A["default"].weight.dtype == dtype
                assert module.lora_B["default"].weight.dtype == dtype

        # Generate output before saving
        test_input = torch.randn(2, 10, dtype=dtype)
        with torch.no_grad():
            output_before = peft_model(test_input)

        # Save and load model
        peft_model.save_pretrained(str(tmp_path))
        loaded_model = PeftModel.from_pretrained(model, str(tmp_path))

        # Generate output after loading
        with torch.no_grad():
            output_after = loaded_model(test_input)

        # Outputs should be close - use looser tolerance for lower precision
        assert torch.allclose(output_before, output_after, atol=1e-2)

    def test_quantized_model_rejection(self):
        """Test that quantized models are properly rejected with clear error."""

        class MockQuantizedLinear(torch.nn.Linear):
            """Mock quantized layer that simulates bitsandbytes quantized layers."""

            def __init__(self, in_features, out_features):
                super().__init__(in_features, out_features)
                # Simulate quantized layer by adding quant_state attribute
                self.quant_state = "mock_quantized"

        # Create model with quantized layer
        model = torch.nn.Sequential(MockQuantizedLinear(10, 10))
        model.train()

        def train_step():
            for _ in range(4):
                inputs = torch.randn(2, 10)
                outputs = model(inputs)
                loss = outputs.sum()
                model.zero_grad()
                loss.backward()

        lora_ga_config = LoraGAConfig(direction="ArB2r", scale="stable")
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["0"],
            init_lora_weights="lora_ga",
            lora_ga_config=lora_ga_config,
        )

        # Should raise ValueError mentioning quantization
        with pytest.raises(ValueError, match="quantized"):
            preprocess_loraga(model, lora_config, train_step)

    def test_unsupported_layer_types_no_error(self):
        """Test that unsupported layer types don't cause errors."""

        class MixedModel(torch.nn.Module):
            """Model with both supported and unsupported layer types."""

            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)  # Supported
                self.conv2d = torch.nn.Conv2d(3, 16, 3)  # Unsupported
                self.embedding = torch.nn.Embedding(100, 10)  # Unsupported

            def forward(self, x):
                return self.linear(x)

        model = MixedModel()
        model.train()

        def train_step():
            for _ in range(4):
                inputs = torch.randn(2, 10)
                outputs = model(inputs)
                loss = outputs.sum()
                model.zero_grad()
                loss.backward()

        lora_ga_config = LoraGAConfig(direction="ArB2r", scale="stable")
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["linear", "conv2d", "embedding"],  # Mix of supported and unsupported
            init_lora_weights="lora_ga",
            lora_ga_config=lora_ga_config,
        )

        # Should not raise error - unsupported layers are silently skipped
        preprocess_loraga(model, lora_config, train_step)

        # Verify that linear layer has LoRA-GA gradient attached during preprocessing
        assert hasattr(model.linear, "_peft_loraga_grad")
        # Unsupported layers won't have gradients attached
        assert not hasattr(model.conv2d, "_peft_loraga_grad")
        assert not hasattr(model.embedding, "_peft_loraga_grad")

        # Now create PEFT model - should work without errors
        peft_model = get_peft_model(model, lora_config)

        # Verify model still works
        test_input = torch.randn(2, 10)
        with torch.no_grad():
            output = peft_model(test_input)
        assert output.shape == (2, 10)

    def test_no_supported_layers_raises_error(self):
        """Test that having no supported layers raises clear error."""

        class UnsupportedModel(torch.nn.Module):
            """Model with only unsupported layer types."""

            def __init__(self):
                super().__init__()
                self.conv2d = torch.nn.Conv2d(3, 16, 3)
                self.embedding = torch.nn.Embedding(100, 10)

            def forward(self, x):
                return x

        model = UnsupportedModel()
        model.train()

        def train_step():
            model.zero_grad()

        lora_ga_config = LoraGAConfig(direction="ArB2r", scale="stable")
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["conv2d", "embedding"],  # Only unsupported layers
            init_lora_weights="lora_ga",
            lora_ga_config=lora_ga_config,
        )

        # Should raise ValueError about no supported layers
        with pytest.raises(ValueError, match="No supported layers found"):
            preprocess_loraga(model, lora_config, train_step)
