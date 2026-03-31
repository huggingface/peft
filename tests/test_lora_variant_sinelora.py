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

import math

import torch
from torch import nn

from peft import LoraConfig, get_peft_model


class MLP(nn.Module):
    def __init__(self, in_features=16, out_features=16):
        super().__init__()
        self.lin0 = nn.Linear(in_features, out_features)
        self.lin1 = nn.Linear(out_features, out_features)

    def forward(self, x):
        return self.lin1(self.lin0(x))


class EmbeddingModel(nn.Module):
    def __init__(self, num_embeddings=100, embedding_dim=16):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.lin = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):
        return self.lin(self.emb(x))


class TestSineLoRA:
    def test_sinelora_output_differs_from_plain_lora(self):
        """SineLoRA output must differ from plain LoRA when weights are non-zero.

        Use init_lora_weights=False so both A and B are non-zero, making the difference between sin(A^T @ B^T) and B @
        A visible without training.
        """
        torch.manual_seed(0)
        base_model_lora = MLP()
        base_model_sinelora = MLP()
        base_model_sinelora.load_state_dict(base_model_lora.state_dict())

        lora_config = LoraConfig(target_modules=["lin0", "lin1"], r=4, lora_alpha=8, init_lora_weights=False)
        sinelora_config = LoraConfig(
            target_modules=["lin0", "lin1"], r=4, lora_alpha=8, use_sinelora=True, init_lora_weights=False
        )

        peft_lora = get_peft_model(base_model_lora, lora_config)
        peft_sinelora = get_peft_model(base_model_sinelora, sinelora_config)

        x = torch.randn(4, 16)
        with torch.no_grad():
            out_lora = peft_lora(x)
            out_sinelora = peft_sinelora(x)

        assert not torch.allclose(out_lora, out_sinelora), "SineLoRA and plain LoRA outputs should differ"

    def test_sinelora_frequency_affects_output(self):
        """Different sinelora_frequency values must produce different outputs."""
        torch.manual_seed(42)
        base_model_1 = MLP()
        base_model_2 = MLP()
        base_model_2.load_state_dict(base_model_1.state_dict())

        config_low_freq = LoraConfig(
            target_modules=["lin0", "lin1"], r=4, use_sinelora=True, sinelora_frequency=1.0, init_lora_weights=False
        )
        config_high_freq = LoraConfig(
            target_modules=["lin0", "lin1"], r=4, use_sinelora=True, sinelora_frequency=1000.0, init_lora_weights=False
        )

        peft_low = get_peft_model(base_model_1, config_low_freq)
        peft_high = get_peft_model(base_model_2, config_high_freq)

        x = torch.randn(4, 16)
        with torch.no_grad():
            out_low = peft_low(x)
            out_high = peft_high(x)

        assert not torch.allclose(out_low, out_high), "Different frequencies should produce different outputs"

    def test_sinelora_scaling_affects_output(self):
        """Different sinelora_scaling values must produce different outputs."""
        torch.manual_seed(7)
        base_model_1 = MLP()
        base_model_2 = MLP()
        base_model_2.load_state_dict(base_model_1.state_dict())

        config_small_scale = LoraConfig(
            target_modules=["lin0", "lin1"], r=4, use_sinelora=True, sinelora_scaling=1.0, init_lora_weights=False
        )
        config_large_scale = LoraConfig(
            target_modules=["lin0", "lin1"], r=4, use_sinelora=True, sinelora_scaling=100.0, init_lora_weights=False
        )

        peft_small = get_peft_model(base_model_1, config_small_scale)
        peft_large = get_peft_model(base_model_2, config_large_scale)

        x = torch.randn(4, 16)
        with torch.no_grad():
            out_small = peft_small(x)
            out_large = peft_large(x)

        assert not torch.allclose(out_small, out_large), "Different scaling values should produce different outputs"

    def test_sinelora_default_scaling_is_sqrt_in_features(self):
        """When sinelora_scaling is None, it defaults to sqrt(in_features)."""
        torch.manual_seed(0)
        model = MLP(in_features=16)
        config = LoraConfig(target_modules=["lin0"], r=4, use_sinelora=True, sinelora_scaling=None)
        peft_model = get_peft_model(model, config)

        for name, module in peft_model.named_modules():
            if hasattr(module, "sinelora_scaling") and "lin0" in name:
                assert math.isclose(module.sinelora_scaling, math.sqrt(16), rel_tol=1e-6), (
                    f"Default sinelora_scaling should be sqrt(in_features)=sqrt(16)={math.sqrt(16)}, "
                    f"got {module.sinelora_scaling}"
                )
                break

    def test_sinelora_merge_unmerge_roundtrip(self):
        """Merging then unmerging weights must restore original output (with non-zero weights)."""
        torch.manual_seed(0)
        model = MLP()
        config = LoraConfig(target_modules=["lin0", "lin1"], r=4, use_sinelora=True, init_lora_weights=False)
        peft_model = get_peft_model(model, config)

        x = torch.randn(4, 16)
        with torch.no_grad():
            out_before = peft_model(x).clone()

        peft_model.merge_adapter()
        peft_model.unmerge_adapter()

        with torch.no_grad():
            out_after = peft_model(x)

        assert torch.allclose(out_before, out_after, atol=1e-4), (
            "Output after merge+unmerge should match original output"
        )

    def test_sinelora_merge_changes_base_weights(self):
        """After merging, the base weights must have changed (with non-zero weights)."""
        torch.manual_seed(0)
        model = MLP()
        config = LoraConfig(target_modules=["lin0"], r=4, use_sinelora=True, init_lora_weights=False)
        peft_model = get_peft_model(model, config)

        orig_weight = peft_model.base_model.model.lin0.weight.detach().clone()

        peft_model.merge_adapter()

        merged_weight = peft_model.base_model.model.lin0.base_layer.weight.detach()
        assert not torch.allclose(orig_weight, merged_weight), "Merge should modify the base layer weights"

    def test_sinelora_embedding_forward(self):
        """SineLoRA on Embedding layers must run without errors."""
        torch.manual_seed(0)
        model = EmbeddingModel()
        config = LoraConfig(target_modules=["emb"], r=4, use_sinelora=True)
        peft_model = get_peft_model(model, config)

        x = torch.randint(0, 100, (4, 8))
        out = peft_model(x)
        assert out.shape == (4, 8, 16)

    def test_sinelora_embedding_merge_unmerge(self):
        """Merge/unmerge roundtrip must work for Embedding layers (with non-zero weights)."""
        torch.manual_seed(0)
        model = EmbeddingModel()
        config = LoraConfig(target_modules=["emb"], r=4, use_sinelora=True, init_lora_weights=False)
        peft_model = get_peft_model(model, config)

        x = torch.randint(0, 100, (4, 8))
        with torch.no_grad():
            out_before = peft_model(x).clone()

        peft_model.merge_adapter()
        peft_model.unmerge_adapter()

        with torch.no_grad():
            out_after = peft_model(x)

        assert torch.allclose(out_before, out_after, atol=1e-4), (
            "Embedding output after merge+unmerge should match original"
        )
