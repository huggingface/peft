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
from torch import nn

from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import Conv1d as LoraConv1d
from peft.tuners.lora.layer import Conv2d as LoraConv2d
from peft.tuners.lora.layer import Embedding as LoraEmbedding
from peft.tuners.lora.layer import Linear as LoraLinear
from peft.tuners.lora.variants import (
    DoraConv1dVariant,
    DoraConv2dVariant,
    DoraEmbeddingVariant,
    DoraLinearVariant,
    calculate_alora_offsets,
    get_alora_offsets_for_forward,
)


# Custom model featuring embeddings and a 'visual stack'
class CustomModel(nn.Module):
    """pytorch module that contains common targetable layers (linear, embedding, conv, ...)"""

    def __init__(self, num_embeddings=100, embedding_dim=16, num_classes=10):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=32, kernel_size=3, padding=1)
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.dummy_conv1d_output_dim = 32 * 10
        self.dummy_conv2d_output_dim = 16 * 10 * 10
        self.linear1 = nn.Linear(self.dummy_conv1d_output_dim + self.dummy_conv2d_output_dim, 64)
        self.linear2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_ids, dummy_image_input):
        # Path 1: Embedding -> Conv1d
        x1 = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        x1 = x1.transpose(1, 2)  # (batch_size, embedding_dim, seq_len)
        x1 = self.relu(self.conv1d(x1))  # (batch_size, 32, seq_len)
        x1_flat = self.flatten(x1)
        # Path 2: Conv2d -> Linear
        x2 = self.relu(self.conv2d(dummy_image_input))  # (batch_size, 16, H, W)
        x2_flat = self.flatten(x2)  # (batch_size, 16*H*W)
        # Combine or select paths if making a functional model.
        # For this test, we mainly care about layer types, so forward might not be fully executed.
        # Let's use x2_flat for subsequent linear layers.
        output = self.relu(self.linear1(torch.concat([x1_flat, x2_flat], dim=1)))
        output = self.linear2(output)
        return output


# Used for testing alora_offsets for aLoRA
class DummyLM(nn.Module):
    def __init__(self, vocab_size: int = 10, hidden_dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, X=None, embeds=None):
        if X is not None:
            embeds = self.embed(X)
        return self.linear(embeds)


class MockTransformerWrapper:
    """Mock class to behave like a transformers model.

    This is needed because the tests initialize the model by calling transformers_class.from_pretrained.

    """

    @classmethod
    def from_pretrained(cls):
        # set the seed so that from_pretrained always returns the same model
        torch.manual_seed(0)

        torch_dtype = torch.float32

        return DummyLM().to(torch_dtype)


VARIANT_MAP = {
    "dora": {
        LoraLinear: DoraLinearVariant,
        LoraEmbedding: DoraEmbeddingVariant,
        LoraConv1d: DoraConv1dVariant,
        LoraConv2d: DoraConv2dVariant,
    }
}


TEST_CASES = [
    (
        "dora",
        LoraConfig,
        {"target_modules": ["linear1", "linear2", "conv1d", "conv2d", "embedding"], "use_dora": True},
    ),
]


class TestLoraVariants:
    @pytest.mark.parametrize("variant_name, config_cls, config_kwargs", TEST_CASES)
    def test_variant_is_applied_to_layers(self, variant_name, config_cls, config_kwargs):
        # This test assumes that targeting and replacing layers works and that after `get_peft_model` we
        # have a model with LoRA layers. We just make sure that each LoRA layer has its variant set and
        # it is also the correct variant for that layer.
        base_model = CustomModel()
        peft_config = config_cls(**config_kwargs)
        peft_model = get_peft_model(base_model, peft_config)

        layer_type_map = VARIANT_MAP[variant_name]

        for _, module in peft_model.named_modules():
            if not hasattr(module, "lora_variant"):
                continue

            # Note that not every variant supports every layer. If it is not mapped it is deemed unsupported and
            # will not be tested.
            expected_variant_type = layer_type_map.get(type(module), None)
            if not expected_variant_type:
                continue

            assert isinstance(module.lora_variant["default"], expected_variant_type)

    def custom_model_with_loss_backpropagated(self, peft_config):
        """Returns the CustomModel + PEFT model instance with a dummy loss that was backpropagated once."""
        base_model = CustomModel()
        peft_model = get_peft_model(base_model, peft_config)

        x, y = torch.ones(10, 10).long(), torch.ones(10, 1, 10, 10)
        out = peft_model(x, y)
        loss = out.sum()
        loss.backward()

        return base_model, peft_model

    def test_dora_params_have_gradients(self):
        """Ensure that the parameters added by the DoRA variant are participating in the output computation."""
        layer_names = ["linear1", "linear2", "conv1d", "conv2d", "embedding"]
        peft_config = LoraConfig(target_modules=layer_names, use_dora=True)
        base_model, peft_model = self.custom_model_with_loss_backpropagated(peft_config)

        for layer in layer_names:
            assert getattr(peft_model.base_model.model, layer).lora_magnitude_vector["default"].weight.grad is not None


# Make sure warning is sent when invocation sequence is not present
def test_calculate_alora_offsets_basic_and_warning():
    config = LoraConfig(alora_invocation_tokens=[1, 2])
    peft_config = {"default": config}
    input_ids = torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6]])

    # second row lacks invocation sequence -> warning and None offset
    with pytest.warns(UserWarning):
        offsets = calculate_alora_offsets(peft_config, "default", input_ids)

    assert offsets[0] == 4
    assert offsets[1] is None


# Verify alora_offsets are correct with multiple adapters
def test_calculate_alora_offsets_with_adapter_names():
    cfg1 = LoraConfig(alora_invocation_tokens=[1])
    cfg2 = LoraConfig(alora_invocation_tokens=[2])
    peft_config = {"a1": cfg1, "a2": cfg2}
    input_ids = torch.tensor([[0, 1, 1], [0, 2, 2]])

    offsets = calculate_alora_offsets(peft_config, "a1", input_ids, adapter_names=["a1", "a2"])

    assert offsets == [2, 2]


# Verify that the adapter does not modify outputs prior to invocation point
def test_alora_activation_matches_base_until_invocation():
    transformers_class = MockTransformerWrapper
    base_model = transformers_class.from_pretrained()
    cfg = LoraConfig(target_modules=["linear"], alora_invocation_tokens=[2], init_lora_weights=False)
    lora_model = get_peft_model(base_model, cfg)
    lora_model.eval()

    input_ids = torch.tensor([[0, 1, 2, 3]])
    start = 2 #index of invocation token
    with lora_model.disable_adapter():
        with torch.no_grad():
            base_out = lora_model(X=input_ids)

    kwargs = get_alora_offsets_for_forward(lora_model, input_ids)
    with torch.no_grad():
        lora_out = lora_model(X=input_ids, **kwargs)
    assert torch.allclose(lora_out[:, :start], base_out[:, :start])
    assert not torch.allclose(lora_out[:, start:], base_out[:, start:])

# Verify that warning is given for alora when providing embeddings only
def test_input_embeds_warning():
    transformers_class = MockTransformerWrapper
    base_model = transformers_class.from_pretrained()
    cfg = LoraConfig(target_modules=["linear"], alora_invocation_tokens=[2], init_lora_weights=False)
    lora_model = get_peft_model(base_model, cfg)
    lora_model.eval()

    input_ids = torch.tensor([[0, 1, 2, 3]])
    input_embeds = base_model.embed(input_ids)
    with pytest.warns(UserWarning):
        with torch.no_grad():
            lora_out = lora_model(embeds=input_embeds)

# Verify that error is raised when requesting num_beams > 1 for alora
def test_num_beams_error():
    transformers_class = MockTransformerWrapper
    base_model = transformers_class.from_pretrained()
    cfg = LoraConfig(target_modules=["linear"], alora_invocation_tokens=[2], init_lora_weights=False)
    lora_model = get_peft_model(base_model, cfg)
    lora_model.eval()

    input_ids = torch.tensor([[0, 1, 2, 3]])
    with pytest.pytest.raises(ValueError):
        with torch.no_grad():
            lora_out = lora_model(X=input_ids,num_beams=2)
