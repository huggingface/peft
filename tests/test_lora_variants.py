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
)


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


class ModelConv2DGroups(nn.Module):
    """For testing when groups argument is used in conv layer"""

    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(16, 32, 3, padding=1, groups=2)
        self.relu = nn.ReLU()
        self.flat = nn.Flatten()
        self.lin0 = nn.Linear(12800, 2)
        self.sm = nn.LogSoftmax(dim=-1)
        self.dtype = torch.float

    def forward(self, X):
        # This is ignoring input since main usage is for checking raising of error when peft is applied
        X = torch.arange(9 * 16 * 20 * 20).view([9, 16, 20, 20]).to(self.conv2d.weight.device)
        X = X.to(self.dtype)
        X = self.conv2d(X)
        X = self.relu(X)
        X = self.flat(X)
        X = self.lin0(X)
        X = self.sm(X)
        return X


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


CONV_GROUPS_TEST_CASES = [
    ("lora with rank divisible by groups", LoraConfig, {"r": 8, "target_modules": ["conv2d"]}),
    ("lora with rank equal to groups", LoraConfig, {"r": 2, "target_modules": ["conv2d"]}),
    ("lora with rank not divisible by groups", LoraConfig, {"r": 1, "target_modules": ["conv2d"]}),
    ("dora with rank divisible by groups", LoraConfig, {"r": 8, "target_modules": ["conv2d"], "use_dora": True}),
    ("dora with rank equal to groups", LoraConfig, {"r": 2, "target_modules": ["conv2d"], "use_dora": True}),
    ("dora with rank not divisible by groups", LoraConfig, {"r": 1, "target_modules": ["conv2d"], "use_dora": True}),
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


class TestConvGroups:
    @pytest.mark.parametrize("test_name, config_cls, config_kwargs", CONV_GROUPS_TEST_CASES)
    def test_error_raised_if_rank_not_divisible_by_groups(self, test_name, config_cls, config_kwargs):
        # This test checks if error is raised when rank is not divisible by groups for conv layer since
        # currently, support is limited to conv layers where the rank is divisible by groups in lora and dora
        base_model = ModelConv2DGroups()
        peft_config = config_cls(**config_kwargs)
        r = config_kwargs["r"]
        base_layer = base_model.conv2d
        groups = base_layer.groups
        if r % groups != 0:
            with pytest.raises(
                ValueError,
                match=(
                    f"Targeting a {base_layer.__class__.__name__} with groups={base_layer.groups} and rank {r}. "
                    "Currently, support is limited to conv layers where the rank is divisible by groups. "
                    "Either choose a different rank or do not target this specific layer."
                ),
            ):
                peft_model = get_peft_model(base_model, peft_config)
        else:
            peft_model = get_peft_model(base_model, peft_config)
