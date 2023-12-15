import unittest

import torch
import torch.nn as nn

from peft.tuners.lora import Conv2d as LoraConv2d
from peft.tuners.lora import Embedding as LoraEmbedding
from peft.tuners.lora import Linear as LoraLinear


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        self.embedding_layer = nn.Embedding(1000, 768)
        self.layer_norm = nn.LayerNorm(768)
        self.linear_transform = nn.Linear(768, 256)

    def forward(self, input_ids):
        embedded_output = self.embedding_layer(input_ids)
        norm_output = self.layer_norm(embedded_output)
        linear_output = self.linear_transform(norm_output)

        return linear_output


class SimpleConv2DModel(nn.Module):
    def __init__(self):
        super(SimpleConv2DModel, self).__init__()

        self.embedding_layer = nn.Embedding(1000, 768)
        self.layer_norm = nn.LayerNorm(768)
        self.conv2d_transform = nn.Conv2d(1, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, input_ids):
        # Additional layers for your custom model
        embedded_output = self.embedding_layer(input_ids)
        norm_output = self.layer_norm(embedded_output)

        # Reshape for Conv2d input (add batch size dimension)
        norm_output = norm_output.unsqueeze(1)
        conv_output = self.conv2d_transform(norm_output)

        # Remove batch size dimension
        conv_output = conv_output.squeeze(1)

        return conv_output


class SimpleLorALinearModel(nn.Module):
    """Same as SimpleModel but wraps Linear in Lora layer"""

    def __init__(self):
        super(SimpleLorALinearModel, self).__init__()

        self.embedding_layer = nn.Embedding(1000, 768)
        self.layer_norm = nn.LayerNorm(768)
        self.linear_transform_base = nn.Linear(768, 256)
        self.linear_transform = LoraLinear(
            self.linear_transform_base, adapter_name="test_linear", r=8, lora_alpha=16, lora_dropout=0.05
        )

    def forward(self, input_ids):
        embedded_output = self.embedding_layer(input_ids)
        norm_output = self.layer_norm(embedded_output)
        linear_output = self.linear_transform(norm_output)

        return linear_output


class SimpleLorAEmbeddingModel(nn.Module):
    """Same as SimpleModel but wraps Embedding in Lora layer"""

    def __init__(self):
        super(SimpleLorAEmbeddingModel, self).__init__()

        self.embedding_layer_base = nn.Embedding(1000, 768)
        self.embedding_layer = LoraEmbedding(
            self.embedding_layer_base, adapter_name="test_embedding", r=8, lora_alpha=16, lora_dropout=0.05
        )
        self.layer_norm = nn.LayerNorm(768)
        self.linear_transform = nn.Linear(768, 256)

    def forward(self, input_ids):
        embedded_output = self.embedding_layer(input_ids)
        norm_output = self.layer_norm(embedded_output)
        linear_output = self.linear_transform(norm_output)

        return linear_output


class SimpleLorAConv2DModel(nn.Module):
    """Same as SimpleModel but wraps Conv2D in Lora layer"""

    def __init__(self):
        super(SimpleLorAConv2DModel, self).__init__()

        self.embedding_layer = nn.Embedding(1000, 768)
        self.layer_norm = nn.LayerNorm(768)
        self.conv2d_transform_base = nn.Conv2d(1, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2d_transform = LoraConv2d(
            self.conv2d_transform_base, adapter_name="test_conv2d", r=8, lora_alpha=16, lora_dropout=0.05
        )

    def forward(self, input_ids):
        # Additional layers for your custom model
        embedded_output = self.embedding_layer(input_ids)
        norm_output = self.layer_norm(embedded_output)

        # Reshape for Conv2d input (add batch size dimension)
        norm_output = norm_output.unsqueeze(1)
        conv_output = self.conv2d_transform(norm_output)

        # Remove batch size dimension
        conv_output = conv_output.squeeze(1)

        return conv_output


class TestAutoCast(unittest.TestCase):
    def test_simple_model(self):
        self._test_model(SimpleModel)

    def test_simple_conv2d_model(self):
        self._test_model(SimpleConv2DModel)

    def test_simple_lora_linear_model(self):
        self._test_model(SimpleLorALinearModel)

    def test_simple_lora_embedding_model(self):
        self._test_model(SimpleLorAEmbeddingModel)

    def test_simple_lora_conv2d_model(self):
        self._test_model(SimpleLorAConv2DModel)

    def _test_model(self, model_class):
        # Instantiate the model
        model = model_class().cuda()

        # Prepare dummy inputs
        input_ids = torch.randint(0, 1000, (2, 10)).cuda()

        # Forward pass with torch.bfloat16
        with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
            outputs = model(input_ids)
            self.assertEqual(outputs.dtype, torch.bfloat16)

        # Forward pass with torch.float32
        with torch.autocast(enabled=True, dtype=torch.float32, device_type="cuda"):
            outputs = model(input_ids)
            self.assertEqual(outputs.dtype, torch.float32)

        # Forward pass with torch.float16
        with torch.autocast(enabled=True, dtype=torch.float16, device_type="cuda"):
            outputs = model(input_ids)
            self.assertEqual(outputs.dtype, torch.float16)
