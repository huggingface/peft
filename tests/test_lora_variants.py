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
from transformers import AutoModelForCausalLM

from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora.layer import Conv1d as LoraConv1d
from peft.tuners.lora.layer import Conv2d as LoraConv2d
from peft.tuners.lora.layer import Embedding as LoraEmbedding
from peft.tuners.lora.layer import Linear as LoraLinear
from peft.tuners.lora.variants import (
    ALoraLinearVariant,
    DoraConv1dVariant,
    DoraConv2dVariant,
    DoraEmbeddingVariant,
    DoraLinearVariant,
    calculate_alora_offsets,
    get_alora_offsets_for_forward,
    get_alora_offsets_for_generate,
)

from .testing_common import hub_online_once


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

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return kwargs

    def forward(self, X=None, embeds=None, num_beams=None, alora_offsets=None):
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

        dtype = torch.float32

        return DummyLM().to(dtype)


VARIANT_MAP = {
    "dora": {
        LoraLinear: DoraLinearVariant,
        LoraEmbedding: DoraEmbeddingVariant,
        LoraConv1d: DoraConv1dVariant,
        LoraConv2d: DoraConv2dVariant,
    },
    "alora": {
        LoraLinear: ALoraLinearVariant,
    },
}


TEST_CASES = [
    (
        "dora",
        LoraConfig,
        {"target_modules": ["linear1", "linear2", "conv1d", "conv2d", "embedding"], "use_dora": True},
    ),
    (
        "alora",
        LoraConfig,
        {"target_modules": ["linear1", "linear2"], "alora_invocation_tokens": [1]},
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


class TestActivatedLora:
    @pytest.mark.parametrize(
        "input_ids, alora_invocation_tokens, expected_offsets",
        [
            ([[0, 1, 2, 3], [0, 4, 5, 6]], [1, 2], [3, None]),
            ([[1, 2, 1, 2], [0, 4, 1, 2]], [1, 2], [2, 2]),
            ([[1, 2, 3, 4], [0, 4, 1, 4]], [1, 2], [4, None]),
            ([[1, 2, 3, 4]], None, [None]),
        ],
    )
    # Verify alora_offsets are calculated correctly
    def test_calculate_alora_offsets(self, input_ids, alora_invocation_tokens, expected_offsets):
        config = LoraConfig(task_type=TaskType.CAUSAL_LM, alora_invocation_tokens=alora_invocation_tokens)
        peft_config = {"default": config}

        # compute offsets
        offsets = calculate_alora_offsets(peft_config, "default", torch.tensor(input_ids))

        assert offsets == expected_offsets

    @pytest.mark.parametrize(
        "input_ids, alora_invocations, expected_offsets",
        [
            ([[0, 1, 1], [0, 2, 2]], {"a1": [1], "a2": [2]}, [1, 1]),
            ([[0, 1, 1], [0, 2, 2]], {"a1": [1], "a2": None}, [1, None]),
        ],
    )
    # Verify alora_offsets are correct with adapter names
    def test_calculate_alora_offsets_with_adapter_names(self, input_ids, alora_invocations, expected_offsets):
        peft_config = {}
        for alora_name in alora_invocations.keys():
            peft_config[alora_name] = LoraConfig(alora_invocation_tokens=alora_invocations[alora_name])

        adapter_names = list(alora_invocations.keys())
        offsets = calculate_alora_offsets(
            peft_config, adapter_names[0], torch.tensor(input_ids), adapter_names=adapter_names
        )

        assert offsets == expected_offsets

    # Verify that the adapter does not modify outputs prior to invocation point
    def test_alora_activation_matches_base_until_invocation(self):
        transformers_class = MockTransformerWrapper
        base_model = transformers_class.from_pretrained()
        cfg = LoraConfig(target_modules=["linear"], alora_invocation_tokens=[2], init_lora_weights=False)
        lora_model = get_peft_model(base_model, cfg)
        lora_model.eval()

        input_ids = torch.tensor([[0, 1, 2, 3]])
        start = 2
        with lora_model.disable_adapter():
            with torch.no_grad():
                base_out = lora_model(X=input_ids)

        kwargs = get_alora_offsets_for_forward(lora_model, input_ids)
        with torch.no_grad():
            lora_out = lora_model(X=input_ids, **kwargs)
        assert torch.allclose(lora_out[:, :start], base_out[:, :start])
        assert not torch.allclose(lora_out[:, start:], base_out[:, start:])

    # Verify that warning is given for alora when providing embeddings only
    def test_input_embeds_warning(self):
        transformers_class = MockTransformerWrapper
        base_model = transformers_class.from_pretrained()
        cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["linear"],
            alora_invocation_tokens=[2],
            init_lora_weights=False,
        )
        lora_model = get_peft_model(base_model, cfg)
        lora_model.eval()

        input_ids = torch.tensor([[0, 1, 2, 3]])
        input_embeds = base_model.embed(input_ids)
        with pytest.warns(
            UserWarning,
            match="Cannot calculate aLoRA offsets when only inputs_embeds are provided. Disabling aLoRA for this forward pass.",
        ):
            kwargs = get_alora_offsets_for_forward(lora_model, inputs_embeds=input_embeds)
        assert kwargs.get("alora_offsets") is None
        with pytest.warns(
            UserWarning,
            match="Cannot calculate aLoRA offsets during generate as input_ids are not available. Disabling aLoRA.",
        ):
            kwargs = get_alora_offsets_for_generate(lora_model, inputs_embeds=input_embeds)
        assert kwargs.get("alora_offsets") is None

    # Verify that error is raised when requesting num_beams > 1 for alora
    def test_num_beams_error(self):
        transformers_class = MockTransformerWrapper
        base_model = transformers_class.from_pretrained()
        cfg = LoraConfig(target_modules=["linear"], alora_invocation_tokens=[2], init_lora_weights=False)
        lora_model = get_peft_model(base_model, cfg)
        lora_model.eval()

        input_ids = torch.tensor([[0, 1, 2, 3]])
        with pytest.raises(ValueError) as e:
            with torch.no_grad():
                lora_out = lora_model(X=input_ids, num_beams=2, alora_offsets=[3])
        assert "Beam search not yet supported for aLoRA." in str(e.value)

    def test_gradient_checkpointing_double_forward_raises(self):
        model_id = "trl-internal-testing/tiny-random-LlamaForCausalLM"

        with hub_online_once(model_id):
            base_model = AutoModelForCausalLM.from_pretrained(model_id)
            cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules="all-linear", alora_invocation_tokens=[0])
            lora_model = get_peft_model(base_model, cfg)
            lora_model.train()

            lora_model.prepare_model_for_gradient_checkpointing(lora_model)
            lora_model.gradient_checkpointing_enable()

            inputs = {"input_ids": torch.tensor([[0, 1, 2, 3]])}

            lora_model.forward(**inputs)

            with pytest.raises(ValueError, match="Multiple invocations of PEFT forward hooks.*"):
                lora_model.forward(**inputs)

    def test_gradient_checkpointing_dpo_doesnt_raise(self):
        model_id = "trl-internal-testing/tiny-random-LlamaForCausalLM"

        with hub_online_once(model_id):
            base_model = AutoModelForCausalLM.from_pretrained(model_id)
            cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules="all-linear", alora_invocation_tokens=[0])
            lora_model = get_peft_model(base_model, cfg)
            lora_model.train()

            lora_model.prepare_model_for_gradient_checkpointing(lora_model)
            lora_model.gradient_checkpointing_enable()

            inputs = {"input_ids": torch.tensor([[0, 1, 2, 3]])}

            with lora_model.disable_adapter():
                lora_model.forward(**inputs)

            lora_model.forward(**inputs)
