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

import torch
from torch import nn

from peft import MonteCLoraConfig, PeftModel, PeftType, get_peft_model
from peft.tuners.monteclora import MonteCLoraLinear


class SimpleNet(nn.Module):
    """
    A simple dummy model for testing Linear layer replacement.
    """

    def __init__(self, in_features=10, out_features=10):
        super().__init__()
        self.lin0 = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(out_features, out_features)

    def forward(self, x):
        x = self.lin0(x)
        x = self.relu(x)
        x = self.lin1(x)
        return x


class TestMonteCLora(unittest.TestCase):
    def setUp(self):
        self.input_dim = 10
        self.output_dim = 10
        self.model = SimpleNet(self.input_dim, self.output_dim)

        # Standard Config for testing
        self.config = MonteCLoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["lin0", "lin1"],
            use_monteclora=True,
            monteclora_n=4,
            monteclora_targets=["lin0"],  # Only apply MonteCLoRA to first layer
            mc_training=True,
            # Set task_type to None so PEFT uses the generic PeftModel
            task_type=None,
        )

    def test_initialization(self):
        """
        Test if the model initializes correctly and replaces layers with MonteCLoraLinear.
        """
        model = get_peft_model(self.model, self.config)

        # Check PEFT type
        assert model.peft_config["default"].peft_type == PeftType.MONTECLORA

        # Check Layer Replacement
        # lin0 should be MonteCLoraLinear (because it is in monteclora_targets)
        assert isinstance(model.base_model.model.lin0, MonteCLoraLinear)
        assert model.base_model.model.lin0.use_monteclora

        # lin1 should be Standard LoRA (because it is NOT in monteclora_targets)
        # Note: MonteCLoraLinear inherits from LoraLayer, but use_monteclora should be False
        assert isinstance(model.base_model.model.lin1, MonteCLoraLinear)
        assert not model.base_model.model.lin1.use_monteclora

    def test_trainable_parameters(self):
        """
        Test that base weights are frozen and MonteCLoRA specific parameters are trainable.
        """
        model = get_peft_model(self.model, self.config)

        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]

        # 1. Check Base weights are frozen
        assert not model.base_model.model.lin0.weight.requires_grad

        # 2. Check LoRA weights are trainable
        assert any("lora_A" in n for n in trainable_names)
        assert any("lora_B" in n for n in trainable_names)

        # 3. Check MonteCLoRA Sampler parameters are trainable
        # We named the dict "lora_mc_sampler_A" in layer.py
        assert any("lora_mc_sampler_A" in n for n in trainable_names)

        # 4. Check that std_prior or expert_weights are specifically trainable
        assert any("expert_weights_prior" in n for n in trainable_names)

    def test_forward_pass(self):
        """
        Test that the forward pass runs without errors and produces correct shape.
        """
        model = get_peft_model(self.model, self.config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        input_data = torch.randn(2, self.input_dim).to(device)
        output = model(input_data)

        # Check shape
        assert output.shape == (2, self.output_dim)

        # Check backward pass
        loss = output.sum()
        loss.backward()

    def test_eval_consistency(self):
        """
        Test that Eval mode turns off sampling and produces deterministic results.
        """
        model = get_peft_model(self.model, self.config)
        model.eval()  # Disable MC sampling

        input_data = torch.randn(1, self.input_dim)

        out1 = model(input_data)
        out2 = model(input_data)

        # Outputs should be identical in Eval mode
        assert torch.allclose(out1, out2, atol=1e-8)

    def test_save_and_load(self):
        """
        Test saving the adapter and loading it back.
        """
        model = get_peft_model(self.model, self.config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)

            # Load back
            model_loaded = PeftModel.from_pretrained(self.model, tmp_dir)

            # Check config matches
            assert model_loaded.peft_config["default"].use_monteclora
            assert model_loaded.peft_config["default"].monteclora_n == 4

            # Check if specific MonteCLoRA keys exist in state dict
            state_dict = model_loaded.state_dict()
            assert any("lora_mc_sampler_A" in k for k in state_dict.keys())


if __name__ == "__main__":
    unittest.main()
