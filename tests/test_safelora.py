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

# This test file is for tests specific to SafeLoRA

import pytest
import torch
from unittest.mock import MagicMock, patch
from peft import PeftModel, SafeLoRAConfig, SafeLoRA


class MockConfig:
    def __init__(self):
        self.base_model_path = './LLM_Models/llama-2-7b-hf/'
        self.aligned_model_path = './LLM_Models/llama-2-7b-chat-fp16/'
        self.select_layers_type = 'threshold'
        self.threshold = 0.5
        self.num_proj_layers = 1
        self.devices = 'cpu'
        self.target_modules = ['q_proj', 'v_proj']
        self.r = 4

@patch('transformers.AutoModelForCausalLM.from_pretrained')
def test_get_aligned_matrix(mock_from_pretrained):
    mock_base_model = MagicMock()
    mock_aligned_model = MagicMock()
    mock_base_model.named_parameters.return_value = [
        ('q_proj', torch.randn(4, 4)),
        ('v_proj', torch.randn(4, 4))
    ]
    mock_aligned_model.named_parameters.return_value = [
        ('q_proj', torch.randn(4, 4)),
        ('v_proj', torch.randn(4, 4))
    ]
    mock_from_pretrained.side_effect = [mock_base_model, mock_aligned_model]
    mock_peft_model = MagicMock(spec=torch.nn.Module)
    mock_peft_model.peft_config = {"default": MockConfig()}
    safelora = SafeLoRA(mock_peft_model, MockConfig())
    aligned_matrix = safelora.get_aligned_matrix()
    assert len(aligned_matrix) == 2
    assert all(vec.shape == (4, 4) for vec in aligned_matrix)

def test_safelora_with_temp_model(temp_model_paths):
    config = MockConfig()
    config.base_model_path = temp_model_paths
    config.aligned_model_path = temp_model_paths
    mock_peft_model = MagicMock(spec=torch.nn.Module)
    mock_peft_model.peft_config = {"default": config}
    safelora = SafeLoRA(mock_peft_model, config)
    aligned_matrix = safelora.get_aligned_matrix()
    assert len(aligned_matrix) == 2