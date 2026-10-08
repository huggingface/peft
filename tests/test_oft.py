# Copyright 2026-present the HuggingFace Inc. team.
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

from peft import OFTConfig, get_peft_model
from peft.utils import infer_device


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin0 = nn.Linear(16, 12)

    def forward(self, X):
        return self.lin0(X)


class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(8, 6, 1)

    def forward(self, X):
        return self.conv2d(X)


class TestOft:
    device = infer_device()

    @pytest.mark.parametrize(
        "model_cls, target_module, input_shape", [(LinearModel, "lin0", (3, 16)), (ConvModel, "conv2d", (2, 8, 5, 5))]
    )
    def test_oft_module_dropout_is_applied_in_train_mode(self, model_cls, target_module, input_shape):
        # module_dropout replaces a fraction of the rotation blocks by the identity during training, it must not be a
        # no-op in train mode and it must not leak into eval mode or into merging
        torch.manual_seed(0)
        model = model_cls().to(self.device)
        X = torch.randn(*input_shape, device=self.device)

        config = OFTConfig(
            r=4, oft_block_size=0, target_modules=[target_module], init_weights=False, module_dropout=0.5
        )
        peft_model = get_peft_model(model, config)

        peft_model.train()
        with torch.no_grad():
            outputs = [peft_model(X) for _ in range(10)]
        assert not all(torch.allclose(outputs[0], output) for output in outputs[1:])

        peft_model.eval()
        with torch.no_grad():
            output_eval = peft_model(X)
            assert torch.allclose(output_eval, peft_model(X))
            peft_model.merge_adapter()
            output_merged = peft_model(X)
        assert torch.allclose(output_eval, output_merged, atol=1e-6, rtol=1e-6)
