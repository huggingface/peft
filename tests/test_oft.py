# Copyright 2023-present the HuggingFace Inc. team.
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
import torch
from torch import nn

from peft import OFTConfig, get_peft_model


class TestOft:
    def test_oft_linear_accepts_4d_inputs(self):
        # Regression for #3968: Linear OFT used to treat any 4-D tensor as a Conv2d
        # activation and unfold it, which fails for shapes like ConvNeXt pwconv.
        base = nn.Sequential(nn.Linear(8, 4))
        model = get_peft_model(base, OFTConfig(r=2, oft_block_size=0, target_modules=["0"])).eval()

        out_3d = model(torch.randn(2, 5, 8))
        out_4d = model(torch.randn(2, 3, 4, 8))
        out_5d = model(torch.randn(2, 3, 4, 5, 8))

        assert out_3d.shape == (2, 5, 4)
        assert out_4d.shape == (2, 3, 4, 4)
        assert out_5d.shape == (2, 3, 4, 5, 4)

    def test_oft_conv2d_4d_still_works(self):
        # Conv2d OFT must keep the unfold/fold path for NCHW inputs.
        base = nn.Sequential(nn.Conv2d(4, 8, kernel_size=3, padding=1))
        model = get_peft_model(base, OFTConfig(r=4, oft_block_size=0, target_modules=["0"])).eval()
        out = model(torch.randn(2, 4, 5, 5))
        assert out.shape == (2, 8, 5, 5)
