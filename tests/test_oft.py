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

import pytest
import torch

from peft.tuners.oft.layer import MultiplicativeDropoutLayer


class TestOft:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_oft_multiplicative_dropout_preserves_dtype(self, dtype):
        # MultiplicativeDropoutLayer built its mask/identity buffers with device= but no dtype=, so mixing them
        # with a half-precision input silently upcast the layer's output to float32.
        torch.manual_seed(0)
        layer = MultiplicativeDropoutLayer(p=0.5).train()
        x = torch.randn(4, 3, 3, dtype=dtype)
        out = layer(x)
        assert out.dtype == dtype
