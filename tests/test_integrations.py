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

import torch
from torch import nn

from peft.utils.integrations import init_empty_weights, skip_init_on_device


class MLP(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.lin0 = nn.Linear(10, 20, bias=bias)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.lin1 = nn.Linear(20, 2, bias=bias)


def get_mlp():
    return MLP()


class TestInitEmptyWeights:
    def test_init_empty_weights_works(self):
        # this is a very rudimentary test, as init_empty_weights is copied almost 1:1 from accelerate and is tested
        # there
        with init_empty_weights():
            mlp = get_mlp()

        expected = torch.device("meta")
        assert all(p.device == expected for p in mlp.parameters())

    def test_skip_init_on_device_works(self):
        # when a function is decorated with skip_init_on_device, the parameters are not moved to meta device, even when
        # inside the context
        decorated_fn = skip_init_on_device(get_mlp)
        with init_empty_weights():
            mlp = decorated_fn()

        expected = torch.device("cpu")
        assert all(p.device == expected for p in mlp.parameters())

    def test_skip_init_on_device_works_outside_context(self):
        # same as before, but ensure that skip_init_on_device does not break when no init_empty_weights context is used
        decorated_fn = skip_init_on_device(get_mlp)
        mlp = decorated_fn()
        expected = torch.device("cpu")
        assert all(p.device == expected for p in mlp.parameters())

    def test_skip_init_on_device_not_permanent(self):
        # ensure that after skip_init_on_device has been used, init_empty_weights reverts to its original functionality

        # with decorator => cpu
        decorated_fn = skip_init_on_device(get_mlp)
        with init_empty_weights():
            mlp = decorated_fn()

        expected = torch.device("cpu")
        assert all(p.device == expected for p in mlp.parameters())

        # without decorator => meta
        with init_empty_weights():
            mlp = get_mlp()

        expected = torch.device("meta")
        assert all(p.device == expected for p in mlp.parameters())

    def test_skip_init_on_device_nested(self):
        # ensure that skip_init_on_device works even if the decorated function is nested inside another decorated
        # function
        @skip_init_on_device
        def outer_fn():
            @skip_init_on_device
            def inner_fn():
                return get_mlp()

            mlp0 = inner_fn()
            mlp1 = get_mlp()
            return mlp0, mlp1

        with init_empty_weights():
            mlp0, mlp1 = outer_fn()

        expected = torch.device("cpu")
        assert all(p.device == expected for p in mlp0.parameters())
        assert all(p.device == expected for p in mlp1.parameters())
