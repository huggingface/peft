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

import pytest
import torch
from torch import nn

from peft import LoraConfig, get_peft_model


class ModelWithModuleDict(nn.Module):
    def __init__(self):
        super().__init__()
        self.other_layer = nn.Linear(10, 10)
        self.module = nn.ModuleDict({"foo": nn.Linear(10, 10)})

    def forward(self):
        return self.module["foo"](torch.rand(1, 10))


class ModelWithModuleList(nn.Module):
    def __init__(self):
        super().__init__()
        self.other_layer = nn.Linear(10, 10)
        self.module = nn.ModuleList([nn.Linear(10, 10)])

    def forward(self):
        return self.module[0](torch.rand(1, 10))


class ModelWithParameterDict(nn.Module):
    def __init__(self):
        super().__init__()
        self.other_layer = nn.Linear(10, 10)
        self.module = nn.ParameterDict({"foo": nn.Parameter(torch.rand(10, 10))})

    def forward(self):
        return self.module["foo"]


class ModelWithParameterList(nn.Module):
    def __init__(self):
        super().__init__()
        self.other_layer = nn.Linear(10, 10)
        self.module = nn.ParameterList([nn.Parameter(torch.rand(10, 10))])

    def forward(self):
        return self.module[0]


@pytest.mark.parametrize(
    "cls", [ModelWithModuleDict, ModelWithModuleList, ModelWithParameterDict, ModelWithParameterList]
)
def test_modules_to_save_targets_module_dict_raises(cls):
    model = cls()
    peft_config = LoraConfig(
        target_modules=["other_layer"],
        modules_to_save=["module"],
    )
    model()  # sanity check that the model would normally work

    msg = "modules_to_save cannot be applied to modules of type"
    with pytest.raises(TypeError, match=msg):
        get_peft_model(model=model, peft_config=peft_config)
