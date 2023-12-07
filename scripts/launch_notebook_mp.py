# coding=utf-8
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

# This is a minimal example of launching PEFT with Accelerate. This used to cause issues because PEFT would eagerly
# import bitsandbytes, which initializes CUDA, resulting in:
# > RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the
# > 'spawn' start method
# This script exists to ensure that this issue does not reoccur.

import torch
import peft
from accelerate import notebook_launcher


def init():
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 2)

        def forward(self, x):
            return self.linear(x)

    model = MyModule().to("cuda")
    peft.get_peft_model(model, peft.LoraConfig(target_modules=["linear"]))


def main():
    notebook_launcher(init, (), num_processes=2)


if __name__ == "__main__":
    main()
