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
"""This is a standalone script that checks that we can hotswap a LoRA adapter on a compiled model

By itself, this script is not super interesting but when we collect the compile logs, we can check that hotswapping
does not trigger recompilation. This is done in the TestLoraHotSwapping class in test_pipelines.py.

Running this script with `check_hotswap(False)` will load the LoRA adapter without hotswapping, which will result in
recompilation.

"""

import os
import sys
import tempfile

import torch
from transformers import AutoModelForCausalLM

from peft import LoraConfig, PeftModel, get_peft_model
from peft.utils import infer_device
from peft.utils.hotswap import hotswap_adapter


torch_device = infer_device()


def check_hotswap(do_hotswap=True):
    torch.manual_seed(0)
    inputs = torch.arange(10).view(-1, 1).to(torch_device)
    model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-OPTForCausalLM").to(torch_device)
    config = LoraConfig(init_lora_weights=False)
    model = get_peft_model(model, config, adapter_name="adapter0").eval()
    model.add_adapter("adapter1", config)

    with tempfile.TemporaryDirectory() as tmp_dirname:
        model.save_pretrained(tmp_dirname)
        del model

        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-OPTForCausalLM").to(torch_device)
        model = PeftModel.from_pretrained(model, os.path.join(tmp_dirname, "adapter0")).eval()
        model = torch.compile(model, mode="reduce-overhead")
        model(inputs).logits

        # swap and check that we get the output from adapter1
        if do_hotswap:
            hotswap_adapter(model, os.path.join(tmp_dirname, "adapter1"), adapter_name="default")
        else:
            model.load_adapter(os.path.join(tmp_dirname, "adapter1"), adapter_name="other")
            model.set_adapter("other")

        # we need to call forward to potentially trigger recompilation
        model(inputs).logits


if __name__ == "__main__":
    # check_hotswap(False) will trigger recompilation
    check_hotswap(do_hotswap=sys.argv[1] == "1")
