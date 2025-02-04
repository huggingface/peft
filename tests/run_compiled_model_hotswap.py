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
from peft.utils.hotswap import hotswap_adapter, prepare_model_for_compiled_hotswap


torch_device = infer_device()
inputs = torch.arange(10).view(-1, 1)


def check_hotswap(do_hotswap=True, ranks=(8, 8), alpha_scalings=(16, 16)):
    torch.manual_seed(0)
    inputs = torch.arange(10).view(-1, 1).to(torch_device)
    model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-OPTForCausalLM").to(torch_device)
    rank0, rank1 = ranks
    alpha0, alpha1 = alpha_scalings
    # note that the 2nd adapter targeting a subset of the 1st adapter is okay, but not the other way round
    config0 = LoraConfig(init_lora_weights=False, r=rank0, lora_alpha=alpha0, target_modules=["q_proj", "v_proj"])
    config1 = LoraConfig(init_lora_weights=False, r=rank1, lora_alpha=alpha1, target_modules=["q_proj"])
    model = get_peft_model(model, config0, adapter_name="adapter0").eval()
    with torch.inference_mode():
        output0 = model(inputs).logits

    model.add_adapter("adapter1", config1)
    model.set_adapter("adapter1")
    with torch.inference_mode():
        output1 = model(inputs).logits

    # sanity check:
    tol = 1e-5
    assert not torch.allclose(output0, output1, atol=tol, rtol=tol)

    with tempfile.TemporaryDirectory() as tmp_dirname:
        model.save_pretrained(tmp_dirname)
        del model

        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-OPTForCausalLM").to(torch_device)
        model = PeftModel.from_pretrained(model, os.path.join(tmp_dirname, "adapter0")).eval()
        if do_hotswap:
            prepare_model_for_compiled_hotswap(model, config=model.peft_config, target_rank=max(ranks))
        model.compile(mode="reduce-overhead")
        output_after0 = model(inputs).logits
        assert torch.allclose(output0, output_after0, atol=tol, rtol=tol)

        # swap and check that we get the output from adapter1
        if do_hotswap:
            hotswap_adapter(model, os.path.join(tmp_dirname, "adapter1"), adapter_name="default")
        else:
            model.load_adapter(os.path.join(tmp_dirname, "adapter1"), adapter_name="other")
            model.set_adapter("other")

        # we need to call forward to potentially trigger recompilation
        output_after1 = model(inputs).logits
        assert torch.allclose(output1, output_after1, atol=tol, rtol=tol)


if __name__ == "__main__":
    # check_hotswap(False) will trigger recompilation
    do_hotswap = sys.argv[1] == "1"
    # ranks is a string like '13,7'
    ranks = sys.argv[2].split(",")
    ranks = int(ranks[0]), int(ranks[1])
    check_hotswap(do_hotswap=do_hotswap, ranks=ranks, alpha_scalings=(8, 16))
