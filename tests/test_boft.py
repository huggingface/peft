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
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from peft import BOFTConfig, PeftModel, get_peft_model
from peft.utils import infer_device


class TestBoft:
    device = infer_device()

    def test_boft_state_dict(self, tmp_path):
        # see #2050
        # ensure that the boft_P buffer is not stored in the checkpoint file and is not necessary to load the model
        # correctly
        torch.manual_seed(0)

        inputs = torch.arange(10).view(-1, 1).to(self.device)
        model_id = "hf-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
        model.eval()
        output_base = model(inputs).logits

        config = BOFTConfig(init_weights=False)
        model = get_peft_model(model, config)
        model.eval()
        output_peft = model(inputs).logits

        atol, rtol = 1e-5, 1e-8
        # sanity check: loading boft changed the output
        assert not torch.allclose(output_base, output_peft, atol=atol, rtol=rtol)

        model.save_pretrained(tmp_path)
        del model

        # check that the boft_P buffer is not present
        state_dict = load_file(tmp_path / "adapter_model.safetensors")
        assert not any("boft_P" in key for key in state_dict)

        # sanity check: the model still produces the same output after loading
        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
        model = PeftModel.from_pretrained(model, tmp_path)
        output_loaded = model(inputs).logits
        assert torch.allclose(output_peft, output_loaded, atol=atol, rtol=rtol)

    def test_boft_old_checkpoint_including_boft_P(self, tmp_path):
        # see #2050
        # This test exists to ensure that after the boft_P buffer was made non-persistent, old checkpoints can still be
        # loaded successfully.
        torch.manual_seed(0)

        inputs = torch.arange(10).view(-1, 1).to(self.device)
        model_id = "hf-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)

        # first create the expected output
        config = BOFTConfig(init_weights=False)
        model = get_peft_model(model, config)
        model.eval()
        output_peft = model(inputs).logits
        del model

        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
        # checkpoint from before the PR whose state_dict still contains boft_P
        hub_id = "peft-internal-testing/boft-tiny-opt-peft-v0.12"
        model = PeftModel.from_pretrained(model, hub_id)
        output_old = model(inputs).logits

        atol, rtol = 1e-5, 1e-8
        assert torch.allclose(output_peft, output_old, atol=atol, rtol=rtol)
