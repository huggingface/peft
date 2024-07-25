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
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model
from peft.helpers import check_if_peft_model, set_adapter_scale
from peft.tuners.lora.layer import LoraLayer


class TestCheckIsPeftModel:
    def test_valid_hub_model(self):
        result = check_if_peft_model("peft-internal-testing/gpt2-lora-random")
        assert result is True

    def test_invalid_hub_model(self):
        result = check_if_peft_model("gpt2")
        assert result is False

    def test_nonexisting_hub_model(self):
        result = check_if_peft_model("peft-internal-testing/non-existing-model")
        assert result is False

    def test_local_model_valid(self, tmp_path):
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        config = LoraConfig()
        model = get_peft_model(model, config)
        model.save_pretrained(tmp_path / "peft-gpt2-valid")
        result = check_if_peft_model(tmp_path / "peft-gpt2-valid")
        assert result is True

    def test_local_model_invalid(self, tmp_path):
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        model.save_pretrained(tmp_path / "peft-gpt2-invalid")
        result = check_if_peft_model(tmp_path / "peft-gpt2-invalid")
        assert result is False

    def test_local_model_broken_config(self, tmp_path):
        with open(tmp_path / "adapter_config.json", "w") as f:
            f.write('{"foo": "bar"}')

        result = check_if_peft_model(tmp_path)
        assert result is False

    def test_local_model_non_default_name(self, tmp_path):
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        config = LoraConfig()
        model = get_peft_model(model, config, adapter_name="other")
        model.save_pretrained(tmp_path / "peft-gpt2-other")

        # no default adapter here
        result = check_if_peft_model(tmp_path / "peft-gpt2-other")
        assert result is False

        # with adapter name
        result = check_if_peft_model(tmp_path / "peft-gpt2-other" / "other")
        assert result is True

    def test_set_adapter_scale(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Get the PeftModel
        model = AutoPeftModelForCausalLM.from_pretrained("ybelkada/opt-350m-lora")
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        # Prepare inputs
        input_prompt = "Preheat the oven to 350 degrees"
        inputs = tokenizer(input_prompt, return_tensors="pt")

        # Get logits before scaling
        with torch.no_grad():
            inputs = inputs.to(device)
            model = model.to(device)
            logits_before_scaling = model(**inputs).logits

        # Store the scale of the lora layers
        scales_before_scaling = scale_from_modules(model)

        with set_adapter_scale(model=model, alpha=0.5):
            # Check for scaling
            scales_during_scaling = scale_from_modules(model)
            for key in scales_before_scaling.keys():
                assert scales_before_scaling[key] != scales_during_scaling[key]

            # Generate logits after scaling
            with torch.no_grad():
                inputs = inputs.to(device)
                model = model.to(device)
                logits_after_scaling = model(**inputs).logits

            assert torch.allclose(logits_before_scaling, logits_after_scaling)

        # Check for restored sclaes
        scales_after_scaling = scale_from_modules(model)
        for key in scales_before_scaling.keys():
            assert scales_before_scaling[key] == scales_after_scaling[key]


def scale_from_modules(model):
    layer_to_scale_map = {}
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            layer_to_scale_map[name] = module.scaling

    return layer_to_scale_map
