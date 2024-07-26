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
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig, get_peft_model
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


class TestScalingAdapters:
    def get_scale_from_modules(self, model):
        layer_to_scale_map = {}
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                layer_to_scale_map[name] = module.scaling

        return layer_to_scale_map

    def test_set_adapter_scale(self):
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        lora_config = LoraConfig(
            r=4,
            lora_alpha=4,
            target_modules=["k_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            init_lora_weights=False,
        )

        model = get_peft_model(model, lora_config)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        inputs = tokenizer("hello world", return_tensors="pt")

        with torch.no_grad():
            logits_before_scaling = model(
                **inputs,
            ).logits

        scales_before_scaling = self.get_scale_from_modules(model)

        with set_adapter_scale(model=model, alpha=0.5):
            scales_during_scaling = self.get_scale_from_modules(model)
            for key in scales_before_scaling.keys():
                assert scales_before_scaling[key] != scales_during_scaling[key]

            with torch.no_grad():
                logits_during_scaling = model(**inputs).logits

            assert not torch.allclose(logits_before_scaling, logits_during_scaling)

        scales_after_scaling = self.get_scale_from_modules(model)
        for key in scales_before_scaling.keys():
            assert scales_before_scaling[key] == scales_after_scaling[key]

        with torch.no_grad():
            logits_after_scaling = model(**inputs).logits

        assert torch.allclose(logits_before_scaling, logits_after_scaling)

    def test_wrong_scaling_datatype(self):
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        lora_config = LoraConfig(
            r=4,
            lora_alpha=4,
            target_modules=["k_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            init_lora_weights=False,
        )

        model = get_peft_model(model, lora_config)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        inputs = tokenizer("hello world", return_tensors="pt")

        with torch.no_grad():
            logits_before_scaling = model(**inputs).logits

        scales_before_scaling = self.get_scale_from_modules(model)

        # we expect a type error here becuase of wrong datatpye of alpha
        with pytest.raises(TypeError):
            with set_adapter_scale(model=model, alpha="a"):
                pass

        scales_after_scaling = self.get_scale_from_modules(model)
        for key in scales_before_scaling.keys():
            assert scales_before_scaling[key] == scales_after_scaling[key]

        with torch.no_grad():
            logits_after_scaling = model(**inputs).logits

        assert torch.allclose(logits_before_scaling, logits_after_scaling)

    def test_not_lora_model(self):
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

        # we expect a value error here because the model
        # does not have lora layers
        with pytest.raises(ValueError):
            with set_adapter_scale(model=model, alpha=0.5):
                pass
