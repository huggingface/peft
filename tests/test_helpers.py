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
from diffusers import StableDiffusionPipeline
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
    @pytest.fixture(scope="class")
    def tokenizer(self):
        return AutoTokenizer.from_pretrained("facebook/opt-125m")

    def get_scale_from_modules(self, model):
        layer_to_scale_map = {}
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                layer_to_scale_map[name] = module.scaling

        return layer_to_scale_map

    def test_set_adapter_scale(self, tokenizer):
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
        inputs = tokenizer("hello world", return_tensors="pt")

        with torch.no_grad():
            logits_before_scaling = model(**inputs).logits

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

        # we expect a type error here becuase of wrong datatpye of alpha
        alpha = "a"
        with pytest.raises(TypeError, match=f"{alpha} should be of type float, got {type(alpha)}"):
            with set_adapter_scale(model=model, alpha=alpha):
                pass

    def test_not_lora_model(self):
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

        # we expect a value error here because the model
        # does not have lora layers
        with pytest.raises(ValueError, match="scaling is only supported for models with `LoraLayer`s"):
            with set_adapter_scale(model=model, alpha=0.5):
                pass

    def test_scaling_set_to_zero(self, tokenizer):
        base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        inputs = tokenizer("hello world", return_tensors="pt")

        base_model.eval()

        with torch.no_grad():
            logits_base_model = base_model(**inputs).logits

        lora_config = LoraConfig(
            r=4,
            lora_alpha=4,
            target_modules=["k_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            init_lora_weights=False,
        )
        lora_model = get_peft_model(base_model, lora_config)
        lora_model.eval()

        with set_adapter_scale(model=lora_model, alpha=0.0):
            with torch.no_grad():
                logits_lora_model = lora_model(**inputs).logits

        assert torch.allclose(logits_base_model, logits_lora_model)

    def test_diffusers_pipeline(self):
        model_id = "hf-internal-testing/tiny-stable-diffusion-torch"
        pipeline = StableDiffusionPipeline.from_pretrained(model_id)

        text_encoder_kwargs = {
            "r": 8,
            "lora_alpha": 32,
            "target_modules": ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
            "lora_dropout": 0.0,
            "bias": "none",
        }
        unet_kwargs = {
            "r": 8,
            "lora_alpha": 32,
            "target_modules": ["proj_in", "proj_out", "to_k", "to_q", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"],
            "lora_dropout": 0.0,
            "bias": "none",
        }

        # Instantiate text_encoder adapter
        config_text_encoder = LoraConfig(**text_encoder_kwargs)
        pipeline.text_encoder = get_peft_model(pipeline.text_encoder, config_text_encoder)

        # Instantiate unet adapter
        config_unet = LoraConfig(**unet_kwargs)
        pipeline.unet = get_peft_model(pipeline.unet, config_unet)

        text_scales_before_scaling = self.get_scale_from_modules(pipeline.text_encoder)
        unet_scales_before_scaling = self.get_scale_from_modules(pipeline.unet)

        with set_adapter_scale(model=pipeline.text_encoder, alpha=0.5), set_adapter_scale(
            model=pipeline.unet, alpha=0.5
        ):
            text_scales_during_scaling = self.get_scale_from_modules(pipeline.text_encoder)
            unet_scales_during_scaling = self.get_scale_from_modules(pipeline.unet)
            for key in text_scales_before_scaling.keys():
                assert text_scales_before_scaling[key] != text_scales_during_scaling[key]
            for key in unet_scales_before_scaling.keys():
                assert unet_scales_before_scaling[key] != unet_scales_during_scaling[key]

        text_scales_fter_scaling = self.get_scale_from_modules(pipeline.text_encoder)
        unet_scales_after_scaling = self.get_scale_from_modules(pipeline.unet)
        for key in text_scales_before_scaling.keys():
            assert text_scales_before_scaling[key] == text_scales_fter_scaling[key]
        for key in unet_scales_before_scaling.keys():
            assert unet_scales_before_scaling[key] == unet_scales_after_scaling[key]

    def test_transformers_pipeline(self, tmp_path, tokenizer):
        # this uses a transformers model that loads the adapter directly
        model_id = "facebook/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        config = LoraConfig(init_lora_weights=False)
        model = get_peft_model(model, config)
        model.save_pretrained(tmp_path / "opt-lora")
        del model

        # load directly into transformers model
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.load_adapter(tmp_path / "opt-lora")

        inputs = tokenizer("hello world", return_tensors="pt")

        model = model.eval()

        with torch.no_grad():
            logits_before_scaling = model(**inputs).logits
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

    def test_multi_adapters(self, tokenizer):
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
        inputs = tokenizer("hello world", return_tensors="pt")

        # add another adaper and activate it
        model.add_adapter("other", lora_config)
        model.set_adapter("other")

        scales_before_scaling = self.get_scale_from_modules(model)
        model.eval()
        with torch.no_grad():
            logits_before = model(**inputs).logits

        with set_adapter_scale(model=model, alpha=0.5):
            scales_during_scaling = self.get_scale_from_modules(model)
            for key in scales_before_scaling.keys():
                assert scales_before_scaling[key] != scales_during_scaling[key]

            with torch.no_grad():
                logits_during = model(**inputs).logits

            assert not torch.allclose(logits_before, logits_during)

        scales_after_scaling = self.get_scale_from_modules(model)
        for key in scales_before_scaling.keys():
            assert scales_before_scaling[key] == scales_after_scaling[key]

        with torch.no_grad():
            logits_after = model(**inputs).logits

        assert torch.allclose(logits_before, logits_after)

    def test_rank_alpha_pattern(self, tokenizer):
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        lora_config = LoraConfig(
            r=4,
            lora_alpha=4,
            target_modules=["k_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            init_lora_weights=False,
            rank_pattern={"k_proj": 2},
            alpha_pattern={"k_proj": 8},
        )

        model = get_peft_model(model, lora_config)
        model.eval()
        inputs = tokenizer("hello world", return_tensors="pt")

        with torch.no_grad():
            logits_before_scaling = model(**inputs).logits

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

    def test_merging_adapter(self, tokenizer):
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
        inputs = tokenizer("hello world", return_tensors="pt")

        with set_adapter_scale(model=model, alpha=0.5):
            with torch.no_grad():
                logits_unmerged_scaling = model(**inputs).logits
            model = model.merge_and_unload()

        with torch.no_grad():
            logits_merged_scaling = model(**inputs).logits

        assert torch.allclose(logits_merged_scaling, logits_unmerged_scaling, atol=1e-4, rtol=1e-4)
