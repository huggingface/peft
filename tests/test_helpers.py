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


from unittest.mock import Mock

import pytest
import torch
from diffusers import StableDiffusionPipeline
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from peft import (
    AdaLoraConfig,
    AdamssConfig,
    KasaConfig,
    LoraConfig,
    TaskType,
    get_peft_model,
)
from peft.helpers import (
    DoraCaching,
    MontecloraTrainerMixin,
    PeftTrainerMixin,
    check_if_peft_model,
    disable_input_dtype_casting,
    rescale_adapter_scale,
)
from peft.tuners.adamss.asa_callback import AdamssAsaCallback
from peft.tuners.adamss.layer import AdamssLayer
from peft.tuners.lora.config import MontecloraConfig
from peft.tuners.lora.layer import LoraLayer
from peft.tuners.lora.monteclora import MontecloraSampler
from peft.utils import infer_device

from .testing_utils import hub_online_once


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
        return AutoTokenizer.from_pretrained("peft-internal-testing/opt-125m")

    def get_scale_from_modules(self, model):
        layer_to_scale_map = {}
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                layer_to_scale_map[name] = module.scaling

        return layer_to_scale_map

    def test_rescale_adapter_scale(self, tokenizer):
        model = AutoModelForCausalLM.from_pretrained("peft-internal-testing/opt-125m")
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

        with rescale_adapter_scale(model=model, multiplier=0.5):
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
        model = AutoModelForCausalLM.from_pretrained("peft-internal-testing/opt-125m")
        lora_config = LoraConfig(
            r=4,
            lora_alpha=4,
            target_modules=["k_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            init_lora_weights=False,
        )

        model = get_peft_model(model, lora_config)

        # we expect a type error here because of wrong datatype of multiplier
        multiplier = "a"
        with pytest.raises(TypeError, match=f"Argument multiplier should be of type float, got {type(multiplier)}"):
            with rescale_adapter_scale(model=model, multiplier=multiplier):
                pass

    def test_not_lora_model(self):
        model = AutoModelForCausalLM.from_pretrained("peft-internal-testing/opt-125m")

        # we expect a value error here because the model
        # does not have lora layers
        with pytest.raises(ValueError, match="scaling is only supported for models with `LoraLayer`s"):
            with rescale_adapter_scale(model=model, multiplier=0.5):
                pass

    def test_scaling_set_to_zero(self, tokenizer):
        base_model = AutoModelForCausalLM.from_pretrained("peft-internal-testing/opt-125m")
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

        with rescale_adapter_scale(model=lora_model, multiplier=0.0):
            with torch.no_grad():
                logits_lora_model = lora_model(**inputs).logits

        assert torch.allclose(logits_base_model, logits_lora_model)

    def test_diffusers_pipeline(self):
        model_id = "hf-internal-testing/tiny-sd-pipe"
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

        with (
            rescale_adapter_scale(model=pipeline.text_encoder, multiplier=0.5),
            rescale_adapter_scale(model=pipeline.unet, multiplier=0.5),
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
        model_id = "peft-internal-testing/opt-125m"
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

        with rescale_adapter_scale(model=model, multiplier=0.5):
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
        model = AutoModelForCausalLM.from_pretrained("peft-internal-testing/opt-125m")
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

        # add another adapter and activate it
        model.add_adapter("other", lora_config)
        model.set_adapter("other")

        scales_before_scaling = self.get_scale_from_modules(model)
        model.eval()
        with torch.no_grad():
            logits_before = model(**inputs).logits

        with rescale_adapter_scale(model=model, multiplier=0.5):
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
        model = AutoModelForCausalLM.from_pretrained("peft-internal-testing/opt-125m")
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

        with rescale_adapter_scale(model=model, multiplier=0.5):
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
        model = AutoModelForCausalLM.from_pretrained("peft-internal-testing/opt-125m")
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

        with rescale_adapter_scale(model=model, multiplier=0.5):
            with torch.no_grad():
                logits_unmerged_scaling = model(**inputs).logits
            model = model.merge_and_unload()

        with torch.no_grad():
            logits_merged_scaling = model(**inputs).logits

        assert torch.allclose(logits_merged_scaling, logits_unmerged_scaling, atol=1e-4, rtol=1e-4)


class TestDisableInputDtypeCasting:
    """Test the context manager `disable_input_dtype_casting` that temporarily disables input dtype casting
    in the model.

    The test works as follows:

    We create a simple MLP and convert it to a PeftModel. The model dtype is set to float16. Then a pre-foward hook is
    added that casts the model parameters to float32. Moreover, a post-forward hook is added that casts the weights
    back to float16. The input dtype is float32.

    Without the disable_input_dtype_casting context, what would happen is that PEFT detects that the input dtype is
    float32 but the weight dtype is float16, so it casts the input to float16. Then the pre-forward hook casts the
    weight to float32, which results in a RuntimeError.

    With the disable_input_dtype_casting context, the input dtype is left as float32 and there is no error. We also add
    a hook to record the dtype of the result from the LoraLayer to ensure that it is indeed float32.

    """

    device = infer_device()
    dtype_record = []

    @torch.no_grad()
    def cast_params_to_fp32_pre_hook(self, module, input):
        for param in module.parameters(recurse=False):
            param.data = param.data.float()
        return input

    @torch.no_grad()
    def cast_params_to_fp16_hook(self, module, input, output):
        for param in module.parameters(recurse=False):
            param.data = param.data.half()
        return output

    def record_dtype_hook(self, module, input, output):
        self.dtype_record.append(output[0].dtype)

    @pytest.fixture
    def inputs(self):
        return torch.randn(4, 10, device=self.device, dtype=torch.float32)

    @pytest.fixture
    def base_model(self):
        class MLP(nn.Module):
            def __init__(self, bias=True):
                super().__init__()
                self.lin0 = nn.Linear(10, 20, bias=bias)
                self.lin1 = nn.Linear(20, 2, bias=bias)
                self.sm = nn.LogSoftmax(dim=-1)

            def forward(self, X):
                X = self.lin0(X)
                X = self.lin1(X)
                X = self.sm(X)
                return X

        return MLP()

    @pytest.fixture
    def model(self, base_model):
        config = LoraConfig(target_modules=["lin0"], modules_to_save=["lin1"])
        model = get_peft_model(base_model, config).to(device=self.device, dtype=torch.float16)
        # Register hooks on the submodule that holds parameters
        for module in model.modules():
            if sum(p.numel() for p in module.parameters()) > 0:
                module.register_forward_pre_hook(self.cast_params_to_fp32_pre_hook)
                module.register_forward_hook(self.cast_params_to_fp16_hook)
            if isinstance(module, LoraLayer):
                module.register_forward_hook(self.record_dtype_hook)
        return model

    def test_disable_input_dtype_casting_active(self, model, inputs):
        self.dtype_record.clear()
        with disable_input_dtype_casting(model, active=True):
            model(inputs)
        assert self.dtype_record == [torch.float32]

    def test_no_disable_input_dtype_casting(self, model, inputs):
        msg = r"expected m.*1 and m.*2 to have the same dtype"
        with pytest.raises(RuntimeError, match=msg):
            model(inputs)

    def test_disable_input_dtype_casting_inactive(self, model, inputs):
        msg = r"expected m.*1 and m.*2 to have the same dtype"
        with pytest.raises(RuntimeError, match=msg):
            with disable_input_dtype_casting(model, active=False):
                model(inputs)

    def test_disable_input_dtype_casting_inactive_after_existing_context(self, model, inputs):
        # this is to ensure that when the context is left, we return to the previous behavior
        with disable_input_dtype_casting(model, active=True):
            model(inputs)

        # after the context exited, we're back to the error
        msg = r"expected m.*1 and m.*2 to have the same dtype"
        with pytest.raises(RuntimeError, match=msg):
            model(inputs)


class TestDoraCaching:
    # Check that DoRA caching works (same results with and without caching, cache is filled/cleared). Note that this test
    # does not check the actual runtime benefit of caching, because this could be flaky and measuring it reliably and in
    # realistic conditions is expensive. Run examples/dora_finetuning/dora-caching.py instead to measure this.
    device = infer_device()

    @pytest.fixture(autouse=True)
    def disable_dora_caching(self):
        # auto-fixture to ensure that no test accidentally permanently enables DoRA caching
        DoraCaching()(enabled=False)

    def get_caches(self, model):
        # utility function to collect all the caches in the model
        caches = []
        for module in model.modules():
            if hasattr(module, "_dora_cache"):
                caches.append(module._dora_cache)
        return caches

    def get_output(self, model, inputs):
        output = model(inputs)
        if hasattr(output, "logits"):
            return output.logits
        return output

    def test_dora_caching_linear(self):
        # ensure that the results don't change due to caching
        inputs = torch.arange(10).view(1, -1).to(self.device)
        model_id = "peft-internal-testing/tiny-random-LlamaForCausalLM"
        config = LoraConfig(init_lora_weights=False, use_dora=True)
        with hub_online_once(model_id):
            model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
            self.check_dora_caching(model, config, inputs)

    def test_dora_caching_embedding(self):
        # ensure that the results don't change due to caching
        inputs = torch.arange(10).view(1, -1).to(self.device)
        model_id = "peft-internal-testing/tiny-random-LlamaForCausalLM"
        config = LoraConfig(init_lora_weights=False, use_dora=True, target_modules=["model.embed_tokens"])
        with hub_online_once(model_id):
            model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
            self.check_dora_caching(model, config, inputs)

    def test_dora_caching_conv(self):
        # ensure that the results don't change due to caching
        # note: don't use something like small resnet, because batch norm affects outputs in train mode

        class ModelConv2D(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv0 = nn.Conv2d(3, 5, kernel_size=3, stride=1, padding=1)
                self.conv1 = nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=1)
                self.linear = nn.Linear(5 * 3 * 3, 10)

            def forward(self, X):
                X = self.conv0(X)
                X = nn.functional.relu(X)
                X = self.conv1(X)
                X = nn.functional.relu(X)
                X = X.view(X.size(0), -1)
                X = self.linear(X)
                return X

        inputs = torch.randn(1, 3, 3, 3).to(self.device)
        config = LoraConfig(init_lora_weights=False, use_dora=True, target_modules=["conv0", "conv1"])
        model = ModelConv2D().to(self.device)
        self.check_dora_caching(model, config, inputs)

    def check_dora_caching(self, model, config, inputs):
        atol, rtol = 1e-6, 1e-6

        # BASE RESULT
        base_result = self.get_output(model, inputs)

        # DEFAULT: WITHOUT DoRA CACHING
        model = get_peft_model(model, config)
        caches = self.get_caches(model)
        dora_result = self.get_output(model, inputs)

        # sanity check: the results should be different
        assert not torch.allclose(base_result, dora_result, atol=atol, rtol=rtol)
        # ensure that there are dora caches but they're all empty
        assert caches
        assert not any(cache for cache in caches)

        # ENABLE DORA CACHING
        model.eval()
        with DoraCaching():
            cached_result = self.get_output(model, inputs)
            # the caches should be populated now
            assert all(cache for cache in caches)
        # the results should be the same
        assert torch.allclose(cached_result, dora_result, atol=atol, rtol=rtol)

        # AFTER EXITING THE CONTEXT
        cached_result_after_context = self.get_output(model, inputs)
        assert torch.allclose(cached_result_after_context, dora_result, atol=atol, rtol=rtol)
        # since we called forward outside of the context, the caches should be cleared
        assert not any(cache for cache in caches)

        # NO CACHING IN TRAIN MODE
        model.train()
        # switching to train model immediately clears the caches
        assert not any(cache for cache in caches)
        with DoraCaching():
            results_train_mode = self.get_output(model, inputs)
            # the caches should still be empty
            assert not any(cache for cache in caches)
        # results should not change
        assert torch.allclose(results_train_mode, dora_result, atol=atol, rtol=rtol)
        # still not any caches expected
        assert not any(cache for cache in caches)

        # PERMANENTLY ENABLE DORA CACHING
        DoraCaching()(enabled=True)
        model.eval()
        # putting the model in eval mode clears the caches
        assert not any(cache for cache in caches)
        # the results should be the same
        cached_result_permanent = self.get_output(model, inputs)
        assert torch.allclose(cached_result_permanent, dora_result, atol=atol, rtol=rtol)
        DoraCaching()(enabled=False)


class _TinyLinearModel(nn.Module):
    """Tiny module containing several Linear layers with common projection names."""

    def __init__(self, d=16):
        super().__init__()
        # Common Transformer projection names
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.gate_proj = nn.Linear(d, d, bias=False)
        self.up_proj = nn.Linear(d, d, bias=False)
        self.down_proj = nn.Linear(d, d, bias=False)


@pytest.fixture
def small_model():
    """
    Small CPU-only model fixture for KappaTuneSelector tests.

    Keeping this tiny ensures it runs quickly on CPU-only CI.
    """
    torch.manual_seed(0)
    return _TinyLinearModel(d=16)


class TestKappaTuneSelector:
    """Tests for KappaTuneSelector and find_kappa_target_modules helper."""

    def test_find_kappa_target_modules_returns_dict(self, small_model):
        """Test the new return format of find_kappa_target_modules."""
        from peft.helpers import find_kappa_target_modules

        result = find_kappa_target_modules(small_model, top_p=0.5)

        assert isinstance(result, dict)
        assert "target_modules" in result
        assert "target_parameters" in result
        assert isinstance(result["target_modules"], list)
        assert result["target_parameters"] is None

    def test_find_kappa_target_modules_selects_modules(self, small_model):
        """Basic functionality test on regular nn.Linear layers."""
        from peft.helpers import find_kappa_target_modules

        result = find_kappa_target_modules(small_model, top_p=0.3)

        assert len(result["target_modules"]) > 0
        # All returned modules should exist in the model
        for name in result["target_modules"]:
            assert any(name in module_name for module_name, _ in small_model.named_modules())

    def test_kappatune_with_moe_layers(self):
        """Test support for fused MoE 3D parameters (target_parameters)."""
        import torch
        from torch import nn

        from peft.helpers import KappaTuneSelector, find_kappa_target_modules

        # Create a minimal dummy MoE model with fused 3D weights
        class DummyMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_up_proj = nn.Parameter(torch.randn(2, 16, 32))
                self.down_proj = nn.Parameter(torch.randn(2, 32, 16))

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([DummyMoE() for _ in range(2)])

        model = DummyModel()

        # Test selector directly
        selector = KappaTuneSelector(model)
        target_params = selector.get_best_target_parameters(top_p=0.5)
        assert len(target_params) > 0
        assert any("gate_up_proj" in name or "down_proj" in name for name in target_params)

        # Test convenience function
        result = find_kappa_target_modules(model, top_p=0.5)
        assert isinstance(result["target_parameters"], list)
        assert len(result["target_parameters"]) > 0
        assert any("gate_up_proj" in name or "down_proj" in name for name in result["target_parameters"])


class TestMontecloraTrainerMixin:
    # TODO remove MontecloraTrainerMixin and this warning after Oct 2027
    def test_mixin_adds_variational_loss_to_trainer(self, tmp_path):
        class MontecloraTrainer(MontecloraTrainerMixin, Trainer):
            pass

        model_id = "peft-internal-testing/tiny-random-OPTForCausalLM"
        with hub_online_once(model_id):
            base_model = AutoModelForCausalLM.from_pretrained(model_id)
            config = LoraConfig(
                target_modules=["q_proj", "v_proj"],
                monteclora_config=MontecloraConfig(num_samples=2),
            )
            model = get_peft_model(base_model, config).train()
            assert any(isinstance(m, MontecloraSampler) for m in model.modules())

            training_args = TrainingArguments(
                output_dir=str(tmp_path),
                per_device_train_batch_size=2,
                num_train_epochs=1,
                save_strategy="no",
                report_to=[],
                use_cpu=True,
            )
            vanilla_trainer = Trainer(model=model, args=training_args)
            with pytest.warns(FutureWarning, match="MontecloraTrainerMixin is deprecated"):
                monteclora_trainer = MontecloraTrainer(model=model, args=training_args)

            torch.manual_seed(0)
            input_ids = torch.randint(0, base_model.config.vocab_size, (2, 8))
            inputs = {"input_ids": input_ids, "labels": input_ids.clone()}

            torch.manual_seed(0)
            task_loss = vanilla_trainer.compute_loss(model, inputs)
            torch.manual_seed(0)
            total_loss = monteclora_trainer.compute_loss(model, inputs)

            assert total_loss.item() != task_loss.item()
            total_loss.backward()


class TestPeftTrainerMixin:
    @pytest.fixture
    def base_model(self):
        model_id = "trl-internal-testing/tiny-random-LlamaForCausalLM"
        with hub_online_once(model_id):
            model = AutoModelForCausalLM.from_pretrained(model_id)
            yield model

    @pytest.fixture
    def inputs(self):
        input_ids = torch.arange(8).reshape(2, 4)
        return {"input_ids": input_ids, "labels": input_ids.clone()}

    def make_trainer(self, model, tmp_path, **kwargs):
        class PeftTrainer(PeftTrainerMixin, Trainer):
            pass

        args = TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            max_steps=3,
            learning_rate=0.01,
            max_grad_norm=0,
            optim="sgd",
            save_strategy="no",
            logging_strategy="no",
            report_to=[],
            disable_tqdm=True,
            use_cpu=True,
        )
        dataset = [{"input_ids": torch.arange(4), "labels": torch.arange(4)} for _ in range(6)]
        return PeftTrainer(model=model, args=args, train_dataset=dataset, **kwargs)

    @pytest.mark.parametrize("return_outputs", [False, True])
    @pytest.mark.parametrize("variant", ["lora", "kasa", "monteclora", "adalora"])
    def test_compute_loss(self, base_model, inputs, tmp_path, variant, return_outputs):
        kwargs = {}
        if variant == "kasa":
            kwargs["kasa_config"] = KasaConfig()
        elif variant == "monteclora":
            kwargs["monteclora_config"] = MontecloraConfig(num_samples=2)

        config = (
            AdaLoraConfig(target_modules=["q_proj"], total_step=3, task_type=TaskType.CAUSAL_LM)
            if variant == "adalora"
            else LoraConfig(target_modules=["q_proj"], r=2, **kwargs)
        )

        model = get_peft_model(base_model, config).train()
        trainer = self.make_trainer(model, tmp_path)
        result = trainer.compute_loss(model, inputs.copy(), return_outputs=return_outputs)
        loss, outputs = result if return_outputs else (result, None)
        assert loss.ndim == 0
        if return_outputs:
            torch.testing.assert_close(loss, outputs.loss + model.get_peft_loss())

        loss.backward()
        assert any(param.grad is not None for param in model.parameters() if param.requires_grad)

        model.eval()
        loss, outputs = trainer.compute_loss(model, inputs.copy(), return_outputs=True)
        torch.testing.assert_close(loss, outputs.loss)

    def test_get_peft_loss_kwargs_forwarded(self, base_model, monkeypatch):
        model = get_peft_model(base_model, LoraConfig(target_modules=["q_proj"]))
        context = object()
        result = torch.tensor(0.5, requires_grad=True)
        hook = Mock(return_value=result)
        monkeypatch.setattr(model.base_model, "get_peft_loss", hook)

        assert model.get_peft_loss(adapter_names="default", training_context=context) is result
        hook.assert_called_once_with(adapter_names="default", training_context=context)

    def test_on_peft_optimizer_step_kwargs_forwarded(self, base_model, monkeypatch):
        model = get_peft_model(base_model, LoraConfig(target_modules=["q_proj"]))
        context = object()
        result = None
        hook = Mock(return_value=result)
        monkeypatch.setattr(model.base_model, "on_peft_optimizer_step", hook)

        assert model.on_peft_optimizer_step(3, training_context=context) is None
        hook.assert_called_once_with(3, training_context=context)

    def test_plain_lora_ignores_training_context(self, base_model):
        model = get_peft_model(base_model, LoraConfig(target_modules=["q_proj"]))
        assert model.get_peft_loss(training_context=object()) == 0.0
        assert model.on_peft_optimizer_step(0, training_context=object()) is None

    @pytest.mark.parametrize("normalized", [False, True])
    @pytest.mark.parametrize("accumulation_steps", [1, 2])
    def test_accumulation_normalizes_auxiliary_gradients(
        self, base_model, inputs, tmp_path, normalized, accumulation_steps
    ):
        model = get_peft_model(
            base_model, LoraConfig(target_modules=["q_proj"], r=2, kasa_config=KasaConfig(beta=0.1, gamma=0.1))
        )
        trainer = self.make_trainer(model, tmp_path)
        # sanity check, as we rely in Trainer attributes here
        assert trainer.loss_is_scaled_for_ga is None
        assert trainer.model_accepts_loss_kwargs
        assert getattr(trainer, "current_gradient_accumulation_steps", None) is None

        trainer.loss_is_scaled_for_ga = normalized
        trainer.model_accepts_loss_kwargs = normalized
        trainer.current_gradient_accumulation_steps = accumulation_steps

        expected_loss = model.get_peft_loss()
        params = [param for param in model.parameters() if param.requires_grad]
        assert params
        expected_grads = torch.autograd.grad(expected_loss, params, allow_unused=True)

        # sequence length 4, shift by 1 -> 3, batch size 2 -> 6; None -> let the model calculate it
        num_items_in_batch = 3 * 2 * accumulation_steps if normalized else None
        for _ in range(accumulation_steps):
            # loss including PEFT loss
            loss = trainer.compute_loss(model, inputs.copy(), num_items_in_batch=num_items_in_batch)
            # original loss without PEFT contribution
            orig_loss = Trainer.compute_loss(trainer, model, inputs.copy(), num_items_in_batch=num_items_in_batch)
            assert not torch.allclose(loss, orig_loss)

            auxiliary_loss = loss - orig_loss
            if not normalized:
                auxiliary_loss = auxiliary_loss / accumulation_steps
            auxiliary_loss.backward()

        for param, expected in zip(params, expected_grads):
            if expected is not None:
                torch.testing.assert_close(param.grad, expected)

    @pytest.mark.parametrize("method", ["adalora", "adamss"])
    def test_updates_after_accumulation_before_zero_grad(self, base_model, tmp_path, monkeypatch, method):
        if method == "adalora":
            config = AdaLoraConfig(target_modules=["q_proj"], init_r=4, target_r=2, total_step=3)
        else:
            config = AdamssConfig(
                target_modules=["q_proj"],
                r=4,
                num_subspaces=2,
                subspace_rank=1,
                use_asa=True,
                asa_target_subspaces=1,
                init_warmup=0,
                final_warmup=10,
                mask_interval=10,
            )
        model = get_peft_model(base_model, config)
        trainer = self.make_trainer(model, tmp_path)
        steps = []
        update = model.on_peft_optimizer_step

        def record_update(step):
            assert any(param.grad is not None for param in model.parameters() if param.requires_grad)
            steps.append(step)
            update(step)

        monkeypatch.setattr(model, "on_peft_optimizer_step", record_update)
        trainer.train()
        assert steps == [0, 1, 2]
        if method == "adalora":
            assert model.rankallocator.exp_avg_ipt
        else:
            layers = [module for module in model.modules() if isinstance(module, AdamssLayer)]
            assert any(value is not None for layer in layers for value in layer.exp_avg_ipt_A["default"])

    def test_skipped_optimizer_update(self, base_model, tmp_path, monkeypatch):
        # When scaling the gradient leads to non-finite values, the gradient step is skipped. Check that AMP overflow
        # also skips AdaLoRA’s optimizer-step hook.
        model = get_peft_model(
            base_model, AdaLoraConfig(target_modules=["q_proj"], init_r=4, target_r=2, total_step=3)
        )
        trainer = self.make_trainer(model, tmp_path)
        trainer.accelerator.scaler = torch.amp.GradScaler("cpu")
        scale_before = trainer.accelerator.scaler.get_scale()
        steps = []
        update = model.on_peft_optimizer_step
        backward_calls = 0

        def overflow_first_backward(grad):
            nonlocal backward_calls
            backward_calls += 1
            return torch.full_like(grad, float("inf")) if backward_calls == 1 else grad

        def record_update(step):
            steps.append(step)
            update(step)

        parameter = next(param for name, param in model.named_parameters() if "lora_E" in name)
        parameter.register_hook(overflow_first_backward)
        monkeypatch.setattr(model, "on_peft_optimizer_step", record_update)
        trainer.train()

        assert steps == [1, 2]  # step 0 overflows
        assert trainer.accelerator.scaler.get_scale() < scale_before  # reduce scale on overlow
        assert all(torch.isfinite(value).all() for value in model.rankallocator.exp_avg_ipt.values())

    def test_legacy_callback_does_not_duplicate_updates(self, base_model, tmp_path, monkeypatch):
        # TODO remove AdamssAsaCallback and this test after Oct 2027
        model = get_peft_model(base_model, LoraConfig(target_modules=["q_proj"]))
        with pytest.warns(FutureWarning, match="AdamssAsaCallback is deprecated"):
            callback = AdamssAsaCallback()

        trainer = self.make_trainer(model, tmp_path, callbacks=[callback])
        update = Mock()
        monkeypatch.setattr(model, "on_peft_optimizer_step", update)
        trainer.train()
        assert update.call_count == 3

    def test_legacy_callback_added_after_initialization(self, base_model, tmp_path):
        model = get_peft_model(base_model, LoraConfig(target_modules=["q_proj"]))
        trainer = self.make_trainer(model, tmp_path)
        with pytest.warns(FutureWarning, match="already handles optimizer-step updates"):
            trainer.add_callback(AdamssAsaCallback)

        assert not any(isinstance(callback, AdamssAsaCallback) for callback in trainer.callback_handler.callbacks)
