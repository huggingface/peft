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

import os
from functools import wraps

import huggingface_hub
import pytest
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig, PeftType, TaskType, XLoraConfig, get_peft_model
from peft.peft_model import PeftModel
from peft.tuners.xlora.layer import XLoraLayer
from peft.utils import infer_device

from .testing_utils import hub_online_once


def flaky(num_tries: int):
    """Decorator for test functions that are flaky"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(num_tries):
                try:
                    return func(*args, **kwargs)
                except AssertionError as e:
                    print(f"Failed test {func.__name__} with error: {e}")
                    continue
            raise AssertionError(f"Failed test {func.__name__} after {num_tries} tries")

        return wrapper

    return decorator


class TestXlora:
    torch_device = infer_device()

    model_id = "peft-internal-testing/opt-125m"
    num_loras = 4

    @pytest.fixture(scope="class")
    def lora_dir(self, tmp_path_factory):
        return tmp_path_factory.mktemp("lora")

    @pytest.fixture(scope="class")
    def lora_embedding_dir(self, tmp_path_factory):
        return tmp_path_factory.mktemp("lora_embedding")

    @pytest.fixture(scope="class")
    def saved_lora_adapters(self, lora_dir):
        file_names = []

        lora_configs = [
            LoraConfig(task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"], init_lora_weights=False)
            for _ in range(self.num_loras)
        ]
        # have 1 LoRA with different target modules
        lora_configs[-1] = LoraConfig(
            task_type="CAUSAL_LM", target_modules=["k_proj", "q_proj", "v_proj"], init_lora_weights=False
        )

        for i, lora_config in enumerate(lora_configs, start=1):
            torch.manual_seed(i)
            model = AutoModelForCausalLM.from_pretrained(self.model_id)
            peft_model = get_peft_model(model, lora_config)
            file_name = os.path.join(lora_dir, f"checkpoint-{i}")
            peft_model.save_pretrained(file_name)
            file_names.append(file_name)
        return file_names

    @pytest.fixture(scope="class")
    def saved_lora_embedding_adapters(self, lora_embedding_dir):
        file_names = []
        for i in range(1, self.num_loras + 1):
            torch.manual_seed(i)
            lora_config = LoraConfig(task_type="CAUSAL_LM", init_lora_weights=False, target_modules=["embed_tokens"])
            model = AutoModelForCausalLM.from_pretrained(self.model_id)
            peft_model = get_peft_model(model, lora_config)
            file_name = os.path.join(lora_embedding_dir, f"checkpoint-{i}")
            peft_model.save_pretrained(file_name)
            file_names.append(file_name)
        return file_names

    @pytest.fixture(scope="class")
    def tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True, device_map=self.torch_device)
        return tokenizer

    @pytest.fixture(scope="function")
    def embedding_model(self, saved_lora_embedding_adapters):
        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        model.config.use_cache = False
        adapters = {str(i): file_name for i, file_name in enumerate(saved_lora_embedding_adapters)}

        peft_config = XLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            peft_type=PeftType.XLORA,
            hidden_size=model.config.hidden_size,
            xlora_depth=8,
            adapters=adapters,
        )
        model = get_peft_model(model, peft_config).to(self.torch_device)
        return model

    @pytest.fixture(scope="function")
    def model(self, saved_lora_adapters):
        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        model.config.use_cache = False
        adapters = {str(i): file_name for i, file_name in enumerate(saved_lora_adapters)}

        peft_config = XLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            peft_type=PeftType.XLORA,
            hidden_size=model.config.hidden_size,
            xlora_depth=8,
            adapters=adapters,
        )
        model = get_peft_model(model, peft_config).to(self.torch_device)
        return model

    @pytest.fixture(scope="function")
    def model_layerwise(self, saved_lora_adapters):
        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        model.config.use_cache = False
        adapters = {str(i): file_name for i, file_name in enumerate(saved_lora_adapters)}

        peft_config = XLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            peft_type=PeftType.XLORA,
            hidden_size=model.config.hidden_size,
            xlora_depth=8,
            adapters=adapters,
            layerwise_scalings=True,
        )
        model = get_peft_model(model, peft_config).to(self.torch_device)
        return model

    def test_functional(self, tokenizer, model):
        model.enable_scalings_logging()
        inputs = tokenizer.encode("Python is a", add_special_tokens=False, return_tensors="pt")
        outputs = model.generate(
            input_ids=inputs.to(self.torch_device),
            max_new_tokens=32,
        )
        assert torch.isfinite(outputs[: inputs.shape[1] :]).all()

    def test_forward_hooks_are_cleaned_up(self, tokenizer, model):
        # There was an issue that forward hooks would accumulate during generation, since one hook per forward step was
        # being registered and generate would call forward multiple times. This is already undesirable, but to make it
        # worse, only the last hook was removed, resulting in hooks accumulating.
        # See https://github.com/huggingface/peft/issues/1472#issuecomment-3235817807
        inputs = tokenizer.encode("Python is a", add_special_tokens=False, return_tensors="pt")
        model.generate(input_ids=inputs.to(self.torch_device), max_new_tokens=10)
        num_hooks_gen1 = len(model.base_model.model.model.decoder.layers[0].self_attn.k_proj._forward_pre_hooks)

        model.generate(input_ids=inputs.to(self.torch_device), max_new_tokens=10)
        num_hooks_gen2 = len(model.base_model.model.model.decoder.layers[0].self_attn.k_proj._forward_pre_hooks)
        assert num_hooks_gen1 == num_hooks_gen2 == 0

    def test_scalings_logging_methods(self, tokenizer, model):
        model.enable_scalings_logging()

        inputs = tokenizer.encode("Python is a", add_special_tokens=False, return_tensors="pt")
        outputs = model.generate(
            input_ids=inputs.to(self.torch_device),
            max_new_tokens=32,
        )
        assert torch.isfinite(outputs[: inputs.shape[1] :]).all()

        _ = model.get_latest_scalings()
        # 32 is the numeber of max scalings. 3 is the number of prompt tokens.
        assert 32 + 3 >= len(model.get_scalings_log()) > 0

        model.disable_scalings_logging()

        inputs = tokenizer.encode("Python is a", add_special_tokens=False, return_tensors="pt")
        outputs = model.generate(
            input_ids=inputs.to(self.torch_device),
            max_new_tokens=32,
        )
        assert torch.isfinite(outputs[: inputs.shape[1] :]).all()

        assert 32 >= len(model.get_scalings_log()) > 0

        bucketed = model.get_bucketed_scalings_log()
        keys = bucketed.keys()
        # Once bucket for each token as we aren't using cache
        assert len(bucketed) == 32 == len(keys)
        seq_len = inputs.shape[1]
        for key in keys:
            assert len(bucketed[key][0]) == 1
            assert len(bucketed[key][1]) == 1
            assert bucketed[key][0][0] == key - seq_len

        model.clear_scalings_log()
        assert len(model.get_scalings_log()) == 0

    def test_misc_methods(self, tokenizer, model):
        model.set_global_scaling_weight(1.5)
        assert model.internal_xlora_classifier.config.global_scaling_weight == 1.5
        assert model.get_global_scaling_weight() == 1.5

        inputs = tokenizer.encode("Python is a", add_special_tokens=False, return_tensors="pt")
        outputs = model.generate(
            input_ids=inputs.to(self.torch_device),
            max_new_tokens=32,
        )
        assert torch.isfinite(outputs[: inputs.shape[1] :]).all()

        assert str(model) is not None

    # On CI (but not locally), this test is flaky since transformers v4.45.0.
    @flaky(num_tries=5)
    def test_save_load_functional(self, tokenizer, model, tmp_path):
        inputs = tokenizer.encode("Python is a", add_special_tokens=False, return_tensors="pt")
        outputs = model.generate(
            input_ids=inputs.to(self.torch_device),
            max_new_tokens=32,
        )
        before_logits = outputs[: inputs.shape[1] :]
        assert torch.isfinite(before_logits).all()

        model.save_pretrained(save_directory=tmp_path)

        del model

        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        model.config.use_cache = False
        model = PeftModel.from_pretrained(model=model, model_id=tmp_path).to(self.torch_device)

        inputs = tokenizer.encode("Python is a", add_special_tokens=False, return_tensors="pt")
        outputs = model.generate(
            input_ids=inputs.to(self.torch_device),
            max_new_tokens=32,
        )
        after_logits = outputs[: inputs.shape[1] :]
        assert torch.isfinite(after_logits).all()
        assert torch.equal(after_logits, before_logits)

    def test_save_load_functional_pt(self, tokenizer, model, tmp_path):
        inputs = tokenizer.encode("Python is a", add_special_tokens=False, return_tensors="pt")
        outputs = model.generate(
            input_ids=inputs.to(self.torch_device),
            max_new_tokens=32,
        )
        before_logits = outputs[: inputs.shape[1] :]
        assert torch.isfinite(before_logits).all()

        model.save_pretrained(save_directory=tmp_path, safe_serialization=False)

        del model

        model = AutoModelForCausalLM.from_pretrained(self.model_id)
        model.config.use_cache = False
        model = PeftModel.from_pretrained(model=model, model_id=tmp_path, safe_serialization=False).to(
            self.torch_device
        )

        inputs = tokenizer.encode("Python is a", add_special_tokens=False, return_tensors="pt")
        outputs = model.generate(
            input_ids=inputs.to(self.torch_device),
            max_new_tokens=32,
        )
        after_logits = outputs[: inputs.shape[1] :]
        assert torch.isfinite(after_logits).all()
        assert torch.equal(after_logits, before_logits), (after_logits, before_logits)

    def test_topk_lora(self, tokenizer, model):
        model.set_topk_lora(2)
        assert model.internal_xlora_classifier.config.top_k_lora == 2

        inputs = tokenizer.encode("Python is a", add_special_tokens=False, return_tensors="pt")
        outputs = model.generate(
            input_ids=inputs.to(self.torch_device),
            max_new_tokens=32,
        )
        assert torch.isfinite(outputs[: inputs.shape[1] :]).all()

    def test_softmax_topk(self, tokenizer, model):
        # Just reach in to set the config
        model.internal_xlora_classifier.config.top_k_lora = 2
        model.internal_xlora_classifier.config.enable_softmax = False
        model.internal_xlora_classifier.config.enable_softmax_topk = True

        inputs = tokenizer.encode("Python is a", add_special_tokens=False, return_tensors="pt")
        outputs = model.generate(
            input_ids=inputs.to(self.torch_device),
            max_new_tokens=32,
        )
        assert torch.isfinite(outputs[: inputs.shape[1] :]).all()

    def test_set_override_scaling_pass_value(self, model):
        # Defaults to 0
        assert model.internal_xlora_classifier.override_scaling_pass_value == 0.0

        # Set it to 2 and make sure it actually is
        model.set_scaling_pass_value(2)
        assert model.internal_xlora_classifier.override_scaling_pass_value == 2
        assert model.internal_xlora_classifier.config.scaling_pass_value == 2

        # Set it to None and make sure it is 1/n
        model.set_scaling_pass_value(None)
        assert model.internal_xlora_classifier.override_scaling_pass_value == 1 / self.num_loras
        assert model.internal_xlora_classifier.config.scaling_pass_value == 1 / self.num_loras

    def test_functional_layerwise(self, tokenizer, model_layerwise):
        model_layerwise.enable_scalings_logging()
        inputs = tokenizer.encode("Python is a", add_special_tokens=False, return_tensors="pt")
        outputs = model_layerwise.generate(
            input_ids=inputs.to(self.torch_device),
            max_new_tokens=32,
        )
        assert torch.isfinite(outputs[: inputs.shape[1] :]).all()

    def test_disable_adapter(self, tokenizer, model):
        model.enable_scalings_logging()
        inputs = tokenizer.encode("Python is a", add_special_tokens=False, return_tensors="pt")
        with model.disable_adapter():
            outputs_disabled = model.generate(
                input_ids=inputs.to(self.torch_device),
                max_new_tokens=32,
            )
        outputs = model.generate(
            input_ids=inputs.to(self.torch_device),
            max_new_tokens=32,
        )
        assert torch.isfinite(outputs_disabled[: inputs.shape[1] :]).all()
        assert torch.isfinite(outputs[: inputs.shape[1] :]).all()
        assert not torch.equal(outputs, outputs_disabled)

    def test_functional_embedding(self, tokenizer, embedding_model):
        inputs = tokenizer.encode("Python is a", add_special_tokens=False, return_tensors="pt")
        outputs = embedding_model.generate(
            input_ids=inputs.to(self.torch_device),
            max_new_tokens=32,
        )
        assert torch.isfinite(outputs[: inputs.shape[1] :]).all()

    def test_xlora_loading_valid(self):
        # This test also simulatenously tests the loading-from-hub functionality!
        torch.manual_seed(123)

        model_id = "peft-internal-testing/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.config.use_cache = False

        adapters = [
            "peft-internal-testing/opt-125m-dummy-lora",
            "peft-internal-testing/opt-125m-dummy-lora",
        ]
        adapters = {str(i): file_name for i, file_name in enumerate(adapters)}

        peft_config = XLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            peft_type=PeftType.XLORA,
            hidden_size=model.config.hidden_size,
            adapters=adapters,
            xlora_depth=8,
            xlora_size=2048,
            layerwise_scalings=True,
            xlora_dropout_p=0.2,
        )
        model = get_peft_model(model, peft_config)

        downloaded = huggingface_hub.hf_hub_download(repo_id=adapters["0"], filename="adapter_model.safetensors")
        sd = load_file(downloaded)
        w0 = model.base_model.model.model.decoder.layers[0].self_attn.q_proj.lora_A["0"].weight
        w1 = sd["base_model.model.model.decoder.layers.0.self_attn.q_proj.lora_A.weight"]

        assert torch.allclose(w0, w1)

    def test_scalings_storage(self, tokenizer, model):
        model.enable_scalings_logging()
        inputs = tokenizer.encode("Python is a", add_special_tokens=False, return_tensors="pt")
        outputs = model.generate(
            input_ids=inputs.to(self.torch_device),
            max_new_tokens=10,
        )

        latest_scalings = model.get_latest_scalings()
        assert latest_scalings is not None, "get_latest_scalings() should not return None after generation"
        assert isinstance(latest_scalings, torch.Tensor)
        assert torch.isfinite(latest_scalings).all(), "Scalings should contain finite values"

    def test_per_token_normalization_with_softmax_topk(self, tokenizer, model, monkeypatch):
        model.internal_xlora_classifier.config.top_k_lora = 2
        model.internal_xlora_classifier.config.enable_softmax = False
        model.internal_xlora_classifier.config.enable_softmax_topk = True

        captured_data = []
        orig_get_maybe_topk_scalings = XLoraLayer.get_maybe_topk_scalings

        def mock_get_maybe_topk_scalings(self, scalings):
            result = orig_get_maybe_topk_scalings(self, scalings)
            if getattr(model, "internal_xlora_scalings", None) is not None:
                captured_data.append(result)
            return result

        monkeypatch.setattr(XLoraLayer, "get_maybe_topk_scalings", mock_get_maybe_topk_scalings)

        model.enable_scalings_logging()
        inputs = tokenizer.encode("Test per token normalization", add_special_tokens=False, return_tensors="pt")
        outputs = model.generate(
            input_ids=inputs.to(self.torch_device),
            max_new_tokens=1,
        )

        for scaling in captured_data:
            weight_sums = scaling.sum(dim=-1)
            assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), (
                "Per-token scaling weights are not normalized to sum to 1."
            )

    def test_xlora_embed_scale_is_applied(self, tmp_path):
        """Test that X-LoRA correctly handles embeddings with scaling (e.g., Gemma3)."""
        model_id = "hf-internal-testing/tiny-random-Gemma3ForCausalLM"
        with hub_online_once(model_id):
            # Create and save Gemma3-compatible LoRA adapters
            adapters = {}
            for i in range(2):
                torch.manual_seed(i + 1)
                lora_config = LoraConfig(
                    task_type="CAUSAL_LM", init_lora_weights=False, target_modules=["embed_tokens"]
                )
                model = AutoModelForCausalLM.from_pretrained(model_id)
                peft_model = get_peft_model(model, lora_config)
                adapter_path = os.path.join(tmp_path, f"checkpoint-{i + 1}")
                peft_model.save_pretrained(adapter_path)
                adapters[str(i)] = adapter_path

            # Load base model and test X-LoRA with embed_scale
            base_model = AutoModelForCausalLM.from_pretrained(model_id).to(self.torch_device)
            base_model.config.use_cache = False
            orig_embedding = base_model.get_input_embeddings()

            xlora_config = XLoraConfig(
                task_type=TaskType.CAUSAL_LM,
                hidden_size=base_model.config.hidden_size,
                adapters=adapters,
            )
            xlora_model = get_peft_model(base_model, xlora_config)

            x = torch.arange(10).to(self.torch_device)
            xlora_embedding = xlora_model.base_model.model.get_input_embeddings()
            max_embedding_output = xlora_embedding(x).abs().max(0)[0]
            assert (max_embedding_output < 100.0).all()

            # set embed_scale to an absurdly high value, then check that the embedding output is also scaled to a high
            # value
            orig_embedding.embed_scale.fill_(10000.0)
            max_embedding_output = xlora_embedding(x).abs().max(0)[0]
            assert (max_embedding_output > 100.0).all()

            # set embed_scale to zero, then check that the embedding output is also zero
            orig_embedding.embed_scale.fill_(0)
            embedding_output = xlora_embedding(x)
            assert (embedding_output == 0.0).all()
