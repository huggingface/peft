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

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig, PeftType, TaskType, XLoraConfig, get_peft_model
from peft.peft_model import PeftModel
from peft.utils import infer_device


class TestXlora:
    torch_device = infer_device()

    model_id = "facebook/opt-125m"
    num_loras = 4

    @pytest.fixture(scope="function")
    def lora_dir(self, tmp_path_factory):
        return tmp_path_factory.mktemp("lora")

    @pytest.fixture(scope="function")
    def lora_embedding_dir(self, tmp_path_factory):
        return tmp_path_factory.mktemp("lora_embedding")

    @pytest.fixture(scope="function")
    def saved_lora_adapters(self, lora_dir):
        file_names = []
        for i in range(1, self.num_loras + 1):
            torch.manual_seed(i)
            lora_config = LoraConfig(task_type="CAUSAL_LM", init_lora_weights=False)
            model = AutoModelForCausalLM.from_pretrained(self.model_id)
            peft_model = get_peft_model(model, lora_config)
            file_name = os.path.join(lora_dir, f"checkpoint-{i}")
            peft_model.save_pretrained(file_name)
            file_names.append(file_name)
        return file_names

    @pytest.fixture(scope="function")
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

    @pytest.fixture(scope="function")
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
        # One bucket for prompt (seqlen=...) and one for the completion (seqlen=1)
        assert len(bucketed) == 2
        # One bucket for prompt (which has 1 elem)
        assert len(bucketed[max(keys)][0]) == 1
        assert len(bucketed[max(keys)][1]) == 1
        assert bucketed[max(keys)][0][0] == 0
        # One bucket for completions with bucket name 1
        assert len(bucketed[1][0]) > 1
        assert len(bucketed[1][1]) > 1
        assert bucketed[1][0][0] > 0

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

        # Set it to 2 and make sure it is 1/a
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
