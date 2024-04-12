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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig, PeftType, TaskType, XLoraConfig, get_peft_model
import pytest


class TestXlora:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_id = "facebook/opt-125m"
    num_loras = 4

    @pytest.fixture(scope="class")
    def tmp_dir(self, tmp_path_factory):
        # create a class-scoped temp directory
        return tmp_path_factory.mktemp("xlora")

    @pytest.fixture(scope="class")
    def saved_lora_adapters(self, tmp_dir):
        file_names = []
        for i in range(1, self.num_loras + 1):
            torch.manual_seed(i)
            lora_config = LoraConfig(task_type="CAUSAL_LM", init_lora_weights=False)
            model = AutoModelForCausalLM.from_pretrained(self.model_id)
            peft_model = get_peft_model(model, lora_config)
            file_name = os.path.join(tmp_dir, f"checkpoint-{i}")
            peft_model.save_pretrained(file_name)
            file_names.append(file_name)
        return file_names

    def test_functional(self, saved_lora_adapters):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True, device_map=self.device)

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
        model = get_peft_model(model, peft_config).to(self.device)

        model.enable_scalings_logging()
        inputs = tokenizer.encode("Python is a", add_special_tokens=False, return_tensors="pt")
        outputs = model.generate(
            input_ids=inputs.to("cuda"),
            max_new_tokens=32,
        )
        text = tokenizer.batch_decode(outputs[: inputs.shape[1] :].detach().cpu().numpy(), skip_special_tokens=True)
        # TODO: do any check on the text?

    def test_scalings_logging_methods(self, saved_lora_adapters):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True, device_map="cuda:0")

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
        model = get_peft_model(model, peft_config).to("cuda")

        model.enable_scalings_logging()

        inputs = tokenizer.encode("Python is a", add_special_tokens=False, return_tensors="pt")
        outputs = model.generate(
            input_ids=inputs.to("cuda"),
            max_new_tokens=32,
        )
        text = tokenizer.batch_decode(outputs[: inputs.shape[1] :].detach().cpu().numpy(), skip_special_tokens=True)
        print(text[0])

        _ = model.get_latest_scalings()
        assert 32 >= len(model.get_scalings_log()) > 0

        model.disable_scalings_logging()

        inputs = tokenizer.encode("Python is a", add_special_tokens=False, return_tensors="pt")
        outputs = model.generate(
            input_ids=inputs.to("cuda"),
            max_new_tokens=32,
        )
        text = tokenizer.batch_decode(outputs[: inputs.shape[1] :].detach().cpu().numpy(), skip_special_tokens=True)
        print(text[0])

        assert 32 >= len(model.get_scalings_log()) > 0

        model.clear_scalings_log()
        assert len(model.get_scalings_log()) == 0

        bucketed = model.get_bucketed_scalings_log()
        keys = bucketed.keys()
        assert len(bucketed) == 2
        # One bucket for prompt (which has 1 elem)
        assert len(bucketed[max(keys)][0]) == 1
        assert len(bucketed[max(keys)][1]) == 1
        assert bucketed[max(keys)][0][0] == 0
        # One bucket for completions with bucket name 1
        assert len(bucketed[1][0]) > 1
        assert len(bucketed[1][1]) > 1
        assert bucketed[1][0][0] > 0

    def test_misc_methods(self, saved_lora_adapters):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True, device_map="cuda:0")

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
        model = get_peft_model(model, peft_config).to("cuda")

        model.set_global_scaling_weight(1.5)
        assert model.internal_xlora_classifier.config.global_scaling_weight == 1.5
        assert model.get_global_scaling_weight() == 1.5

        inputs = tokenizer.encode("Python is a", add_special_tokens=False, return_tensors="pt")
        outputs = model.generate(
            input_ids=inputs.to("cuda"),
            max_new_tokens=32,
        )
        text = tokenizer.batch_decode(outputs[: inputs.shape[1] :].detach().cpu().numpy(), skip_special_tokens=True)
        print(text[0])

        model.set_use_trainable_adapters(True)
        assert model.xlora_config.use_trainable_adapters

        model.set_use_trainable_adapters(False)
        model.get_use_trainable_adapters()
        assert not model.xlora_config.use_trainable_adapters

        assert str(model) is not None
