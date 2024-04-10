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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

from peft import LoraConfig, PeftType, TaskType, XLoraConfig, get_peft_model

model_id = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, device_map="cuda:0")
tmp_path = "/tmp/xlora_test"
os.makedirs(tmp_path, exist_ok=True)

for i in range(1, 9):
    torch.manual_seed(i)
    lora_config = LoraConfig(task_type="CAUSAL_LM", init_lora_weights=False)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    peft_model = get_peft_model(model, lora_config)
    peft_model.save_pretrained(f"{tmp_path}/checkpoint-{i}")
    print(f"finished {i} of 8")


class TestXlora:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_functional(self):
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, device_map="cuda:0")

        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.config.use_cache = False
        adapters = {str(i): f"{tmp_path}/checkpoint-{i}" for i in range(1, 9)}

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

    def test_scalings_logging_methods(self):
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, device_map="cuda:0")

        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.config.use_cache = False
        adapters = {str(i): f"{tmp_path}/checkpoint-{i}" for i in range(1, 9)}

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

    def test_scalings_flush(self):
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, device_map="cuda:0")

        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.config.use_cache = False
        adapters = {str(i): f"{tmp_path}/checkpoint-{i}" for i in range(1, 9)}

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

        model.flush_log_scalings("output")  # writes to output.npy and a json file

    def test_misc_methods(self):
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, device_map="cuda:0")

        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.config.use_cache = False
        adapters = {str(i): f"{tmp_path}/checkpoint-{i}" for i in range(1, 9)}

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

        assert str(model) != None
