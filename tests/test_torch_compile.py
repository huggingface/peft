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

# The intent of the tests contained in this file is to check as many PEFT features as possible with torch.compile. This
# is thus a document on how well torch.compile is supported by PEFT. Currently, we know that certain features do not
# work with torch.compile. The corresponding tests should be marked with `@pytest.mark.xfail(strict=True)`.
#
# When adding a new test that fails with torch.compile, please make sure first that it does NOT fail without
# torch.compile.

import gc
import os

import pytest
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from peft import (
    AdaLoraConfig,
    BOFTConfig,
    HRAConfig,
    IA3Config,
    LNTuningConfig,
    LoHaConfig,
    LoKrConfig,
    LoraConfig,
    OFTConfig,
    PeftModel,
    TaskType,
    VeraConfig,
    get_peft_model,
)

from .testing_utils import require_bitsandbytes


# only run (very slow) torch.compile tests when explicitly asked to
if os.environ.get("PEFT_DEBUG_WITH_TORCH_COMPILE") != "1":
    pytest.skip(allow_module_level=True)


# Mapping: name of the setting -> (Peft config instance, torch.compile kwargs)
SETTINGS = {
    "adalora": (AdaLoraConfig(task_type=TaskType.CAUSAL_LM), {}),
    "boft": (BOFTConfig(task_type=TaskType.CAUSAL_LM), {}),
    "dora": (LoraConfig(task_type=TaskType.CAUSAL_LM, use_dora=True), {}),
    "ia3": (IA3Config(task_type=TaskType.CAUSAL_LM), {}),
    "ln_tuning": (LNTuningConfig(task_type=TaskType.CAUSAL_LM, target_modules=["final_layer_norm"]), {}),
    "loha": (LoHaConfig(task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "v_proj"]), {}),
    "lokr": pytest.param(
        (LoKrConfig(task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "v_proj"]), {}),
        marks=pytest.mark.xfail(strict=True),
    ),
    "lora": (LoraConfig(task_type=TaskType.CAUSAL_LM), {}),
    "lora-target-embeddings": pytest.param(
        (LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=["embed_tokens"]), {}),
        marks=pytest.mark.xfail(strict=True),
    ),
    "lora-with-modules-to-save": (LoraConfig(task_type=TaskType.CAUSAL_LM, modules_to_save=["embed_tokens"]), {}),
    "oft": (OFTConfig(task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "v_proj"]), {}),
    "vera": (VeraConfig(task_type=TaskType.CAUSAL_LM), {}),
    "hra": (HRAConfig(task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "v_proj"]), {}),
}


@pytest.mark.single_gpu_tests
class TestTorchCompileCausalLM:
    """
    Tests for using torch.compile with causal LM.

    Tip: When adding a new test, set `fake_compile = False` below. With this setting, torch.compile is being skipped.
    This is useful for two reasons:

    - compile is slow, so to quickly iterate on the test, it's best to disable it and only enable it at the very end
    - even if you expect the test to fail with compile, as compile does not work with every PEFT feature, it still MUST
      succeed without compile, otherwise the test is incorrect.

    Before creating the PR, disable `fake_compile`.
    """

    fake_compile = False
    model_id = "hf-internal-testing/tiny-random-OPTForCausalLM"
    max_train_loss = 15.0  # generous threshold for maximum loss after training

    @pytest.fixture(autouse=True)
    def teardown(self):
        r"""
        Efficient mechanism to free GPU memory after each test. Based on
        https://github.com/huggingface/transformers/issues/21094
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_id)

    @pytest.fixture(scope="class")
    def data(self, tokenizer):
        def tokenize(samples):
            # For some reason, the max sequence length is not honored by the tokenizer, resulting in IndexErrors. Thus,
            # manually ensure that sequences are not too long.
            tokenized = tokenizer(samples["quote"])
            tokenized["input_ids"] = [input_ids[: tokenizer.model_max_length] for input_ids in tokenized["input_ids"]]
            tokenized["attention_mask"] = [
                input_ids[: tokenizer.model_max_length] for input_ids in tokenized["attention_mask"]
            ]
            return tokenized

        data = load_dataset("ybelkada/english_quotes_copy")
        data = data.map(tokenize, batched=True)
        # We need to manually remove unused columns. This is because we cannot use remove_unused_columns=True in the
        # Trainer, as this leads to errors with torch.compile. We also cannot just leave them in, as they contain
        # strings. Therefore, manually remove all unused columns.
        data = data.remove_columns(["quote", "author", "tags"])
        return data

    def compile(self, model, compile_kwargs):
        compile_kwargs = compile_kwargs.copy()
        # those are only for the Trainer arguments
        compile_kwargs.pop("torch_compile_backend", None)
        compile_kwargs.pop("torch_compile_mode", None)
        if self.fake_compile:
            return model
        return torch.compile(model, **compile_kwargs)

    @pytest.mark.parametrize("settings", SETTINGS.values(), ids=SETTINGS.keys())
    def test_causal_lm_training_trainer_compile(self, settings, tokenizer, data, tmp_path):
        r"""Train a PEFT model with torch.compile using Trainer"""
        tmp_dir = tmp_path / "model"
        config, compile_kwargs = settings
        if isinstance(config, AdaLoraConfig):
            pytest.skip(reason="AdaLora does not work correctly with Trainer")

        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
        )
        model = get_peft_model(model, config)

        # record outputs before training
        model.eval()
        sample = torch.tensor(data["train"][:1]["input_ids"]).to(model.device)
        with torch.inference_mode():
            output_before = model(sample)
        model.train()

        train_kwargs = {
            "per_device_train_batch_size": 4,
            "max_steps": 5,
            "learning_rate": 1e-3,
            "logging_steps": 1,
            "output_dir": tmp_dir,
            "seed": 0,
        }
        training_args = TrainingArguments(
            torch_compile=not self.fake_compile,
            torch_compile_backend=compile_kwargs.get("torch_compile_backend", None),
            torch_compile_mode=compile_kwargs.get("torch_compile_mode", None),
            **train_kwargs,
        )
        trainer = Trainer(
            model=model,
            train_dataset=data["train"],
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        model.config.use_cache = False
        trainer.train()

        model.eval()
        atol, rtol = 1e-4, 1e-4
        with torch.inference_mode():
            output_after = model(sample)
            tokens_after = model.generate(sample)
        assert torch.isfinite(output_after.logits).all()
        # sanity check: model was updated
        assert not torch.allclose(output_before.logits, output_after.logits, atol=atol, rtol=rtol)
        assert trainer.state.log_history[-1]["train_loss"] < self.max_train_loss

        # check saving the model and loading it without compile
        model.save_pretrained(tmp_path)
        del model
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="auto")
        model = PeftModel.from_pretrained(model, tmp_path)
        with torch.inference_mode():
            output_loaded = model(sample)
            tokens_loaded = model.generate(sample)
        assert torch.allclose(output_after.logits, output_loaded.logits, atol=atol, rtol=rtol)
        assert (tokens_after == tokens_loaded).all()

    @pytest.mark.parametrize("settings", SETTINGS.values(), ids=SETTINGS.keys())
    def test_causal_lm_training_pytorch_compile(self, settings, tokenizer, data, tmp_path):
        r"""Train a PEFT model with torch.compile using PyTorch training loop"""
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
        )
        config, compile_kwargs = settings
        model = get_peft_model(model, config)
        if isinstance(config, AdaLoraConfig):
            model.base_model.peft_config["default"].total_step = 5
        model = self.compile(model, compile_kwargs)

        # record outputs before training
        model.eval()
        sample = torch.tensor(data["train"][:1]["input_ids"]).to(model.device)
        with torch.inference_mode():
            output_before = model(sample)
        model.train()

        model.config.use_cache = False
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batch_size = 4
        losses = []
        max_steps = 5 * batch_size
        for i in range(0, max_steps, batch_size):
            batch = tokenizer.pad(data["train"][i : i + batch_size], return_tensors="pt").to(model.device)
            # add targets
            batch["labels"] = batch["input_ids"].clone()
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if isinstance(config, AdaLoraConfig):
                model.base_model.update_and_allocate(i)

        model.eval()
        with torch.inference_mode():
            output_after = model(sample)
            tokens_after = model.generate(sample)
        assert torch.isfinite(output_after.logits).all()
        atol, rtol = 1e-4, 1e-4
        # sanity check: model was updated
        assert not torch.allclose(output_before.logits, output_after.logits, atol=atol, rtol=rtol)
        assert losses[-1] < self.max_train_loss

        # check saving the model and loading it without compile
        model.save_pretrained(tmp_path)
        del model
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="auto")
        model = PeftModel.from_pretrained(model, tmp_path)
        with torch.inference_mode():
            output_loaded = model(sample)
            tokens_loaded = model.generate(sample)
        assert torch.allclose(output_after.logits, output_loaded.logits, atol=atol, rtol=rtol)
        assert (tokens_after == tokens_loaded).all()

    @require_bitsandbytes
    @pytest.mark.xfail(strict=True)
    def test_causal_lm_training_lora_bnb_compile(self, tokenizer, data, tmp_path):
        r"""Train a bnb quantized LoRA model with torch.compile using PyTorch training loop"""
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        )
        config = LoraConfig(task_type=TaskType.CAUSAL_LM)
        model = get_peft_model(model, config)
        model = self.compile(model, {})

        # record outputs before training
        model.eval()
        sample = torch.tensor(data["train"][:1]["input_ids"]).to(model.device)
        with torch.inference_mode():
            output_before = model(sample)
        model.train()

        model.config.use_cache = False
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batch_size = 4
        losses = []
        max_steps = 5 * batch_size
        for i in range(0, max_steps, batch_size):
            batch = tokenizer.pad(data["train"][i : i + batch_size], return_tensors="pt").to(model.device)
            # add targets
            batch["labels"] = batch["input_ids"].clone()
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        with torch.inference_mode():
            output_after = model(sample)
        assert torch.isfinite(output_after.logits).all()
        atol, rtol = 1e-4, 1e-4
        # sanity check: model was updated
        assert not torch.allclose(output_before.logits, output_after.logits, atol=atol, rtol=rtol)
        assert losses[-1] < self.max_train_loss

        # check saving the model and loading it without compile
        model.save_pretrained(tmp_path)
        del model
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, device_map="auto", quantization_config=BitsAndBytesConfig(load_in_4bit=True)
        )
        model = PeftModel.from_pretrained(model, tmp_path)

        with torch.inference_mode():
            # after loading, outputs are float32 for some reason
            output_loaded = model(sample)
        assert torch.allclose(output_after.logits, output_loaded.logits, atol=atol, rtol=rtol)

    @pytest.mark.xfail(strict=True)
    @require_bitsandbytes
    def test_causal_lm_multiple_lora_adapter_compile(self, tokenizer, data):
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        ).eval()
        sample = torch.tensor(data["train"][:1]["input_ids"]).to(model.device)
        with torch.inference_mode():
            output_base = model(sample)

        config = LoraConfig(task_type=TaskType.CAUSAL_LM, init_lora_weights=False)
        model = get_peft_model(model, config).eval()
        model = self.compile(model, {})
        model.add_adapter("other", config)
        model = self.compile(model, {})

        with torch.inference_mode():
            output_default_adapter = model(sample)
        model.set_adapter("other")
        with torch.inference_mode():
            output_other_adapter = model(sample)

        atol, rtol = 1e-4, 1e-4
        # outputs of the base model != output of default adapter != output of other adapter
        assert not torch.allclose(output_base.logits, output_default_adapter.logits, atol=atol, rtol=rtol)
        assert not torch.allclose(output_base.logits, output_other_adapter.logits, atol=atol, rtol=rtol)
        assert not torch.allclose(output_default_adapter.logits, output_other_adapter.logits, atol=atol, rtol=rtol)

        # now delete the other adapter
        model.delete_adapter("other")
        model.set_adapter("default")
        with torch.inference_mode():
            output_after_delete = model(sample)

        # outputs after delete == output of default adapter
        assert torch.allclose(output_default_adapter.logits, output_after_delete.logits, atol=atol, rtol=rtol)

    @pytest.mark.xfail(strict=True)
    def test_causal_lm_disable_lora_adapter_compile(self, tokenizer, data):
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        ).eval()
        sample = torch.tensor(data["train"][:1]["input_ids"]).to(model.device)
        with torch.inference_mode():
            output_base = model(sample)

        config = LoraConfig(task_type=TaskType.CAUSAL_LM, init_lora_weights=False)
        model = get_peft_model(model, config).eval()
        model = self.compile(model, {})
        output_lora = model(sample)

        with model.disable_adapter():
            with torch.inference_mode():
                output_disabled = model(sample)

        atol, rtol = 1e-4, 1e-4
        # outputs of the base model == output disabled adapter != output of lora adapter
        assert torch.allclose(output_base.logits, output_disabled.logits, atol=atol, rtol=rtol)
        assert not torch.allclose(output_base.logits, output_lora.logits, atol=atol, rtol=rtol)

    @require_bitsandbytes
    def test_causal_lm_merging_lora_adapter_compile(self, tokenizer, data):
        # merge the adapter
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        ).eval()
        sample = torch.tensor(data["train"][:1]["input_ids"]).to(model.device)
        with torch.inference_mode():
            output_base = model(sample)

        config = LoraConfig(task_type=TaskType.CAUSAL_LM, init_lora_weights=False)
        model = get_peft_model(model, config).eval()
        with torch.inference_mode():
            output_lora = model(sample)

        model.merge_adapter()
        with torch.inference_mode():
            output_merged = model(sample)

        # merging is less precise, be more tolerant
        atol, rtol = 1e-1, 1e-1
        # outputs of the base model != output of lora adapter == output of merged adapter
        assert not torch.allclose(output_base.logits, output_lora.logits, atol=atol, rtol=rtol)
        assert torch.allclose(output_lora.logits, output_merged.logits, atol=atol, rtol=rtol)

    @require_bitsandbytes
    def test_causal_lm_merging_multiple_lora_adapters_compile(self, tokenizer, data):
        # merge multiple adapters at once
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        ).eval()
        sample = torch.tensor(data["train"][:1]["input_ids"]).to(model.device)
        with torch.inference_mode():
            output_base = model(sample)

        config = LoraConfig(task_type=TaskType.CAUSAL_LM, init_lora_weights=False)
        model = get_peft_model(model, config).eval()
        model.add_adapter("other", config)
        with torch.inference_mode():
            output_default = model(sample)

        model.set_adapter("other")
        with torch.inference_mode():
            output_other = model(sample)

        model.base_model.merge_adapter(["default", "other"])
        with torch.inference_mode():
            output_merged = model(sample)

        # merging is less precise, be more tolerant
        atol, rtol = 1e-1, 1e-1
        # outputs of the base model != output of default adapter != output of other adapter
        assert not torch.allclose(output_base.logits, output_default.logits, atol=atol, rtol=rtol)
        assert not torch.allclose(output_base.logits, output_other.logits, atol=atol, rtol=rtol)
        assert not torch.allclose(output_default.logits, output_other.logits, atol=atol, rtol=rtol)
        # outputs of merged adapter != all others
        assert not torch.allclose(output_base.logits, output_merged.logits, atol=atol, rtol=rtol)
        assert not torch.allclose(output_default.logits, output_merged.logits, atol=atol, rtol=rtol)
        assert not torch.allclose(output_other.logits, output_merged.logits, atol=atol, rtol=rtol)

    @require_bitsandbytes
    @pytest.mark.xfail(strict=True)
    def test_causal_lm_merge_and_unload_lora_adapter_compile(self, tokenizer, data):
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        ).eval()
        sample = torch.tensor(data["train"][:1]["input_ids"]).to(model.device)
        with torch.inference_mode():
            output_base = model(sample)

        config = LoraConfig(task_type=TaskType.CAUSAL_LM, init_lora_weights=False)
        model = get_peft_model(model, config).eval()
        model = self.compile(model, {})
        with torch.inference_mode():
            output_lora = model(sample)

        unloaded = model.merge_and_unload()
        with torch.inference_mode():
            output_unloaded = unloaded(sample)

        # merging is less precise, be more tolerant
        atol, rtol = 1e-1, 1e-1
        # outputs of the base model != output of lora adapter == output of unloaded adapter
        assert not torch.allclose(output_base.logits, output_lora.logits, atol=atol, rtol=rtol)
        assert torch.allclose(output_lora.logits, output_unloaded.logits, atol=atol, rtol=rtol)

    @require_bitsandbytes
    @pytest.mark.xfail(strict=True)
    def test_causal_lm_mixed_batch_lora_adapter_compile(self, tokenizer, data):
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        ).eval()

        # we need at least 3 samples for this to work!
        sample = {
            "input_ids": torch.arange(12).reshape(3, 4).to("cuda"),
            "attention_mask": torch.ones(3, 4).long().to("cuda"),
        }

        with torch.inference_mode():
            output_base = model(**sample)

        config = LoraConfig(task_type=TaskType.CAUSAL_LM, init_lora_weights=False)
        model = get_peft_model(model, config).eval()
        with torch.inference_mode():
            output_default = model(**sample)

        model.add_adapter("other", config)
        model.set_adapter("other")
        with torch.inference_mode():
            output_other = model(**sample)

        model = self.compile(model, {})

        # set adapter_indices so that it alternates between 0 (base), lora 1, and lora 2
        adapter_names = ["__base__", "default", "other"]
        with torch.inference_mode():
            output_mixed = model(**sample, adapter_names=adapter_names)

        atol, rtol = 1e-4, 1e-4
        # outputs of the base model != output of lora adapter 1 != output of other adapter
        assert not torch.allclose(output_base.logits, output_default.logits, atol=atol, rtol=rtol)
        assert not torch.allclose(output_default.logits, output_other.logits, atol=atol, rtol=rtol)
        assert not torch.allclose(output_other.logits, output_mixed.logits, atol=atol, rtol=rtol)
        # outputs of mixed adapter is mix of all 3
        assert torch.allclose(output_base.logits[0], output_mixed.logits[0], atol=atol, rtol=rtol)
        assert torch.allclose(output_default.logits[1], output_mixed.logits[1], atol=atol, rtol=rtol)
        assert torch.allclose(output_other.logits[2], output_mixed.logits[2], atol=atol, rtol=rtol)

    @require_bitsandbytes
    def test_causal_lm_add_weighted_adapter_lora_adapter_compile(self, tokenizer, data):
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        ).eval()
        sample = torch.tensor(data["train"][:1]["input_ids"]).to(model.device)
        with torch.inference_mode():
            output_base = model(sample)

        config = LoraConfig(task_type=TaskType.CAUSAL_LM, init_lora_weights=False)
        model = get_peft_model(model, config).eval()
        model.add_adapter("other", config)
        with torch.inference_mode():
            output_default = model(sample)

        model.set_adapter("other")
        with torch.inference_mode():
            output_other = model(sample)

        model.add_weighted_adapter(["default", "other"], [0.5, 0.5], adapter_name="combined")
        model.set_adapter("combined")
        with torch.inference_mode():
            output_combined = model(sample)

        atol, rtol = 1e-4, 1e-4
        # outputs of the base model != output of default adapter != output of other adapter
        assert not torch.allclose(output_base.logits, output_default.logits, atol=atol, rtol=rtol)
        assert not torch.allclose(output_base.logits, output_other.logits, atol=atol, rtol=rtol)
        assert not torch.allclose(output_default.logits, output_other.logits, atol=atol, rtol=rtol)
        # outputs of combined adapter != all others
        assert not torch.allclose(output_base.logits, output_combined.logits, atol=atol, rtol=rtol)
        assert not torch.allclose(output_default.logits, output_combined.logits, atol=atol, rtol=rtol)
        assert not torch.allclose(output_other.logits, output_combined.logits, atol=atol, rtol=rtol)
