# Note: These tests were copied from test_common_gpu.py and test_gpu_examples.py as they can run on CPU too.
#
# Copyright 2025-present the HuggingFace Inc. team.
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
import gc
import os
import tempfile
import unittest

import pytest
import torch
from accelerate.utils.memory import clear_device_cache
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from peft import (
    AdaLoraConfig,
    LoraConfig,
    OFTConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from peft.tuners.lora import GPTQLoraLinear
from peft.utils import SAFETENSORS_WEIGHTS_NAME, infer_device

from .testing_utils import (
    device_count,
    load_dataset_english_quotes,
    require_gptqmodel,
    require_optimum,
    require_torch_multi_accelerator,
)


@require_gptqmodel
class PeftGPTQModelCommonTests(unittest.TestCase):
    r"""
    A common tester to run common operations that are performed on GPU/CPU such as generation, loading in 8bit, etc.
    """

    def setUp(self):
        self.causal_lm_model_id = "facebook/opt-350m"
        self.device = infer_device()

    def tearDown(self):
        r"""
        Efficient mechanism to free GPU memory after each test. Based on
        https://github.com/huggingface/transformers/issues/21094
        """
        clear_device_cache(garbage_collection=True)
        gc.collect()

    def test_lora_gptq_quantization_from_pretrained_safetensors(self):
        r"""
        Tests that the gptqmodel quantization using LoRA works as expected with safetensors weights.
        """
        from transformers import GPTQConfig

        model_id = "marcsun13/opt-350m-gptq-4bit"
        quantization_config = GPTQConfig(bits=4, use_exllama=False)
        kwargs = {
            "pretrained_model_name_or_path": model_id,
            "dtype": torch.float16,
            "device_map": "auto",
            "quantization_config": quantization_config,
        }
        model = AutoModelForCausalLM.from_pretrained(**kwargs)
        model = prepare_model_for_kbit_training(model)

        config = LoraConfig(task_type="CAUSAL_LM")
        peft_model = get_peft_model(model, config)
        peft_model.generate(input_ids=torch.LongTensor([[0, 2, 3, 1]]).to(peft_model.device))

        with tempfile.TemporaryDirectory() as tmp_dir:
            peft_model.save_pretrained(tmp_dir)
            model = AutoModelForCausalLM.from_pretrained(**kwargs)
            model = PeftModel.from_pretrained(model, tmp_dir)
            model = prepare_model_for_kbit_training(model)
            model.generate(input_ids=torch.LongTensor([[0, 2, 3, 1]]).to(peft_model.device))

            # loading a 2nd adapter works, #1239
            model.load_adapter(tmp_dir, "adapter2")
            model.set_adapter("adapter2")
            model.generate(input_ids=torch.LongTensor([[0, 2, 3, 1]]).to(peft_model.device))

            # check that both adapters are in the same layer
            assert "default" in model.base_model.model.model.decoder.layers[0].self_attn.q_proj.lora_A
            assert "adapter2" in model.base_model.model.model.decoder.layers[0].self_attn.q_proj.lora_A

    def test_oft_gptq_quantization_from_pretrained_safetensors(self):
        r"""
        Tests that the gptqmodel quantization using OFT works as expected with safetensors weights.
        """
        from transformers import GPTQConfig

        model_id = "marcsun13/opt-350m-gptq-4bit"
        quantization_config = GPTQConfig(bits=4, use_exllama=False)
        kwargs = {
            "pretrained_model_name_or_path": model_id,
            "dtype": torch.float16,
            "device_map": "auto",
            "quantization_config": quantization_config,
        }
        model = AutoModelForCausalLM.from_pretrained(**kwargs)
        model = prepare_model_for_kbit_training(model)

        config = OFTConfig(task_type="CAUSAL_LM")
        peft_model = get_peft_model(model, config)
        peft_model.generate(input_ids=torch.LongTensor([[0, 2, 3, 1]]).to(peft_model.device))

        with tempfile.TemporaryDirectory() as tmp_dir:
            peft_model.save_pretrained(tmp_dir)
            model = AutoModelForCausalLM.from_pretrained(**kwargs)
            model = PeftModel.from_pretrained(model, tmp_dir)
            model = prepare_model_for_kbit_training(model)
            model.generate(input_ids=torch.LongTensor([[0, 2, 3, 1]]).to(peft_model.device))

            # loading a 2nd adapter works, #1239
            model.load_adapter(tmp_dir, "adapter2")
            model.set_adapter("adapter2")
            model.generate(input_ids=torch.LongTensor([[0, 2, 3, 1]]).to(peft_model.device))

            # check that both adapters are in the same layer
            assert "default" in model.base_model.model.model.decoder.layers[0].self_attn.q_proj.oft_R
            assert "adapter2" in model.base_model.model.model.decoder.layers[0].self_attn.q_proj.oft_R


@require_gptqmodel
@require_optimum
class PeftGPTQModelTests(unittest.TestCase):
    r"""
    GPTQ + peft tests
    """

    def setUp(self):
        from transformers import GPTQConfig

        self.causal_lm_model_id = "marcsun13/opt-350m-gptq-4bit"
        self.quantization_config = GPTQConfig(bits=4, backend="auto_trainable")
        self.tokenizer = AutoTokenizer.from_pretrained(self.causal_lm_model_id)

    def tearDown(self):
        r"""
        Efficient mechanism to free GPU memory after each test. Based on
        https://github.com/huggingface/transformers/issues/21094
        """
        clear_device_cache(garbage_collection=True)

    def _check_inference_finite(self, model, batch):
        # try inference without Trainer class
        training = model.training
        model.eval()
        output = model(**batch.to(model.device))
        assert torch.isfinite(output.logits).all()
        model.train(training)

    def test_causal_lm_training(self):
        r"""
        Test the CausalLM training on a single GPU device. The test would simply fail if the adapters are not set
        correctly.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = AutoModelForCausalLM.from_pretrained(
                self.causal_lm_model_id,
                dtype=torch.float16,
                device_map="auto",
                quantization_config=self.quantization_config,
            )

            model = prepare_model_for_kbit_training(model)
            config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

            data = load_dataset_english_quotes()
            data = data.map(lambda samples: self.tokenizer(samples["quote"]), batched=True)

            trainer = Trainer(
                model=model,
                train_dataset=data["train"],
                args=TrainingArguments(
                    per_device_train_batch_size=4,
                    gradient_accumulation_steps=4,
                    warmup_steps=2,
                    max_steps=3,
                    learning_rate=2e-4,
                    fp16=True,
                    logging_steps=1,
                    output_dir=tmp_dir,
                ),
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            )
            model.config.use_cache = False
            trainer.train()

            model.cpu().save_pretrained(tmp_dir)

            assert "adapter_config.json" in os.listdir(tmp_dir)
            assert SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir)

            # assert loss is not None
            assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_oft_causal_lm_training(self):
        r"""
        Test the CausalLM training on a single GPU device. The test would simply fail if the adapters are not set
        correctly.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = AutoModelForCausalLM.from_pretrained(
                self.causal_lm_model_id,
                dtype=torch.float16,
                device_map="auto",
                quantization_config=self.quantization_config,
            )

            model = prepare_model_for_kbit_training(model)
            config = OFTConfig(
                r=0,
                oft_block_size=8,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

            data = load_dataset_english_quotes()
            data = data.map(lambda samples: self.tokenizer(samples["quote"]), batched=True)

            trainer = Trainer(
                model=model,
                train_dataset=data["train"],
                args=TrainingArguments(
                    per_device_train_batch_size=4,
                    gradient_accumulation_steps=4,
                    warmup_steps=2,
                    max_steps=3,
                    learning_rate=2e-4,
                    fp16=True,
                    logging_steps=1,
                    output_dir=tmp_dir,
                ),
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            )
            model.config.use_cache = False
            trainer.train()

            model.cpu().save_pretrained(tmp_dir)

            assert "adapter_config.json" in os.listdir(tmp_dir)
            assert SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir)

            # assert loss is not None
            assert trainer.state.log_history[-1]["train_loss"] is not None

    @pytest.mark.single_gpu_tests
    def test_adalora_causalLM(self):
        r"""
        Tests the gptq training with adalora
        """

        model = AutoModelForCausalLM.from_pretrained(
            self.causal_lm_model_id,
            dtype=torch.float16,
            device_map="auto",
            quantization_config=self.quantization_config,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.causal_lm_model_id)
        model = prepare_model_for_kbit_training(model)

        peft_config = AdaLoraConfig(
            total_step=40,
            init_r=6,
            target_r=4,
            tinit=10,
            tfinal=20,
            deltaT=5,
            beta1=0.3,
            beta2=0.3,
            orth_reg_weight=0.2,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, peft_config)

        data = load_dataset_english_quotes()
        data = data.map(lambda samples: self.tokenizer(samples["quote"]), batched=True)
        batch = tokenizer(data["train"][:3]["quote"], return_tensors="pt", padding=True)
        self._check_inference_finite(model, batch)

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = Trainer(
                model=model,
                train_dataset=data["train"],
                args=TrainingArguments(
                    per_device_train_batch_size=4,
                    gradient_accumulation_steps=4,
                    warmup_steps=2,
                    max_steps=3,
                    learning_rate=2e-4,
                    fp16=True,
                    logging_steps=1,
                    output_dir=tmp_dir,
                ),
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            )
            model.config.use_cache = False
            trainer.train()

            model.cpu().save_pretrained(tmp_dir)

            assert "adapter_config.json" in os.listdir(tmp_dir)
            assert SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir)

            # assert loss is not None
            assert trainer.state.log_history[-1]["train_loss"] is not None

    @pytest.mark.multi_gpu_tests
    @require_torch_multi_accelerator
    def test_causal_lm_training_multi_accelerator(self):
        r"""
        Test the CausalLM training on a multi-accelerator device. The test would simply fail if the adapters are not
        set correctly.
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            model = AutoModelForCausalLM.from_pretrained(
                self.causal_lm_model_id,
                dtype=torch.float16,
                device_map="auto",
                quantization_config=self.quantization_config,
            )

            assert set(model.hf_device_map.values()) == set(range(device_count))

            model = prepare_model_for_kbit_training(model)

            setattr(model, "model_parallel", True)
            setattr(model, "is_parallelizable", True)

            config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )

            model = get_peft_model(model, config)

            data = load_dataset_english_quotes()
            data = data.map(lambda samples: self.tokenizer(samples["quote"]), batched=True)

            trainer = Trainer(
                model=model,
                train_dataset=data["train"],
                args=TrainingArguments(
                    per_device_train_batch_size=4,
                    gradient_accumulation_steps=4,
                    warmup_steps=2,
                    max_steps=3,
                    learning_rate=2e-4,
                    fp16=True,
                    logging_steps=1,
                    output_dir=tmp_dir,
                ),
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            )
            model.config.use_cache = False
            trainer.train()

            model.cpu().save_pretrained(tmp_dir)

            assert "adapter_config.json" in os.listdir(tmp_dir)
            assert SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir)

            # assert loss is not None
            assert trainer.state.log_history[-1]["train_loss"] is not None

    @pytest.mark.multi_gpu_tests
    @require_torch_multi_accelerator
    def test_oft_causal_lm_training_multi_accelerator(self):
        r"""
        Test the CausalLM training on a multi-accelerator device. The test would simply fail if the adapters are not
        set correctly.
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            model = AutoModelForCausalLM.from_pretrained(
                self.causal_lm_model_id,
                dtype=torch.float16,
                device_map="auto",
                quantization_config=self.quantization_config,
            )

            assert set(model.hf_device_map.values()) == set(range(device_count))

            model = prepare_model_for_kbit_training(model)

            setattr(model, "model_parallel", True)
            setattr(model, "is_parallelizable", True)

            config = OFTConfig(
                r=0,
                oft_block_size=8,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )

            model = get_peft_model(model, config)

            data = load_dataset_english_quotes()
            data = data.map(lambda samples: self.tokenizer(samples["quote"]), batched=True)

            trainer = Trainer(
                model=model,
                train_dataset=data["train"],
                args=TrainingArguments(
                    per_device_train_batch_size=4,
                    gradient_accumulation_steps=4,
                    warmup_steps=2,
                    max_steps=3,
                    learning_rate=2e-4,
                    fp16=True,
                    logging_steps=1,
                    output_dir=tmp_dir,
                ),
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            )
            model.config.use_cache = False
            trainer.train()

            model.cpu().save_pretrained(tmp_dir)

            assert "adapter_config.json" in os.listdir(tmp_dir)
            assert SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir)

            # assert loss is not None
            assert trainer.state.log_history[-1]["train_loss"] is not None

    def test_non_default_adapter_name(self):
        # See issue 1346
        config = LoraConfig(
            r=16,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )

        # default adapter name
        model = AutoModelForCausalLM.from_pretrained(
            self.causal_lm_model_id,
            dtype=torch.float16,
            device_map="auto",
            quantization_config=self.quantization_config,
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, config)
        n_trainable_default, n_total_default = model.get_nb_trainable_parameters()

        # other adapter name
        model = AutoModelForCausalLM.from_pretrained(
            self.causal_lm_model_id,
            dtype=torch.float16,
            device_map="auto",
            quantization_config=self.quantization_config,
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, config, adapter_name="other")
        n_trainable_other, n_total_other = model.get_nb_trainable_parameters()

        assert n_trainable_other > 0
        # sanity check
        assert n_trainable_default == n_trainable_other
        assert n_total_default == n_total_other

    def test_oft_non_default_adapter_name(self):
        # See issue 1346
        config = OFTConfig(
            r=0,
            oft_block_size=8,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )

        # default adapter name
        model = AutoModelForCausalLM.from_pretrained(
            self.causal_lm_model_id,
            dtype=torch.float16,
            device_map="auto",
            quantization_config=self.quantization_config,
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, config)
        n_trainable_default, n_total_default = model.get_nb_trainable_parameters()

        # other adapter name
        model = AutoModelForCausalLM.from_pretrained(
            self.causal_lm_model_id,
            dtype=torch.float16,
            device_map="auto",
            quantization_config=self.quantization_config,
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, config, adapter_name="other")
        n_trainable_other, n_total_other = model.get_nb_trainable_parameters()

        assert n_trainable_other > 0
        # sanity check
        assert n_trainable_default == n_trainable_other
        assert n_total_default == n_total_other

    def test_load_lora(self):
        model_id = "ModelCloud/Llama-3.2-1B-gptqmodel-ci-4bit"
        adapter_id = "ModelCloud/Llama-3.2-1B-gptqmodel-ci-4bit-lora"

        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        model.load_adapter(adapter_id)

        # assert dynamic rank
        v_proj_module = model.model.layers[5].self_attn.v_proj
        assert isinstance(v_proj_module, GPTQLoraLinear)
        assert v_proj_module.lora_A["default"].weight.data.shape[0] == 128
        assert v_proj_module.lora_B["default"].weight.data.shape[1] == 128
        gate_proj_module = model.model.layers[5].mlp.gate_proj
        assert isinstance(gate_proj_module, GPTQLoraLinear)
        assert gate_proj_module.lora_A["default"].weight.data.shape[0] == 256
        assert gate_proj_module.lora_B["default"].weight.data.shape[1] == 256

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inp = tokenizer("Capital of France is", return_tensors="pt").to(model.device)
        tokens = model.generate(**inp)[0]
        result = tokenizer.decode(tokens)

        assert "paris" in result.lower()
