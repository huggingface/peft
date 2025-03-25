# Copyright 2023-present the HuggingFace Inc. team.
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
import importlib
import itertools
import os
import re
import tempfile
import unittest
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import pytest
import torch
from accelerate import infer_auto_device_map
from accelerate.test_utils.testing import run_command
from accelerate.utils import patch_environment
from accelerate.utils.memory import clear_device_cache
from datasets import Audio, Dataset, DatasetDict, load_dataset
from packaging import version
from parameterized import parameterized
from torch.distributed import init_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)
from transformers.pytorch_utils import Conv1D

from peft import (
    AdaLoraConfig,
    EvaConfig,
    LoftQConfig,
    LoraConfig,
    PeftModel,
    PrefixTuningConfig,
    PromptEncoderConfig,
    TaskType,
    VeraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    initialize_lora_eva_weights,
    inject_adapter_in_model,
    prepare_model_for_kbit_training,
    replace_lora_weights_loftq,
    set_peft_model_state_dict,
)
from peft.import_utils import is_diffusers_available, is_xpu_available
from peft.tuners import boft
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import SAFETENSORS_WEIGHTS_NAME, infer_device
from peft.utils.hotswap import hotswap_adapter, prepare_model_for_compiled_hotswap
from peft.utils.loftq_utils import NFQuantizer
from peft.utils.other import fsdp_auto_wrap_policy

from .testing_utils import (
    device_count,
    require_aqlm,
    require_auto_awq,
    require_auto_gptq,
    require_bitsandbytes,
    require_eetq,
    require_hqq,
    require_multi_accelerator,
    require_non_cpu,
    require_non_xpu,
    require_optimum,
    require_torch_gpu,
    require_torch_multi_gpu,
    require_torchao,
    torch_device,
)


# A full testing suite that tests all the necessary features on GPU. The tests should
# rely on the example scripts to test the features.


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    r"""
    Directly copied from:
    https://github.com/huggingface/peft/blob/main/examples/int8_training/peft_bnb_whisper_large_v2_training.ipynb
    """

    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


@require_non_cpu
@require_bitsandbytes
class PeftBnbGPUExampleTests(unittest.TestCase):
    r"""
    A single GPU int8 + fp4 test suite, this will test if training fits correctly on a single GPU device (1x NVIDIA T4
    16GB) using bitsandbytes.

    The tests are the following:

    - Seq2Seq model training based on:
      https://github.com/huggingface/peft/blob/main/examples/int8_training/Finetune_flan_t5_large_bnb_peft.ipynb
    - Causal LM model training based on:
      https://github.com/huggingface/peft/blob/main/examples/int8_training/Finetune_opt_bnb_peft.ipynb
    - Audio model training based on:
      https://github.com/huggingface/peft/blob/main/examples/int8_training/peft_bnb_whisper_large_v2_training.ipynb

    """

    def setUp(self):
        self.seq2seq_model_id = "google/flan-t5-base"
        self.causal_lm_model_id = "facebook/opt-6.7b"
        self.tokenizer = AutoTokenizer.from_pretrained(self.causal_lm_model_id)
        self.audio_model_id = "openai/whisper-large"

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

    @pytest.mark.single_gpu_tests
    def test_causal_lm_training(self):
        r"""
        Test the CausalLM training on a single GPU device. This test is a converted version of
        https://github.com/huggingface/peft/blob/main/examples/int8_training/Finetune_opt_bnb_peft.ipynb where we train
        `opt-6.7b` on `english_quotes` dataset in few steps. The test would simply fail if the adapters are not set
        correctly.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = AutoModelForCausalLM.from_pretrained(
                self.causal_lm_model_id,
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                device_map="auto",
            )

            tokenizer = AutoTokenizer.from_pretrained(self.causal_lm_model_id)
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

            data = load_dataset("ybelkada/english_quotes_copy")
            data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

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
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            )
            model.config.use_cache = False
            trainer.train()

            model.cpu().save_pretrained(tmp_dir)

            assert "adapter_config.json" in os.listdir(tmp_dir)
            assert SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir)

            # assert loss is not None
            assert trainer.state.log_history[-1]["train_loss"] is not None

    @pytest.mark.single_gpu_tests
    def test_causal_lm_training_4bit(self):
        r"""
        Test the CausalLM training on a single GPU device. This test is a converted version of
        https://github.com/huggingface/peft/blob/main/examples/int8_training/Finetune_opt_bnb_peft.ipynb where we train
        `opt-6.7b` on `english_quotes` dataset in few steps using 4bit base model. The test would simply fail if the
        adapters are not set correctly.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = AutoModelForCausalLM.from_pretrained(
                self.causal_lm_model_id,
                quantization_config=BitsAndBytesConfig(load_in_4bit=True),
                device_map="auto",
            )

            tokenizer = AutoTokenizer.from_pretrained(self.causal_lm_model_id)
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

            data = load_dataset("ybelkada/english_quotes_copy")
            data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

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
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            )
            model.config.use_cache = False
            trainer.train()

            model.cpu().save_pretrained(tmp_dir)

            assert "adapter_config.json" in os.listdir(tmp_dir)
            assert SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir)

            # assert loss is not None
            assert trainer.state.log_history[-1]["train_loss"] is not None

    @pytest.mark.multi_gpu_tests
    def test_causal_lm_training_multi_gpu_4bit(self):
        r"""
        Test the CausalLM training on a multi-GPU device with 4bit base model. The test would simply fail if the
        adapters are not set correctly.
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            model = AutoModelForCausalLM.from_pretrained(
                self.causal_lm_model_id,
                device_map="auto",
                quantization_config=BitsAndBytesConfig(load_in_4bit=True),
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

            data = load_dataset("Abirate/english_quotes")
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
    @require_non_cpu
    def test_4bit_adalora_causalLM(self):
        r"""
        Tests the 4bit training with adalora
        """
        model_id = "facebook/opt-350m"

        # for >3 GPUs, might need: device_map={"": "cuda:0"}
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=BitsAndBytesConfig(load_in_4bit=True)
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        peft_config = AdaLoraConfig(
            init_r=6,
            target_r=4,
            tinit=2,
            tfinal=2,
            total_step=6,
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

        data = load_dataset("ybelkada/english_quotes_copy")
        data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
        batch = tokenizer(data["train"][:3]["quote"], return_tensors="pt", padding=True)
        self._check_inference_finite(model, batch)

        class OptimizerStepCallback(TrainerCallback):
            def on_optimizer_step(self, args, state, control, **kwargs):
                model.update_and_allocate(state.global_step)

        step_callback = OptimizerStepCallback()

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = Trainer(
                model=model,
                train_dataset=data["train"],
                args=TrainingArguments(
                    per_device_train_batch_size=4,
                    gradient_accumulation_steps=4,
                    warmup_steps=2,
                    max_steps=6,
                    learning_rate=2e-4,
                    fp16=True,
                    logging_steps=1,
                    output_dir=tmp_dir,
                ),
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            )
            model.config.use_cache = False
            trainer.add_callback(step_callback)
            trainer.train()

            model.cpu().save_pretrained(tmp_dir)

            assert "adapter_config.json" in os.listdir(tmp_dir)
            assert SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir)

            # assert loss is not None
            assert trainer.state.log_history[-1]["train_loss"] is not None

    @pytest.mark.single_gpu_tests
    @require_non_cpu
    def test_8bit_adalora_causalLM(self):
        r"""
        Tests the 8bit training with adalora
        """
        model_id = "facebook/opt-350m"

        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=BitsAndBytesConfig(load_in_8bit=True)
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        peft_config = AdaLoraConfig(
            init_r=6,
            target_r=4,
            tinit=2,
            tfinal=2,
            total_step=6,
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

        data = load_dataset("ybelkada/english_quotes_copy")
        data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
        batch = tokenizer(data["train"][:3]["quote"], return_tensors="pt", padding=True)
        self._check_inference_finite(model, batch)

        class OptimizerStepCallback(TrainerCallback):
            def on_optimizer_step(self, args, state, control, **kwargs):
                model.update_and_allocate(state.global_step)

        step_callback = OptimizerStepCallback()

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = Trainer(
                model=model,
                train_dataset=data["train"],
                args=TrainingArguments(
                    per_device_train_batch_size=4,
                    gradient_accumulation_steps=4,
                    warmup_steps=2,
                    max_steps=6,
                    learning_rate=2e-4,
                    fp16=True,
                    logging_steps=1,
                    output_dir=tmp_dir,
                ),
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            )
            model.config.use_cache = False
            trainer.add_callback(step_callback)
            trainer.train()

            model.cpu().save_pretrained(tmp_dir)

            assert "adapter_config.json" in os.listdir(tmp_dir)
            assert SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir)

            # assert loss is not None
            assert trainer.state.log_history[-1]["train_loss"] is not None

    @pytest.mark.multi_gpu_tests
    @require_multi_accelerator
    def test_causal_lm_training_multi_gpu(self):
        r"""
        Test the CausalLM training on a multi-GPU device. This test is a converted version of
        https://github.com/huggingface/peft/blob/main/examples/int8_training/Finetune_opt_bnb_peft.ipynb where we train
        `opt-6.7b` on `english_quotes` dataset in few steps. The test would simply fail if the adapters are not set
        correctly.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = AutoModelForCausalLM.from_pretrained(
                self.causal_lm_model_id,
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                device_map="auto",
            )
            print(f"device map: {model.hf_device_map}")
            assert set(model.hf_device_map.values()) == set(range(device_count))

            tokenizer = AutoTokenizer.from_pretrained(self.causal_lm_model_id)
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

            data = load_dataset("Abirate/english_quotes")
            data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

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
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            )
            model.config.use_cache = False
            trainer.train()

            model.cpu().save_pretrained(tmp_dir)

            assert "adapter_config.json" in os.listdir(tmp_dir)
            assert SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir)

            # assert loss is not None
            assert trainer.state.log_history[-1]["train_loss"] is not None

    @pytest.mark.single_gpu_tests
    def test_seq2seq_lm_training_single_gpu(self):
        r"""
        Test the Seq2SeqLM training on a single GPU device. This test is a converted version of
        https://github.com/huggingface/peft/blob/main/examples/int8_training/Finetune_opt_bnb_peft.ipynb where we train
        `flan-large` on `english_quotes` dataset in few steps. The test would simply fail if the adapters are not set
        correctly.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.seq2seq_model_id,
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                device_map={"": 0},
            )

            assert set(model.hf_device_map.values()) == {0}

            tokenizer = AutoTokenizer.from_pretrained(self.seq2seq_model_id)
            model = prepare_model_for_kbit_training(model)

            config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q", "v"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )

            model = get_peft_model(model, config)

            data = load_dataset("ybelkada/english_quotes_copy")
            data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

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
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            )
            model.config.use_cache = False
            trainer.train()

            model.cpu().save_pretrained(tmp_dir)

            assert "adapter_config.json" in os.listdir(tmp_dir)
            assert SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir)

            # assert loss is not None
            assert trainer.state.log_history[-1]["train_loss"] is not None

    @pytest.mark.multi_gpu_tests
    @require_multi_accelerator
    def test_seq2seq_lm_training_multi_gpu(self):
        r"""
        Test the Seq2SeqLM training on a multi-GPU device. This test is a converted version of
        https://github.com/huggingface/peft/blob/main/examples/int8_training/Finetune_opt_bnb_peft.ipynb where we train
        `flan-large` on `english_quotes` dataset in few steps. The test would simply fail if the adapters are not set
        correctly.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.seq2seq_model_id,
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                device_map="balanced",
            )

            assert set(model.hf_device_map.values()) == set(range(device_count))

            tokenizer = AutoTokenizer.from_pretrained(self.seq2seq_model_id)
            model = prepare_model_for_kbit_training(model)

            config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q", "v"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )

            model = get_peft_model(model, config)

            data = load_dataset("ybelkada/english_quotes_copy")
            data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

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
                    output_dir="outputs",
                ),
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            )
            model.config.use_cache = False
            trainer.train()

            model.cpu().save_pretrained(tmp_dir)

            assert "adapter_config.json" in os.listdir(tmp_dir)
            assert SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir)

            # assert loss is not None
            assert trainer.state.log_history[-1]["train_loss"] is not None

    # TODO skipping to see if this leads to single GPU tests passing
    @pytest.mark.skip
    @pytest.mark.single_gpu_tests
    def test_audio_model_training(self):
        r"""
        Test the audio model training on a single GPU device. This test is a converted version of
        https://github.com/huggingface/peft/blob/main/examples/int8_training/peft_bnb_whisper_large_v2_training.ipynb
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_name = "ybelkada/common_voice_mr_11_0_copy"
            task = "transcribe"
            language = "Marathi"
            common_voice = DatasetDict()

            common_voice["train"] = load_dataset(dataset_name, split="train+validation")

            common_voice = common_voice.remove_columns(
                ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"]
            )

            feature_extractor = WhisperFeatureExtractor.from_pretrained(self.audio_model_id)
            tokenizer = WhisperTokenizer.from_pretrained(self.audio_model_id, language=language, task=task)
            processor = WhisperProcessor.from_pretrained(self.audio_model_id, language=language, task=task)

            common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

            def prepare_dataset(batch):
                # load and resample audio data from 48 to 16kHz
                audio = batch["audio"]

                # compute log-Mel input features from input audio array
                batch["input_features"] = feature_extractor(
                    audio["array"], sampling_rate=audio["sampling_rate"]
                ).input_features[0]

                # encode target text to label ids
                batch["labels"] = tokenizer(batch["sentence"]).input_ids
                return batch

            common_voice = common_voice.map(
                prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=2
            )
            data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

            model = WhisperForConditionalGeneration.from_pretrained(
                self.audio_model_id, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto"
            )

            model.config.forced_decoder_ids = None
            model.config.suppress_tokens = []

            model = prepare_model_for_kbit_training(model)

            # as Whisper model uses Conv layer in encoder, checkpointing disables grad computation
            # to avoid this, make the inputs trainable
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

            config = LoraConfig(
                r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none"
            )

            model = get_peft_model(model, config)
            model.print_trainable_parameters()

            training_args = Seq2SeqTrainingArguments(
                output_dir=tmp_dir,  # change to a repo name of your choice
                per_device_train_batch_size=8,
                gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
                learning_rate=1e-3,
                warmup_steps=2,
                max_steps=3,
                fp16=True,
                per_device_eval_batch_size=8,
                generation_max_length=128,
                logging_steps=25,
                remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
                label_names=["labels"],  # same reason as above
            )

            trainer = Seq2SeqTrainer(
                args=training_args,
                model=model,
                train_dataset=common_voice["train"],
                data_collator=data_collator,
                tokenizer=processor.feature_extractor,
            )

            trainer.train()

            model.cpu().save_pretrained(tmp_dir)

            assert "adapter_config.json" in os.listdir(tmp_dir)
            assert SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir)

            # assert loss is not None
            assert trainer.state.log_history[-1]["train_loss"] is not None

    @pytest.mark.single_gpu_tests
    def test_4bit_non_default_adapter_name(self):
        # See PR 1294
        config = LoraConfig(
            r=16,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        # default adapter name
        model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, config)
        n_trainable_default, n_total_default = model.get_nb_trainable_parameters()

        # other adapter name
        model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, config, adapter_name="other")
        n_trainable_other, n_total_other = model.get_nb_trainable_parameters()

        assert n_trainable_other > 0
        # sanity check
        assert n_trainable_default == n_trainable_other
        assert n_total_default == n_total_other

    @pytest.mark.single_gpu_tests
    def test_8bit_non_default_adapter_name(self):
        # See PR 1294
        config = LoraConfig(
            r=16,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

        # default adapter name
        model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, config)
        n_trainable_default, n_total_default = model.get_nb_trainable_parameters()

        # other adapter name
        model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            device_map="auto",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, config, adapter_name="other")
        n_trainable_other, n_total_other = model.get_nb_trainable_parameters()

        assert n_trainable_other > 0
        # sanity check
        assert n_trainable_default == n_trainable_other
        assert n_total_default == n_total_other

    @pytest.mark.single_gpu_tests
    def test_causal_lm_training_4bit_dora(self):
        r"""
        Same as test_causal_lm_training_4bit but with DoRA
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = AutoModelForCausalLM.from_pretrained(
                self.causal_lm_model_id,
                quantization_config=BitsAndBytesConfig(load_in_4bit=True),
                device_map="auto",
            )

            tokenizer = AutoTokenizer.from_pretrained(self.causal_lm_model_id)
            model = prepare_model_for_kbit_training(model)

            config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                use_dora=True,
            )

            model = get_peft_model(model, config)

            data = load_dataset("ybelkada/english_quotes_copy")
            data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

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
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            )
            model.config.use_cache = False
            trainer.train()

            model.cpu().save_pretrained(tmp_dir)

            assert "adapter_config.json" in os.listdir(tmp_dir)
            assert SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir)

            # assert loss is not None
            assert trainer.state.log_history[-1]["train_loss"] is not None

    @pytest.mark.multi_gpu_tests
    def test_causal_lm_training_multi_gpu_4bit_dora(self):
        r"""
        Same as test_causal_lm_training_multi_gpu_4bit but with DoRA
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            model = AutoModelForCausalLM.from_pretrained(
                self.causal_lm_model_id,
                device_map="auto",
                quantization_config=BitsAndBytesConfig(load_in_4bit=True),
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
                use_dora=True,
            )

            model = get_peft_model(model, config)

            data = load_dataset("Abirate/english_quotes")
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
    def test_causal_lm_training_8bit_dora(self):
        r"""
        Same as test_causal_lm_training_4bit_dora but with 8bit
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = AutoModelForCausalLM.from_pretrained(
                self.causal_lm_model_id,
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                device_map="auto",
            )

            tokenizer = AutoTokenizer.from_pretrained(self.causal_lm_model_id)
            model = prepare_model_for_kbit_training(model)

            config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                use_dora=True,
            )

            model = get_peft_model(model, config)

            data = load_dataset("ybelkada/english_quotes_copy")
            data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

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
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            )
            model.config.use_cache = False
            trainer.train()

            model.cpu().save_pretrained(tmp_dir)

            assert "adapter_config.json" in os.listdir(tmp_dir)
            assert SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir)

            # assert loss is not None
            assert trainer.state.log_history[-1]["train_loss"] is not None

    @pytest.mark.multi_gpu_tests
    def test_causal_lm_training_multi_gpu_8bit_dora(self):
        r"""
        Same as test_causal_lm_training_multi_gpu_4bit_dora but with 8bit
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            model = AutoModelForCausalLM.from_pretrained(
                self.causal_lm_model_id,
                device_map="auto",
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
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
                use_dora=True,
            )

            model = get_peft_model(model, config)

            data = load_dataset("Abirate/english_quotes")
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
    def test_causal_lm_training_gpt2_dora(self):
        r"""
        Same as test_causal_lm_training_4bit but with DoRA
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="auto")

            tokenizer = AutoTokenizer.from_pretrained(self.causal_lm_model_id)
            model = prepare_model_for_kbit_training(model)

            config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                use_dora=True,
            )

            model = get_peft_model(model, config)

            data = load_dataset("ybelkada/english_quotes_copy")
            data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

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
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            )
            model.config.use_cache = False
            trainer.train()

            model.cpu().save_pretrained(tmp_dir)

            assert "adapter_config.json" in os.listdir(tmp_dir)
            assert SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir)

            # assert loss is not None
            assert trainer.state.log_history[-1]["train_loss"] is not None

    @parameterized.expand(["4bit", "8bit"])
    def test_initialize_dora_with_bnb_on_cpu(self, kbit):
        # 1674
        # The issue is that to initialize DoRA, we need to dequantize the weights. That only works on GPU for bnb.
        # Therefore, intializing DoRA with bnb on CPU used to fail.
        model_id = "facebook/opt-125m"
        if kbit == "4bit":
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
        elif kbit == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError("Only 4bit and 8bit bnb allowed")

        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
        model = model.cpu()  # ensure that we're on CPU
        # sanity check that all weights are on CPU
        weights_not_cpu = [name for name, p in model.named_parameters() if p.device != torch.device("cpu")]
        assert not weights_not_cpu

        lora_config = LoraConfig(use_dora=True)

        # should not raise
        peft_model = get_peft_model(model, lora_config)
        # check that the weights are still on CPU
        weights_not_cpu = [name for name, p in peft_model.named_parameters() if p.device != torch.device("cpu")]
        assert not weights_not_cpu

    @pytest.mark.single_gpu_tests
    def test_causal_lm_training_vera(self):
        r"""
        Same as test_causal_lm_training but with VeRA
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = AutoModelForCausalLM.from_pretrained(
                self.causal_lm_model_id,
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                device_map="auto",
            )

            tokenizer = AutoTokenizer.from_pretrained(self.causal_lm_model_id)
            model = prepare_model_for_kbit_training(model)

            config = VeraConfig(
                r=16,
                target_modules=["q_proj", "v_proj"],
                vera_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )

            model = get_peft_model(model, config)

            data = load_dataset("ybelkada/english_quotes_copy")
            data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

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
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            )
            model.config.use_cache = False
            trainer.train()

            model.cpu().save_pretrained(tmp_dir)

            assert "adapter_config.json" in os.listdir(tmp_dir)
            assert SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir)

            # assert loss is not None
            assert trainer.state.log_history[-1]["train_loss"] is not None

    @pytest.mark.single_gpu_tests
    def test_causal_lm_training_4bit_vera(self):
        r"""
        Same as test_causal_lm_training_4bit but with VeRA
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = AutoModelForCausalLM.from_pretrained(
                self.causal_lm_model_id,
                quantization_config=BitsAndBytesConfig(load_in_4bit=True),
                device_map="auto",
            )

            tokenizer = AutoTokenizer.from_pretrained(self.causal_lm_model_id)
            model = prepare_model_for_kbit_training(model)

            config = VeraConfig(
                r=16,
                target_modules=["q_proj", "v_proj"],
                vera_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )

            model = get_peft_model(model, config)

            data = load_dataset("ybelkada/english_quotes_copy")
            data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

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
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            )
            model.config.use_cache = False
            trainer.train()

            model.cpu().save_pretrained(tmp_dir)

            assert "adapter_config.json" in os.listdir(tmp_dir)
            assert SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir)

            # assert loss is not None
            assert trainer.state.log_history[-1]["train_loss"] is not None

    @pytest.mark.multi_gpu_tests
    def test_causal_lm_training_multi_gpu_vera(self):
        r"""
        Same as test_causal_lm_training_multi_gpu but with VeRA
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            model = AutoModelForCausalLM.from_pretrained(
                self.causal_lm_model_id,
                device_map="auto",
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            )

            assert set(model.hf_device_map.values()) == set(range(device_count))

            model = prepare_model_for_kbit_training(model)

            setattr(model, "model_parallel", True)
            setattr(model, "is_parallelizable", True)

            config = VeraConfig(
                r=16,
                target_modules=["q_proj", "v_proj"],
                vera_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )

            model = get_peft_model(model, config)

            data = load_dataset("Abirate/english_quotes")
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
    def test_causal_lm_training_multi_gpu_4bit_vera(self):
        r"""
        Same as test_causal_lm_training_multi_gpu_4bit but with VeRA
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            model = AutoModelForCausalLM.from_pretrained(
                self.causal_lm_model_id,
                device_map="auto",
                quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            )

            assert set(model.hf_device_map.values()) == set(range(device_count))

            model = prepare_model_for_kbit_training(model)

            setattr(model, "model_parallel", True)
            setattr(model, "is_parallelizable", True)

            config = VeraConfig(
                r=16,
                target_modules=["q_proj", "v_proj"],
                vera_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )

            model = get_peft_model(model, config)

            data = load_dataset("Abirate/english_quotes")
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
    def test_causal_lm_training_lora_resize_embeddings_trainable_tokens(self):
        r"""
        Test LoRA with trainable tokens on a resized embedding matrix
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_storage=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            model = AutoModelForCausalLM.from_pretrained(
                self.causal_lm_model_id,
                quantization_config=bnb_config,
                device_map="auto",
            )

            # add 2 new tokens
            tokenizer = AutoTokenizer.from_pretrained(self.causal_lm_model_id)
            new_tokens = ["<think>", "</think>"]
            tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
            trainable_token_indices = [tokenizer.vocab[token] for token in new_tokens]

            cur_emb_size = model.model.decoder.embed_tokens.weight.shape[0]
            model.resize_token_embeddings(max(tokenizer.vocab_size, cur_emb_size))
            model = prepare_model_for_kbit_training(model)

            config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                trainable_token_indices={"embed_tokens": trainable_token_indices},
            )

            model = get_peft_model(model, config)

            data = load_dataset("ybelkada/english_quotes_copy")

            def tokenize(samples):
                # add new tokens to samples
                samples = [f"<think>{row}</think>" for row in samples["quote"]]
                return tokenizer(samples)

            data = data.map(tokenize, batched=True)

            trainer = Trainer(
                model=model,
                train_dataset=data["train"],
                args=TrainingArguments(
                    per_device_train_batch_size=4,
                    gradient_accumulation_steps=4,
                    warmup_steps=2,
                    max_steps=3,
                    # higher learning rate, as embeddings are a bit slow to update
                    learning_rate=1e-3,
                    fp16=True,
                    logging_steps=1,
                    output_dir=tmp_dir,
                ),
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            )
            model.config.use_cache = False
            trainer.train()

            model.cpu().save_pretrained(tmp_dir)

            # ensure that the new trainable tokens have been updated
            embedding = model.base_model.model.model.decoder.embed_tokens
            tol = 1e-4
            assert not torch.allclose(
                embedding.token_adapter.trainable_tokens_delta["default"],
                embedding.original_module.weight[trainable_token_indices],
                atol=tol,
                rtol=tol,
            )

            # check size of the checkpoint, should be small since the embedding matrix does not need to be stored
            stat = os.stat(os.path.join(tmp_dir, SAFETENSORS_WEIGHTS_NAME))
            embed_params = model.base_model.model.model.decoder.embed_tokens.original_module.weight.numel()
            # fp32 -> 4x
            emb_file_size = 4 * embed_params
            assert stat.st_size < emb_file_size

            # sanity check: assert loss is not None
            assert trainer.state.log_history[-1]["train_loss"] is not None


@require_torch_gpu
@require_auto_gptq
@require_optimum
class PeftGPTQGPUTests(unittest.TestCase):
    r"""
    GPTQ + peft tests
    """

    def setUp(self):
        from transformers import GPTQConfig

        self.causal_lm_model_id = "marcsun13/opt-350m-gptq-4bit"
        # TODO : check if it works for Exllamav2 kernels
        self.quantization_config = GPTQConfig(bits=4, use_exllama=False)
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

    @pytest.mark.single_gpu_tests
    def test_causal_lm_training(self):
        r"""
        Test the CausalLM training on a single GPU device. The test would simply fail if the adapters are not set
        correctly.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = AutoModelForCausalLM.from_pretrained(
                self.causal_lm_model_id,
                torch_dtype=torch.float16,
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

            data = load_dataset("ybelkada/english_quotes_copy")
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
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=self.quantization_config,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.causal_lm_model_id)
        model = prepare_model_for_kbit_training(model)

        peft_config = AdaLoraConfig(
            init_r=6,
            target_r=4,
            tinit=2,
            tfinal=2,
            total_step=6,
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

        data = load_dataset("ybelkada/english_quotes_copy")
        data = data.map(lambda samples: self.tokenizer(samples["quote"]), batched=True)
        batch = tokenizer(data["train"][:3]["quote"], return_tensors="pt", padding=True)
        self._check_inference_finite(model, batch)

        class OptimizerStepCallback(TrainerCallback):
            def on_optimizer_step(self, args, state, control, **kwargs):
                model.update_and_allocate(state.global_step)

        step_callback = OptimizerStepCallback()

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = Trainer(
                model=model,
                train_dataset=data["train"],
                args=TrainingArguments(
                    per_device_train_batch_size=4,
                    gradient_accumulation_steps=4,
                    warmup_steps=2,
                    max_steps=6,
                    learning_rate=2e-4,
                    fp16=True,
                    logging_steps=1,
                    output_dir=tmp_dir,
                ),
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            )
            trainer.add_callback(step_callback)
            model.config.use_cache = False
            trainer.train()

            model.cpu().save_pretrained(tmp_dir)

            assert "adapter_config.json" in os.listdir(tmp_dir)
            assert SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir)

            # assert loss is not None
            assert trainer.state.log_history[-1]["train_loss"] is not None

    @pytest.mark.multi_gpu_tests
    @require_torch_multi_gpu
    def test_causal_lm_training_multi_gpu(self):
        r"""
        Test the CausalLM training on a multi-GPU device. The test would simply fail if the adapters are not set
        correctly.
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            model = AutoModelForCausalLM.from_pretrained(
                self.causal_lm_model_id,
                torch_dtype=torch.float16,
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

            data = load_dataset("Abirate/english_quotes")
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
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=self.quantization_config,
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, config)
        n_trainable_default, n_total_default = model.get_nb_trainable_parameters()

        # other adapter name
        model = AutoModelForCausalLM.from_pretrained(
            self.causal_lm_model_id,
            torch_dtype=torch.float16,
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


@require_non_cpu
class OffloadSaveTests(unittest.TestCase):
    def setUp(self):
        self.causal_lm_model_id = "gpt2"

    def tearDown(self):
        r"""
        Efficient mechanism to free GPU memory after each test. Based on
        https://github.com/huggingface/transformers/issues/21094
        """
        clear_device_cache(garbage_collection=True)

    def test_offload_load(self):
        r"""
        Test the loading of a LoRA model with CPU- and disk-offloaded modules
        """
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id)
        tokenizer = AutoTokenizer.from_pretrained(self.causal_lm_model_id)
        memory_limits = {"cpu": "0.4GIB"}  # no "disk" for PeftModel.from_pretrained() compatibility

        # offload around half of all transformer modules to the disk
        device_map = infer_auto_device_map(model, max_memory=memory_limits)
        assert "cpu" in device_map.values()
        assert "disk" in device_map.values()

        config = LoraConfig(task_type="CAUSAL_LM", init_lora_weights=False, target_modules=["c_attn"])

        model = get_peft_model(model, config)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id, device_map="cpu")
            lora_model = PeftModel.from_pretrained(model, tmp_dir).eval()
            input_tokens = tokenizer.encode("Four score and seven years ago", return_tensors="pt")
            output = lora_model(input_tokens)[0]

            # load the model with device_map
            offloaded_model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id, device_map=device_map)
            assert len({p.device for p in offloaded_model.parameters()}) == 2  # 'cpu' and 'meta'
            offloaded_lora_model = PeftModel.from_pretrained(offloaded_model, tmp_dir, max_memory=memory_limits).eval()
            offloaded_output = offloaded_lora_model(input_tokens)[0]
        assert torch.allclose(output, offloaded_output, atol=1e-5)

    @pytest.mark.single_gpu_tests
    def test_offload_merge(self):
        r"""
        Test merging, unmerging, and unloading of a model with CPU- and disk- offloaded modules.
        """
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id)
        tokenizer = AutoTokenizer.from_pretrained(self.causal_lm_model_id)
        memory_limits = {0: "0.2GIB", "cpu": "0.2GIB"}  # no "disk" for PeftModel.from_pretrained() compatibility
        # offloads around half of all transformer modules
        device_map = infer_auto_device_map(model, max_memory=memory_limits)
        assert 0 in device_map.values()
        assert "cpu" in device_map.values()
        assert "disk" in device_map.values()

        config = LoraConfig(task_type="CAUSAL_LM", init_lora_weights=False, target_modules=["c_attn"])

        model = get_peft_model(model, config)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            # load the model with device_map
            model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id, device_map=device_map).eval()
            assert len({p.device for p in model.parameters()}) == 2

            model = PeftModel.from_pretrained(model, tmp_dir, max_memory=memory_limits)

        input_tokens = tokenizer.encode("Four score and seven years ago", return_tensors="pt")
        model.eval()

        # test peft model adapter merge
        pre_merge_olayer = model(input_tokens)[0]
        model.merge_adapter()
        post_merge_olayer = model(input_tokens)[0]
        assert torch.allclose(post_merge_olayer, pre_merge_olayer)

        # test peft model adapter unmerge
        model.unmerge_adapter()
        post_unmerge_olayer = model(input_tokens)[0]
        assert torch.allclose(post_unmerge_olayer, pre_merge_olayer)

        # test LoRA merge and unload
        model = model.merge_and_unload()
        post_unload_merge_olayer = model(input_tokens)[0]
        assert torch.allclose(post_unload_merge_olayer, pre_merge_olayer)


@pytest.mark.skipif(not (torch.cuda.is_available() or is_xpu_available()), reason="test requires a GPU")
@pytest.mark.single_gpu_tests
class TestPiSSA:
    r"""
    Tests for PiSSA to ensure that it reduces the quantization error compared to normal LoRA quantization.
    """

    # The error factor indicates by how much the quantization error should be decreased when using PiSSA compared to
    # quantization without PiSSA. Thus 1.03 means that the error should be decreased by 3% at least. This is a very
    # conservative value to prevent flakiness, in practice most gains are > 1.5
    error_factor = 1.03

    def quantize_model(self, model, num_bits=4, device="cuda"):
        # Quantize the `weight.data` of the linear layer in the model to `num_bits` and store it with full precision.
        quantizer = NFQuantizer(num_bits=num_bits, device=device, method="normal", block_size=64)
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, Conv1D)) and "lm_head" not in name:
                quantized_weight, max_abs, shape = quantizer.quantize_block(module.weight.data.to(device))
                module.weight.data = quantizer.dequantize_block(quantized_weight, max_abs, shape)
        return model

    def nuclear_norm(self, base_model, quantized_model):
        # Calculate the nuclear norm (sum of singular values) of the error matrices between the `quantized_model` and the `base_model`.
        error_list = []
        for name, module in base_model.named_modules():
            if isinstance(module, (torch.nn.Linear, Conv1D)) and "lm_head" not in name:
                quant_module = quantized_model.get_submodule(name)
                error_list.append(torch.linalg.svdvals(module.weight.data - quant_module.weight.data).sum())
        return torch.Tensor(error_list).sum()

    def get_errors(
        self,
        tmp_path,
        bits=4,
        device="cuda",
        model_id="hf-internal-testing/tiny-random-BloomForCausalLM",
    ):
        # Comparing the quantized LoRA model to the base model, vs the PiSSA quantized model to the base model.
        # We expect the PiSSA quantized model to have less error than the normal LoRA quantized model.

        cls = AutoModelForSeq2SeqLM if "t5" in str(model_id) else AutoModelForCausalLM
        base_model = cls.from_pretrained(model_id).eval().to(device)
        task_type = TaskType.SEQ_2_SEQ_LM if base_model.config.is_encoder_decoder else TaskType.CAUSAL_LM

        # logits from the normal quantized LoRA model
        target_modules = "all-linear" if task_type != TaskType.SEQ_2_SEQ_LM else ["o", "k", "wi", "q", "v"]
        lora_config = LoraConfig(task_type=task_type, target_modules=target_modules)

        qlora_model = self.quantize_model(cls.from_pretrained(model_id).eval().to(device), bits, device)
        qlora_model = get_peft_model(
            qlora_model,
            lora_config,
        )
        qlora_model = qlora_model.merge_and_unload()
        qlora_error = self.nuclear_norm(base_model, qlora_model)
        del qlora_model
        clear_device_cache(garbage_collection=True)

        # logits from quantized LoRA model using PiSSA
        lora_config = LoraConfig(
            task_type=task_type,
            init_lora_weights="pissa",
            target_modules=target_modules,
        )
        pissa_model = cls.from_pretrained(model_id).eval().to(device)
        pissa_model = get_peft_model(pissa_model, lora_config)

        # save LoRA weights, they should be initialized such that they minimize the quantization error
        pissa_model.base_model.peft_config["default"].init_lora_weights = True
        pissa_model.save_pretrained(tmp_path / "pissa_model")

        pissa_model = pissa_model.unload()
        pissa_model.save_pretrained(tmp_path / "residual_model")

        del pissa_model
        clear_device_cache(garbage_collection=True)

        # now load quantized model and apply PiSSA-initialized weights on top
        qpissa_model = self.quantize_model(
            cls.from_pretrained(tmp_path / "residual_model").eval().to(device), bits, device
        )
        qpissa_model = PeftModel.from_pretrained(qpissa_model, tmp_path / "pissa_model")
        qpissa_model = qpissa_model.merge_and_unload()
        qpissa_error = self.nuclear_norm(base_model, qpissa_model)
        del qpissa_model
        clear_device_cache(garbage_collection=True)

        assert qlora_error > 0.0
        assert qpissa_error > 0.0

        # next, check that PiSSA quantization errors are smaller than LoRA errors by a certain margin
        assert qpissa_error < (qlora_error / self.error_factor)

    @pytest.mark.parametrize("device", [torch_device, "cpu"])
    def test_bloomz_pissa_4bit(self, device, tmp_path):
        # In this test, we compare the logits of the base model, the quantized LoRA model, and the quantized model
        # using PiSSA. When quantizing, we expect a certain level of error. However, we expect the PiSSA quantized
        # model to have less error than the normal LoRA quantized model. Note that when using normal LoRA, the
        # quantization error is simply the error from quantization without LoRA, as LoRA is a no-op before training.
        # We still apply LoRA for the test for consistency.

        self.get_errors(bits=4, device=device, tmp_path=tmp_path)

    @pytest.mark.parametrize("device", [torch_device, "cpu"])
    def test_bloomz_pissa_8bit(self, device, tmp_path):
        # Same test as test_bloomz_pissa_4bit but with 8 bits.
        self.get_errors(bits=8, device=device, tmp_path=tmp_path)

    @pytest.mark.parametrize("device", [torch_device, "cpu"])
    def test_t5_pissa_4bit(self, device, tmp_path):
        self.get_errors(bits=4, device=device, model_id="t5-small", tmp_path=tmp_path)

    @pytest.mark.parametrize("device", [torch_device, "cpu"])
    def test_t5_pissa_8bit(self, device, tmp_path):
        self.get_errors(bits=8, device=device, model_id="t5-small", tmp_path=tmp_path)

    @pytest.mark.parametrize("device", [torch_device, "cpu"])
    def test_gpt2_pissa_4bit(self, device, tmp_path):
        # see 2104
        self.get_errors(bits=4, device=device, model_id="gpt2", tmp_path=tmp_path)

    @pytest.mark.parametrize("device", [torch_device, "cpu"])
    def test_gpt2_pissa_8bit(self, device, tmp_path):
        # see 2104
        self.get_errors(bits=8, device=device, model_id="gpt2", tmp_path=tmp_path)

    @require_bitsandbytes
    def test_lora_pissa_conversion_same_output_after_loading_with_quantization(self, tmp_path):
        # A copy of the test `test_lora_pissa_conversion_same_output_after_loading` in peft/tests/test_initialization.py,
        # that would fail if bitsandbytes quantization is used because Quant(W_res) + AB !=Quant(W) + \Delta(AB).
        import bitsandbytes as bnb

        torch.manual_seed(0)
        data = torch.rand(10, 1000).to(torch_device)

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # choose a large weight so that averages are close to expected values
                self.linear = torch.nn.Linear(1000, 1000)
                self.embed = torch.nn.Embedding(1000, 1000)
                self.conv2d = torch.nn.Conv2d(100, 100, 3)

            def forward(self, x):
                x_int = (100 * x).int()
                x_4d = x.flatten().reshape(1, 100, 10, 10)
                return self.linear(x), self.embed(x_int), self.conv2d(x_4d)

        model = MyModule().to(torch_device)
        output_base = model(data)[0]

        config = LoraConfig(init_lora_weights="pissa", target_modules=["linear"], r=8)
        peft_model = get_peft_model(deepcopy(model), config)
        # save the initial model
        peft_model.peft_config["default"].init_lora_weights = True
        peft_model.save_pretrained(tmp_path / "init-model")
        peft_model = peft_model.unload()
        torch.save(peft_model.state_dict(), tmp_path / "residual-model")
        del peft_model

        # create 4bit base model
        base_model = deepcopy(model)
        base_model.load_state_dict(torch.load(tmp_path / "residual-model"))
        # sanity check: the base model weights were indeed changed
        tol = 1e-06
        assert not torch.allclose(model.linear.weight, base_model.linear.weight, atol=tol, rtol=tol)
        # quantize the linear layer
        linear4bit = bnb.nn.Linear4bit(base_model.linear.in_features, base_model.linear.out_features)
        linear4bit.load_state_dict(base_model.linear.state_dict())
        linear4bit.to(0)
        base_model.linear = linear4bit
        peft_model = PeftModel.from_pretrained(deepcopy(base_model), tmp_path / "init-model")
        output_quantized_pissa = peft_model(data)[0]
        # sanity check
        tol = 1e-06
        assert not torch.allclose(output_base, output_quantized_pissa, atol=tol, rtol=tol)

        # modify the weights, or else the adapter performs an identity transformation
        peft_model.base_model.linear.lora_B["default"].weight.data *= 2.0
        output_finetuned_pissa = peft_model(data)[0]
        # sanity check
        tol = 1e-06
        assert not torch.allclose(output_quantized_pissa, output_finetuned_pissa, atol=tol, rtol=tol)

        # save the model normally
        peft_model.save_pretrained(tmp_path / "pissa-model")
        model_loaded = PeftModel.from_pretrained(deepcopy(base_model), tmp_path / "pissa-model")
        output_loaded = model_loaded(data)[0]

        assert torch.allclose(output_finetuned_pissa, output_loaded, atol=tol, rtol=tol)
        # sanity check: ranks should still be 8 as initially
        assert model_loaded.peft_config["default"].r == 8
        assert model_loaded.base_model.model.linear.lora_A["default"].weight.shape[0] == 8

        # save the model with conversion
        peft_model.save_pretrained(
            tmp_path / "pissa-model-converted", path_initial_model_for_weight_conversion=tmp_path / "init-model"
        )
        model_converted = PeftModel.from_pretrained(deepcopy(model), tmp_path / "pissa-model-converted")
        output_converted = model_converted(data)[0]

        # rank should be double of what it was initially
        assert model_converted.peft_config["default"].r == 16
        assert model_converted.base_model.model.linear.lora_A["default"].weight.shape[0] == 16
        # base model weights should be the same as the initial model
        assert torch.allclose(
            model.linear.weight, model_converted.base_model.model.linear.base_layer.weight, atol=tol, rtol=tol
        )
        # This check is expected to fail when using bnb
        assert not torch.allclose(output_finetuned_pissa, output_converted, atol=tol, rtol=tol)


@pytest.mark.skipif(not (torch.cuda.is_available() or is_xpu_available()), reason="test requires a GPU")
@pytest.mark.single_gpu_tests
class TestOLoRA:
    r"""
    Tests for OLoRA to ensure that it reduces the quantization error compared to normal LoRA quantization.
    """

    # The error factor indicates by how much the quantization error should be decreased when using OLoRA compared to
    # quantization without OLoRA. Thus 1.03 means that the error should be decreased by 3% at least. This is a very
    # conservative value to prevent flakiness, in practice most gains are > 1.5
    error_factor = 1.2

    def quantize_model(self, model, num_bits=4, device="cuda"):
        # Quantize the `weight.data` of the linear layer in the model to `num_bits` and store it with full precision.
        quantizer = NFQuantizer(num_bits=num_bits, device=device, method="normal", block_size=64)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
                quantized_weight, max_abs, shape = quantizer.quantize_block(module.weight.data.to(device))
                module.weight.data = quantizer.dequantize_block(quantized_weight, max_abs, shape)
        return model

    def nuclear_norm(self, base_model, quantized_model):
        # Calculate the nuclear norm (sum of singular values) of the error matrices between the `quantized_model` and the `base_model`.
        error_list = []
        for name, module in base_model.named_modules():
            if isinstance(module, torch.nn.Linear) and "lm_head" not in name:
                quant_module = quantized_model.get_submodule(name)
                error_list.append(torch.linalg.svdvals(module.weight.data - quant_module.weight.data).sum())
        return torch.Tensor(error_list).sum()

    def get_errors(
        self,
        tmp_path,
        bits=4,
        device="cuda",
        model_id="hf-internal-testing/tiny-random-BloomForCausalLM",
    ):
        # Comparing the quantized LoRA model to the base model, vs the OLoRA quantized model to the base model.
        # We expect the OLoRA quantized model to have less error than the normal LoRA quantized model.

        cls = AutoModelForSeq2SeqLM if "t5" in str(model_id) else AutoModelForCausalLM
        base_model = cls.from_pretrained(model_id).eval().to(device)
        task_type = TaskType.SEQ_2_SEQ_LM if base_model.config.is_encoder_decoder else TaskType.CAUSAL_LM

        # logits from the normal quantized LoRA model
        target_modules = "all-linear" if task_type != TaskType.SEQ_2_SEQ_LM else ["o", "k", "wi", "q", "v"]
        lora_config = LoraConfig(task_type=task_type, target_modules=target_modules)

        qlora_model = self.quantize_model(cls.from_pretrained(model_id).eval().to(device), bits, device)
        qlora_model = get_peft_model(
            qlora_model,
            lora_config,
        )
        qlora_model = qlora_model.merge_and_unload()
        qlora_error = self.nuclear_norm(base_model, qlora_model)
        del qlora_model
        clear_device_cache(garbage_collection=True)

        # logits from quantized LoRA model using OLoRA
        lora_config = LoraConfig(
            task_type=task_type,
            init_lora_weights="olora",
            target_modules=target_modules,
        )
        olora_model = cls.from_pretrained(model_id).eval().to(device)
        olora_model = get_peft_model(olora_model, lora_config)

        # save LoRA weights, they should be initialized such that they minimize the quantization error
        olora_model.base_model.peft_config["default"].init_lora_weights = True
        olora_model.save_pretrained(tmp_path / "olora_model")

        olora_model = olora_model.unload()
        olora_model.save_pretrained(tmp_path / "residual_model")

        del olora_model
        clear_device_cache(garbage_collection=True)

        # now load quantized model and apply OLoRA-initialized weights on top
        qolora_model = self.quantize_model(
            cls.from_pretrained(tmp_path / "residual_model").eval().to(device), bits, device
        )
        qolora_model = PeftModel.from_pretrained(qolora_model, tmp_path / "olora_model")
        qolora_model = qolora_model.merge_and_unload()
        qolora_error = self.nuclear_norm(base_model, qolora_model)
        del qolora_model
        clear_device_cache(garbage_collection=True)

        assert qlora_error > 0.0
        assert qolora_error > 0.0

        # next, check that OLoRA quantization errors are smaller than LoRA errors by a certain margin
        assert qolora_error < (qlora_error / self.error_factor)

    @pytest.mark.parametrize("device", [torch_device, "cpu"])
    def test_bloomz_olora_4bit(self, device, tmp_path):
        # In this test, we compare the logits of the base model, the quantized LoRA model, and the quantized model
        # using OLoRA. When quantizing, we expect a certain level of error. However, we expect the OLoRA quantized
        # model to have less error than the normal LoRA quantized model. Note that when using normal LoRA, the
        # quantization error is simply the error from quantization without LoRA, as LoRA is a no-op before training.
        # We still apply LoRA for the test for consistency.

        self.get_errors(bits=4, device=device, tmp_path=tmp_path)

    @pytest.mark.parametrize("device", [torch_device, "cpu"])
    def test_bloomz_olora_8bit(self, device, tmp_path):
        # Same test as test_bloomz_olora_4bit but with 8 bits.
        self.get_errors(bits=8, device=device, tmp_path=tmp_path)

    @pytest.mark.parametrize("bits", [4, 8])
    def test_olora_with_quantized_model(self, bits):
        import bitsandbytes as bnb

        # issue 1999
        model_id = "hf-internal-testing/tiny-random-OPTForCausalLM"
        if bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_storage=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif bits == 8:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError("bits must be 4 or 8")

        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
        model = prepare_model_for_kbit_training(model)
        config = LoraConfig(init_lora_weights="olora")
        model = get_peft_model(model, config)

        # check that the correct type is used for the weights
        base_layer = model.base_model.model.model.decoder.layers[0].self_attn.v_proj.base_layer.weight
        if bits == 4:
            assert isinstance(base_layer, bnb.nn.modules.Params4bit)
        else:
            assert isinstance(base_layer, bnb.nn.modules.Int8Params)

        inputs = torch.arange(10).unsqueeze(0).to(model.device)
        logits = model(inputs).logits  # does not raise
        assert torch.isfinite(logits).all()


@pytest.mark.skipif(
    not (torch.cuda.is_available() or is_xpu_available()), reason="test requires a hardware accelerator"
)
@require_bitsandbytes
class TestLoftQ:
    r"""
    Tests for LoftQ to ensure that it reduces the quantization error compared to normal LoRA quantization.
    """

    # The error factor indicates by how much the quantization error should be decreased when using LoftQ compared to
    # quantization without LoftQ. Thus 1.03 means that the error should be decreased by 3% at least. This is a very
    # conservative value to prevent flakiness, in practice most gains are > 1.5
    error_factor = 1.03
    device = infer_device()

    def get_input(self, model_id, device):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer("All I want is", padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        return inputs

    def get_base_model(self, model_id, device, **kwargs):
        cls = AutoModelForSeq2SeqLM if "t5" in str(model_id) else AutoModelForCausalLM
        model = cls.from_pretrained(model_id, **kwargs).eval()
        model = model.to(self.device)
        return model

    def get_logits(self, model, inputs):
        if model.config.is_encoder_decoder:
            input_ids = inputs["input_ids"]
            return model(input_ids=input_ids, decoder_input_ids=input_ids).logits
        return model(**inputs).logits

    def get_errors(
        self,
        tmp_path,
        bits=4,
        loftq_iter=1,
        device="cuda",
        model_id="hf-internal-testing/tiny-random-BloomForCausalLM",
        use_dora=False,
    ):
        # Helper function that returns the quantization errors (MAE and MSE) when comparing the quantized LoRA model
        # to the base model, vs the LoftQ quantized model to the base model. We expect the LoftQ quantized model to
        # have less error than the normal LoRA quantized model. Since we compare logits, the observed error is
        # already somewhat dampened because of the softmax.
        torch.manual_seed(0)
        model = self.get_base_model(model_id, device)
        task_type = TaskType.SEQ_2_SEQ_LM if model.config.is_encoder_decoder else TaskType.CAUSAL_LM
        inputs = self.get_input(model_id, device)
        # the base logits are the reference, we try to match those as closely as possible
        logits_base = self.get_logits(model, inputs)
        # clean up
        del model
        clear_device_cache(garbage_collection=True)

        # logits from the normal quantized LoRA model
        target_modules = "all-linear" if task_type != TaskType.SEQ_2_SEQ_LM else ["o", "k", "wi", "q", "v"]
        lora_config = LoraConfig(task_type=task_type, use_dora=use_dora, target_modules=target_modules)
        kwargs = {}
        if bits == 4:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
        elif bits == 8:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError("bits must be 4 or 8")

        quantized_model = get_peft_model(
            self.get_base_model(model_id, device=None, **kwargs),
            lora_config,
        )
        torch.manual_seed(0)
        logits_quantized = self.get_logits(quantized_model, inputs)
        del quantized_model
        clear_device_cache(garbage_collection=True)

        # logits from quantized LoRA model using LoftQ
        loftq_config = LoftQConfig(loftq_bits=bits, loftq_iter=loftq_iter)
        lora_config = LoraConfig(
            task_type=task_type,
            init_lora_weights="loftq",
            loftq_config=loftq_config,
            use_dora=use_dora,
            target_modules=target_modules,
        )
        model = self.get_base_model(model_id, device)
        if device != "cpu":
            model = model.to(torch_device)
        loftq_model = get_peft_model(model, lora_config)
        if device != "cpu":
            loftq_model = loftq_model.to(torch_device)

        # save LoRA weights, they should be initialized such that they minimize the quantization error
        loftq_model.base_model.peft_config["default"].init_lora_weights = True
        loftq_model.save_pretrained(tmp_path / "loftq_model")

        loftq_model = loftq_model.unload()
        loftq_model.save_pretrained(tmp_path / "base_model")

        del loftq_model
        clear_device_cache(garbage_collection=True)

        # now load quantized model and apply LoftQ-initialized weights on top
        base_model = self.get_base_model(tmp_path / "base_model", device=None, **kwargs, torch_dtype=torch.float32)
        loftq_model = PeftModel.from_pretrained(base_model, tmp_path / "loftq_model", is_trainable=True)

        # TODO sanity check: model is quantized

        torch.manual_seed(0)
        logits_loftq = self.get_logits(loftq_model, inputs)
        del loftq_model
        clear_device_cache(garbage_collection=True)

        mae_quantized = torch.abs(logits_base - logits_quantized).mean()
        mse_quantized = torch.pow(logits_base - logits_quantized, 2).mean()
        mae_loftq = torch.abs(logits_base - logits_loftq).mean()
        mse_loftq = torch.pow(logits_base - logits_loftq, 2).mean()
        return mae_quantized, mse_quantized, mae_loftq, mse_loftq

    @pytest.mark.parametrize("device", [torch_device, "cpu"])
    def test_bloomz_loftq_4bit(self, device, tmp_path):
        # In this test, we compare the logits of the base model, the quantized LoRA model, and the quantized model
        # using LoftQ. When quantizing, we expect a certain level of error. However, we expect the LoftQ quantized
        # model to have less error than the normal LoRA quantized model. Note that when using normal LoRA, the
        # quantization error is simply the error from quantization without LoRA, as LoRA is a no-op before training.
        # We still apply LoRA for the test for consistency.

        mae_quantized, mse_quantized, mae_loftq, mse_loftq = self.get_errors(bits=4, device=device, tmp_path=tmp_path)
        # first, sanity check that all errors are > 0.0
        assert mae_quantized > 0.0
        assert mse_quantized > 0.0
        assert mae_loftq > 0.0
        assert mse_loftq > 0.0

        # next, check that LoftQ quantization errors are smaller than LoRA errors by a certain margin
        assert mse_loftq < (mse_quantized / self.error_factor)
        assert mae_loftq < (mae_quantized / self.error_factor)

    @pytest.mark.parametrize("device", [torch_device, "cpu"])
    def test_bloomz_loftq_4bit_iter_5(self, device, tmp_path):
        # Same test as the previous one but with 5 iterations. We should expect the error to be even smaller with more
        # iterations, but in practice the difference is not that large, at least not for this small base model.
        mae_quantized, mse_quantized, mae_loftq, mse_loftq = self.get_errors(
            bits=4, loftq_iter=5, device=device, tmp_path=tmp_path
        )
        # first, sanity check that all errors are > 0.0
        assert mae_quantized > 0.0
        assert mse_quantized > 0.0
        assert mae_loftq > 0.0
        assert mse_loftq > 0.0

        # next, check that LoftQ quantization errors are smaller than LoRA errors by a certain margin
        assert mse_loftq < (mse_quantized / self.error_factor)
        assert mae_loftq < (mae_quantized / self.error_factor)

    @pytest.mark.parametrize("device", [torch_device, "cpu"])
    def test_bloomz_loftq_8bit(self, device, tmp_path):
        # Same test as test_bloomz_loftq_4bit but with 8 bits.
        mae_quantized, mse_quantized, mae_loftq, mse_loftq = self.get_errors(bits=8, device=device, tmp_path=tmp_path)

        # first, sanity check that all errors are > 0.0
        assert mae_quantized > 0.0
        assert mse_quantized > 0.0
        assert mae_loftq > 0.0
        assert mse_loftq > 0.0

        # next, check that LoftQ quantization errors are smaller than LoRA errors by a certain margin
        assert mse_loftq < (mse_quantized / self.error_factor)
        assert mae_loftq < (mae_quantized / self.error_factor)

    @pytest.mark.parametrize("device", [torch_device, "cpu"])
    def test_bloomz_loftq_8bit_iter_5(self, device, tmp_path):
        # Same test as test_bloomz_loftq_4bit_iter_5 but with 8 bits.
        mae_quantized, mse_quantized, mae_loftq, mse_loftq = self.get_errors(
            bits=8, loftq_iter=5, device=device, tmp_path=tmp_path
        )

        # first, sanity check that all errors are > 0.0
        assert mae_quantized > 0.0
        assert mse_quantized > 0.0
        assert mae_loftq > 0.0
        assert mse_loftq > 0.0

        # next, check that LoftQ quantization errors are smaller than LoRA errors by a certain margin
        assert mse_loftq < (mse_quantized / self.error_factor)
        assert mae_loftq < (mae_quantized / self.error_factor)

    @pytest.mark.parametrize("device", [torch_device, "cpu"])
    def test_t5_loftq_4bit(self, device, tmp_path):
        mae_quantized, mse_quantized, mae_loftq, mse_loftq = self.get_errors(
            bits=4, device=device, model_id="t5-small", tmp_path=tmp_path
        )
        # first, sanity check that all errors are > 0.0
        assert mae_quantized > 0.0
        assert mse_quantized > 0.0
        assert mae_loftq > 0.0
        assert mse_loftq > 0.0

        # next, check that LoftQ quantization errors are smaller than LoRA errors by a certain margin
        assert mse_loftq < (mse_quantized / self.error_factor)
        assert mae_loftq < (mae_quantized / self.error_factor)

    @pytest.mark.parametrize("device", [torch_device, "cpu"])
    def test_t5_loftq_8bit(self, device, tmp_path):
        mae_quantized, mse_quantized, mae_loftq, mse_loftq = self.get_errors(
            bits=8, device=device, model_id="t5-small", tmp_path=tmp_path
        )
        # first, sanity check that all errors are > 0.0
        assert mae_quantized > 0.0
        assert mse_quantized > 0.0
        assert mae_loftq > 0.0
        assert mse_loftq > 0.0

        # next, check that LoftQ quantization errors are smaller than LoRA errors by a certain margin
        assert mse_loftq < (mse_quantized / self.error_factor)
        assert mae_loftq < (mae_quantized / self.error_factor)

    @pytest.mark.xfail  # failing for now, but having DoRA pass is only a nice-to-have, not a must, so we're good
    @pytest.mark.parametrize("device", [torch_device, "cpu"])
    def test_bloomz_loftq_4bit_dora(self, device, tmp_path):
        # same as test_bloomz_loftq_4bit but with DoRA
        mae_quantized, mse_quantized, mae_loftq, mse_loftq = self.get_errors(
            bits=4, device=device, use_dora=True, tmp_path=tmp_path
        )
        # first, sanity check that all errors are > 0.0
        assert mae_quantized > 0.0
        assert mse_quantized > 0.0
        assert mae_loftq > 0.0
        assert mse_loftq > 0.0

        # next, check that LoftQ quantization errors are smaller than LoRA errors by a certain margin
        factor = 3
        assert mae_loftq < (mae_quantized / factor)
        assert mse_loftq < (mse_quantized / factor)

    @pytest.mark.parametrize("device", [torch_device, "cpu"])
    def test_bloomz_loftq_8bit_dora(self, device, tmp_path):
        # same as test_bloomz_loftq_8bit but with DoRA
        mae_quantized, mse_quantized, mae_loftq, mse_loftq = self.get_errors(
            bits=8, device=device, use_dora=True, tmp_path=tmp_path
        )

        # first, sanity check that all errors are > 0.0
        assert mae_quantized > 0.0
        assert mse_quantized > 0.0
        assert mae_loftq > 0.0
        assert mse_loftq > 0.0

        # next, check that LoftQ quantization errors are smaller than LoRA errors by a certain margin
        assert mae_loftq < (mae_quantized / self.error_factor)
        assert mse_loftq < (mse_quantized / self.error_factor)

    def test_replace_lora_weights_with_loftq_using_callable(self):
        """
        Test replacing LoRa weights with LoFTQ using a callable.

        Using the replace_lora_weights_loftq function, we replace the LoRa weights of a bnb-quantized model with LoRA
        weights initialized by LoftQ on the fly. We use a callable to decide whether to replace the weights or not.
        This callable checks, for each weight, if replacing it would actually result in logits that are closer to the
        original logits of the non-quantized model.

        """
        torch.manual_seed(0)
        model_id = "bigscience/bloomz-560m"
        device = torch_device
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer("The dog was", padding=True, return_tensors="pt").to(device)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
            logits_base = model(**inputs).logits
            model.save_pretrained(tmp_dir)

            # load in 4bit
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config)
            model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", target_modules="all-linear"))
            logits_lora = model(**inputs).logits

            current_mse = float("inf")
            logs = []

            def my_callback(model, module_name):
                """Callable to replace weights with LoFTQ if the mse is lower than the current best one."""
                nonlocal current_mse

                logits = model(**inputs).logits
                mse = ((logits_base - logits) ** 2).mean()
                if mse < current_mse:
                    current_mse = mse
                    logs.append(True)
                    return True
                logs.append(False)
                return False

            replace_lora_weights_loftq(model, model_path=tmp_dir, callback=my_callback)
            logits_loftq = model(**inputs).logits

            mae_lora = (logits_base - logits_lora).abs().mean()
            mae_loftq = (logits_base - logits_loftq).abs().mean()
            mse_lora = ((logits_base - logits_lora) ** 2).mean()
            mse_loftq = ((logits_base - logits_loftq) ** 2).mean()

            # check that the error was reduced by a certain margin
            assert mae_loftq * 1.5 < mae_lora
            assert mse_loftq * 2.5 < mse_lora

            # check that the callback has returned some True and some False values
            assert any(logs)
            assert not all(logs)

        del model
        clear_device_cache(garbage_collection=True)

    def test_replace_lora_weights_with_local_model(self):
        # see issue 2020
        torch.manual_seed(0)
        model_id = "hf-internal-testing/tiny-random-OPTForCausalLM"
        device = torch_device

        with tempfile.TemporaryDirectory() as tmp_dir:
            # save base model locally
            model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
            model.save_pretrained(tmp_dir)
            del model

            # load in 4bit
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
            )

            # load the base model from local directory
            model = AutoModelForCausalLM.from_pretrained(tmp_dir, quantization_config=bnb_config)
            model = get_peft_model(model, LoraConfig())

            # passing the local path directly works
            replace_lora_weights_loftq(model, model_path=tmp_dir)
            del model

            # load the base model from local directory
            model = AutoModelForCausalLM.from_pretrained(tmp_dir, quantization_config=bnb_config)
            model = get_peft_model(model, LoraConfig())

            # when not passing, ensure that users are made aware of the `model_path` argument
            with pytest.raises(ValueError, match="model_path"):
                replace_lora_weights_loftq(model)

        del model
        clear_device_cache(garbage_collection=True)

    def test_config_no_loftq_init(self):
        with pytest.warns(
            UserWarning,
            match="`loftq_config` specified but will be ignored when `init_lora_weights` is not 'loftq'.",
        ):
            LoraConfig(loftq_config=LoftQConfig())

    def test_config_no_loftq_config(self):
        with pytest.raises(ValueError, match="`loftq_config` must be specified when `init_lora_weights` is 'loftq'."):
            LoraConfig(init_lora_weights="loftq")


@require_bitsandbytes
@require_non_cpu
class MultiprocessTester(unittest.TestCase):
    def test_notebook_launcher(self):
        script_path = os.path.join("scripts", "launch_notebook_mp.py")
        cmd = ["python", script_path]
        with patch_environment(omp_num_threads=1):
            run_command(cmd, env=os.environ.copy())


@require_non_cpu
class MixedPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.causal_lm_model_id = "facebook/opt-125m"
        self.tokenizer = AutoTokenizer.from_pretrained(self.causal_lm_model_id)
        self.config = LoraConfig(
            r=16,
            lora_alpha=32,
            task_type="CAUSAL_LM",
        )

        data = load_dataset("ybelkada/english_quotes_copy")
        self.data = data.map(lambda samples: self.tokenizer(samples["quote"]), batched=True)

    def tearDown(self):
        r"""
        Efficient mechanism to free GPU memory after each test. Based on
        https://github.com/huggingface/transformers/issues/21094
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif is_xpu_available():
            torch.xpu.empty_cache()
        gc.collect()

    @pytest.mark.single_gpu_tests
    def test_model_using_float16_with_amp_raises(self):
        # This test shows the issue with using a model in fp16 and then trying to use it with mixed precision training,
        # which should not use fp16.
        model = AutoModelForCausalLM.from_pretrained(
            self.causal_lm_model_id,
            torch_dtype=torch.float16,
        )
        model = get_peft_model(model, self.config, autocast_adapter_dtype=False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = Trainer(
                model=model,
                train_dataset=self.data["train"],
                args=TrainingArguments(
                    fp16=True,  # <= this is required for the error to be raised
                    output_dir=tmp_dir,
                    max_steps=3,
                ),
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            )
            with pytest.raises(ValueError, match="Attempting to unscale FP16 gradients."):
                trainer.train()

    @pytest.mark.single_gpu_tests
    def test_model_using_float16_autocast_dtype(self):
        # Here we use autocast_adapter_dtype=True (the default) to automatically promote the adapter weights to float32.
        # No exception should be raised.
        model = AutoModelForCausalLM.from_pretrained(
            self.causal_lm_model_id,
            torch_dtype=torch.float16,
        )
        model = get_peft_model(model, self.config, autocast_adapter_dtype=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = Trainer(
                model=model,
                train_dataset=self.data["train"],
                args=TrainingArguments(
                    fp16=True,  # <= this is required for the error to be raised
                    output_dir=tmp_dir,
                    max_steps=3,
                ),
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            )
            trainer.train()  # does not raise

    @pytest.mark.single_gpu_tests
    def test_model_using_float16_explicit_cast(self):
        # Same test as above but containing the fix to make it work
        model = AutoModelForCausalLM.from_pretrained(
            self.causal_lm_model_id,
            torch_dtype=torch.float16,
        )
        model = get_peft_model(model, self.config, autocast_adapter_dtype=False)

        # here we manually promote the adapter weights to float32
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.float()

        dtype_counts_before = Counter(p.dtype for p in model.parameters())
        model = AutoModelForCausalLM.from_pretrained(
            self.causal_lm_model_id,
            torch_dtype=torch.float16,
        )

        model = get_peft_model(model, self.config, autocast_adapter_dtype=True)
        dtype_counts_after = Counter(p.dtype for p in model.parameters())
        assert dtype_counts_before == dtype_counts_after

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = Trainer(
                model=model,
                train_dataset=self.data["train"],
                args=TrainingArguments(
                    fp16=True,  # <= this is required for the error to be raised
                    max_steps=3,
                    output_dir=tmp_dir,
                ),
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            )
            trainer.train()  # does not raise

    @pytest.mark.single_gpu_tests
    def test_load_model_using_float16_with_amp_raises(self):
        # Same as previous tests, but loading the adapter with PeftModel.from_pretrained instead
        model = AutoModelForCausalLM.from_pretrained(
            self.causal_lm_model_id,
            torch_dtype=torch.float16,
        )
        model = get_peft_model(model, self.config, autocast_adapter_dtype=False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id, torch_dtype=torch.float16)
            model = PeftModel.from_pretrained(model, tmp_dir, autocast_adapter_dtype=False, is_trainable=True)

            trainer = Trainer(
                model=model,
                train_dataset=self.data["train"],
                args=TrainingArguments(
                    fp16=True,  # <= this is required for the error to be raised
                    output_dir=tmp_dir,
                    max_steps=3,
                ),
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            )
            with pytest.raises(ValueError, match="Attempting to unscale FP16 gradients."):
                trainer.train()

    @pytest.mark.single_gpu_tests
    def test_load_model_using_float16_autocast_dtype(self):
        # Same as previous tests, but loading the adapter with PeftModel.from_pretrained instead
        model = AutoModelForCausalLM.from_pretrained(
            self.causal_lm_model_id,
            torch_dtype=torch.float16,
        )
        # Below, we purposefully set autocast_adapter_dtype=False so that the saved adapter uses float16. We still want
        # the loaded adapter to use float32 when we load it with autocast_adapter_dtype=True.
        model = get_peft_model(model, self.config, autocast_adapter_dtype=False)
        # sanity check: this should have float16 adapter weights:
        assert (
            model.base_model.model.model.decoder.layers[0].self_attn.v_proj.lora_A["default"].weight.dtype
            == torch.float16
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id, torch_dtype=torch.float16)
            model = PeftModel.from_pretrained(model, tmp_dir, autocast_adapter_dtype=True, is_trainable=True)
            # sanity check: this should NOT have float16 adapter weights:
            assert (
                model.base_model.model.model.decoder.layers[0].self_attn.v_proj.lora_A["default"].weight.dtype
                == torch.float32
            )

            trainer = Trainer(
                model=model,
                train_dataset=self.data["train"],
                args=TrainingArguments(
                    fp16=True,  # <= this is required for the error to be raised
                    output_dir=tmp_dir,
                    max_steps=3,
                ),
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            )
            trainer.train()  # does not raise

    @pytest.mark.single_gpu_tests
    def test_load_adapter_using_float16_autocast_dtype(self):
        # Here we test the load_adapter method with autocast_adapter_dtype. We show that autocasting is prevented when
        # calling load_model(..., autocast_adapter_dtype=False) and that it is enabled when calling
        # load_model(..., autocast_adapter_dtype=True) (the default).
        model = AutoModelForCausalLM.from_pretrained(
            self.causal_lm_model_id,
            torch_dtype=torch.float16,
        )
        # Below, we purposefully set autocast_adapter_dtype=False so that the saved adapter uses float16. We still want
        # the loaded adapter to use float32 when we load it with autocast_adapter_dtype=True.
        model = get_peft_model(model, self.config, autocast_adapter_dtype=False)
        # sanity check: this should have float16 adapter weights:
        assert (
            model.base_model.model.model.decoder.layers[0].self_attn.v_proj.lora_A["default"].weight.dtype
            == torch.float16
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id, torch_dtype=torch.float16)
            # the default adapter is now in float16
            model = get_peft_model(model, self.config, autocast_adapter_dtype=False)
            # sanity check: this should NOT have float16 adapter weights:
            assert (
                model.base_model.model.model.decoder.layers[0].self_attn.v_proj.lora_A["default"].weight.dtype
                == torch.float16
            )

            # now load the first adapter in float16 using the adapter name "loaded16"
            model.load_adapter(tmp_dir, "loaded16", autocast_adapter_dtype=False)
            assert (
                model.base_model.model.model.decoder.layers[0].self_attn.v_proj.lora_A["loaded16"].weight.dtype
                == torch.float16
            )

            # now load the first adapter in float32 using the adapter name "loaded32"
            model.load_adapter(tmp_dir, "loaded32", autocast_adapter_dtype=True)
            assert (
                model.base_model.model.model.decoder.layers[0].self_attn.v_proj.lora_A["loaded32"].weight.dtype
                == torch.float32
            )

            # training with the default adapter, which is in float16, should raise
            model.set_adapter("default")
            trainer = Trainer(
                model=model,
                train_dataset=self.data["train"],
                args=TrainingArguments(
                    fp16=True,  # <= this is required for the error to be raised
                    output_dir=tmp_dir,
                    max_steps=3,
                ),
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            )
            with pytest.raises(ValueError, match="Attempting to unscale FP16 gradients."):
                trainer.train()

            # training the model with the adapter "loaded16", which is in float16, should also raise
            model.set_adapter("loaded16")
            trainer = Trainer(
                model=model,
                train_dataset=self.data["train"],
                args=TrainingArguments(
                    fp16=True,  # <= this is required for the error to be raised
                    output_dir=tmp_dir,
                    max_steps=3,
                ),
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            )
            with pytest.raises(ValueError, match="Attempting to unscale FP16 gradients."):
                trainer.train()

            # training the model with the adapter "loaded32", which is in float32, should not raise
            model.set_adapter("loaded32")
            trainer = Trainer(
                model=model,
                train_dataset=self.data["train"],
                args=TrainingArguments(
                    fp16=True,  # <= this is required for the error to be raised
                    output_dir=tmp_dir,
                    max_steps=3,
                ),
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            )
            trainer.train()  # does not raise


@require_non_xpu
@require_torch_gpu
@require_aqlm
@unittest.skipUnless(
    version.parse(importlib.metadata.version("transformers")) >= version.parse("4.38.0"),
    "test requires `transformers>=4.38.0`",
)
class PeftAqlmGPUTests(unittest.TestCase):
    r"""
    AQLM + peft tests
    """

    def setUp(self):
        self.causal_lm_model_id = "BlackSamorez/TinyLlama-1_1B-Chat-v1_0-AQLM-2Bit-1x16-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(self.causal_lm_model_id)

    def tearDown(self):
        r"""
        Efficient mechanism to free GPU memory after each test. Based on
        https://github.com/huggingface/transformers/issues/21094
        """
        gc.collect()
        torch.cuda.empty_cache()

    def _check_inference_finite(self, model, batch):
        # try inference without Trainer class
        training = model.training
        model.eval()
        output = model(**batch.to(model.device))
        assert torch.isfinite(output.logits).all()
        model.train(training)

    @pytest.mark.single_gpu_tests
    # see https://github.com/Vahe1994/AQLM/pull/139
    @pytest.mark.xfail(reason="AQLM does not work with PyTorch 2.5 (yet)", strict=True, raises=AttributeError)
    def test_causal_lm_training_aqlm(self):
        r"""
        Test the CausalLM training on a single GPU device. The test would simply fail if the adapters are not set
        correctly.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = AutoModelForCausalLM.from_pretrained(
                self.causal_lm_model_id,
                device_map="cuda",
                torch_dtype="auto",
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

            data = load_dataset("ybelkada/english_quotes_copy")
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
                    logging_steps=1,
                    output_dir=tmp_dir,
                    fp16=True,
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


@require_non_xpu
@require_torch_gpu
@require_hqq
@unittest.skipUnless(
    version.parse(importlib.metadata.version("transformers")) >= version.parse("4.36.1"),
    "test requires `transformers>=4.36.1`",
)
class PeftHqqGPUTests(unittest.TestCase):
    r"""
    HQQ + peft tests
    """

    def setUp(self):
        self.causal_lm_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(self.causal_lm_model_id)

    def tearDown(self):
        r"""
        Efficient mechanism to free GPU memory after each test. Based on
        https://github.com/huggingface/transformers/issues/21094
        """
        gc.collect()
        torch.cuda.empty_cache()

    @pytest.mark.single_gpu_tests
    @parameterized.expand([False, True])
    def test_causal_lm_training_hqq(self, use_dora):
        r"""
        Test the CausalLM training on a single GPU device. The test would simply fail if the adapters are not set
        correctly.
        """

        from transformers import HqqConfig

        with tempfile.TemporaryDirectory() as tmp_dir:
            device = "cuda"
            compute_dtype = torch.float16

            quant_config = HqqConfig(nbits=4, group_size=64)

            model = AutoModelForCausalLM.from_pretrained(
                self.causal_lm_model_id,
                device_map=device,
                torch_dtype=compute_dtype,
                quantization_config=quant_config,
            )

            model = prepare_model_for_kbit_training(model)
            config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                use_dora=use_dora,
            )
            model = get_peft_model(model, config)

            data = load_dataset("ybelkada/english_quotes_copy")
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
                    logging_steps=1,
                    output_dir=tmp_dir,
                    fp16=True,
                ),
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            )
            model.config.use_cache = False
            trainer.train()

            model.save_pretrained(tmp_dir)

            assert "adapter_config.json" in os.listdir(tmp_dir)
            assert SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir)

            # assert loss is not None
            assert trainer.state.log_history[-1]["train_loss"] is not None

    @pytest.mark.single_gpu_tests
    def test_hqq_lora_model_outputs(self):
        # check that the outputs generated by HQQ with LoRA are similar to those without HQQ
        from transformers import HqqConfig

        device = "cuda"
        compute_dtype = torch.float16

        # first load the model without HQQ
        model = AutoModelForCausalLM.from_pretrained(
            self.causal_lm_model_id,
            device_map=device,
            torch_dtype=compute_dtype,
        )
        config = LoraConfig(
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
            init_lora_weights=False,
        )
        torch.manual_seed(0)
        model = get_peft_model(model, config).eval()
        inputs = self.tokenizer("The meaning of unit tests is", return_tensors="pt").to(model.device)

        with torch.inference_mode():
            output_normal = model(**inputs).logits
        assert torch.isfinite(output_normal).all()

        del model
        gc.collect()
        torch.cuda.empty_cache()

        # now load with HQQ
        quant_config = HqqConfig(nbits=4, group_size=64)
        model = AutoModelForCausalLM.from_pretrained(
            self.causal_lm_model_id,
            device_map=device,
            torch_dtype=compute_dtype,
            quantization_config=quant_config,
        )
        torch.manual_seed(0)
        model = get_peft_model(model, config).eval()
        with torch.inference_mode():
            output_hqq = model(**inputs).logits

        # check that outputs of HQQ are highly correlated; there are outliers, so don't check for equality
        cc_matrix = torch.corrcoef(torch.stack((output_normal.float().flatten(), output_hqq.float().flatten())))
        assert cc_matrix.min() > 0.97

        # check that outputs are the same after merging
        cc_matrix = torch.corrcoef(torch.stack((output_normal.float().flatten(), output_hqq.float().flatten())))
        assert cc_matrix.min() > 0.97

        # check outputs are the same after unmerging
        model.unmerge_adapter()
        with torch.inference_mode():
            output_unmerged = model(**inputs).logits
        cc_matrix = torch.corrcoef(torch.stack((output_normal.float().flatten(), output_unmerged.float().flatten())))
        assert cc_matrix.min() > 0.97

        # check that the results are the same after saving and loading
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            del model
            gc.collect()
            torch.cuda.empty_cache()

            quant_config = HqqConfig(nbits=4, group_size=64)
            model = AutoModelForCausalLM.from_pretrained(
                self.causal_lm_model_id,
                device_map=device,
                torch_dtype=compute_dtype,
                quantization_config=quant_config,
            )
            model = PeftModel.from_pretrained(model, tmp_dir)
            with torch.inference_mode():
                output_loaded = model(**inputs).logits

            # for loading, we expect high precision, so check for equality and not just correlation
            atol, rtol = 1e-6, 1e-6
            assert torch.allclose(output_hqq, output_loaded, atol=atol, rtol=rtol)

        # check that outputs are the same after merge_and_unload
        model = model.merge_and_unload()
        with torch.inference_mode():
            output_merged_unloaded = model(**inputs).logits
        cc_matrix = torch.corrcoef(
            torch.stack((output_normal.float().flatten(), output_merged_unloaded.float().flatten()))
        )
        assert cc_matrix.min() > 0.97


@require_torch_gpu
@require_auto_awq
class PeftAwqGPUTests(unittest.TestCase):
    r"""
    Awq + peft tests
    """

    def setUp(self):
        self.causal_lm_model_id = "peft-internal-testing/opt-125m-awq"
        self.tokenizer = AutoTokenizer.from_pretrained(self.causal_lm_model_id)

    def tearDown(self):
        r"""
        Efficient mechanism to free GPU memory after each test. Based on
        https://github.com/huggingface/transformers/issues/21094
        """
        gc.collect()
        torch.cuda.empty_cache()

    def _check_inference_finite(self, model, batch):
        # try inference without Trainer class
        training = model.training
        model.eval()
        output = model(**batch.to(model.device))
        assert torch.isfinite(output.logits).all()
        model.train(training)

    @pytest.mark.single_gpu_tests
    def test_causal_lm_training_awq(self):
        r"""
        Test the CausalLM training on a single GPU device. The test would simply fail if the adapters are not set
        correctly.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = AutoModelForCausalLM.from_pretrained(
                self.causal_lm_model_id,
                device_map="auto",
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

            data = load_dataset("ybelkada/english_quotes_copy")
            data = data.map(lambda samples: self.tokenizer(samples["quote"]), batched=True)

            # TODO: deal correctly with this case in transformers
            model._is_quantized_training_enabled = True

            trainer = Trainer(
                model=model,
                train_dataset=data["train"],
                args=TrainingArguments(
                    per_device_train_batch_size=4,
                    gradient_accumulation_steps=4,
                    warmup_steps=2,
                    max_steps=3,
                    learning_rate=2e-4,
                    logging_steps=1,
                    output_dir=tmp_dir,
                    fp16=True,
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
    @require_torch_multi_gpu
    def test_causal_lm_training_multi_gpu(self):
        r"""
        Test the CausalLM training on a multi-GPU device. The test would simply fail if the adapters are not set
        correctly.
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            model = AutoModelForCausalLM.from_pretrained(
                self.causal_lm_model_id,
                device_map="auto",
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

            data = load_dataset("Abirate/english_quotes")
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


@require_non_xpu
@require_torch_gpu
@require_eetq
class PeftEetqGPUTests(unittest.TestCase):
    r"""
    EETQ + peft tests
    """

    def setUp(self):
        self.causal_lm_model_id = "facebook/opt-125m"
        self.tokenizer = AutoTokenizer.from_pretrained(self.causal_lm_model_id)

    def tearDown(self):
        r"""
        Efficient mechanism to free GPU memory after each test. Based on
        https://github.com/huggingface/transformers/issues/21094
        """
        gc.collect()
        torch.cuda.empty_cache()

    def _check_inference_finite(self, model, batch):
        # try inference without Trainer class
        training = model.training
        model.eval()
        output = model(**batch.to(model.device))
        assert torch.isfinite(output.logits).all()
        model.train(training)

    @pytest.mark.single_gpu_tests
    def test_causal_lm_training_eetq(self):
        r"""
        Test the CausalLM training on a single GPU device. The test would simply fail if the adapters are not set
        correctly.
        """
        from transformers import EetqConfig

        with tempfile.TemporaryDirectory() as tmp_dir:
            quantization_config = EetqConfig("int8")

            model = AutoModelForCausalLM.from_pretrained(
                self.causal_lm_model_id, device_map="auto", quantization_config=quantization_config
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

            data = load_dataset("ybelkada/english_quotes_copy")
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
    @require_torch_multi_gpu
    def test_causal_lm_training_multi_gpu_eetq(self):
        r"""
        Test the CausalLM training on a multi-GPU device. The test would simply fail if the adapters are not set
        correctly.
        """
        from transformers import EetqConfig

        with tempfile.TemporaryDirectory() as tmp_dir:
            quantization_config = EetqConfig("int8")

            model = AutoModelForCausalLM.from_pretrained(
                self.causal_lm_model_id,
                device_map="auto",
                quantization_config=quantization_config,
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

            data = load_dataset("Abirate/english_quotes")
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


@require_non_xpu
@require_torch_gpu
@require_torchao
class PeftTorchaoGPUTests(unittest.TestCase):
    r"""
    torchao + peft tests
    """

    supported_quant_types = [
        "int8_weight_only",
        "int8_dynamic_activation_int8_weight",
        # int4_weight_only raises an error:
        # RuntimeError: derivative for aten::_weight_int4pack_mm is not implemented
        # "int4_weight_only",
    ]

    def setUp(self):
        self.causal_lm_model_id = "facebook/opt-125m"
        self.tokenizer = AutoTokenizer.from_pretrained(self.causal_lm_model_id)
        # torchao breaks with fp16 and if a previous test uses fp16, transformers will set this env var, which affects
        # subsequent tests, therefore the env var needs to be cleared explicitly
        #
        # TODO: remove this once https://github.com/huggingface/transformers/pull/34886 is merged
        os.environ.pop("ACCELERATE_MIXED_PRECISION", None)

    def tearDown(self):
        r"""
        Efficient mechanism to free GPU memory after each test. Based on
        https://github.com/huggingface/transformers/issues/21094
        """
        gc.collect()
        torch.cuda.empty_cache()

    @parameterized.expand(supported_quant_types)
    @pytest.mark.single_gpu_tests
    def test_causal_lm_training_single_gpu_torchao(self, quant_type):
        from transformers import TorchAoConfig

        device = 0

        with tempfile.TemporaryDirectory() as tmp_dir:
            quantization_config = TorchAoConfig(quant_type=quant_type)
            model = AutoModelForCausalLM.from_pretrained(
                self.causal_lm_model_id, device_map=device, quantization_config=quantization_config
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

            data = load_dataset("ybelkada/english_quotes_copy")
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
                    logging_steps=1,
                    output_dir=tmp_dir,
                ),
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            )
            trainer.model.config.use_cache = False
            trainer.train()

            model.save_pretrained(tmp_dir)

            assert "adapter_config.json" in os.listdir(tmp_dir)
            assert SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir)

            # assert loss is not None
            assert trainer.state.log_history[-1]["train_loss"] is not None

    @pytest.mark.single_gpu_tests
    def test_causal_lm_training_single_gpu_torchao_dora_int8_weight_only(self):
        from transformers import TorchAoConfig

        device = 0

        with tempfile.TemporaryDirectory() as tmp_dir:
            quantization_config = TorchAoConfig(quant_type="int8_weight_only")
            model = AutoModelForCausalLM.from_pretrained(
                self.causal_lm_model_id, device_map=device, quantization_config=quantization_config
            )
            model = prepare_model_for_kbit_training(model)

            config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                use_dora=True,
            )
            model = get_peft_model(model, config)

            data = load_dataset("ybelkada/english_quotes_copy")
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
                    logging_steps=1,
                    output_dir=tmp_dir,
                ),
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            )
            trainer.model.config.use_cache = False
            trainer.train()

            model.save_pretrained(tmp_dir)

            assert "adapter_config.json" in os.listdir(tmp_dir)
            assert SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir)

            # assert loss is not None
            assert trainer.state.log_history[-1]["train_loss"] is not None

    @pytest.mark.single_gpu_tests
    def test_causal_lm_training_single_gpu_torchao_dora_int8_dynamic_activation_int8_weight_raises(self):
        from transformers import TorchAoConfig

        device = 0

        quantization_config = TorchAoConfig(quant_type="int8_dynamic_activation_int8_weight")
        model = AutoModelForCausalLM.from_pretrained(
            self.causal_lm_model_id, device_map=device, quantization_config=quantization_config
        )
        model = prepare_model_for_kbit_training(model)

        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            use_dora=True,
        )
        with pytest.raises(NotImplementedError):
            get_peft_model(model, config)

    @pytest.mark.single_gpu_tests
    def test_causal_lm_training_single_gpu_torchao_int4_raises(self):
        # int4_weight_only raises an error:
        # RuntimeError: derivative for aten::_weight_int4pack_mm is not implemented
        # TODO: Once proper torchao support for int4 is added, remove this test and add int4 to supported_quant_types
        from transformers import TorchAoConfig

        device = 0

        quantization_config = TorchAoConfig(quant_type="int4_weight_only")
        model = AutoModelForCausalLM.from_pretrained(
            self.causal_lm_model_id, device_map=device, quantization_config=quantization_config
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

        msg = re.escape("TorchaoLoraLinear only supports int8 weights for now")
        with pytest.raises(ValueError, match=msg):
            get_peft_model(model, config)

    @parameterized.expand(supported_quant_types)
    @pytest.mark.multi_gpu_tests
    @require_torch_multi_gpu
    def test_causal_lm_training_multi_gpu_torchao(self, quant_type):
        from transformers import TorchAoConfig

        with tempfile.TemporaryDirectory() as tmp_dir:
            quantization_config = TorchAoConfig(quant_type=quant_type)
            model = AutoModelForCausalLM.from_pretrained(
                self.causal_lm_model_id,
                device_map="auto",
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
            )

            assert set(model.hf_device_map.values()) == set(range(device_count))

            model = prepare_model_for_kbit_training(model)
            model.model_parallel = True
            model.is_parallelizable = True

            config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

            data = load_dataset("ybelkada/english_quotes_copy")
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
                    logging_steps=1,
                    output_dir=tmp_dir,
                ),
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            )
            trainer.model.config.use_cache = False
            trainer.train()

            model.save_pretrained(tmp_dir)

            assert "adapter_config.json" in os.listdir(tmp_dir)
            assert SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir)

            # assert loss is not None
            assert trainer.state.log_history[-1]["train_loss"] is not None

    @pytest.mark.multi_gpu_tests
    @require_torch_multi_gpu
    def test_causal_lm_training_multi_gpu_torchao_int4_raises(self):
        # int4_weight_only raises an error:
        # RuntimeError: derivative for aten::_weight_int4pack_mm is not implemented
        # TODO: Once proper torchao support for int4 is added, remove this test and add int4 to supported_quant_types
        from transformers import TorchAoConfig

        quantization_config = TorchAoConfig(quant_type="int4_weight_only")
        model = AutoModelForCausalLM.from_pretrained(
            self.causal_lm_model_id,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
        )

        assert set(model.hf_device_map.values()) == set(range(device_count))

        model = prepare_model_for_kbit_training(model)
        model.model_parallel = True
        model.is_parallelizable = True

        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        msg = re.escape("TorchaoLoraLinear only supports int8 weights for now")
        with pytest.raises(ValueError, match=msg):
            get_peft_model(model, config)

    @pytest.mark.single_gpu_tests
    def test_torchao_merge_layers_int8_weight_only(self):
        from torchao.dtypes import AffineQuantizedTensor
        from transformers import TorchAoConfig

        quant_type = "int8_weight_only"
        torch.manual_seed(0)
        device = 0
        dummy_input = torch.arange(10).view(-1, 1).to(device)

        quantization_config = TorchAoConfig(quant_type=quant_type)
        model = AutoModelForCausalLM.from_pretrained(
            self.causal_lm_model_id, device_map=device, quantization_config=quantization_config
        ).eval()
        logits_base = model(dummy_input)[0]

        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            init_lora_weights=False,
        )
        model = get_peft_model(model, config)

        model.eval()
        logits = model(dummy_input)[0]

        # sanity check: outputs changed
        # precision is quite low, so we need to use high atol and rtol
        atol, rtol = 1e-1, 1e-1
        assert not torch.allclose(logits, logits_base, atol=atol, rtol=rtol)

        model.merge_adapter()
        logits_merged = model(dummy_input)[0]
        for name, module in model.named_modules():
            if "base_layer" in name:
                assert isinstance(module.weight, AffineQuantizedTensor)

        model.unmerge_adapter()
        logits_unmerged = model(dummy_input)[0]
        for name, module in model.named_modules():
            if "base_layer" in name:
                assert isinstance(module.weight, AffineQuantizedTensor)

        model = model.merge_and_unload()
        logits_merged_unloaded = model(dummy_input)[0]

        assert torch.allclose(logits, logits_merged, atol=atol, rtol=rtol)
        assert torch.allclose(logits, logits_unmerged, atol=atol, rtol=rtol)
        assert torch.allclose(logits, logits_merged_unloaded, atol=atol, rtol=rtol)

    @pytest.mark.single_gpu_tests
    def test_torchao_merge_layers_int8_dynamic_activation_int8_weight_raises(self):
        # int8_dynamic_activation_int8_weight does not support dequantize, thus merging does not work
        from transformers import TorchAoConfig

        quant_type = "int8_dynamic_activation_int8_weight"
        torch.manual_seed(0)
        device = 0

        quantization_config = TorchAoConfig(quant_type=quant_type)
        model = AutoModelForCausalLM.from_pretrained(
            self.causal_lm_model_id, device_map=device, quantization_config=quantization_config
        ).eval()

        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            init_lora_weights=False,
        )
        model = get_peft_model(model, config)

        msg = re.escape(
            "Weights of type LinearActivationQuantizedTensor do not support dequantization (yet), which is needed to "
            "support merging."
        )
        with pytest.raises(NotImplementedError, match=msg):
            model.merge_adapter()


PRECISIONS = [(torch.float32), (torch.float16), (torch.bfloat16)]

LORA_PARAMS = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
}


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding_layer = torch.nn.Embedding(1000, 768)
        self.layer_norm = torch.nn.LayerNorm(768)
        self.linear_transform = torch.nn.Linear(768, 256)

    def forward(self, input_ids):
        embedded_output = self.embedding_layer(input_ids)
        norm_output = self.layer_norm(embedded_output)
        linear_output = self.linear_transform(norm_output)

        return linear_output


class SimpleConv2DModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding_layer = torch.nn.Embedding(1000, 768)
        self.layer_norm = torch.nn.LayerNorm(768)
        self.conv2d_transform = torch.nn.Conv2d(1, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, input_ids):
        # Additional layers for your custom model
        embedded_output = self.embedding_layer(input_ids)
        norm_output = self.layer_norm(embedded_output)

        # Reshape for Conv2d input (add batch size dimension)
        norm_output = norm_output.unsqueeze(1)
        conv_output = self.conv2d_transform(norm_output)

        # Remove batch size dimension
        conv_output = conv_output.squeeze(1)

        return conv_output


@require_non_cpu
class TestAutoCast(unittest.TestCase):
    device = infer_device()

    # This test makes sure, that Lora dtypes are consistent with the types
    # infered by torch.autocast under tested PRECISIONS
    @parameterized.expand(PRECISIONS)
    def test_simple_model(self, *args, **kwargs):
        self._test_model(SimpleModel(), *args, **kwargs)

    @parameterized.expand(PRECISIONS)
    def test_simple_lora_linear_model(self, *args, **kwargs):
        simple_model = SimpleModel()
        config = LoraConfig(
            **LORA_PARAMS,
            target_modules=["linear_transform"],
        )

        lora_model = get_peft_model(simple_model, config)

        self._test_model(lora_model, *args, **kwargs)

    @parameterized.expand(PRECISIONS)
    def test_simple_lora_embedding_model(self, *args, **kwargs):
        simple_model = SimpleModel()
        config = LoraConfig(
            **LORA_PARAMS,
            target_modules=["embedding_layer"],
        )
        lora_model = get_peft_model(simple_model, config)

        self._test_model(lora_model, *args, **kwargs)

    @parameterized.expand(PRECISIONS)
    def test_simple_conv2d_model(self, *args, **kwargs):
        self._test_model(SimpleConv2DModel(), *args, **kwargs)

    @parameterized.expand(PRECISIONS)
    def test_simple_lora_conv2d_model(self, *args, **kwargs):
        simple_model = SimpleConv2DModel()
        config = LoraConfig(
            **LORA_PARAMS,
            target_modules=["conv2d_transform"],
        )
        lora_model = get_peft_model(simple_model, config)
        self._test_model(lora_model, *args, **kwargs)

    def _test_model(self, model, precision):
        # Move model to GPU
        model = model.to(self.device)

        # Prepare dummy inputs
        input_ids = torch.randint(0, 1000, (2, 10)).to(self.device)
        if precision == torch.bfloat16:
            is_xpu = self.device == "xpu"
            is_cuda_bf16 = self.device == "cuda" and torch.cuda.is_bf16_supported()
            if not (is_xpu or is_cuda_bf16):
                self.skipTest("Bfloat16 not supported on this device")

        # Forward pass with test precision
        with torch.autocast(enabled=True, dtype=precision, device_type=self.device):
            outputs = model(input_ids)
            assert outputs.dtype == precision


class TestFSDPWrap:
    """
    Test that we can successfully initialize an FSDP instance of the module.

    This is a very simple test, as it does not perform actual FSDP training. Here we just ensure that the FSDP instance
    can be created. This can fail for several reasons, e.g. int dtype from BNB or inconsistent requires_grad settings
    due to the auto wrap policy.

    """

    @pytest.mark.single_gpu_tests
    @require_bitsandbytes
    def test_bnb_4bit_wrap_fsdp(self):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            # float32 must be used, or else FSDP will complain about mixed int and float dtypes
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_quant_storage=torch.float32,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            quantization_config=quant_config,
            torch_dtype=torch.float32,
        )
        # model = prepare_model_for_kbit_training(model)
        config = LoraConfig(
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
            use_dora=True,
        )
        model = get_peft_model(model, config)

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29501"

        init_process_group(world_size=1, rank=0)
        # check that this does not raise:
        FSDP(model, auto_wrap_policy=fsdp_auto_wrap_policy(model), use_orig_params=False, sync_module_states=True)

    def test_fsdp_auto_wrap_policy_does_not_raise_on_custom_model(self):
        # See #2167
        # Avoid raising on custom models since Trainer uses fsdp_auto_wrap_policy automatically for PEFT + FSDP
        fsdp_auto_wrap_policy(SimpleModel())  # does not raise


class TestBOFT:
    """
    Test that we can correctly use half-precision models with BOFT.
    """

    device = infer_device()

    @require_non_cpu
    @pytest.mark.single_gpu_tests
    def test_boft_half_linear(self):
        # Check that we can use BoFT with model loaded in half precision
        layer = torch.nn.Linear(160, 160).to(self.device)
        layer = boft.layer.Linear(layer, "layer", boft_n_butterfly_factor=2).to(dtype=torch.bfloat16)
        x = torch.randn(160, 160, device=self.device, dtype=torch.bfloat16)
        layer(x)  # does not raise

    @require_non_cpu
    @pytest.mark.single_gpu_tests
    def test_boft_half_conv(self):
        conv = torch.nn.Conv2d(1, 1, 4).to(self.device)
        conv = boft.layer.Conv2d(conv, "conv", boft_n_butterfly_factor=2).to(dtype=torch.bfloat16)
        x = torch.randn(1, 160, 160, device=self.device, dtype=torch.bfloat16)
        conv(x)  # does not raise


class TestPTuningReproducibility:
    device = infer_device()

    @require_non_cpu
    def test_p_tuning_exactly_reproducible_after_loading(self, tmp_path):
        # See: https://github.com/huggingface/peft/issues/2043#issuecomment-2321522577
        # Ensure that after loading a p-tuning checkpoint, results are exactly reproducible (before the patch, they were
        # only _almost_ identical).

        # The model must be sufficiently large for the effect to be measurable, which is why this test requires is not
        # run on CPU.
        model_id = "facebook/opt-125m"
        inputs = torch.arange(10).view(-1, 1).to(self.device)

        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
        peft_config = PromptEncoderConfig(task_type="CAUSAL_LM", num_virtual_tokens=20, encoder_hidden_size=128)
        model = get_peft_model(model, peft_config).eval()

        with torch.inference_mode():
            output_peft = model(inputs).logits
            gen_peft = model.generate(inputs, min_new_tokens=10, max_new_tokens=10)

        model.save_pretrained(tmp_path)
        del model
        clear_device_cache(garbage_collection=True)

        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
        model = PeftModel.from_pretrained(model, tmp_path)

        with torch.inference_mode():
            output_loaded = model(inputs).logits
            gen_loaded = model.generate(inputs, min_new_tokens=10, max_new_tokens=10)

        torch.testing.assert_close(output_loaded, output_peft)
        torch.testing.assert_close(gen_loaded, gen_peft)


@pytest.mark.single_gpu_tests
class TestLowCpuMemUsageDifferentDevices:
    """Test for the low CPU memory usage option for loading PEFT models.

    There are already tests for low_cpu_mem_usage=True in test_initialization.py but here we want to run tests that
    require a GPU.

    """

    model_id = "hf-internal-testing/tiny-random-OPTForCausalLM"
    device = infer_device()

    @require_non_cpu
    @pytest.mark.parametrize("device_model, device_sd", [("cpu", infer_device()), (infer_device(), "cpu")])
    def test_low_cpu_mem_usage_model_model_on_gpu_state_dict_on_cpu_works(self, device_model, device_sd):
        # specifically test diverging devices for the model and state_dict
        inputs = {"input_ids": torch.randint(0, 100, (1, 10)), "attention_mask": torch.ones(1, 10)}
        inputs = {k: v.to(device_model) for k, v in inputs.items()}

        model = AutoModelForCausalLM.from_pretrained(self.model_id).to(device_model)
        lora_config = LoraConfig(init_lora_weights=False, target_modules="all-linear")
        model = get_peft_model(model, lora_config)
        model.eval()
        logits_not_low_cpu_mem = model(**inputs).logits

        state_dict = get_peft_model_state_dict(model)
        peft_model_state_dict = {}
        # remap the state dict so that it can be correctly loaded, and move weights to the other device
        prefix = "base_model.model."
        for k, v in state_dict.items():
            k = k[len(prefix) :]
            peft_model_state_dict[k] = v.to(device_sd)

        del model

        model = AutoModelForCausalLM.from_pretrained(self.model_id).to(device_model)
        model.eval()
        inject_adapter_in_model(lora_config, model, low_cpu_mem_usage=True)
        load_result = set_peft_model_state_dict(model, peft_model_state_dict, low_cpu_mem_usage=True)

        # sanity check: all lora keys are matched
        assert not any("lora" in k for k in load_result.missing_keys)
        assert not any("lora" in k for k in load_result.unexpected_keys)

        logits_low_cpu_mem = model(**inputs).logits

        assert torch.allclose(logits_low_cpu_mem, logits_not_low_cpu_mem)
        assert {p.device.type for p in model.parameters()} == {device_model}

    @require_bitsandbytes
    @pytest.mark.parametrize("quantization_method", ["bnb-4bit", "bnb-8bit"])
    def test_low_cpu_mem_usage_with_quantization(self, quantization_method):
        # Ensure that low_cpu_mem_usage works with quantization
        # See also https://github.com/huggingface/diffusers/issues/10550
        if quantization_method == "bnb-4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float32,
                bnb_4bit_quant_storage=torch.float32,
                bnb_4bit_use_double_quant=True,
            )
        elif quantization_method == "bnb-8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(f"Unknown quantization method {quantization_method}")

        model = AutoModelForCausalLM.from_pretrained(self.model_id, quantization_config=quantization_config)
        if model.device.type != self.device:
            # calling model.to("cuda") with 8 bit bnb raises an error, thus guard against it
            model = model.to(self.device)

        lora_config = LoraConfig(init_lora_weights=False, target_modules="all-linear")

        # We use get_peft_model with low_cpu_mem_usage=True here. This is not typically done in practice (the option is
        # mostly interesting for loading trained adapters), but it does the job for testing purposes.
        model = get_peft_model(model, lora_config, low_cpu_mem_usage=True)  # this should not raise
        assert {p.device.type for p in model.parameters()} == {self.device, "meta"}


class TestEvaInitializationGPU:
    """GPU tests for the Eva initialization method."""

    # Constants for test configuration
    COSINE_SIMILARITY_THRESHOLD = 0.75
    NUM_SEEDS = 3
    BATCH_SIZE = 4
    MAX_LENGTH = 256
    LORA_DIM = 8
    LORA_ALPHA = 1
    DEVICE = infer_device()

    @pytest.fixture
    def tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @pytest.fixture
    def dataset(self, tokenizer):
        dataset = load_dataset("ybelkada/english_quotes_copy", split="train")
        # concatenate examples
        examples = []
        example = ""
        for data in dataset:
            if len(example) >= self.MAX_LENGTH:
                examples.append(example)
                example = ""
            example = example + " " + data["quote"]
        dataset = Dataset.from_dict({"text": examples})
        # tokenize
        dataset = dataset.map(
            lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=self.MAX_LENGTH),
            batched=True,
            remove_columns=dataset.column_names,
        )
        dataset.set_format(type="torch")
        return dataset

    @pytest.fixture
    def model(self):
        model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        model.transformer.h = model.transformer.h[:2]  # truncate to 2 layers
        return model.to(self.DEVICE)

    @pytest.fixture
    def model_bnb(self):
        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            "openai-community/gpt2",
            quantization_config=bnb_config,
            attn_implementation="eager",  # gpt2 doesnt support flash attention
        )
        model.transformer.h = model.transformer.h[:2]  # truncate to 2 layers
        model = prepare_model_for_kbit_training(model)
        return model

    @pytest.fixture
    def model_fixture(self, request):
        return request.getfixturevalue(request.param)

    @pytest.fixture
    def peft_config(self):
        return LoraConfig(
            r=self.LORA_DIM,
            lora_alpha=self.LORA_ALPHA,
            target_modules=["c_attn"],
            init_lora_weights="eva",
            eva_config=EvaConfig(rho=2),
        )

    def is_bnb_model(self, model):
        return hasattr(model.config, "quantization_config")

    @staticmethod
    def collate_fn(examples):
        return {k: torch.stack([v[k] for v in examples], dim=0) for k in examples[0].keys()}

    @require_non_cpu
    @require_bitsandbytes
    @pytest.mark.single_gpu_tests
    @pytest.mark.parametrize("model_fixture", ["model", "model_bnb"], indirect=True)
    def test_eva_initialization_consistency(self, model_fixture, dataset, peft_config):
        """Test that the state dict returned by get_eva_state_dict loaded correctly and is consistent across different seeds based
        on the cosine similarity of the svd components."""
        state_dicts = []
        for seed in range(self.NUM_SEEDS):
            shuffled_dataset = dataset.shuffle(seed=seed)
            dataloader = DataLoader(
                shuffled_dataset,
                batch_size=self.BATCH_SIZE,
                collate_fn=lambda examples: {
                    k: torch.stack([v[k] for v in examples], dim=0) for k in examples[0].keys()
                },
                shuffle=False,
            )
            peft_model = get_peft_model(deepcopy(model_fixture), peft_config)
            initialize_lora_eva_weights(peft_model, dataloader)
            state_dicts.append(
                {k: v.cpu() for k, v in peft_model.state_dict().items() if "lora_A.default.weight" in k}
            )

        cos_sims = defaultdict(list)
        for i, j in itertools.combinations(range(self.NUM_SEEDS), 2):
            for k, v1 in state_dicts[i].items():
                v2 = state_dicts[j][k]
                min_size = min(v1.size(0), v2.size(0))
                cos_sims[k].extend(torch.cosine_similarity(v1[:min_size], v2[:min_size], dim=1).abs().tolist())

        mean_cosine_similarities = {k: torch.tensor(v).mean() for k, v in cos_sims.items()}
        for layer_name, mean_cosine_similarity in mean_cosine_similarities.items():
            assert mean_cosine_similarity > self.COSINE_SIMILARITY_THRESHOLD, (
                f"Mean absolute cosine similarity {mean_cosine_similarity:.4f} "
                f"is not greater than {self.COSINE_SIMILARITY_THRESHOLD}"
            )


@pytest.mark.multi_gpu_tests
class TestPrefixTuning:
    device = infer_device()

    @require_multi_accelerator
    def test_prefix_tuning_multiple_devices_decoder_model(self):
        # See issue 2134
        model_id = "hf-internal-testing/tiny-random-MistralForCausalLM"
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding="left")
        inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to(self.device)

        device_map = {
            "model.embed_tokens": 0,
            "model.layers.0": 0,
            "model.layers.1": 1,
            "model.norm": 1,
            "model.rotary_emb": 1,
            "lm_head": 1,
        }
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map)
        # sanity check, as the test passes trivially for a single device
        assert len({p.device for p in model.parameters()}) > 1
        # sanity check: this should work without peft
        model.generate(**inputs)  # does not raise

        peft_config = PrefixTuningConfig(num_virtual_tokens=10, task_type="CAUSAL_LM")
        model = get_peft_model(model, peft_config)
        model.generate(**inputs)  # does not raise

    @require_multi_accelerator
    def test_prefix_tuning_multiple_devices_encoder_decoder_model(self):
        # See issue 2134
        model_id = "hf-internal-testing/tiny-random-T5Model"
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding="left")
        inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to(self.device)
        device_map = {
            "shared": 0,
            "encoder.embed_tokens": 0,
            "encoder.block.0": 0,
            "encoder.block.1": 0,
            "encoder.block.2": 1,
            "encoder.block.3": 1,
            "encoder.block.4": 1,
            "encoder.final_layer_norm": 1,
            "decoder.embed_tokens": 0,
            "decoder.block.0": 0,
            "decoder.block.1": 0,
            "decoder.block.2": 1,
            "decoder.block.3": 1,
            "decoder.block.4": 1,
            "decoder.final_layer_norm": 1,
            "lm_head": 0,
        }
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map=device_map)
        # sanity check, as the test passes trivially for a single device
        assert len({p.device for p in model.parameters()}) > 1
        # sanity check: this should work without peft
        model.generate(**inputs)  # does not raise

        peft_config = PrefixTuningConfig(num_virtual_tokens=10, task_type="SEQ_2_SEQ_LM")
        model = get_peft_model(model, peft_config)
        model.generate(**inputs)  # does not raise


@pytest.mark.skipif(
    not (torch.cuda.is_available() or is_xpu_available()), reason="test requires a hardware accelerator"
)
@pytest.mark.single_gpu_tests
class TestHotSwapping:
    """
    Test hotswapping on compiled models.

    This test suite is only run on GPU as it is quite slow.
    """

    torch_device = infer_device()

    @pytest.fixture(scope="class", autouse=True)
    def reset_float32_matmul_precision(self):
        # Earlier tests may run torchao, which, at the time this was added, sets the float32 matmul precision to 'high'.
        # This in turn results in some models producing different outputs when compiled (but only for some seeds).
        # Therefore, we need to ensure that the precision is reset to "highest", which is the default.
        # TODO: if torchao removes the side effect, this fixture can be deleted.
        # https://github.com/pytorch/ao/blob/ffb4350640e76c7e7f449dd1e36d33f19fe384c8/torchao/quantization/utils.py#L589
        torch.set_float32_matmul_precision("highest")

    @pytest.fixture(autouse=True)
    def reset_dynamo_cache(self):
        # It is critical that the dynamo cache is reset for each test. Otherwise, if the test re-uses the same model,
        # there will be recompilation errors, as torch caches the model when run in the same process.
        yield
        torch._dynamo.reset()

    #######
    # LLM #
    #######

    def check_hotswap(self, do_hotswap, ranks, alpha_scalings):
        """
        Test hotswapping with a compiled model.

        Passing do_hotswap=False should trigger recompilation. Use the raise_error_on_recompile context manager to
        raise an error when recompilation occurs.

        """
        torch.manual_seed(0)
        inputs = torch.arange(10).view(-1, 1).to(self.torch_device)
        model_id = "hf-internal-testing/tiny-random-OPTForCausalLM"
        model = AutoModelForCausalLM.from_pretrained(model_id).to(self.torch_device)
        rank0, rank1 = ranks
        alpha0, alpha1 = alpha_scalings

        # note that the 2nd adapter targeting a subset of the 1st adapter is okay, but not the other way round
        config0 = LoraConfig(init_lora_weights=False, r=rank0, lora_alpha=alpha0, target_modules=["q_proj", "v_proj"])
        config1 = LoraConfig(init_lora_weights=False, r=rank1, lora_alpha=alpha1, target_modules=["q_proj"])
        model = get_peft_model(model, config0, adapter_name="adapter0").eval()
        with torch.inference_mode():
            output0 = model(inputs).logits

        model.add_adapter("adapter1", config1)
        model.set_adapter("adapter1")
        with torch.inference_mode():
            output1 = model(inputs).logits

        # sanity check:
        tol = 1e-4
        assert not torch.allclose(output0, output1, atol=tol, rtol=tol)

        with tempfile.TemporaryDirectory() as tmp_dirname:
            model.save_pretrained(tmp_dirname)
            del model

            model = AutoModelForCausalLM.from_pretrained(model_id).to(self.torch_device)
            model = PeftModel.from_pretrained(model, os.path.join(tmp_dirname, "adapter0")).eval()
            if do_hotswap:
                prepare_model_for_compiled_hotswap(model, config=model.peft_config, target_rank=max(ranks))
            model = torch.compile(model, mode="reduce-overhead")
            output_after0 = model(inputs).logits
            assert torch.allclose(output0, output_after0, atol=tol, rtol=tol)

            # swap and check that we get the output from adapter1
            if do_hotswap:
                hotswap_adapter(model, os.path.join(tmp_dirname, "adapter1"), adapter_name="default")
            else:
                model.load_adapter(os.path.join(tmp_dirname, "adapter1"), adapter_name="other")
                model.set_adapter("other")

            # we need to call forward to potentially trigger recompilation
            output_after1 = model(inputs).logits
            assert torch.allclose(output1, output_after1, atol=tol, rtol=tol)

    # it is important to check hotswapping small to large ranks and large to small ranks
    @pytest.mark.parametrize("ranks", [(11, 11), (7, 13), (13, 7)])
    def test_hotswapping_compiled_model_does_not_trigger_recompilation(self, ranks):
        with torch._dynamo.config.patch(error_on_recompile=True):  # raise an error on recompilation
            self.check_hotswap(do_hotswap=True, ranks=ranks, alpha_scalings=ranks)

    def test_no_hotswapping_compiled_model_triggers_recompilation(self):
        # contingency test to ensure that hotswapping is actually needed to prevent recompilation
        ranks = 7, 13
        with torch._dynamo.config.patch(error_on_recompile=True):
            with pytest.raises(torch._dynamo.exc.RecompileError):  # raise an error on recompilation
                self.check_hotswap(do_hotswap=False, ranks=ranks, alpha_scalings=ranks)

    ###################
    # DIFFUSION MODEL #
    ###################

    def get_small_unet(self):
        # from diffusers UNet2DConditionModelTests
        from diffusers import UNet2DConditionModel

        torch.manual_seed(0)
        init_dict = {
            "block_out_channels": (4, 8),
            "norm_num_groups": 4,
            "down_block_types": ("CrossAttnDownBlock2D", "DownBlock2D"),
            "up_block_types": ("UpBlock2D", "CrossAttnUpBlock2D"),
            "cross_attention_dim": 8,
            "attention_head_dim": 2,
            "out_channels": 4,
            "in_channels": 4,
            "layers_per_block": 1,
            "sample_size": 16,
        }
        model = UNet2DConditionModel(**init_dict)
        return model.to(self.torch_device)

    def get_unet_lora_config(self, lora_rank, lora_alpha, target_modules):
        # from diffusers test_models_unet_2d_condition.py
        # note that this only targets linear layers by default
        unet_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            init_lora_weights=False,
            use_dora=False,
        )
        return unet_lora_config

    def get_dummy_input(self):
        # from UNet2DConditionModelTests
        from diffusers.utils.testing_utils import floats_tensor

        batch_size = 4
        num_channels = 4
        sizes = (16, 16)

        noise = floats_tensor((batch_size, num_channels) + sizes).to(self.torch_device)
        time_step = torch.tensor([10]).to(self.torch_device)
        encoder_hidden_states = floats_tensor((batch_size, 4, 8)).to(self.torch_device)

        return {"sample": noise, "timestep": time_step, "encoder_hidden_states": encoder_hidden_states}

    def set_lora_device(self, model, adapter_names, device):
        # copied from diffusers LoraBaseMixin.set_lora_device
        for module in model.modules():
            if isinstance(module, BaseTunerLayer):
                for adapter_name in adapter_names:
                    module.lora_A[adapter_name].to(device)
                    module.lora_B[adapter_name].to(device)
                    # this is a param, not a module, so device placement is not in-place -> re-assign
                    if hasattr(module, "lora_magnitude_vector") and module.lora_magnitude_vector is not None:
                        if adapter_name in module.lora_magnitude_vector:
                            module.lora_magnitude_vector[adapter_name] = module.lora_magnitude_vector[adapter_name].to(
                                device
                            )

    def check_hotswap_diffusion(self, do_hotswap, ranks, alpha_scalings, target_modules):
        dummy_input = self.get_dummy_input()
        unet = self.get_small_unet()
        rank0, rank1 = ranks
        alpha0, alpha1 = alpha_scalings
        lora_config0 = self.get_unet_lora_config(rank0, alpha0, target_modules=target_modules)
        lora_config1 = self.get_unet_lora_config(rank1, alpha1, target_modules=target_modules)
        unet.add_adapter(lora_config0, adapter_name="adapter0")
        unet.add_adapter(lora_config1, adapter_name="adapter1")

        with tempfile.TemporaryDirectory() as tmp_dirname:
            unet.save_lora_adapter(os.path.join(tmp_dirname, "0"), safe_serialization=True, adapter_name="adapter0")
            unet.save_lora_adapter(os.path.join(tmp_dirname, "1"), safe_serialization=True, adapter_name="adapter1")
            del unet

            unet = self.get_small_unet()
            file_name0 = os.path.join(os.path.join(tmp_dirname, "0"), "pytorch_lora_weights.safetensors")
            file_name1 = os.path.join(os.path.join(tmp_dirname, "1"), "pytorch_lora_weights.safetensors")
            unet.load_lora_adapter(file_name0, safe_serialization=True, adapter_name="adapter0")

            prepare_model_for_compiled_hotswap(
                unet, config={"adapter0": lora_config0, "adapter1": lora_config1}, target_rank=max(ranks)
            )
            unet = torch.compile(unet, mode="reduce-overhead")
            unet(**dummy_input)["sample"]

            if do_hotswap:
                unet.load_lora_adapter(file_name1, adapter_name="adapter0", hotswap=True)
            else:
                # offloading the old and loading the new adapter will result in recompilation
                self.set_lora_device(unet, adapter_names=["adapter0"], device="cpu")
                unet.load_lora_adapter(file_name1, adapter_name="other_name", hotswap=False)

            # we need to call forward to potentially trigger recompilation
            unet(**dummy_input)["sample"]

    @pytest.mark.skipif(not is_diffusers_available(), reason="Test requires diffusers to be installed")
    @pytest.mark.xfail(
        strict=True,
        reason="Requires hotswap to be implemented in diffusers",
        raises=ValueError,
    )
    # it is important to check hotswapping small to large ranks and large to small ranks
    @pytest.mark.parametrize("ranks", [(11, 11), (7, 13), (13, 7)])
    @pytest.mark.parametrize(
        "target_modules",
        [
            ["to_q", "to_k", "to_v", "to_out.0"],  # Linear layers
            ["conv", "conv1", "conv2"],  # Conv2d layers
            ["to_q", "conv"],  # mix of Linear and Conv2d
        ],
    )
    def test_hotswapping_compiled_diffusers_model_does_not_trigger_recompilation(self, ranks, target_modules):
        with torch._dynamo.config.patch(error_on_recompile=True):  # raise an error on recompilation
            self.check_hotswap_diffusion(
                do_hotswap=True, ranks=ranks, alpha_scalings=ranks, target_modules=target_modules
            )
