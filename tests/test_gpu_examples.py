# coding=utf-8
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
import gc
import os
import tempfile
import unittest
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import pytest
import torch
from accelerate.test_utils.testing import run_command
from accelerate.utils import patch_environment
from datasets import Audio, DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

from peft import (
    AdaLoraConfig,
    LoftQConfig,
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
)
from peft.utils import SAFETENSORS_WEIGHTS_NAME

from .testing_utils import (
    require_auto_gptq,
    require_bitsandbytes,
    require_optimum,
    require_torch_gpu,
    require_torch_multi_gpu,
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


@require_torch_gpu
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
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _check_inference_finite(self, model, batch):
        # try inference without Trainer class
        training = model.training
        model.eval()
        output = model(**batch.to(model.device))
        self.assertTrue(torch.isfinite(output.logits).all())
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
                load_in_8bit=True,
                device_map="auto",
            )

            tokenizer = AutoTokenizer.from_pretrained(self.causal_lm_model_id)
            model = prepare_model_for_int8_training(model)

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

            self.assertTrue("adapter_config.json" in os.listdir(tmp_dir))
            self.assertTrue(SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir))

            # assert loss is not None
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

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
                load_in_4bit=True,
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

            self.assertTrue("adapter_config.json" in os.listdir(tmp_dir))
            self.assertTrue(SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir))

            # assert loss is not None
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

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
                load_in_4bit=True,
            )

            self.assertEqual(set(model.hf_device_map.values()), set(range(torch.cuda.device_count())))

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

            self.assertTrue("adapter_config.json" in os.listdir(tmp_dir))
            self.assertTrue(SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir))

            # assert loss is not None
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

    @pytest.mark.single_gpu_tests
    @require_torch_gpu
    def test_4bit_adalora_causalLM(self):
        r"""
        Tests the 4bit training with adalora
        """
        model_id = "facebook/opt-350m"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            load_in_4bit=True,
            device_map={"": "cuda:0"},  # fix for >3 GPUs
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        peft_config = AdaLoraConfig(
            init_r=6,
            target_r=4,
            tinit=50,
            tfinal=100,
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
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            )
            model.config.use_cache = False
            trainer.train()

            model.cpu().save_pretrained(tmp_dir)

            self.assertTrue("adapter_config.json" in os.listdir(tmp_dir))
            self.assertTrue(SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir))

            # assert loss is not None
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

    @pytest.mark.single_gpu_tests
    @require_torch_gpu
    def test_8bit_adalora_causalLM(self):
        r"""
        Tests the 8bit training with adalora
        """
        model_id = "facebook/opt-350m"

        model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        peft_config = AdaLoraConfig(
            init_r=6,
            target_r=4,
            tinit=50,
            tfinal=100,
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
                data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            )
            model.config.use_cache = False
            trainer.train()

            model.cpu().save_pretrained(tmp_dir)

            self.assertTrue("adapter_config.json" in os.listdir(tmp_dir))
            self.assertTrue(SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir))

            # assert loss is not None
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

    @pytest.mark.multi_gpu_tests
    @require_torch_multi_gpu
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
                load_in_8bit=True,
                device_map="auto",
            )

            self.assertEqual(set(model.hf_device_map.values()), set(range(torch.cuda.device_count())))

            tokenizer = AutoTokenizer.from_pretrained(self.causal_lm_model_id)
            model = prepare_model_for_int8_training(model)

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

            self.assertTrue("adapter_config.json" in os.listdir(tmp_dir))
            self.assertTrue(SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir))

            # assert loss is not None
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

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
                load_in_8bit=True,
                device_map={"": 0},
            )

            self.assertEqual(set(model.hf_device_map.values()), {0})

            tokenizer = AutoTokenizer.from_pretrained(self.seq2seq_model_id)
            model = prepare_model_for_int8_training(model)

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

            self.assertTrue("adapter_config.json" in os.listdir(tmp_dir))
            self.assertTrue(SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir))

            # assert loss is not None
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

    @pytest.mark.multi_gpu_tests
    @require_torch_multi_gpu
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
                load_in_8bit=True,
                device_map="balanced",
            )

            self.assertEqual(set(model.hf_device_map.values()), set(range(torch.cuda.device_count())))

            tokenizer = AutoTokenizer.from_pretrained(self.seq2seq_model_id)
            model = prepare_model_for_int8_training(model)

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

            self.assertTrue("adapter_config.json" in os.listdir(tmp_dir))
            self.assertTrue(SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir))

            # assert loss is not None
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

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
                self.audio_model_id, load_in_8bit=True, device_map="auto"
            )

            model.config.forced_decoder_ids = None
            model.config.suppress_tokens = []

            model = prepare_model_for_int8_training(model)

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

            self.assertTrue("adapter_config.json" in os.listdir(tmp_dir))
            self.assertTrue(SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir))

            # assert loss is not None
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])


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
        gc.collect()
        torch.cuda.empty_cache()

    def _check_inference_finite(self, model, batch):
        # try inference without Trainer class
        training = model.training
        model.eval()
        output = model(**batch.to(model.device))
        self.assertTrue(torch.isfinite(output.logits).all())
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

            self.assertTrue("adapter_config.json" in os.listdir(tmp_dir))
            self.assertTrue(SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir))

            # assert loss is not None
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

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
            tinit=50,
            tfinal=100,
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

            self.assertTrue("adapter_config.json" in os.listdir(tmp_dir))
            self.assertTrue(SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir))

            # assert loss is not None
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

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

            self.assertEqual(set(model.hf_device_map.values()), set(range(torch.cuda.device_count())))

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

            self.assertTrue("adapter_config.json" in os.listdir(tmp_dir))
            self.assertTrue(SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir))

            # assert loss is not None
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])


@require_torch_gpu
class LoftQTests(unittest.TestCase):
    r"""
    Tests for LoftQ
    """

    def setUp(self):
        self.error_factor = 3
        self.model_id = "hf-internal-testing/tiny-random-BloomForCausalLM"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.inputs = self.tokenizer("All I want is", padding=True, return_tensors="pt").to("cuda")

    def get_errors(self, bits=4, loftq_iter=1):
        # Helper function that returns the quantization errors (MAE and MSE) when comparing the quantized LoRA model
        # to the base model, vs the LoftQ quantized model to the base model. We expect the LoftQ quantized model to
        # have less error than the normal LoRA quantized model. Since we compare logits, the observed error is
        # already somewhat dampened because of the softmax.
        model = AutoModelForCausalLM.from_pretrained(self.model_id).cuda().eval()
        torch.manual_seed(0)
        logits_base = model(**self.inputs).logits
        # clean up
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # logits from the normal quantized LoRA model
        lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM)
        kwargs = {}
        if bits == 4:
            kwargs["load_in_4bit"] = True
        elif bits == 8:
            kwargs["load_in_8bit"] = True
        else:
            raise ValueError("bits must be 4 or 8")

        quantized_model = get_peft_model(
            AutoModelForCausalLM.from_pretrained(self.model_id, device_map="auto", **kwargs).eval(),
            lora_config,
        )
        torch.manual_seed(0)
        logits_quantized = quantized_model(**self.inputs).logits
        del quantized_model
        gc.collect()
        torch.cuda.empty_cache()

        # logits from quantized LoRA model using LoftQ
        loftq_config = LoftQConfig(loftq_bits=bits, loftq_iter=loftq_iter)
        lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, init_lora_weights="loftq", loftq_config=loftq_config)
        loftq_model = get_peft_model(AutoModelForCausalLM.from_pretrained(self.model_id).cuda().eval(), lora_config)
        torch.manual_seed(0)
        logits_loftq = loftq_model(**self.inputs).logits
        del loftq_model
        gc.collect()
        torch.cuda.empty_cache()

        mae_quantized = torch.abs(logits_base - logits_quantized).mean()
        mse_quantized = torch.pow(logits_base - logits_quantized, 2).mean()
        mae_loftq = torch.abs(logits_base - logits_loftq).mean()
        mse_loftq = torch.pow(logits_base - logits_loftq, 2).mean()
        return mae_quantized, mse_quantized, mae_loftq, mse_loftq

    def test_bloomz_loftq_4bit(self):
        # In this test, we compare the logits of the base model, the quantized LoRA model, and the quantized model
        # using LoftQ. When quantizing, we expect a certain level of error. However, we expect the LoftQ quantized
        # model to have less error than the normal LoRA quantized model. Note that when using normal LoRA, the
        # quantization error is simply the error from quantization without LoRA, as LoRA is a no-op before training.
        # We still apply LoRA for the test for consistency.

        mae_quantized, mse_quantized, mae_loftq, mse_loftq = self.get_errors(bits=4)
        # first, sanity check that all errors are > 0.0
        self.assertTrue(mae_quantized > 0.0)
        self.assertTrue(mse_quantized > 0.0)
        self.assertTrue(mae_loftq > 0.0)
        self.assertTrue(mse_loftq > 0.0)

        # next, check that LoftQ quantization errors are smaller than LoRA errors by a certain margin
        factor = 3
        self.assertTrue(mae_loftq < mae_quantized / factor)
        self.assertTrue(mse_loftq < mse_quantized / factor)

    def test_bloomz_loftq_4bit_iter_5(self):
        # Same test as the previous one but with 5 iterations. We should expect the error to be even smaller with more
        # iterations, but in practice the difference is not that large, at least not for this small base model.
        mae_quantized, mse_quantized, mae_loftq, mse_loftq = self.get_errors(bits=4, loftq_iter=5)
        # first, sanity check that all errors are > 0.0
        self.assertTrue(mae_quantized > 0.0)
        self.assertTrue(mse_quantized > 0.0)
        self.assertTrue(mae_loftq > 0.0)
        self.assertTrue(mse_loftq > 0.0)

        # next, check that LoftQ quantization errors are smaller than LoRA errors by a certain margin
        self.assertTrue(mae_loftq < mae_quantized / self.error_factor)
        self.assertTrue(mse_loftq < mse_quantized / self.error_factor)

    def test_bloomz_loftq_8bit(self):
        # this currently does not work:
        # https://github.com/huggingface/peft/pull/1150#issuecomment-1838891499
        if True:  # TODO: remove as soon as the issue is fixed
            return

        # Same test as test_bloomz_loftq_4bit but with 8 bits.
        mae_quantized, mse_quantized, mae_loftq, mse_loftq = self.get_errors(bits=8)

        # first, sanity check that all errors are > 0.0
        self.assertTrue(mae_quantized > 0.0)
        self.assertTrue(mse_quantized > 0.0)
        self.assertTrue(mae_loftq > 0.0)
        self.assertTrue(mse_loftq > 0.0)

        # next, check that LoftQ quantization errors are smaller than LoRA errors by a certain margin
        self.assertTrue(mae_loftq < mae_quantized / self.error_factor)
        self.assertTrue(mse_loftq < mse_quantized / self.error_factor)

    def test_bloomz_loftq_8bit_iter_5(self):
        # this currently does not work:
        # https://github.com/huggingface/peft/pull/1150#issuecomment-1838891499
        if True:  # TODO: remove as soon as the issue is fixed
            return

        # Same test as test_bloomz_loftq_4bit_iter_5 but with 8 bits.
        mae_quantized, mse_quantized, mae_loftq, mse_loftq = self.get_errors(bits=8, loftq_iter=5)

        # first, sanity check that all errors are > 0.0
        self.assertTrue(mae_quantized > 0.0)
        self.assertTrue(mse_quantized > 0.0)
        self.assertTrue(mae_loftq > 0.0)
        self.assertTrue(mse_loftq > 0.0)

        # next, check that LoftQ quantization errors are smaller than LoRA errors by a certain margin
        self.assertTrue(mae_loftq < mae_quantized / self.error_factor)
        self.assertTrue(mse_loftq < mse_quantized / self.error_factor)


@require_bitsandbytes
@require_torch_gpu
class MultiprocessTester(unittest.TestCase):
    def test_notebook_launcher(self):
        script_path = os.path.join("scripts", "launch_notebook_mp.py")
        cmd = ["python", script_path]
        with patch_environment(omp_num_threads=1):
            run_command(cmd, env=os.environ.copy())
