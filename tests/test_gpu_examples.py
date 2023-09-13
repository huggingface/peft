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
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
)

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
            self.assertTrue("adapter_model.bin" in os.listdir(tmp_dir))

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
            self.assertTrue("adapter_model.bin" in os.listdir(tmp_dir))

            # assert loss is not None
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

    @pytest.mark.multi_gpu_tests
    def test_causal_lm_training_mutli_gpu_4bit(self):
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

            self.assertEqual(set(model.hf_device_map.values()), {0, 1})

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
            self.assertTrue("adapter_model.bin" in os.listdir(tmp_dir))

            # assert loss is not None
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

    @pytest.mark.single_gpu_tests
    @require_torch_gpu
    def test_4bit_adalora_causalLM(self):
        r"""
        Tests the 4bit training with adalora
        """
        model_id = "facebook/opt-350m"

        model = AutoModelForCausalLM.from_pretrained(model_id, load_in_4bit=True)
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
            self.assertTrue("adapter_model.bin" in os.listdir(tmp_dir))

            # assert loss is not None
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

    @pytest.mark.multi_gpu_tests
    @require_torch_multi_gpu
    def test_causal_lm_training_mutli_gpu(self):
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

            self.assertEqual(set(model.hf_device_map.values()), {0, 1})

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
            self.assertTrue("adapter_model.bin" in os.listdir(tmp_dir))

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
            self.assertTrue("adapter_model.bin" in os.listdir(tmp_dir))

            # assert loss is not None
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

    @pytest.mark.multi_gpu_tests
    @require_torch_multi_gpu
    def test_seq2seq_lm_training_mutli_gpu(self):
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

            self.assertEqual(set(model.hf_device_map.values()), {0, 1})

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
            self.assertTrue("adapter_model.bin" in os.listdir(tmp_dir))

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
            self.assertTrue("adapter_model.bin" in os.listdir(tmp_dir))

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
        self.quantization_config = GPTQConfig(bits=4, disable_exllama=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.causal_lm_model_id)

    def tearDown(self):
        r"""
        Efficient mechanism to free GPU memory after each test. Based on
        https://github.com/huggingface/transformers/issues/21094
        """
        gc.collect()
        torch.cuda.empty_cache()

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
            self.assertTrue("adapter_model.bin" in os.listdir(tmp_dir))

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
            self.assertTrue("adapter_model.bin" in os.listdir(tmp_dir))

            # assert loss is not None
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])

    @pytest.mark.multi_gpu_tests
    @require_torch_multi_gpu
    def test_causal_lm_training_mutli_gpu(self):
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

            self.assertEqual(set(model.hf_device_map.values()), {0, 1})

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
            self.assertTrue("adapter_model.bin" in os.listdir(tmp_dir))

            # assert loss is not None
            self.assertIsNotNone(trainer.state.log_history[-1]["train_loss"])
