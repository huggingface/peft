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
import importlib
import os
import tempfile
import unittest
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import pytest
import torch
from accelerate import infer_auto_device_map
from accelerate.test_utils.testing import run_command
from accelerate.utils import patch_environment
from datasets import Audio, DatasetDict, load_dataset
from packaging import version
from parameterized import parameterized
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
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
)
from peft.utils import SAFETENSORS_WEIGHTS_NAME

from .testing_utils import (
    require_aqlm,
    require_auto_awq,
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
                load_in_4bit=True,
            )

            assert set(model.hf_device_map.values()) == set(range(torch.cuda.device_count()))

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
    @require_torch_gpu
    def test_4bit_adalora_causalLM(self):
        r"""
        Tests the 4bit training with adalora
        """
        model_id = "facebook/opt-350m"

        # for >3 GPUs, might need: device_map={"": "cuda:0"}
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

            assert "adapter_config.json" in os.listdir(tmp_dir)
            assert SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir)

            # assert loss is not None
            assert trainer.state.log_history[-1]["train_loss"] is not None

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

            assert "adapter_config.json" in os.listdir(tmp_dir)
            assert SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir)

            # assert loss is not None
            assert trainer.state.log_history[-1]["train_loss"] is not None

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

            assert set(model.hf_device_map.values()) == set(range(torch.cuda.device_count()))

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
                load_in_8bit=True,
                device_map={"": 0},
            )

            assert set(model.hf_device_map.values()) == {0}

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

            assert "adapter_config.json" in os.listdir(tmp_dir)
            assert SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir)

            # assert loss is not None
            assert trainer.state.log_history[-1]["train_loss"] is not None

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

            assert set(model.hf_device_map.values()) == set(range(torch.cuda.device_count()))

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

            assert "adapter_config.json" in os.listdir(tmp_dir)
            assert SAFETENSORS_WEIGHTS_NAME in os.listdir(tmp_dir)

            # assert loss is not None
            assert trainer.state.log_history[-1]["train_loss"] is not None

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
            load_in_4bit=True,
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, config)
        n_trainable_default, n_total_default = model.get_nb_trainable_parameters()

        # other adapter name
        model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            device_map="auto",
            load_in_4bit=True,
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
            load_in_8bit=True,
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, config)
        n_trainable_default, n_total_default = model.get_nb_trainable_parameters()

        # other adapter name
        model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            device_map="auto",
            load_in_8bit=True,
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, config, adapter_name="other")
        n_trainable_other, n_total_other = model.get_nb_trainable_parameters()

        assert n_trainable_other > 0
        # sanity check
        assert n_trainable_default == n_trainable_other
        assert n_total_default == n_total_other


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

            assert set(model.hf_device_map.values()) == set(range(torch.cuda.device_count()))

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


@require_torch_gpu
class OffloadSaveTests(unittest.TestCase):
    def setUp(self):
        self.causal_lm_model_id = "gpt2"

    def tearDown(self):
        r"""
        Efficient mechanism to free GPU memory after each test. Based on
        https://github.com/huggingface/transformers/issues/21094
        """
        gc.collect()
        torch.cuda.empty_cache()

    @pytest.mark.single_gpu_tests
    @require_torch_gpu
    def test_offload_merge(self):
        r"""
        Test merging, unmerging, and unloading of a model with CPU- offloaded modules.
        """
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_pretrained(self.causal_lm_model_id)
        tokenizer = AutoTokenizer.from_pretrained(self.causal_lm_model_id)
        # TODO: add disk offload once PeftModel.from_pretrained supports
        memory_limits = {0: "0.4GIB", "cpu": "5GIB"}
        # offloads around half of all transformer modules
        device_map = infer_auto_device_map(model, max_memory=memory_limits)
        assert 0 in device_map.values()
        assert "cpu" in device_map.values()

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


@require_torch_gpu
class LoftQTests(unittest.TestCase):
    r"""
    Tests for LoftQ to ensure that it reduces the quantization error compared to normal LoRA quantization.
    """

    def setUp(self):
        self.error_factor = 3

    def get_input(self, model_id, device):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = tokenizer("All I want is", padding=True, return_tensors="pt")
        if device == "cuda":
            inputs = inputs.to("cuda")
        return inputs

    def get_base_model(self, model_id, device, **kwargs):
        cls = AutoModelForSeq2SeqLM if "t5" in model_id else AutoModelForCausalLM
        model = cls.from_pretrained(model_id, **kwargs).eval()
        if device == "cuda":
            model = model.to("cuda")
        return model

    def get_logits(self, model, inputs):
        if model.config.is_encoder_decoder:
            input_ids = inputs["input_ids"]
            return model(input_ids=input_ids, decoder_input_ids=input_ids).logits
        return model(**inputs).logits

    def get_errors(
        self, bits=4, loftq_iter=1, device="cuda", model_id="hf-internal-testing/tiny-random-BloomForCausalLM"
    ):
        # Helper function that returns the quantization errors (MAE and MSE) when comparing the quantized LoRA model
        # to the base model, vs the LoftQ quantized model to the base model. We expect the LoftQ quantized model to
        # have less error than the normal LoRA quantized model. Since we compare logits, the observed error is
        # already somewhat dampened because of the softmax.
        torch.manual_seed(0)
        model = self.get_base_model(model_id, device)
        task_type = TaskType.SEQ_2_SEQ_LM if model.config.is_encoder_decoder else TaskType.CAUSAL_LM
        inputs = self.get_input(model_id, device)
        logits_base = self.get_logits(model, inputs)
        # clean up
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # logits from the normal quantized LoRA model
        lora_config = LoraConfig(task_type=task_type)
        kwargs = {}
        if bits == 4:
            kwargs["load_in_4bit"] = True
        elif bits == 8:
            kwargs["load_in_8bit"] = True
        else:
            raise ValueError("bits must be 4 or 8")

        quantized_model = get_peft_model(
            self.get_base_model(model_id, device=None, **kwargs),
            lora_config,
        )
        torch.manual_seed(0)
        logits_quantized = self.get_logits(quantized_model, inputs)
        del quantized_model
        gc.collect()
        torch.cuda.empty_cache()

        # logits from quantized LoRA model using LoftQ
        loftq_config = LoftQConfig(loftq_bits=bits, loftq_iter=loftq_iter)
        lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, init_lora_weights="loftq", loftq_config=loftq_config)
        model = self.get_base_model(model_id, device)
        if device == "cuda":
            model = model.to("cuda")
        loftq_model = get_peft_model(model, lora_config)
        if device == "cuda":
            loftq_model = loftq_model.to("cuda")

        torch.manual_seed(0)
        logits_loftq = self.get_logits(loftq_model, inputs)
        del loftq_model
        gc.collect()
        torch.cuda.empty_cache()

        mae_quantized = torch.abs(logits_base - logits_quantized).mean()
        mse_quantized = torch.pow(logits_base - logits_quantized, 2).mean()
        mae_loftq = torch.abs(logits_base - logits_loftq).mean()
        mse_loftq = torch.pow(logits_base - logits_loftq, 2).mean()
        return mae_quantized, mse_quantized, mae_loftq, mse_loftq

    @parameterized.expand(["cuda", "cpu"])
    def test_bloomz_loftq_4bit(self, device):
        # In this test, we compare the logits of the base model, the quantized LoRA model, and the quantized model
        # using LoftQ. When quantizing, we expect a certain level of error. However, we expect the LoftQ quantized
        # model to have less error than the normal LoRA quantized model. Note that when using normal LoRA, the
        # quantization error is simply the error from quantization without LoRA, as LoRA is a no-op before training.
        # We still apply LoRA for the test for consistency.

        mae_quantized, mse_quantized, mae_loftq, mse_loftq = self.get_errors(bits=4, device=device)
        # first, sanity check that all errors are > 0.0
        assert mae_quantized > 0.0
        assert mse_quantized > 0.0
        assert mae_loftq > 0.0
        assert mse_loftq > 0.0

        # next, check that LoftQ quantization errors are smaller than LoRA errors by a certain margin
        factor = 3
        assert mae_loftq < (mae_quantized / factor)
        assert mse_loftq < (mse_quantized / factor)

    @parameterized.expand(["cuda", "cpu"])
    def test_bloomz_loftq_4bit_iter_5(self, device):
        # Same test as the previous one but with 5 iterations. We should expect the error to be even smaller with more
        # iterations, but in practice the difference is not that large, at least not for this small base model.
        mae_quantized, mse_quantized, mae_loftq, mse_loftq = self.get_errors(bits=4, loftq_iter=5, device=device)
        # first, sanity check that all errors are > 0.0
        assert mae_quantized > 0.0
        assert mse_quantized > 0.0
        assert mae_loftq > 0.0
        assert mse_loftq > 0.0

        # next, check that LoftQ quantization errors are smaller than LoRA errors by a certain margin
        assert mae_loftq < (mae_quantized / self.error_factor)
        assert mse_loftq < (mse_quantized / self.error_factor)

    @parameterized.expand(["cuda", "cpu"])
    def test_bloomz_loftq_8bit(self, device):
        # Same test as test_bloomz_loftq_4bit but with 8 bits.
        mae_quantized, mse_quantized, mae_loftq, mse_loftq = self.get_errors(bits=8, device=device)

        # first, sanity check that all errors are > 0.0
        assert mae_quantized > 0.0
        assert mse_quantized > 0.0
        assert mae_loftq > 0.0
        assert mse_loftq > 0.0

        # next, check that LoftQ quantization errors are smaller than LoRA errors by a certain margin
        assert mae_loftq < (mae_quantized / self.error_factor)
        assert mse_loftq < (mse_quantized / self.error_factor)

    @parameterized.expand(["cuda", "cpu"])
    def test_bloomz_loftq_8bit_iter_5(self, device):
        # Same test as test_bloomz_loftq_4bit_iter_5 but with 8 bits.
        mae_quantized, mse_quantized, mae_loftq, mse_loftq = self.get_errors(bits=8, loftq_iter=5, device=device)

        # first, sanity check that all errors are > 0.0
        assert mae_quantized > 0.0
        assert mse_quantized > 0.0
        assert mae_loftq > 0.0
        assert mse_loftq > 0.0

        # next, check that LoftQ quantization errors are smaller than LoRA errors by a certain margin
        assert mae_loftq < (mae_quantized / self.error_factor)
        assert mse_loftq < (mse_quantized / self.error_factor)

    @parameterized.expand(["cuda", "cpu"])
    def test_t5_loftq_4bit(self, device):
        mae_quantized, mse_quantized, mae_loftq, mse_loftq = self.get_errors(
            bits=4, device=device, model_id="t5-small"
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

    @parameterized.expand(["cuda", "cpu"])
    def test_t5_loftq_8bit(self, device):
        mae_quantized, mse_quantized, mae_loftq, mse_loftq = self.get_errors(
            bits=8, device=device, model_id="t5-small"
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


@require_bitsandbytes
@require_torch_gpu
class MultiprocessTester(unittest.TestCase):
    def test_notebook_launcher(self):
        script_path = os.path.join("scripts", "launch_notebook_mp.py")
        cmd = ["python", script_path]
        with patch_environment(omp_num_threads=1):
            run_command(cmd, env=os.environ.copy())


@require_torch_gpu
class MixedPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.causal_lm_model_id = "facebook/opt-350m"
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
        gc.collect()

    @pytest.mark.single_gpu_tests
    def test_model_loaded_in_float16_raises(self):
        # This test shows the issue with loading the model in fp16 and then trying to use it with mixed precision
        # training, which should not use fp16. If this is ever automated in PEFT, this test should fail. In that case,
        # remove this test, adjust the next one, and remove the entry about FP16 usage from troubleshooting.md.
        model = AutoModelForCausalLM.from_pretrained(
            self.causal_lm_model_id,
            torch_dtype=torch.float16,
        )
        model = get_peft_model(model, self.config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = Trainer(
                model=model,
                train_dataset=self.data["train"],
                args=TrainingArguments(
                    fp16=True,  # <= this is required for the error to be raised
                    logging_steps=1,
                    output_dir=tmp_dir,
                ),
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            )
            with pytest.raises(ValueError, match="Attempting to unscale FP16 gradients."):
                trainer.train()

    @pytest.mark.single_gpu_tests
    def test_model_loaded_in_float16_working(self):
        # Same test as before but containing the fix to make it work
        model = AutoModelForCausalLM.from_pretrained(
            self.causal_lm_model_id,
            torch_dtype=torch.float16,
        )
        model = get_peft_model(model, self.config)

        # for now, this is unfortunately necessary to avoid the error:
        # ValueError: Attempting to unscale FP16 gradients.
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.float()

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = Trainer(
                model=model,
                train_dataset=self.data["train"],
                args=TrainingArguments(
                    fp16=True,
                    max_steps=3,
                    output_dir=tmp_dir,
                ),
                data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            )
            trainer.train()


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

            assert set(model.hf_device_map.values()) == set(range(torch.cuda.device_count()))

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


@require_torch_gpu
class TestAutoCast(unittest.TestCase):
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
        model = model.cuda()

        # Prepare dummy inputs
        input_ids = torch.randint(0, 1000, (2, 10)).cuda()
        if precision == torch.bfloat16:
            if not torch.cuda.is_bf16_supported():
                self.skipTest("Bfloat16 not supported on this device")

        # Forward pass with test precision
        with torch.autocast(enabled=True, dtype=precision, device_type="cuda"):
            outputs = model(input_ids)
            assert outputs.dtype == precision
