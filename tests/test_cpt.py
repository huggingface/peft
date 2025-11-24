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

from typing import Any, Union

import pytest
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from peft import CPTConfig, TaskType, get_peft_model


TEMPLATE = {"input": "input: {}", "intra_seperator": " ", "output": "output: {}", "inter_seperator": "\n"}

MODEL_NAME = "peft-internal-testing/tiny-random-OPTForCausalLM"
MAX_INPUT_LENGTH = 1024


@pytest.fixture(scope="module")
def global_tokenizer():
    """Load the tokenizer fixture for the model."""

    return AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")


@pytest.fixture(scope="module")
def config_text():
    """Load the SST2 dataset and prepare it for testing."""
    config = CPTConfig(
        cpt_token_ids=[0, 1, 2, 3, 4, 5, 6, 7],  # Example token IDs for testing
        cpt_mask=[1, 1, 1, 1, 1, 1, 1, 1],
        cpt_tokens_type_mask=[1, 2, 2, 2, 3, 3, 3, 4],
        opt_weighted_loss_type="decay",
        opt_loss_decay_factor=0.95,
        opt_projection_epsilon=0.2,
        opt_projection_format_epsilon=0.1,
        tokenizer_name_or_path=MODEL_NAME,
        task_type=TaskType.CAUSAL_LM,
    )
    return config


@pytest.fixture(scope="module")
def config_random():
    """Load the SST2 dataset and prepare it for testing."""
    config = CPTConfig(
        opt_weighted_loss_type="decay",
        opt_loss_decay_factor=0.95,
        opt_projection_epsilon=0.2,
        opt_projection_format_epsilon=0.1,
        tokenizer_name_or_path=MODEL_NAME,
        task_type=TaskType.CAUSAL_LM,
    )
    return config


@pytest.fixture(scope="module")
def sst_data():
    """Load the SST2 dataset and prepare it for testing."""
    data = load_dataset("glue", "sst2")

    def add_string_labels(example):
        if example["label"] == 0:
            example["label_text"] = "negative"
        elif example["label"] == 1:
            example["label_text"] = "positive"
        return example

    train_dataset = data["train"].select(range(4)).map(add_string_labels)
    test_dataset = data["validation"].select(range(10)).map(add_string_labels)

    return {"train": train_dataset, "test": test_dataset}


@pytest.fixture(scope="module")
def collator(global_tokenizer):
    class CPTDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
        def __init__(self, tokenizer, training=True, mlm=False):
            super().__init__(tokenizer, mlm=mlm)
            self.training = training
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # mk check why needed

        def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
            # Handle dict or lists with proper padding and conversion to tensor.
            list_sample_mask = []
            for i in range(len(examples)):
                if "sample_mask" in examples[i].keys():
                    list_sample_mask.append(examples[i].pop("sample_mask"))

            max_len = max(len(ex["input_ids"]) for ex in examples)

            def pad_sequence(sequence, max_len, pad_value=0):
                return sequence + [pad_value] * (max_len - len(sequence))

            input_ids = torch.tensor([pad_sequence(ex["input_ids"], max_len) for ex in examples])
            attention_mask = torch.tensor([pad_sequence(ex["attention_mask"], max_len) for ex in examples])
            input_type_mask = torch.tensor([pad_sequence(ex["input_type_mask"], max_len) for ex in examples])

            batch = {"input_ids": input_ids, "attention_mask": attention_mask, "input_type_mask": input_type_mask}

            tensor_sample_mask = batch["input_ids"].clone().long()
            tensor_sample_mask[:, :] = 0
            for i in range(len(list_sample_mask)):
                tensor_sample_mask[i, : len(list_sample_mask[i])] = list_sample_mask[i]

            batch["labels"] = batch["input_ids"].clone()
            if not self.training:
                batch["sample_mask"] = tensor_sample_mask

            return batch

    collator = CPTDataCollatorForLanguageModeling(global_tokenizer, training=True, mlm=False)
    return collator


def dataset(data, tokenizer):
    class CPTDataset(Dataset):
        def __init__(self, samples, tokenizer, template, max_length=MAX_INPUT_LENGTH):
            self.template = template
            self.tokenizer = tokenizer
            self.max_length = max_length

            self.attention_mask = []
            self.input_ids = []
            self.input_type_mask = []
            self.inter_seperator_ids = self._get_input_ids(template["inter_seperator"])

            for sample_i in tqdm(samples):
                input_text, label = sample_i["sentence"], sample_i["label_text"]
                input_ids, attention_mask, input_type_mask = self.preprocess_sentence(input_text, label)

                self.input_ids.append(input_ids)
                self.attention_mask.append(attention_mask)
                self.input_type_mask.append(input_type_mask)

        def _get_input_ids(self, text):
            return self.tokenizer(text, add_special_tokens=False)["input_ids"]

        def preprocess_sentence(self, input_text, label):
            input_template_part_1_text, input_template_part_2_text = self.template["input"].split("{}")
            input_template_tokenized_part1 = self._get_input_ids(input_template_part_1_text)
            input_tokenized = self._get_input_ids(input_text)
            input_template_tokenized_part2 = self._get_input_ids(input_template_part_2_text)

            sep_tokenized = self._get_input_ids(self.template["intra_seperator"])

            label_template_part_1, label_template_part_2 = self.template["output"].split("{}")
            label_template_part1_tokenized = self._get_input_ids(label_template_part_1)
            label_tokenized = self._get_input_ids(label)
            label_template_part2_tokenized = self._get_input_ids(label_template_part_2)

            eos = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []
            input_ids = (
                input_template_tokenized_part1
                + input_tokenized
                + input_template_tokenized_part2
                + sep_tokenized
                + label_template_part1_tokenized
                + label_tokenized
                + label_template_part2_tokenized
                + eos
            )

            # determine label tokens, to calculate loss only over them when labels_loss == True
            attention_mask = [1] * len(input_ids)
            input_type_mask = (
                [1] * len(input_template_tokenized_part1)
                + [2] * len(input_tokenized)
                + [1] * len(input_template_tokenized_part2)
                + [0] * len(sep_tokenized)
                + [3] * len(label_template_part1_tokenized)
                + [4] * len(label_tokenized)
                + [3] * len(label_template_part2_tokenized)
                + [0] * len(eos)
            )

            assert len(input_type_mask) == len(input_ids) == len(attention_mask)

            return input_ids, attention_mask, input_type_mask

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "input_type_mask": self.input_type_mask[idx],
            }

    dataset = CPTDataset(data, tokenizer, TEMPLATE)

    return dataset


def test_model_initialization_text(global_tokenizer, config_text):
    """Test model loading and PEFT model initialization."""
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    model = get_peft_model(base_model, config_text)
    assert model is not None, "PEFT model initialization failed"


def test_model_initialization_random(global_tokenizer, config_random):
    """Test model loading and PEFT model initialization."""
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    model = get_peft_model(base_model, config_random)
    assert model is not None, "PEFT model initialization failed"


def test_model_initialization_wrong_task_type_raises():
    msg = "CPTConfig only supports task_type = CAUSAL_LM."
    with pytest.raises(ValueError, match=msg):
        CPTConfig(task_type=TaskType.SEQ_CLS)

    msg = "CPTConfig only supports task_type = CAUSAL_LM."
    with pytest.raises(ValueError, match=msg):
        CPTConfig()


def test_model_training_random(sst_data, global_tokenizer, collator, config_random):
    """Perform a short training run to verify the model and data integration."""

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = get_peft_model(base_model, config_random)
    emb = model.prompt_encoder.default.embedding.weight.data.clone().detach()
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=1,
        num_train_epochs=2,
        remove_unused_columns=False,
        save_strategy="no",
        logging_steps=1,
    )

    train_dataset = dataset(sst_data["train"], global_tokenizer)

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, data_collator=collator)

    trainer.train()
    # Verify that the embedding tensor remains unchanged (frozen)
    assert torch.all(model.prompt_encoder.default.embedding.weight.data.clone().detach().cpu() == emb.cpu())

    delta_emb = model.prompt_encoder.default.get_projection().clone().detach()
    norm_delta = delta_emb.norm(dim=1).cpu()
    epsilon = model.prompt_encoder.default.get_epsilon().cpu()
    # Verify that the change in tokens is constrained to epsilon
    assert torch.all(norm_delta <= epsilon)


def test_model_batch_training_text(sst_data, global_tokenizer, collator, config_text):
    """Perform a short training run to verify the model and data integration."""

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = get_peft_model(base_model, config_text)
    emb = model.prompt_encoder.default.embedding.weight.data.clone().detach()

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=2,
        num_train_epochs=2,
        remove_unused_columns=False,
        save_strategy="no",
        logging_steps=1,
    )

    train_dataset = dataset(sst_data["train"], global_tokenizer)

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, data_collator=collator)

    trainer.train()
    # Verify that the embedding tensor remains unchanged (frozen)
    assert torch.all(model.prompt_encoder.default.embedding.weight.data.clone().detach().cpu() == emb.cpu())

    cpt_tokens_type_mask = torch.Tensor(config_text.cpt_tokens_type_mask).long()
    non_label_idx = (cpt_tokens_type_mask == 1) | (cpt_tokens_type_mask == 2) | (cpt_tokens_type_mask == 3)

    delta_emb = model.prompt_encoder.default.get_projection().clone().detach()
    norm_delta = delta_emb.norm(dim=1).cpu()
    epsilon = model.prompt_encoder.default.get_epsilon().cpu()
    # Verify that the change in tokens is constrained to epsilon
    assert torch.all(norm_delta <= epsilon)
    # Ensure that label tokens remain unchanged
    assert torch.all((norm_delta == 0) == (~non_label_idx))
