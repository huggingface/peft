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


import torch
from datasets import load_dataset


def load_scienceqa(num_train=1000, num_eval=200, seed=42):
    """
    Load ScienceQA dataset for science question answering.

    Args:
        num_train: Number of training samples
        num_eval: Number of evaluation samples
        seed: Random seed for reproducibility

    Returns:
        train_dataset, eval_dataset
    """
    dataset = load_dataset("derek-thomas/ScienceQA", split="train")

    # Shuffle and split
    dataset = dataset.shuffle(seed=seed)
    train_dataset = dataset.select(range(num_train))
    eval_dataset = dataset.select(range(num_train, num_train + num_eval))

    return train_dataset, eval_dataset


def load_numglue(num_train=1000, num_eval=200, seed=42):
    """
    Load NumGLUE dataset for mathematical reasoning.

    Args:
        num_train: Number of training samples
        num_eval: Number of evaluation samples
        seed: Random seed for reproducibility

    Returns:
        train_dataset, eval_dataset
    """
    import json

    from datasets import Dataset
    from huggingface_hub import hf_hub_download

    # Download the NumGLUE JSON file manually
    json_path = hf_hub_download(repo_id="metaeval/num-glue", filename="NumGLUE_train.json", repo_type="dataset")

    # Read and process the JSON file line by line
    data = []
    with open(json_path) as f:
        for line in f:
            if line.strip():  # Skip empty lines
                item = json.loads(line)
                # Extract the number from the answer JSON structure
                answer = item.get("answer", "")
                if isinstance(answer, dict):
                    # NumGLUE answers are JSON with 'number' and 'date' fields
                    # Extract just the number field
                    answer_str = answer.get("number", "")
                else:
                    answer_str = str(answer)

                data.append({"question": item.get("question", ""), "answer": answer_str})

    # Create dataset from processed data
    dataset = Dataset.from_list(data)

    # Shuffle and split
    dataset = dataset.shuffle(seed=seed)
    train_dataset = dataset.select(range(min(num_train, len(dataset))))

    # If not enough samples, use what's available
    eval_start = min(num_train, len(dataset))
    eval_end = min(num_train + num_eval, len(dataset))
    eval_dataset = dataset.select(range(eval_start, eval_end))

    return train_dataset, eval_dataset


def load_fomc(num_train=1000, num_eval=200, seed=42):
    """
    Load FOMC dataset for financial sentiment classification.

    Args:
        num_train: Number of training samples
        num_eval: Number of evaluation samples
        seed: Random seed for reproducibility

    Returns:
        train_dataset, eval_dataset
    """
    dataset = load_dataset("TheFinAI/finben-fomc", split="test")

    # Shuffle and split
    dataset = dataset.shuffle(seed=seed)
    train_dataset = dataset.select(range(min(num_train, len(dataset))))

    eval_start = min(num_train, len(dataset))
    eval_end = min(num_train + num_eval, len(dataset))
    eval_dataset = dataset.select(range(eval_start, eval_end))

    return train_dataset, eval_dataset


def format_scienceqa_for_llama(examples, tokenizer, max_length=512):
    """Format ScienceQA examples for Llama instruction following."""
    prompts = []
    labels_text = []

    for i in range(len(examples["question"])):
        # Build the question with choices
        question = examples["question"][i]
        choices = examples["choices"][i]

        # Format choices
        choices_text = "\n".join([f"{chr(65 + j)}. {choice}" for j, choice in enumerate(choices)])

        prompt = f"""Answer the following science question by selecting the correct option.
                    Question: {question}

                    Choices:
{choices_text}

Answer (just the letter):"""

        # Get the answer (convert index to letter)
        answer_idx = examples["answer"][i]
        answer = chr(65 + answer_idx)

        prompts.append(prompt)
        labels_text.append(answer)

    # Tokenize
    model_inputs = tokenizer(prompts, max_length=max_length, truncation=True, padding=False)

    # Tokenize labels
    labels = tokenizer(labels_text, max_length=10, truncation=True, padding=False)

    # Combine input and label for training
    combined_input_ids = []
    combined_attention_mask = []
    combined_labels = []

    for i in range(len(model_inputs["input_ids"])):
        input_ids = model_inputs["input_ids"][i]
        label_ids = labels["input_ids"][i]

        # Combine input and label
        combined = input_ids + label_ids + [tokenizer.eos_token_id]
        combined_input_ids.append(combined)

        # Attention mask
        combined_attention_mask.append([1] * len(combined))

        # Labels (mask the prompt part, only train on answer)
        label_masked = [-100] * len(input_ids) + label_ids + [tokenizer.eos_token_id]
        combined_labels.append(label_masked)

    return {
        "input_ids": combined_input_ids,
        "attention_mask": combined_attention_mask,
        "labels": combined_labels,
    }


def format_numglue_for_llama(examples, tokenizer, max_length=512):
    """Format NumGLUE examples for Llama instruction following."""
    prompts = []
    labels_text = []

    for i in range(len(examples["question"])):
        question = examples["question"][i]
        answer = str(examples["answer"][i])

        prompt = f"""Solve the following math problem and provide just the numerical answer.

Question: {question}

Answer:"""

        prompts.append(prompt)
        labels_text.append(answer)

    # Tokenize
    model_inputs = tokenizer(prompts, max_length=max_length, truncation=True, padding=False)
    labels = tokenizer(labels_text, max_length=20, truncation=True, padding=False)

    combined_input_ids = []
    combined_attention_mask = []
    combined_labels = []

    for i in range(len(model_inputs["input_ids"])):
        input_ids = model_inputs["input_ids"][i]
        label_ids = labels["input_ids"][i]

        combined = input_ids + label_ids + [tokenizer.eos_token_id]
        combined_input_ids.append(combined)
        combined_attention_mask.append([1] * len(combined))

        label_masked = [-100] * len(input_ids) + label_ids + [tokenizer.eos_token_id]
        combined_labels.append(label_masked)

    return {
        "input_ids": combined_input_ids,
        "attention_mask": combined_attention_mask,
        "labels": combined_labels,
    }


def format_fomc_for_llama(examples, tokenizer, max_length=512):
    """Format FOMC examples for Llama instruction following."""
    prompts = []
    labels_text = []

    for i in range(len(examples["text"])):
        text = examples["text"][i]
        # FOMC dataset has 'answer' column with values like 'dovish', 'hawkish', 'neutral'
        label = examples["answer"][i].capitalize()  # Capitalize first letter

        prompt = f"""Classify the sentiment of the following Federal Reserve statement as Dovish, Hawkish, or Neutral.

Statement: {text}

Sentiment:"""

        prompts.append(prompt)
        labels_text.append(label)

    # Tokenize
    model_inputs = tokenizer(prompts, max_length=max_length, truncation=True, padding=False)
    labels = tokenizer(labels_text, max_length=10, truncation=True, padding=False)

    combined_input_ids = []
    combined_attention_mask = []
    combined_labels = []

    for i in range(len(model_inputs["input_ids"])):
        input_ids = model_inputs["input_ids"][i]
        label_ids = labels["input_ids"][i]

        combined = input_ids + label_ids + [tokenizer.eos_token_id]
        combined_input_ids.append(combined)
        combined_attention_mask.append([1] * len(combined))

        label_masked = [-100] * len(input_ids) + label_ids + [tokenizer.eos_token_id]
        combined_labels.append(label_masked)

    return {
        "input_ids": combined_input_ids,
        "attention_mask": combined_attention_mask,
        "labels": combined_labels,
    }


class DataCollatorForCompletionOnly:
    """Data collator that pads sequences for training."""

    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        # Pad sequences
        max_len = min(max(len(f["input_ids"]) for f in features), self.max_length)

        input_ids = []
        attention_mask = []
        labels = []

        for f in features:
            # Truncate if needed
            curr_input_ids = f["input_ids"][:max_len]
            curr_attention_mask = f["attention_mask"][:max_len]
            curr_labels = f["labels"][:max_len]

            # Pad
            padding_length = max_len - len(curr_input_ids)
            curr_input_ids = curr_input_ids + [self.tokenizer.pad_token_id] * padding_length
            curr_attention_mask = curr_attention_mask + [0] * padding_length
            curr_labels = curr_labels + [-100] * padding_length

            input_ids.append(curr_input_ids)
            attention_mask.append(curr_attention_mask)
            labels.append(curr_labels)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
