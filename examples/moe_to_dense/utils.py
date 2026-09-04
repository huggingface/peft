# Copyright 2026-present the HuggingFace Inc. team.
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

"""
Data loading and evaluation utilities, copied from the MetaMathQA benchmark in `method_comparison/MetaMathQA`.
"""

from collections.abc import Callable
from decimal import Decimal, DivisionByZero, InvalidOperation
from functools import partial
from typing import Optional

import datasets
import numpy as np
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig


# with a token limit of 768 for query + response, we have to exclude all texts with length > 1304; this leaves 93.8% of
# the dataset
CHAR_LIMIT = 1300
# train/valid/test split -- note that evaluation takes quite long, so don't choose too large sizes for the valid set,
# since it's run multiple times during training; test is only run once at the end and thus can be larger
VALID_SIZE = 50


########
# DATA #
########


def get_filtered_dataset(*, ds: datasets.Dataset, print_fn: Callable[..., None]) -> Dataset:
    """Return the filtered dataset, with long queries removed.

    We determined that 99% of queries have 529 or fewer characters. Characters roughly correspond to tokens, so this is
    a good proxy. We cannot use tokens directly, as that depends on the tokenizer, which can be different for each
    model, but we want the same filter for each model.

    """
    char_lengths = [len(f"{q} {r}") for q, r in zip(ds["query"], ds["response"])]
    idx_filtered = [i for i, length in enumerate(char_lengths) if length <= CHAR_LIMIT]
    print_fn(f"Filtered dataset: {100 * len(idx_filtered) / len(ds):.1f}% of the original dataset")
    return ds.select(idx_filtered)


def get_train_valid_test_datasets(
    *, tokenizer, query_template: str, print_fn: Callable[..., None]
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Return the train, valid, and test splits of the dataset.

    The train set is MetaMathQA, the valid set is a random sample of the GSM8K train set, the test set is the GSM8K
    test set.
    """
    metamath = load_dataset("meta-math/MetaMathQA")["train"]
    metamath = get_filtered_dataset(ds=metamath, print_fn=print_fn)

    # gsmk8k does not need to be filtered as query and response are short enough
    gsm8k = load_dataset("openai/gsm8k", "main")
    gsm8k = gsm8k.rename_columns({"question": "query", "answer": "response"})
    gsm8k_train = gsm8k["train"]
    gsm8k_test = gsm8k["test"]

    np.random.seed(0)
    indices = np.arange(len(gsm8k_train))
    np.random.shuffle(indices)
    idx_valid = indices[:VALID_SIZE]

    ds_train = metamath
    ds_valid = gsm8k_train.select(idx_valid)
    ds_test = gsm8k_test

    print_fn(f"Train size: {len(ds_train)}")
    print_fn(f"Valid size: {len(ds_valid)}")
    print_fn(f"Test size: {len(ds_test)}")

    tokenize_with_answer_ = partial(tokenize_with_answer, tokenizer=tokenizer, template=query_template)
    tokenize_wo_answer_ = partial(tokenize_wo_answer, tokenizer=tokenizer, template=query_template)
    ds_train = ds_train.map(tokenize_with_answer_, batched=True).remove_columns(["type", "query", "original_question"])
    ds_valid = ds_valid.map(tokenize_wo_answer_, batched=True).remove_columns(["query"])
    ds_test = ds_test.map(tokenize_wo_answer_, batched=True).remove_columns(["query"])

    return ds_train, ds_valid, ds_test


def tokenize_with_answer(samples, tokenizer, template):
    queries = [template.format(query=sample) + answer for sample, answer in zip(samples["query"], samples["response"])]
    tokenized = tokenizer(queries)
    tokenized["input_ids"] = [input_ids[: tokenizer.model_max_length] for input_ids in tokenized["input_ids"]]
    tokenized["attention_mask"] = [
        input_ids[: tokenizer.model_max_length] for input_ids in tokenized["attention_mask"]
    ]
    return tokenized


def tokenize_wo_answer(samples, tokenizer, template):
    queries = [template.format(query=sample) for sample in samples["query"]]
    tokenized = tokenizer(queries)
    tokenized["input_ids"] = [input_ids[: tokenizer.model_max_length] for input_ids in tokenized["input_ids"]]
    tokenized["attention_mask"] = [
        input_ids[: tokenizer.model_max_length] for input_ids in tokenized["attention_mask"]
    ]
    return tokenized


def get_tokenizer(*, model_id: str, max_seq_length: int):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.model_max_length = max_seq_length
    # inputs are padded on the right side and the padding token is the EOS token
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class BucketIterator:
    """
    Iterator that yields batches of data from a torch Dataset, grouped in buckets by sequence length

    The iterator will yield batches of size `batch_size`, where the samples in each batch are sorted by sequence
    length. This is done to minimize the amount of padding required for each batch. To avoid sorting the entire dataset
    and thus introducing a bias, the dataset is first split into buckets of size `batch_size * bucket_factor`.

    Args:
        ds: The torch Dataset to iterate over
        batch_size: The batch size
        bucket_factor: The factor by which to multiply the batch size to determine the bucket size
        delete_cols: The columns to delete from the dataset before yielding a batch
    """

    def __init__(self, ds, *, batch_size: int, bucket_factor: int, delete_cols: list[str]) -> None:
        self.ds = ds
        self.batch_size = batch_size
        self.bucket_factor = bucket_factor
        self.delete_cols = set(delete_cols)

        assert self.bucket_factor > 0, "bucket_factor must be greater than 0"

    def _batch_iterator(self, bucket):
        tokens_per_sample_bucket = torch.tensor([len(i) for i in bucket["input_ids"]])
        # sort long to short instead to encounter possible OOM errors as early as possible
        sorted = torch.argsort(tokens_per_sample_bucket, descending=True)
        cls = type(bucket)  # conserve the type returned by the ds
        bucket = {k: [v[i] for i in sorted] for k, v in bucket.items() if k not in self.delete_cols}
        num_samples = len(bucket["input_ids"])
        for j in range(0, num_samples, self.batch_size):
            batch = {k: v[j : j + self.batch_size] for k, v in bucket.items()}
            yield cls(batch)

    def __iter__(self):
        bucket_size = self.batch_size * self.bucket_factor
        for i in range(0, len(self.ds), bucket_size):
            bucket = self.ds[i : i + bucket_size]
            yield from self._batch_iterator(bucket)

        # if there is a remainder, we yield the last batch
        if len(self.ds) % bucket_size != 0:
            bucket = self.ds[-(len(self.ds) % bucket_size) :]
            yield from self._batch_iterator(bucket)


##############
# EVALUATION #
##############


def get_generation_config(*, seq_len, generate_kwargs) -> GenerationConfig:
    # filter out None values so that we don't depend on setting correct defaults in the config
    generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}
    if ("max_length" in generate_kwargs) and ("max_new_tokens" in generate_kwargs):
        # transformers does not support setting both max_length and max_new_tokens, but what we want in this case is to
        # take the smaller of the two values
        new_max_length = min(generate_kwargs["max_new_tokens"] + seq_len, generate_kwargs["max_length"])
        del generate_kwargs["max_new_tokens"]
        generate_kwargs["max_length"] = new_max_length
    generation_config = GenerationConfig(**generate_kwargs)
    return generation_config


def evaluate(model, tokenizer, ds, batch_size, generate_kwargs, use_tqdm: bool = False) -> tuple[list[str], list[str]]:
    generate_kwargs = generate_kwargs.copy()

    if "pad_token_id" not in generate_kwargs:
        generate_kwargs["pad_token_id"] = tokenizer.pad_token_id

    with torch.inference_mode():
        predictions = []
        responses = []
        pbar = range(0, len(ds), batch_size)
        if use_tqdm:
            pbar = tqdm(pbar)
        for j in pbar:
            sliced = ds[j : j + batch_size]
            responses += sliced.pop("response")
            batch = tokenizer.pad(sliced, return_tensors="pt", padding_side="left").to(model.device)
            seq_len = batch["input_ids"].shape[1]
            generation_config = get_generation_config(seq_len=seq_len, generate_kwargs=generate_kwargs)
            # the tokenizer is needed in case `stop_strings` are used
            outputs = model.generate(**batch, generation_config=generation_config, tokenizer=tokenizer)
            # only decode the generated tokens, not the prompt (which contains the few-shot example)
            predictions += tokenizer.batch_decode(outputs[:, seq_len:], skip_special_tokens=True)
    return predictions, responses


def parse_answer(text: str) -> Optional[str]:
    """
    A label/prediction can look like this:

    Question: If the magnitude of vector v is equal to 4, what is the dot product of vector v with itself?. Think step
    by step Answer: The dot product of a vector with itself is equal to the square of its magnitude. So, the dot
    product of vector v with itself is equal to $4^2 = \boxed{16}$.The answer is: 16

    We want to extract '16' from this string.

    """
    # This implementation is based on sampling meta-llama/Llama-3.1-8B-Instruct. It may not work for other models.
    candidate_delimiters = [
        # MetaMath:
        "The answer is: ",
        "The answer is ",
        "The final answer is: ",
        "The final answer is ",
        # GSM8K:
        "#### ",
    ]
    text = text.strip()
    text = text.rstrip(".!?")
    for delimiter in candidate_delimiters:
        if delimiter in text:
            break
    else:  # no match
        return None

    text = text.rpartition(delimiter)[-1].strip()
    # if a new paragraph follows after the final answer, we want to remove it
    text = text.split("\n", 1)[0]
    # note: we can just remove % here since the GSM8K dataset just omits it, i.e. 50% -> 50, no need to divide by 100
    text = text.strip(" .!?$%")
    return text


def convert_to_decimal(s: Optional[str]) -> Optional[Decimal]:
    """
    Converts a string representing a number to a Decimal.

    The string may be:
      - A simple number (e.g., "13", "65.33")
      - A fraction (e.g., "20/14")
    """
    if s is None:
        return None

    try:
        s = s.strip()
        # Check if the string represents a fraction.
        if "/" in s:
            parts = s.split("/")
            if len(parts) != 2:
                return None
            numerator = Decimal(parts[0].strip())
            denominator = Decimal(parts[1].strip())
            if denominator == 0:
                return None
            value = numerator / denominator
        else:
            # Parse as a regular decimal or integer string.
            value = Decimal(s)
        return value
    except (DivisionByZero, InvalidOperation, ValueError):
        return None


def get_accuracy(*, predictions: list[str], responses: list[str]) -> float:
    if len(predictions) != len(responses):
        raise ValueError(f"Prediction length mismatch: {len(predictions)} != {len(responses)}")

    y_true: list[str | float | None] = []
    y_pred: list[str | float | None] = []

    for prediction, response in zip(predictions, responses):
        parsed_prediction = parse_answer(prediction)
        parsed_response = parse_answer(response)
        if parsed_response is None:
            raise ValueError(f"Error encountered while trying to parse response: {response}")

        decimal_prediction = convert_to_decimal(parsed_prediction)
        decimal_answer = convert_to_decimal(parsed_response)
        if decimal_prediction is not None:
            y_pred.append(float(decimal_prediction))
        elif parsed_prediction is not None:
            y_pred.append(parsed_prediction)
        else:
            y_pred.append(None)

        # we convert decimals to float so that stuff like this works:
        # float(convert_to_decimal('20/35')) == float(convert_to_decimal('0.5714285714285714'))
        if decimal_answer is not None:
            y_true.append(float(decimal_answer))
        elif parsed_prediction is not None:
            y_true.append(parsed_response)
        else:
            y_true.append(None)

    correct: list[bool] = []
    for true, pred in zip(y_true, y_pred):
        if (true is not None) and (pred is not None):
            correct.append(true == pred)
        else:
            correct.append(False)

    accuracy = sum(correct) / len(correct)
    return accuracy
