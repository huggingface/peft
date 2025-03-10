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

"""
All utilities related to data handling.
"""

from functools import partial
from typing import Callable

import datasets
import numpy as np
from datasets import Dataset, load_dataset


# with a token limit of 768 for query + response, we have to exclude all texts with length > 1304; this leaves 93.8% of
# the dataset
CHAR_LIMIT = 1300
# train/valid/test split -- note that evaluation takes quite long, so don't choose too large sizes for the valid set,
# since it's run multiple times during training; test is only run once at the end and thus can be larger
VALID_SIZE = 50


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
    Return the indices of the train, valid, and test splits of the dataset.

    We cannot use ds.train_test_split(..., stratify_by_column="type") as it gives:

    > ValueError: Stratifying by column is only supported for ClassLabel column, and column type is Value.

    even after calling ds_filtered.class_encode_column("type"). Thus, using sklearn's StratifiedKFold instead.
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
