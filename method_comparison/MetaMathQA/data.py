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
TODO
"""
from typing import Any, Callable

import datasets
from datasets import Dataset, load_dataset
from sklearn.model_selection import StratifiedKFold
import numpy as np


CHAR_LIMIT = 550


def get_filtered_dataset(*, ds: datasets.Dataset, print_fn: Callable[[Any, ...], None]) -> Dataset:
    """Return the filtered dataset, with long queries removed.

    We determined that 99% of queries have 529 or fewer characters. Characters roughly correspond to tokens, so this is
    a good proxy. We cannot use tokens directly, as that depends on the tokenizer, which can be different for each
    model, but we want the same filter for each model.

    """
    char_lengths = [len(x) for x in ds["query"]]
    idx_filtered = [i for i, l in enumerate(char_lengths) if l <= CHAR_LIMIT]
    print_fn(f"Filtered dataset: {100 * len(idx_filtered) / len(ds):.1f}% of the original dataset")
    return ds.select(idx_filtered)


def get_train_valid_test_datasets(
    *,
    ds: datasets.Dataset,
    valid_size: int,
    test_size: int,
    print_fn: Callable[[Any, ...], None],
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Return the indices of the train, valid, and test splits of the dataset.

    We cannot use ds.train_test_split(..., stratify_by_column="type") as it gives:

    > ValueError: Stratifying by column is only supported for ClassLabel column, and column type is Value.

    even after calling ds_filtered.class_encode_column("type"). Thus, using sklearn's StratifiedKFold instead.
    """
    dataset_types = ds["type"]
    total_size = len(ds)
    assert valid_size + test_size < total_size

    n_splits_train = total_size // (test_size + valid_size)
    kfold = StratifiedKFold(n_splits_train, shuffle=True, random_state=0)
    idx_train, idx_rest = next(iter(kfold.split(np.arange(total_size).reshape(-1, 1), y=np.array(dataset_types))))
    print_fn(f"Train size: {len(idx_train)}")

    n_splits_test = (valid_size + test_size) // valid_size
    kfold = StratifiedKFold(n_splits_test, shuffle=True, random_state=0)
    idx_test, idx_valid  = next(iter(kfold.split(np.arange(len(idx_rest)).reshape(-1, 1), y=np.array(dataset_types)[idx_rest])))
    idx_test = idx_rest[idx_test]
    idx_valid = idx_rest[idx_valid]
    print_fn(f"Valid size: {len(idx_valid)}")

    # unused columns must be removed; now we don't need them anymore
    ds_train = ds.select(idx_train).remove_columns(['type', 'query', 'original_question', 'response'])
    ds_valid = ds.select(idx_valid).remove_columns(['type', 'query', 'original_question'])
    ds_test = ds.select(idx_test).remove_columns(['type', 'query', 'original_question'])

    assert set(idx_test) | set(idx_valid) | set(idx_train) == set(range(total_size))
    return ds_train, ds_valid, ds_test


def get_dataset(*, dataset_name, tokenizer, template) -> Dataset:
    def tokenize(samples):
        # For some reason, the max sequence length is not honored by the tokenizer, resulting in IndexErrors. Thus,
        # manually ensure that sequences are not too long.
        queries = [template.format(query=sample) for sample in samples["query"]]
        tokenized = tokenizer(queries)
        tokenized["input_ids"] = [input_ids[: tokenizer.model_max_length] for input_ids in tokenized["input_ids"]]
        tokenized["attention_mask"] = [
            input_ids[: tokenizer.model_max_length] for input_ids in tokenized["attention_mask"]
        ]
        return tokenized

    ds = load_dataset(dataset_name)["train"]
    ds = ds.map(tokenize, batched=True)
    return ds
