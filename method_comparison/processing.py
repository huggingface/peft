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

"""Data processing used for analyzing and presenting the results"""

import json
import os

import pandas as pd


def _preprocess_common(row):
    """Extract fields common to all tasks from a single result row.

    Returns a tuple of metainfo dict and train metrics, or None if the row should be skipped.
    """
    run_info = row["run_info"]
    train_info = row["train_info"]
    meta_info = row["meta_info"]
    if run_info["peft_config"]:
        peft_type = run_info["peft_config"]["peft_type"]
    else:
        peft_type = "full-finetuning"
    if train_info["status"] != "success":
        return None

    train_metrics = train_info["metrics"][-1]

    dct = {
        "experiment_name": run_info["experiment_name"],
        "model_id": run_info["train_config"]["model_id"],
        "train_config": run_info["train_config"],
        "peft_type": peft_type,
        "peft_config": run_info["peft_config"],
        "accelerator_memory_reserved_avg": train_info["accelerator_memory_reserved_avg"],
        "accelerator_memory_max": train_info["accelerator_memory_max"],
        "accelerator_memory_reserved_99th": train_info["accelerator_memory_reserved_99th"],
        "total_time": run_info["total_time"],
        "train_time": train_info["train_time"],
        "file_size": train_info["file_size"],
        "num_trainable_params": train_info["num_trainable_params"],
        "train_loss": train_metrics["train loss"],
        "train_samples": train_metrics["train samples"],
        "peft_version": meta_info["package_info"]["peft-version"],
        "peft_branch": run_info["peft_branch"],
        "transformers_version": meta_info["package_info"]["transformers-version"],
        "datasets_version": meta_info["package_info"]["datasets-version"],
        "torch_version": meta_info["package_info"]["torch-version"],
        "package_info": meta_info["package_info"],
        "system_info": meta_info["system_info"],
        "created_at": run_info["created_at"],
    }
    return dct, train_metrics


def _preprocess_metamathqa(dct, train_metrics, meta_info):
    """Add MetaMathQA-specific fields."""
    dct["test_accuracy"] = train_metrics["test accuracy"]
    dct["train_total_tokens"] = train_metrics["train total tokens"]
    dct["forgetting*"] = train_metrics.get("forgetting", 123)
    dct["bitsandbytes_version"] = meta_info["package_info"]["bitsandbytes-version"]


def _preprocess_image_gen(dct, train_metrics, meta_info):
    """Add image-gen-specific fields."""
    dct["test_dino_similarity"] = train_metrics["test dino_similarity"]
    dct["drift*"] = train_metrics.get("drift", 123)
    dct["diffusers_version"] = meta_info["package_info"]["diffusers-version"]


_TASK_PREPROCESSORS = {
    "MetaMathQA": _preprocess_metamathqa,
    "image-gen": _preprocess_image_gen,
}


def preprocess(rows, task_name: str, print_fn=print):
    task_preprocessor = _TASK_PREPROCESSORS.get(task_name)
    if task_preprocessor is None:
        raise ValueError(f"Unknown task_name: {task_name!r}. Choose from {list(_TASK_PREPROCESSORS)}")

    results = []
    skipped = 0
    for row in rows:
        common = _preprocess_common(row)
        if common is None:
            skipped += 1
            continue

        dct, train_metrics = common
        dct["task_name"] = task_name
        task_preprocessor(dct, train_metrics, row["meta_info"])
        results.append(dct)

    if skipped:
        print_fn(f"Skipped {skipped} of {len(rows)} entries because the train status != success")

    return results


def load_jsons(path):
    results = []
    for fn in os.listdir(path):
        if fn.endswith(".json"):
            with open(os.path.join(path, fn)) as f:
                row = json.load(f)
                results.append(row)
    return results


_COMMON_DTYPES = {
    "task_name": "string",
    "experiment_name": "string",
    "model_id": "string",
    "train_config": "string",
    "peft_type": "string",
    "peft_config": "string",
    "accelerator_memory_reserved_avg": int,
    "accelerator_memory_max": int,
    "accelerator_memory_reserved_99th": int,
    "total_time": float,
    "train_time": float,
    "file_size": int,
    "train_loss": float,
    "train_samples": int,
    "num_trainable_params": int,
    "peft_version": "string",
    "peft_branch": "string",
    "transformers_version": "string",
    "datasets_version": "string",
    "torch_version": "string",
    "package_info": "string",
    "system_info": "string",
    "created_at": "string",
}

_TASK_DTYPES = {
    "MetaMathQA": {
        "test_accuracy": float,
        "train_total_tokens": int,
        "forgetting*": float,
        "bitsandbytes_version": "string",
    },
    "image-gen": {
        "test_dino_similarity": float,
        "drift*": float,
        "diffusers_version": "string",
    },
}

_TASK_IMPORTANT_COLUMNS = {
    "MetaMathQA": [
        "experiment_name",
        "peft_type",
        "total_time",
        "train_time",
        "test_accuracy",
        "train_loss",
        "accelerator_memory_max",
        "accelerator_memory_reserved_99th",
        "accelerator_memory_reserved_avg",
        "num_trainable_params",
        "file_size",
        "created_at",
        "task_name",
        "forgetting*",
    ],
    "image-gen": [
        "experiment_name",
        "peft_type",
        "total_time",
        "train_time",
        "test_dino_similarity",
        "drift*",
        "train_loss",
        "accelerator_memory_max",
        "accelerator_memory_reserved_99th",
        "accelerator_memory_reserved_avg",
        "num_trainable_params",
        "file_size",
        "created_at",
        "task_name",
    ],
}


def load_df(path, task_name, print_fn=print):
    jsons = load_jsons(path)
    preprocessed = preprocess(jsons, task_name=task_name, print_fn=print_fn)
    dtype_dict = {**_COMMON_DTYPES, **_TASK_DTYPES.get(task_name, {})}
    if not preprocessed:
        return pd.DataFrame(columns=dtype_dict.keys())
    df = pd.DataFrame(preprocessed)
    df = df.astype(dtype_dict)
    df["created_at"] = pd.to_datetime(df["created_at"])
    # round training time to nearest second
    df["train_time"] = df["train_time"].round().astype(int)
    df["total_time"] = df["total_time"].round().astype(int)

    # reorder columns for better viewing, pinned_columns arg in Gradio seems not to work correctly
    important_columns = _TASK_IMPORTANT_COLUMNS.get(task_name, ["experiment_name", "peft_type"])
    other_columns = [col for col in df if col not in important_columns]
    df = df[important_columns + other_columns]

    columns = ["experiment_name", "model_id", "peft_type", "created_at"]
    # we want to keep only the most recent run for each experiment
    df = df.sort_values("created_at").drop_duplicates(columns, keep="last")
    return df
