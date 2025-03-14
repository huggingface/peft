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


def preprocess(rows, task_name: str, print_fn=print):
    results = []
    skipped = 0
    for row in rows:
        run_info = row["run_info"]
        train_info = row["train_info"]
        meta_info = row["meta_info"]
        if train_info["status"] != "success":
            skipped += 1
            continue

        test_metrics = train_info["metrics"][-1]

        # extract the fields that make most sense
        dct = {
            "task_name": task_name,
            "experiment_name": run_info["experiment_name"],
            "model_id": run_info["train_config"]["model_id"],
            "train_config": run_info["train_config"],
            "peft_type": run_info["peft_config"]["peft_type"],
            "peft_config": run_info["peft_config"],
            "cuda_memory_reserved_avg": train_info["cuda_memory_reserved_avg"],
            "cuda_memory_max": train_info["cuda_memory_max"],
            "cuda_memory_reserved_99th": train_info["cuda_memory_reserved_99th"],
            "total_time": run_info["total_time"],
            "train_time": train_info["train_time"],
            "file_size": train_info["file_size"],
            "test_accuracy": test_metrics["test accuracy"],
            "test_loss": test_metrics["train loss"],
            "train_samples": test_metrics["train samples"],
            "train_total_tokens": test_metrics["train total tokens"],
            "peft_version": meta_info["package_info"]["peft-version"],
            "peft_branch": run_info["peft_branch"],
            "transformers_version": meta_info["package_info"]["transformers-version"],
            "datasets_version": meta_info["package_info"]["datasets-version"],
            "torch_version": meta_info["package_info"]["torch-version"],
            "bitsandbytes_version": meta_info["package_info"]["bitsandbytes-version"],
            "package_info": meta_info["package_info"],
            "system_info": meta_info["system_info"],
            "created_at": run_info["created_at"],
        }
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


def load_df(path, task_name, print_fn=print):
    jsons = load_jsons(path)
    preprocessed = preprocess(jsons, task_name=task_name, print_fn=print_fn)
    dtype_dict = {
        "task_name": "string",
        "experiment_name": "string",
        "model_id": "string",
        "train_config": "string",
        "peft_type": "string",
        "peft_config": "string",
        "cuda_memory_reserved_avg": int,
        "cuda_memory_max": int,
        "cuda_memory_reserved_99th": int,
        "total_time": float,
        "train_time": float,
        "file_size": int,
        "test_accuracy": float,
        "test_loss": float,
        "train_samples": int,
        "train_total_tokens": int,
        "peft_version": "string",
        "peft_branch": "string",
        "transformers_version": "string",
        "datasets_version": "string",
        "torch_version": "string",
        "bitsandbytes_version": "string",
        "package_info": "string",
        "system_info": "string",
        "created_at": "string",
    }
    df = pd.DataFrame(preprocessed)
    df = df.astype(dtype_dict)
    df["created_at"] = pd.to_datetime(df["created_at"])
    # round training time to nearest second
    df["train_time"] = df["train_time"].round().astype(int)
    df["total_time"] = df["total_time"].round().astype(int)

    # reorder columns for better viewing, pinned_columns arg in Gradio seems not to work correctly
    important_columns = [
        "experiment_name",
        "peft_type",
        "total_time",
        "train_time",
        "test_accuracy",
        "test_loss",
        "cuda_memory_max",
        "cuda_memory_reserved_99th",
        "cuda_memory_reserved_avg",
        "file_size",
        "created_at",
        "task_name",
    ]
    other_columns = [col for col in df if col not in important_columns]
    df = df[important_columns + other_columns]

    size_before_drop_dups = len(df)
    columns = ["experiment_name", "model_id", "peft_type", "created_at"]
    # we want to keep only the most recent run for each experiment
    df = df.sort_values("created_at").drop_duplicates(columns, keep="last")
    return df
