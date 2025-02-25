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

import argparse
import datetime as dt
import enum
import gc
import json
import os
import sys
import tempfile
import time

# import warnings
from dataclasses import asdict, dataclass
from typing import Any

import huggingface_hub
import numpy as np
import torch
from torch import nn
from utils import (
    DATASET_NAME,
    FILE_NAME_TRAIN_PARAMS,
    get_base_model_info,
    get_data,
    get_dataset_info,
    get_meta_info,
    get_model,
    get_peft_branch,
    get_tokenizer,
    get_train_config,
    init_cuda,
    validate_experiment_path,
)

from peft import PeftConfig
from peft.utils import SAFETENSORS_WEIGHTS_NAME


# # suppress all warnings
# warnings.filterwarnings("ignore") # FIXME?

dtype_to_bytes_linear = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "int4": 0.5}
# main results
RESULT_PATH = os.path.join(os.path.dirname(__file__), "results")
# testing results
RESULT_PATH_TEST = os.path.join(os.path.dirname(__file__), "temporary_results")
# cancelled results
RESULT_PATH_CANCELLED = os.path.join(os.path.dirname(__file__), "cancelled_results")


class TrainStatus(enum.Enum):
    FAILED = "failed"
    SUCCESS = "success"
    CANCELED = "canceled"


@dataclass
class TrainResult:
    status: TrainStatus
    train_time: float
    cuda_memory_log: list[int]
    losses: list[float]
    metrics: list[Any]  # TODO


def train(
    *,
    model: nn.Module,
    max_steps: int,
    batch_size: int,
    learning_rate: float,
    data: Any,
    tokenizer: Any,
    cuda_memory_init: int,
) -> TrainResult:
    cuda_memory_log = []
    losses = []
    metrics = []
    sample = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    status = TrainStatus.FAILED
    tic_train = time.perf_counter()

    for i in range(0, max_steps):
        tic = time.perf_counter()
        try:
            batch = tokenizer.pad(data["train"][sample : sample + batch_size], return_tensors="pt").to(model.device)
            sample += batch_size

            # add targets
            batch["labels"] = batch["input_ids"].clone()
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            cuda_memory_log.append(torch.cuda.memory_allocated() - cuda_memory_init)
            torch.cuda.empty_cache()
            gc.collect()
            toc = time.perf_counter()
            print_verbose(f"step {i:3d} loss {loss.item():.6f} time {toc - tic:.2f}s")
        except KeyboardInterrupt:
            print_verbose("canceled training")
            status = TrainStatus.CANCELED
            break

    toc_train = time.perf_counter()
    if status != TrainStatus.CANCELED:
        status = TrainStatus.SUCCESS
    train_result = TrainResult(
        status=status,
        train_time=toc_train - tic_train,
        cuda_memory_log=cuda_memory_log,
        losses=losses,
        metrics=metrics,
    )
    return train_result


def log_to_console(log_data: dict) -> None:
    cuda_memory_max = log_data["train_info"]["cuda_memory_max"]
    cuda_memory_avg = log_data["train_info"]["cuda_memory_avg"]
    cuda_memory_90th = log_data["train_info"]["cuda_memory_90th"]
    time_train = log_data["train_info"]["train_time"]
    time_total = log_data["run_info"]["total_time"]
    file_size = log_data["train_info"]["file_size"]

    print(f"cuda memory max: {cuda_memory_max // 2**20}MB")
    print(f"cuda memory avg: {cuda_memory_avg // 2**20}MB")
    print(f"cuda memory 90th percentile: {cuda_memory_90th // 2**20}MB")
    print(f"train time: {time_train}s")
    print(f"total time: {time_total:.2f}s")
    print(f"file size of checkpoint: {file_size / 2**20:.1f}MB")


def log_to_file(*, log_data: dict, save_dir: str, experiment_name: str, timestamp: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    file_name = f"{experiment_name.replace(os.path.sep, '--')}--{timestamp.replace(':', '-')}.json"
    file_name = os.path.join(save_dir, file_name)
    with open(file_name, "w") as f:
        json.dump(log_data, f, indent=2)
    print_verbose(f"Saved log to {file_name}")


def log_results(
    *,
    experiment_name: str,
    train_result: TrainResult,
    cuda_memory_init: int,
    time_total: float,
    model: nn.Module,
    model_info: huggingface_hub.ModelInfo,
    dataset_info: huggingface_hub.DatasetInfo,
    start_date: str,
    peft_config_sha: str,
    train_params_sha: str,
):
    # collect results
    cuda_memory_final = torch.cuda.max_memory_allocated()
    cuda_memory_avg = int(sum(train_result.cuda_memory_log) / len(train_result.cuda_memory_log))
    cuda_memory_90th = int(np.percentile(train_result.cuda_memory_log, 90))

    with tempfile.TemporaryDirectory() as tmp_dir:
        model.save_pretrained(tmp_dir)
        stat = os.stat(os.path.join(tmp_dir, SAFETENSORS_WEIGHTS_NAME))
        file_size = stat.st_size
        print(f"Saved PEFT checkpoint to {tmp_dir}")

    meta_info = get_meta_info()
    model_sha = model_info.sha
    model_created_at = model_info.created_at.isoformat()
    dataset_sha = dataset_info.sha
    dataset_created_at = dataset_info.created_at.isoformat()

    peft_branch = get_peft_branch()

    if train_result.status == TrainStatus.CANCELED:
        save_dir = RESULT_PATH_CANCELLED
        print_verbose("Experiment run was categorized as canceled")
    if peft_branch != "main":
        save_dir = RESULT_PATH_TEST
        print_verbose(f"Experiment run was categorized as a test run on branch {peft_branch}")
    elif train_result.status == TrainStatus.SUCCESS:
        save_dir = RESULT_PATH
        print_verbose("Experiment run was categorized as successful run")

    log_data = {
        "run_info": {
            "created_at": start_date,
            "total_time": time_total,
            "experiment_name": experiment_name,
            "peft_branch": peft_branch,
            "train_params_sha": train_params_sha,
            "peft_config_sha": peft_config_sha,
        },
        "train_info": {
            "cuda_memory_avg": cuda_memory_avg,
            "cuda_memory_max": (cuda_memory_final - cuda_memory_init),
            "cuda_memory_90th": cuda_memory_90th,
            "train_time": train_result.train_time,
            "file_size": file_size,
            "status": train_result.status.value,
            "metrics": train_result.metrics,
            "losses": train_result.losses,
        },
        "meta_info": {
            "model_sha": model_sha,
            "model_created_at": model_created_at,
            "dataset_sha": dataset_sha,
            "dataset_created_at": dataset_created_at,
            **asdict(meta_info),
        },
    }

    log_to_console(log_data)
    log_to_file(log_data=log_data, save_dir=save_dir, experiment_name=experiment_name, timestamp=start_date)


def main(*, path_experiment: str, experiment_name: str, train_params_sha: str, peft_config_sha: str) -> None:
    tic_total = time.perf_counter()
    start_date = dt.datetime.now(tz=dt.timezone.utc).replace(microsecond=0).isoformat()

    # load configs
    peft_config = PeftConfig.from_pretrained(path_experiment)
    path_train_config = os.path.join(path_experiment, FILE_NAME_TRAIN_PARAMS)
    train_config = get_train_config(path_train_config)

    # initialize objects
    cuda_memory_init = init_cuda()
    tokenizer = get_tokenizer(model_id=train_config.model_id, max_seq_length=train_config.max_seq_length)
    data = get_data(tokenizer=tokenizer)
    model_info = get_base_model_info(train_config.model_id)
    dataset_info = get_dataset_info(DATASET_NAME)
    model = get_model(
        model_id=train_config.model_id,
        dtype=train_config.dtype,
        peft_config=peft_config,
        compile=train_config.compile,
    )
    print_verbose(model)

    # train model
    try:
        train_result = train(
            model=model,
            max_steps=train_config.max_steps,
            batch_size=train_config.batch_size,
            learning_rate=train_config.learning_rate,
            data=data,
            tokenizer=tokenizer,
            cuda_memory_init=cuda_memory_init,
        )
    except Exception as e:
        print_verbose(f"Training failed with error: {e}")
        raise

    if train_result.status == TrainStatus.FAILED:
        print_verbose("Training failed, not logging results")
        sys.exit(1)

    time_total = time.perf_counter() - tic_total

    # log results: print and save to file
    log_results(
        experiment_name=experiment_name,
        train_result=train_result,
        cuda_memory_init=cuda_memory_init,
        time_total=time_total,
        model=model,
        model_info=model_info,
        dataset_info=dataset_info,
        start_date=start_date,
        train_params_sha=train_params_sha,
        peft_config_sha=peft_config_sha,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("path_experiment", type=str, help="Path to the experiment directory")
    args = parser.parse_args()

    experiment_name, train_params_sha, peft_config_sha = validate_experiment_path(args.path_experiment)

    if args.verbose:

        def print_verbose(*args, **kwargs):
            kwargs["file"] = sys.stderr
            print(*args, **kwargs)
    else:

        def print_verbose(*args, **kwargs):
            pass

    main(
        path_experiment=args.path_experiment,
        experiment_name=experiment_name,
        train_params_sha=train_params_sha,
        peft_config_sha=peft_config_sha,
    )
