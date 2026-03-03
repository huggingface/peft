# Copyright 2026-present the HuggingFace Inc. team.

"""Utilities for RLMath experiments."""

from __future__ import annotations

import datetime as dt
import importlib.metadata
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass
from typing import Any


FILE_NAME_DEFAULT_TRAIN_PARAMS = os.path.join(os.path.dirname(__file__), "default_training_params.json")
FILE_NAME_TRAIN_PARAMS = "training_params.json"
RESULT_PATH = os.path.join(os.path.dirname(__file__), "results")
RESULT_PATH_TEST = os.path.join(os.path.dirname(__file__), "temporary_results")
RESULT_PATH_CANCELLED = os.path.join(os.path.dirname(__file__), "cancelled_results")


@dataclass
class RLTrainConfig:
    model_id: str
    dtype: str
    seed: int
    max_steps: int
    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    per_device_eval_batch_size: int
    max_prompt_length: int
    max_completion_length: int
    num_generations: int
    num_generations_eval: int
    temperature: float
    top_p: float
    beta: float
    epsilon: float
    loss_type: str
    scale_rewards: str
    importance_sampling_level: str
    use_vllm: bool
    train_subset_size: int
    eval_subset_size: int
    eval_steps: int
    save_steps: int
    save_total_limit: int
    logging_steps: int
    save_checkpoint: bool
    dataset_name: str
    dataset_config: str | None
    dataset_train_split: str
    dataset_test_split: str
    top_k: int = 0
    min_p: float | None = None
    repetition_penalty: float = 1.0
    mask_truncated_completions: bool = False
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = False
    disable_dropout: bool = False
    generation_batch_size: int | None = None
    num_iterations: int = 1
    epsilon_high: float | None = None
    delta: float | None = None


def load_train_config(exp_dir: str) -> RLTrainConfig:
    with open(FILE_NAME_DEFAULT_TRAIN_PARAMS) as f:
        base = json.load(f)

    train_override_path = os.path.join(exp_dir, FILE_NAME_TRAIN_PARAMS)
    if os.path.exists(train_override_path):
        with open(train_override_path) as f:
            override = json.load(f)
        base.update(override)

    return RLTrainConfig(**base)


def get_peft_branch() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def get_result_dir(status: str, branch: str) -> str:
    if status == "cancelled":
        return RESULT_PATH_CANCELLED
    if status == "success" and branch == "main":
        return RESULT_PATH
    return RESULT_PATH_TEST


def save_result(result: dict[str, Any], out_dir: str, experiment_name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_name = experiment_name.replace("/", "--") + ".json"
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return out_path


def get_package_info() -> dict[str, str | None]:
    def v(pkg: str) -> str | None:
        try:
            return importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "transformers-version": v("transformers"),
        "peft-version": v("peft"),
        "datasets-version": v("datasets"),
        "trl-version": v("trl"),
        "torch-version": v("torch"),
        "bitsandbytes-version": v("bitsandbytes"),
    }


def get_system_info() -> dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def build_base_result(experiment_name: str, cfg: RLTrainConfig, peft_cfg: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "run_info": {
            "created_at": now_iso(),
            "total_time": None,
            "experiment_name": experiment_name,
            "peft_branch": get_peft_branch(),
            "train_config": asdict(cfg),
            "peft_config": peft_cfg,
        },
        "train_info": {
            "status": "running",
            "train_time": None,
            "file_size": None,
            "num_trainable_params": None,
            "metrics": [],
        },
        "rl_eval_info": {},
        "meta_info": {
            "package_info": get_package_info(),
            "system_info": get_system_info(),
        },
    }
