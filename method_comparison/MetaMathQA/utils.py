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
All utilities not related to data handling.
"""

import enum
import json
import os
import platform
import subprocess
import tempfile
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal, Optional

import huggingface_hub
import numpy as np
import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

import peft
from peft import PeftConfig, get_peft_model, prepare_model_for_kbit_training
from peft.utils import CONFIG_NAME, SAFETENSORS_WEIGHTS_NAME


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available, currently only CUDA is supported")

device = "cuda"
CUDA_MEMORY_INIT_THRESHOLD = 500 * 2**20  # 500MB
FILE_NAME_DEFAULT_TRAIN_PARAMS = os.path.join(os.path.dirname(__file__), "default_training_params.json")
FILE_NAME_TRAIN_PARAMS = "training_params.json"  # specific params for this experiment
# main results
RESULT_PATH = os.path.join(os.path.dirname(__file__), "results")
# testing results
RESULT_PATH_TEST = os.path.join(os.path.dirname(__file__), "temporary_results")
# cancelled results
RESULT_PATH_CANCELLED = os.path.join(os.path.dirname(__file__), "cancelled_results")
hf_api = huggingface_hub.HfApi()


@dataclass
class TrainConfig:
    """All configuration parameters associated with training the model

    Args:
        model_id: The model identifier
        dtype: The data type to use for the model
        max_seq_length: The maximum sequence length
        batch_size: The batch size for training
        batch_size_eval: The batch size for eval/test, can be much higher than for training
        max_steps: The maximum number of steps to train for
        eval_steps: The number of steps between evaluations
        compile: Whether to compile the model
        query_template: The template for the query
        seed: The random seed
        grad_norm_clip: The gradient norm clipping value (set to 0 to skip)
        optimizer_kwargs: The optimizer keyword arguments (lr etc.)
        lr_scheduler: The learning rate scheduler (currently only None or 'cosine' are supported)
        use_amp: Whether to use automatic mixed precision
        generation_kwargs: Arguments passed to transformers GenerationConfig (used in evaluation)
        attn_implementation: The attention implementation to use (if any), see transformers docs
    """

    model_id: str
    dtype: Literal["float32", "float16", "bfloat16", "int8", "int4"]
    max_seq_length: int
    batch_size: int
    batch_size_eval: int
    max_steps: int
    eval_steps: int
    compile: bool
    query_template: str
    seed: int
    grad_norm_clip: float  # set to 0 to skip
    optimizer_kwargs: dict[str, Any]
    lr_scheduler: Optional[Literal["cosine"]]
    use_amp: bool
    generation_kwargs: dict[str, Any]
    attn_implementation: Optional[str]

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str):
            raise ValueError(f"Invalid model_id: {self.model_id}")
        if self.dtype not in ["float32", "float16", "bfloat16", "int8", "int4"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")
        if self.max_seq_length < 0:
            raise ValueError(f"Invalid max_seq_length: {self.max_seq_length}")
        if self.batch_size <= 0:
            raise ValueError(f"Invalid batch_size: {self.batch_size}")
        if self.batch_size_eval <= 0:
            raise ValueError(f"Invalid eval batch_size: {self.batch_size_eval}")
        if self.max_steps <= 0:
            raise ValueError(f"Invalid max_steps: {self.max_steps}")
        if self.eval_steps <= 0:
            raise ValueError(f"Invalid eval_steps: {self.eval_steps}")
        if self.eval_steps > self.max_steps:
            raise ValueError(f"Invalid eval_steps: {self.eval_steps} > max_steps: {self.max_steps}")
        if self.grad_norm_clip < 0:
            raise ValueError(f"Invalid grad_norm_clip: {self.grad_norm_clip}")
        if self.lr_scheduler not in [None, "cosine"]:
            raise ValueError(f"Invalid lr_scheduler: {self.lr_scheduler}, must be None or 'cosine'")
        if "{query}" not in self.query_template:
            raise ValueError("Invalid query_template, must contain '{query}'")


def validate_experiment_path(path: str) -> str:
    # the experiment path should take the form of ./experiments/<peft-method>/<experiment-name>
    # e.g. ./experiments/lora/rank32
    # it should contain:
    # - adapter_config.json
    # - optional: training_params.json
    if not os.path.exists(FILE_NAME_DEFAULT_TRAIN_PARAMS):
        raise FileNotFoundError(
            f"Missing default training params file '{FILE_NAME_DEFAULT_TRAIN_PARAMS}' in the ./experiments directory"
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path {path} does not exist")
    if not os.path.exists(os.path.join(path, CONFIG_NAME)):
        raise FileNotFoundError(os.path.join(path, CONFIG_NAME))

    # check path structure
    path_parts = path.rstrip(os.path.sep).split(os.path.sep)
    if (len(path_parts) != 3) or (path_parts[-3] != "experiments"):
        raise ValueError(
            f"Path {path} does not have the correct structure, should be ./experiments/<peft-method>/<experiment-name>"
        )

    experiment_name = os.path.join(*path_parts[-2:])
    return experiment_name


def get_train_config(path: str) -> TrainConfig:
    # first, load the default params, then update with experiment-specific params
    with open(FILE_NAME_DEFAULT_TRAIN_PARAMS) as f:
        default_config_kwargs = json.load(f)

    config_kwargs = {}
    if os.path.exists(path):
        with open(path) as f:
            config_kwargs = json.load(f)

    config_kwargs = {**default_config_kwargs, **config_kwargs}
    return TrainConfig(**config_kwargs)


def init_cuda() -> int:
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.manual_seed_all(0)
    # might not be necessary, but just to be sure
    nn.Linear(1, 1).to(device)

    cuda_memory_init = torch.cuda.max_memory_reserved()
    if cuda_memory_init > CUDA_MEMORY_INIT_THRESHOLD:
        raise RuntimeError(
            f"CUDA memory usage at start is too high: {cuda_memory_init // 2**20}MB, please ensure that no other "
            f"processes are running on {device}."
        )

    torch.cuda.reset_peak_memory_stats()
    cuda_memory_init = torch.cuda.max_memory_reserved()
    return cuda_memory_init


def get_tokenizer(*, model_id: str, max_seq_length: int):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.model_max_length = max_seq_length
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_base_model(
    *,
    model_id: str,
    dtype: Literal["float32", "float16", "bfloat16", "int8", "int4"],
    compile: bool,
    attn_implementation: Optional[str],
) -> nn.Module:
    kwargs: dict[str, Any] = {
        "pretrained_model_name_or_path": model_id,
        "device_map": device,
        "attn_implementation": attn_implementation,
    }
    if dtype == "int4":
        quant_config = BitsAndBytesConfig(load_in_4bit=True)
        kwargs["quantization_config"] = quant_config
    elif dtype == "int8":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["quantization_config"] = quant_config
    elif dtype == "bfloat16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif dtype == "float16":
        kwargs["torch_dtype"] = torch.float16
    elif dtype != "float32":
        raise ValueError(f"Invalid dtype: {dtype}")

    model = AutoModelForCausalLM.from_pretrained(**kwargs)

    if dtype in ["int8", "int4"]:
        model = prepare_model_for_kbit_training(model)

    if compile:
        model = torch.compile(model)

    return model


def get_model(
    *,
    model_id: str,
    dtype: Literal["float32", "float16", "bfloat16", "int8", "int4"],
    compile: bool,
    attn_implementation: Optional[str],
    peft_config: PeftConfig,
) -> nn.Module:
    base_model = get_base_model(
        model_id=model_id, dtype=dtype, compile=compile, attn_implementation=attn_implementation
    )
    model = get_peft_model(base_model, peft_config)
    return model


##################
# ANSWER PARSING #
##################


def parse_answer(text: str) -> Optional[str]:
    """
    A label/prediction can look like this:

    Question: If the magnitude of vector v is equal to 4, what is the dot product of vector v with itself?. Think step
    by step
    Answer: The dot product of a vector with itself is equal to the square of its magnitude. So, the dot product of
    vector v with itself is equal to $4^2 = \boxed{16}$.The answer is: 16

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
    text = text.strip(" .!?")
    return text


def get_accuracy(*, predictions: list[str], responses: list[str]) -> float:
    if len(predictions) != len(responses):
        raise ValueError(f"Prediction length mismatch: {len(predictions)} != {len(responses)}")

    correct = 0
    for prediction, response in zip(predictions, responses):
        parsed_prediction = parse_answer(prediction)
        parsed_response = parse_answer(response)
        if parsed_response is None:
            raise ValueError(f"Error encountered while trying to parse response: {response}")
        if parsed_prediction is not None:
            correct += int(parsed_prediction == parsed_response)

    return correct / len(predictions)


###########
# LOGGING #
###########


def get_base_model_info(model_id: str) -> Optional[huggingface_hub.ModelInfo]:
    try:
        return hf_api.model_info(model_id)
    except Exception as exc:
        warnings.warn(f"Could not retrieve model info, failed with error {exc}")
        return None


def get_dataset_info(dataset_id: str) -> Optional[huggingface_hub.DatasetInfo]:
    try:
        return hf_api.dataset_info(dataset_id)
    except Exception as exc:
        warnings.warn(f"Could not retrieve dataset info, failed with error {exc}")
        return None


def get_git_hash(module) -> Optional[str]:
    if "site-packages" in module.__path__[0]:
        return None

    return subprocess.check_output("git rev-parse HEAD".split(), cwd=os.path.dirname(module.__file__)).decode().strip()


def get_package_info() -> dict[str, Optional[str]]:
    """Get the package versions and commit hashes of transformers, peft, datasets, bnb, and torch"""
    import bitsandbytes
    import datasets
    import torch
    import transformers

    import peft

    package_info = {
        "transformers-version": transformers.__version__,
        "transformers-commit-hash": get_git_hash(transformers),
        "peft-version": peft.__version__,
        "peft-commit-hash": get_git_hash(peft),
        "datasets-version": datasets.__version__,
        "datasets-commit-hash": get_git_hash(datasets),
        "bitsandbytes-version": bitsandbytes.__version__,
        "bitsandbytes-commit-hash": get_git_hash(bitsandbytes),
        "torch-version": torch.__version__,
        "torch-commit-hash": get_git_hash(torch),
    }
    return package_info


def get_system_info() -> dict[str, str]:
    system_info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "gpu": torch.cuda.get_device_name(0),
    }
    return system_info


@dataclass
class MetaInfo:
    package_info: dict[str, Optional[str]]
    system_info: dict[str, str]
    pytorch_info: str


def get_meta_info() -> MetaInfo:
    meta_info = MetaInfo(
        package_info=get_package_info(),
        system_info=get_system_info(),
        pytorch_info=torch.__config__.show(),
    )
    return meta_info


def get_peft_branch() -> str:
    return (
        subprocess.check_output("git rev-parse --abbrev-ref HEAD".split(), cwd=os.path.dirname(peft.__file__))
        .decode()
        .strip()
    )


class TrainStatus(enum.Enum):
    FAILED = "failed"
    SUCCESS = "success"
    CANCELED = "canceled"


@dataclass
class TrainResult:
    status: TrainStatus
    train_time: float
    cuda_memory_reserved_log: list[int]
    losses: list[float]
    metrics: list[Any]  # TODO


def log_to_console(log_data: dict[str, Any], print_fn: Callable[..., None]) -> None:
    cuda_memory_max = log_data["train_info"]["cuda_memory_max"]
    cuda_memory_avg = log_data["train_info"]["cuda_memory_reserved_avg"]
    cuda_memory_reserved_99th = log_data["train_info"]["cuda_memory_reserved_99th"]
    time_train = log_data["train_info"]["train_time"]
    time_total = log_data["run_info"]["total_time"]
    file_size = log_data["train_info"]["file_size"]

    print_fn(f"cuda memory max: {cuda_memory_max // 2**20}MB")
    print_fn(f"cuda memory reserved avg: {cuda_memory_avg // 2**20}MB")
    print_fn(f"cuda memory reserved 99th percentile: {cuda_memory_reserved_99th // 2**20}MB")
    print_fn(f"train time: {time_train}s")
    print_fn(f"total time: {time_total:.2f}s")
    print_fn(f"file size of checkpoint: {file_size / 2**20:.1f}MB")


def log_to_file(
    *, log_data: dict, save_dir: str, experiment_name: str, timestamp: str, print_fn: Callable[..., None]
) -> None:
    file_name = f"{experiment_name.replace(os.path.sep, '--')}--{timestamp.replace(':', '-')}.json"
    file_name = os.path.join(save_dir, file_name)
    with open(file_name, "w") as f:
        json.dump(log_data, f, indent=2)
    print_fn(f"Saved log to: {file_name}")


def log_results(
    *,
    experiment_name: str,
    train_result: TrainResult,
    cuda_memory_init: int,
    time_total: float,
    model: nn.Module,
    model_info: Optional[huggingface_hub.ModelInfo],
    datasets_info: dict[str, Optional[huggingface_hub.DatasetInfo]],
    start_date: str,
    train_config: TrainConfig,
    peft_config: PeftConfig,
    print_fn: Callable[..., None],
) -> None:
    # collect results
    cuda_memory_final = torch.cuda.max_memory_reserved()
    cuda_memory_avg = int(sum(train_result.cuda_memory_reserved_log) / len(train_result.cuda_memory_reserved_log))
    cuda_memory_reserved_99th = int(np.percentile(train_result.cuda_memory_reserved_log, 99))

    with tempfile.TemporaryDirectory() as tmp_dir:
        model.save_pretrained(tmp_dir)
        stat = os.stat(os.path.join(tmp_dir, SAFETENSORS_WEIGHTS_NAME))
        file_size = stat.st_size
        print_fn(f"Saved PEFT checkpoint to {tmp_dir}")

    meta_info = get_meta_info()
    if model_info is not None:
        model_sha = model_info.sha
        model_created_at = model_info.created_at.isoformat()
    else:
        model_sha = None
        model_created_at = None

    dataset_info_log = {}
    for key, dataset_info in datasets_info.items():
        if dataset_info is not None:
            dataset_sha = dataset_info.sha
            dataset_created_at = dataset_info.created_at.isoformat()
        else:
            dataset_sha = None
            dataset_created_at = None
        dataset_info_log[key] = {"sha": dataset_sha, "created_at": dataset_created_at}

    peft_branch = get_peft_branch()

    if train_result.status == TrainStatus.CANCELED:
        save_dir = RESULT_PATH_CANCELLED
        print_fn("Experiment run was categorized as canceled")
    elif peft_branch != "main":
        save_dir = RESULT_PATH_TEST
        print_fn(f"Experiment run was categorized as a test run on branch {peft_branch}")
    elif train_result.status == TrainStatus.SUCCESS:
        save_dir = RESULT_PATH
        print_fn("Experiment run was categorized as successful run")

    peft_config_dict = peft_config.to_dict()
    for key, value in peft_config_dict.items():
        if isinstance(value, set):
            peft_config_dict[key] = list(value)

    log_data = {
        "run_info": {
            "created_at": start_date,
            "total_time": time_total,
            "experiment_name": experiment_name,
            "peft_branch": peft_branch,
            "train_config": asdict(train_config),
            "peft_config": peft_config_dict,
        },
        "train_info": {
            "cuda_memory_reserved_avg": cuda_memory_avg,
            "cuda_memory_max": (cuda_memory_final - cuda_memory_init),
            "cuda_memory_reserved_99th": cuda_memory_reserved_99th,
            "train_time": train_result.train_time,
            "file_size": file_size,
            "status": train_result.status.value,
            "metrics": train_result.metrics,
        },
        "meta_info": {
            "model_info": {"sha": model_sha, "created_at": model_created_at},
            "dataset_info": dataset_info_log,
            **asdict(meta_info),
        },
    }

    log_to_console(log_data, print_fn=print)  # use normal print to be able to redirect if so desired
    log_to_file(
        log_data=log_data, save_dir=save_dir, experiment_name=experiment_name, timestamp=start_date, print_fn=print_fn
    )
