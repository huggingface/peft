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

import json
import os
import platform
import subprocess
from dataclasses import dataclass
from typing import Any, Literal, Optional

import huggingface_hub
import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

import peft
from peft import PeftConfig, get_peft_model, prepare_model_for_kbit_training
from peft.utils import CONFIG_NAME


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available, currently only CUDA is supported")

device = "cuda"
CUDA_MEMORY_INIT_THRESHOLD = 500 * 2**20  # 500MB
FILE_NAME_TRAIN_PARAMS = "training_params.json"
DATASET_NAME = "meta-math/MetaMathQA"
hf_api = huggingface_hub.HfApi()


@dataclass
class TrainConfig:
    """All configuration parameters associated with training the model"""

    model_id: str
    dtype: Literal["float32", "float16", "bfloat16", "int8", "int4"]
    max_seq_length: int
    batch_size: int
    max_steps: int
    eval_steps: int
    compile: bool
    query_template: str
    max_generation_length: int
    seed: int
    grad_norm_clip: float  # set to 0 to skip
    optimizer_kwargs: dict[str, Any]
    lr_scheduler: Optional[Literal["cosine"]]

    def __post_init__(self):
        if not isinstance(self.model_id, str):
            raise ValueError(f"Invalid model_id: {self.model_id}")
        if self.dtype not in ["float32", "float16", "bfloat16", "int8", "int4"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")
        if self.max_seq_length < 0:
            raise ValueError(f"Invalid max_seq_length: {self.max_seq_length}")
        if self.batch_size <= 0:
            raise ValueError(f"Invalid batch_size: {self.batch_size}")
        if self.max_steps <= 0:
            raise ValueError(f"Invalid max_steps: {self.max_steps}")
        if self.eval_steps <= 0:
            raise ValueError(f"Invalid eval_steps: {self.eval_steps}")
        if self.eval_steps > self.max_steps:
            raise ValueError(f"Invalid eval_steps: {self.eval_steps} > max_steps: {self.max_steps}")
        if self.max_generation_length <= 0:
            raise ValueError(f"Invalid max_generation_length: {self.max_generation_length}")
        if self.grad_norm_clip <= 0:
            raise ValueError(f"Invalid grad_norm_clip: {self.grad_norm_clip}")
        if self.lr_scheduler not in [None, "cosine"]:
            raise ValueError(f"Invalid lr_scheduler: {self.lr_scheduler}, must be None or 'cosine'")
        if "{query}" not in self.query_template:
            raise ValueError("Invalid query_template, must contain '{query}'")


def validate_experiment_path(path: str) -> tuple[str, str, str]:
    # the experiment path should take the form of ./experiments/<peft-method>/<experiment-name>
    # e.g. ./experiments/lora/rank32
    # it should contain:
    # - training_params.json
    # - adapter_config.json
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path {path} does not exist")
    if not os.path.exists(os.path.join(path, FILE_NAME_TRAIN_PARAMS)):
        raise FileNotFoundError(os.path.join(path, FILE_NAME_TRAIN_PARAMS))
    if not os.path.exists(os.path.join(path, CONFIG_NAME)):
        raise FileNotFoundError(os.path.join(path, CONFIG_NAME))

    # check path structure
    path_parts = path.rstrip(os.path.sep).split(os.path.sep)
    if (len(path_parts) != 3) or (path_parts[-3] != "experiments"):
        raise ValueError(
            f"Path {path} does not have the correct structure, should be ./experiments/<peft-method>/<experiment-name>"
        )

    experiment_name = os.path.join(*path_parts[-2:])
    train_params_sha = (
        subprocess.check_output(f"sha256sum {os.path.join(path, FILE_NAME_TRAIN_PARAMS)}".split()).decode().split()[0]
    )
    peft_config_sha = (
        subprocess.check_output(f"sha256sum {os.path.join(path, CONFIG_NAME)}".split()).decode().split()[0]
    )
    return experiment_name, train_params_sha, peft_config_sha


def get_train_config(path: str) -> TrainConfig:
    with open(path) as f:
        config_kwargs = json.load(f)

    return TrainConfig(**config_kwargs)


def init_cuda() -> int:
    torch.manual_seed(0)
    if device == "cpu":
        return

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.manual_seed_all(0)
    # might not be necessary, but just to be sure
    nn.Linear(1, 1).to(device)

    cuda_memory_init = torch.cuda.max_memory_allocated()
    if cuda_memory_init > CUDA_MEMORY_INIT_THRESHOLD:
        raise RuntimeError(
            f"CUDA memory usage at start is too high: {cuda_memory_init // 2**20}MB, please ensure that no other "
            f"processes are running on {device}."
        )
    return cuda_memory_init


def get_tokenizer(*, model_id: str, max_seq_length: int):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.model_max_length = max_seq_length
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_base_model(
    *, model_id: str, dtype: Literal["float32", "float16", "bfloat16", "int8", "int4"], compile: bool
) -> nn.Module:
    if dtype == "int4":
        quant_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, quantization_config=quant_config)
        model = prepare_model_for_kbit_training(model)
    elif dtype == "int8":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, quantization_config=quant_config)
        model = prepare_model_for_kbit_training(model)
    elif dtype == "bfloat16":
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, torch_dtype=torch.bfloat16)
    elif dtype == "float16":
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, torch_dtype=torch.float16)
    elif dtype == "float32":
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device)
    else:
        raise ValueError(f"Invalid dtype: {dtype}")
    # model.config.use_cache = False # FIXME needed?

    if compile:
        model = torch.compile(model)

    return model


def get_model(
    *,
    model_id: str,
    dtype: Literal["float32", "float16", "bfloat16", "int8", "int4"],
    compile: bool,
    peft_config: PeftConfig,
) -> nn.Module:
    base_model = get_base_model(model_id=model_id, dtype=dtype, compile=compile)
    model = get_peft_model(base_model, peft_config)
    return model


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
    candidate_delimiters = ["The answer is: ", "The answer is ", "The final answer is: ", "The final answer is "]
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


def get_base_model_info(model_id: str) -> huggingface_hub.ModelInfo:
    return hf_api.model_info(model_id)


def get_dataset_info(dataset_id: str) -> huggingface_hub.DatasetInfo:
    return hf_api.dataset_info(dataset_id)


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


def get_meta_info():
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
