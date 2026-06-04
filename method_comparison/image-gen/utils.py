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

"""Utilities for the image generation benchmark."""

import copy
import enum
import json
import os
import platform
import subprocess
import tempfile
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal, Optional

import datasets
import diffusers
import huggingface_hub
import numpy as np
import torch
import transformers
from diffusers import Flux2KleinPipeline
from torch import nn
from transformers import AutoImageProcessor, AutoModel, get_cosine_schedule_with_warmup

import peft
from peft import PeftConfig, get_peft_model
from peft.optimizers import create_lorafa_optimizer, create_loraplus_optimizer
from peft.utils import SAFETENSORS_WEIGHTS_NAME, infer_device


device = infer_device()

if device not in ["cuda", "xpu"]:
    raise RuntimeError("CUDA or XPU is not available, currently only CUDA or XPU is supported")

ACCELERATOR_MEMORY_INIT_THRESHOLD = 500 * 2**20  # 500MB
FILE_NAME_DEFAULT_TRAIN_PARAMS = os.path.join(os.path.dirname(__file__), "default_training_params.json")
FILE_NAME_TRAIN_PARAMS = "training_params.json"
RESULT_PATH = os.path.join(os.path.dirname(__file__), "results")
RESULT_PATH_TEST = os.path.join(os.path.dirname(__file__), "temporary_results")
RESULT_PATH_CANCELLED = os.path.join(os.path.dirname(__file__), "cancelled_results")
SAMPLE_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "sample-images")
SAMPLE_IMAGE_PATH_MAIN = os.path.join(SAMPLE_IMAGE_PATH, "results")
SAMPLE_IMAGE_PATH_TEST = os.path.join(SAMPLE_IMAGE_PATH, "temporary_results")
SAMPLE_IMAGE_PATH_CANCELLED = os.path.join(SAMPLE_IMAGE_PATH, "cancelled_results")
hf_api = huggingface_hub.HfApi()
WARMUP_STEP_RATIO = 0.1


@dataclass
class TrainConfig:
    """All configuration parameters associated with training the model

    Args:
        model_id: The model identifier, should not be changed
        dataset_id: The dataset identifier, should not be changed
        dataset_split: The dataset split to use, should not be changed
        dtype: The data type to use for the model
        resolution: The image resolution
        batch_size: The batch size for training
        batch_size_eval: The batch size for eval/test
        repeats: The number of repeats for the dataset (if there are more steps than train samples)
        max_steps: The maximum number of steps to train
        eval_steps: The number of steps between evaluations
        compile: Whether to compile the model
        use_gc: Whether to use gradient checkpointing.
        seed: The random seed
        grad_norm_clip: The gradient norm clipping value (set to 0 to skip)
        optimizer_type: The name of a torch optimizer (e.g. AdamW) or a PEFT method ("lora+", "lora-fa")
        optimizer_kwargs: The optimizer keyword arguments (lr etc.)
        lr_scheduler: The learning rate scheduler (currently only None or 'cosine' are supported)
        use_amp: Whether to use automatic mixed precision
        autocast_adapter_dtype: Whether to cast adapter dtype to float32, same argument as in PEFT
        instance_prompts: The prompt(s) used for training instances
        image_column: The column name for images in the dataset
        valid_size: The validation set size
        test_size: The test set size
        num_inference_steps: The number of inference steps for image generation
        guidance_scale: The guidance scale for image generation
        max_sequence_length: The maximum sequence length for the text encoder
        text_encoder_out_layers: The output layers of the text encoder to use
        weighting_scheme: The weighting scheme for the loss
        logit_mean: The logit mean for logit_normal weighting
        logit_std: The logit std for logit_normal weighting
        mode_scale: The mode scale for mode weighting
        dino_model_id: The DINO model identifier for evaluation
        dino_image_size: The image size for the DINO model
        sample_image_prompts: The prompts used for generating sample images, should not be changed
        drift_image_prompts: The prompts used for measuring drift, should not be changed
    """

    model_id: str
    dataset_id: str
    dataset_split: str
    dtype: Literal["float32", "float16", "bfloat16"]
    resolution: int
    batch_size: int
    batch_size_eval: int
    repeats: int
    max_steps: int
    eval_steps: int
    compile: bool
    use_gc: bool
    seed: int
    grad_norm_clip: float
    optimizer_type: str
    optimizer_kwargs: dict[str, Any]
    lr_scheduler: Optional[Literal["cosine"]]
    use_amp: bool
    autocast_adapter_dtype: bool
    instance_prompts: str | list[str]
    image_column: str
    valid_size: int
    test_size: int
    num_inference_steps: int
    guidance_scale: float
    max_sequence_length: int
    text_encoder_out_layers: list[int]
    weighting_scheme: Literal["none", "sigma_sqrt", "logit_normal", "mode"]
    logit_mean: float
    logit_std: float
    mode_scale: float
    dino_model_id: str
    dino_image_size: int
    sample_image_prompts: list[str]
    drift_image_prompts: list[str]

    def __post_init__(self) -> None:
        if self.dtype not in ["float32", "float16", "bfloat16"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")
        if self.batch_size <= 0:
            raise ValueError(f"Invalid batch_size: {self.batch_size}")
        if self.batch_size_eval <= 0:
            raise ValueError(f"Invalid batch_size_eval: {self.batch_size_eval}")
        if self.repeats <= 0:
            raise ValueError(f"Invalid repeats: {self.repeats}")
        if self.max_steps <= 0:
            raise ValueError(f"Invalid max_steps: {self.max_steps}")
        if self.eval_steps <= 0:
            raise ValueError(f"Invalid eval_steps: {self.eval_steps}")
        if self.eval_steps > self.max_steps:
            raise ValueError(f"Invalid eval_steps: {self.eval_steps} > max_steps: {self.max_steps}")
        if self.grad_norm_clip < 0:
            raise ValueError(f"Invalid grad_norm_clip: {self.grad_norm_clip}")
        if self.optimizer_type not in ["lora+", "lora-fa"] and not hasattr(torch.optim, self.optimizer_type):
            raise ValueError(f"Invalid optimizer_type: {self.optimizer_type}")
        if self.lr_scheduler not in [None, "cosine"]:
            raise ValueError(f"Invalid lr_scheduler: {self.lr_scheduler}, must be None or 'cosine'")


def validate_experiment_path(path: str) -> str:
    if not os.path.exists(FILE_NAME_DEFAULT_TRAIN_PARAMS):
        raise FileNotFoundError(f"Missing default training params file '{FILE_NAME_DEFAULT_TRAIN_PARAMS}'")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path {path} does not exist")

    path_parts = path.rstrip(os.path.sep).split(os.path.sep)
    if (len(path_parts) != 3) or (path_parts[-3] != "experiments"):
        raise ValueError(
            f"Path {path} does not have the correct structure, should be ./experiments/<peft-method>/<experiment-name>"
        )

    experiment_name = os.path.join(*path_parts[-2:])
    return experiment_name


def get_train_config(path: str) -> TrainConfig:
    with open(FILE_NAME_DEFAULT_TRAIN_PARAMS) as f:
        default_config_kwargs = json.load(f)

    config_kwargs = {}
    if os.path.exists(path):
        with open(path) as f:
            config_kwargs = json.load(f)

    config_kwargs = {**default_config_kwargs, **config_kwargs}
    return TrainConfig(**config_kwargs)


def init_accelerator() -> int:
    torch_accelerator_module = getattr(torch, device, torch.cuda)
    torch.manual_seed(0)
    torch_accelerator_module.reset_peak_memory_stats()
    torch_accelerator_module.manual_seed_all(0)
    nn.Linear(1, 1).to(device)

    accelerator_memory_init = torch_accelerator_module.max_memory_reserved()
    if accelerator_memory_init > ACCELERATOR_MEMORY_INIT_THRESHOLD:
        raise RuntimeError(
            f"{device} memory usage at start is too high: {accelerator_memory_init // 2**20}MB, "
            f"please ensure that no other processes are running on {device}."
        )

    torch_accelerator_module.reset_peak_memory_stats()
    accelerator_memory_init = torch_accelerator_module.max_memory_reserved()
    return accelerator_memory_init


def get_torch_dtype(dtype: Literal["float32", "float16", "bfloat16"]) -> torch.dtype:
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    return torch.bfloat16


def get_pipeline(
    *,
    model_id: str,
    dtype: Literal["float32", "float16", "bfloat16"],
    compile: bool,
    peft_config: Optional[PeftConfig],
    autocast_adapter_dtype: bool,
    use_gc: bool,
):
    torch_dtype = get_torch_dtype(dtype)
    pipeline = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    pipeline.set_progress_bar_config(disable=True)
    if use_gc:
        pipeline.transformer.enable_gradient_checkpointing()

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)

    transformer = pipeline.transformer
    if peft_config is None:
        transformer.requires_grad_(True)
    else:
        transformer = get_peft_model(transformer, peft_config, autocast_adapter_dtype=autocast_adapter_dtype)
        pipeline.transformer = transformer

    if compile:
        pipeline.transformer = torch.compile(pipeline.transformer, dynamic=True)

    pipeline.transformer.train()
    pipeline.vae.eval()
    pipeline.text_encoder.eval()
    return pipeline


class DummyScheduler:
    def __init__(self, lr):
        self.lr = lr

    def get_last_lr(self):
        return [self.lr]

    def step(self):
        pass


def get_optimizer_and_scheduler(
    model, *, optimizer_type: str, max_steps: int, lr_scheduler_arg: Optional[Literal["cosine"]], **optimizer_kwargs
) -> tuple[torch.optim.Optimizer, Any]:
    if optimizer_type == "lora+":
        optimizer = create_loraplus_optimizer(model, optimizer_cls=torch.optim.AdamW, **optimizer_kwargs)
    elif optimizer_type == "lora-fa":
        optimizer = create_lorafa_optimizer(model, **optimizer_kwargs)
    else:
        cls = getattr(torch.optim, optimizer_type)
        optimizer = cls(model.parameters(), **optimizer_kwargs)

    if lr_scheduler_arg == "cosine":
        warmup_steps = int(WARMUP_STEP_RATIO * max_steps)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps
        )
    elif lr_scheduler_arg is None:
        lr_scheduler = DummyScheduler(optimizer_kwargs["lr"])
    else:
        raise ValueError(f"Invalid lr_scheduler argument: {lr_scheduler_arg}")

    return optimizer, lr_scheduler


def upload_checkpoint_to_bucket(model: nn.Module, experiment_name: str, bucket_name: str):
    """Uploads model checkpoint to Hugging Face Bucket"""
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True, delete=True) as tmp_dir:
            model.save_pretrained(tmp_dir)
            huggingface_hub.batch_bucket_files(
                bucket_name,
                add=[
                    (os.path.join(tmp_dir, fname), f"checkpoints/{experiment_name}/{fname}")
                    for fname in os.listdir(tmp_dir)
                ],
            )
    except Exception as exc:
        print(f"Failed to upload model checkpoint to hub: {exc}")


def upload_images_to_bucket(bucket_name: str):
    """Syncs test images (only main runs) with Hugging Face Bucket"""
    try:
        huggingface_hub.sync_bucket(SAMPLE_IMAGE_PATH, f"hf://buckets/{bucket_name}/sample-images", delete=False)
    except Exception as exc:
        print(f"Failed to upload sample images to hub: {exc}")


def get_file_size(
    transformer: nn.Module, *, peft_config: Optional[PeftConfig], clean: bool, print_fn: Callable[..., None]
) -> int:
    file_size = 99999999
    if peft_config is not None:
        try:
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True, delete=clean) as tmp_dir:
                transformer.save_pretrained(tmp_dir)
                stat = os.stat(os.path.join(tmp_dir, SAFETENSORS_WEIGHTS_NAME))
                file_size = stat.st_size
                if not clean:
                    print_fn(f"Saved PEFT checkpoint to {tmp_dir}")
        except Exception as exc:
            print(f"Failed to save PEFT checkpoint due to the following error: {exc}")
    else:
        print_fn("Not saving full model checkpoint because it is too large, estimating size instead")
        try:
            num_params = sum(param.numel() for param in transformer.parameters())
            dtype_size = next(transformer.parameters()).element_size()
            file_size = num_params * dtype_size
        except Exception as exc:
            print(f"Failed to determine file size for fully finetuned model because of: {exc}")
    return file_size


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
    module_path = module.__path__[0]
    if "site-packages" in module_path:
        return None
    return subprocess.check_output("git rev-parse HEAD".split(), cwd=os.path.dirname(module.__file__)).decode().strip()


def get_package_info() -> dict[str, Optional[str]]:
    package_info = {
        "transformers-version": transformers.__version__,
        "transformers-commit-hash": get_git_hash(transformers),
        "peft-version": peft.__version__,
        "peft-commit-hash": get_git_hash(peft),
        "datasets-version": datasets.__version__,
        "datasets-commit-hash": get_git_hash(datasets),
        "diffusers-version": diffusers.__version__,
        "diffusers-commit-hash": get_git_hash(diffusers),
        "torch-version": torch.__version__,
        "torch-commit-hash": get_git_hash(torch),
    }
    return package_info


def get_system_info() -> dict[str, str]:
    torch_accelerator_module = getattr(torch, device, torch.cuda)
    system_info = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "accelerator": torch_accelerator_module.get_device_name(0),
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
    accelerator_memory_reserved_log: list[int]
    accelerator_memory_max_train: int
    losses: list[float]
    metrics: list[Any]
    error_msg: str
    num_trainable_params: int
    num_total_params: int


def get_dino_encoder(model_id: str, image_size: int):
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model


@torch.inference_mode()
def get_dino_embeddings(images, processor, model, batch_size: int):
    embeddings = []
    for i in range(0, len(images), batch_size):
        batch_images = images[i : i + batch_size]
        inputs = processor(images=batch_images, return_tensors="pt").to(model.device)
        hidden_state = model(**inputs).last_hidden_state[:, 0]
        hidden_state = torch.nn.functional.normalize(hidden_state, dim=-1)
        embeddings.append(hidden_state)
    return torch.cat(embeddings, dim=0)


def log_to_console(log_data: dict[str, Any], print_fn: Callable[..., None]) -> None:
    accelerator_memory_max = log_data["train_info"]["accelerator_memory_max"]
    accelerator_memory_avg = log_data["train_info"]["accelerator_memory_reserved_avg"]
    accelerator_memory_reserved_99th = log_data["train_info"]["accelerator_memory_reserved_99th"]
    time_train = log_data["train_info"]["train_time"]
    time_total = log_data["run_info"]["total_time"]
    file_size = log_data["train_info"]["file_size"]

    print_fn(f"accelerator memory max: {accelerator_memory_max // 2**20}MB")
    print_fn(f"accelerator memory reserved avg: {accelerator_memory_avg // 2**20}MB")
    print_fn(f"accelerator memory reserved 99th percentile: {accelerator_memory_reserved_99th // 2**20}MB")
    print_fn(f"train time: {time_train}s")
    print_fn(f"total time: {time_total:.2f}s")
    print_fn(f"file size of checkpoint: {file_size / 2**20:.1f}MB")


def log_to_file(
    *, log_data: dict, save_dir: str, experiment_name: str, timestamp: str, print_fn: Callable[..., None]
) -> None:
    file_name = os.path.join(save_dir, f"{get_artifact_stem(experiment_name, timestamp, save_dir)}.json")
    with open(file_name, "w") as f:
        json.dump(log_data, f, indent=2)
    print_fn(f"Saved log to: {file_name}")


def get_result_save_dir(*, train_status: TrainStatus, peft_branch: str) -> str:
    if train_status == TrainStatus.CANCELED:
        return RESULT_PATH_CANCELLED
    if peft_branch != "main":
        return RESULT_PATH_TEST
    if train_status == TrainStatus.SUCCESS:
        return RESULT_PATH
    return tempfile.mkdtemp()


def get_sample_image_save_dir(*, train_status: TrainStatus, peft_branch: str) -> str:
    if train_status == TrainStatus.CANCELED:
        return SAMPLE_IMAGE_PATH_CANCELLED
    if peft_branch != "main":
        return SAMPLE_IMAGE_PATH_TEST
    if train_status == TrainStatus.SUCCESS:
        return SAMPLE_IMAGE_PATH_MAIN
    return tempfile.mkdtemp()


def get_artifact_stem(experiment_name: str, timestamp: str, save_dir: str) -> str:
    experiment_name = experiment_name.replace(os.path.sep, "--")
    if save_dir.endswith(RESULT_PATH) or save_dir.endswith(SAMPLE_IMAGE_PATH_MAIN):
        return experiment_name
    return f"{experiment_name}--{timestamp.replace(':', '-')}"


def log_results(
    *,
    experiment_name: str,
    train_result: TrainResult,
    time_total: float,
    file_size: int,
    model_info: Optional[huggingface_hub.ModelInfo],
    dataset_info: Optional[huggingface_hub.DatasetInfo],
    start_date: str,
    train_config: TrainConfig,
    peft_config: Optional[PeftConfig],
    print_fn: Callable[..., None],
) -> None:
    if train_result.accelerator_memory_reserved_log:
        accelerator_memory_avg = int(
            sum(train_result.accelerator_memory_reserved_log) / len(train_result.accelerator_memory_reserved_log)
        )
        accelerator_memory_reserved_99th = int(np.percentile(train_result.accelerator_memory_reserved_log, 99))
    else:
        accelerator_memory_avg = 0
        accelerator_memory_reserved_99th = 0

    meta_info = get_meta_info()
    if model_info is not None:
        model_sha = model_info.sha
        model_created_at = model_info.created_at.isoformat()
    else:
        model_sha = None
        model_created_at = None

    if dataset_info is not None:
        dataset_sha = dataset_info.sha
        dataset_created_at = dataset_info.created_at.isoformat()
    else:
        dataset_sha = None
        dataset_created_at = None

    peft_branch = get_peft_branch()

    save_dir = get_result_save_dir(train_status=train_result.status, peft_branch=peft_branch)

    if save_dir == RESULT_PATH_CANCELLED:
        print_fn("Experiment run was categorized as canceled")
    elif save_dir == RESULT_PATH_TEST:
        print_fn(f"Experiment run was categorized as a test run on branch {peft_branch}")
    elif save_dir == RESULT_PATH:
        print_fn("Experiment run was categorized as successful run")
    else:
        print_fn(f"Experiment could not be categorized, writing results to {save_dir}. Please open an issue on PEFT.")

    if peft_config is None:
        peft_config_dict: Optional[dict[str, Any]] = None
    else:
        peft_config_dict = copy.deepcopy(peft_config.to_dict())
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
            "error_msg": train_result.error_msg,
        },
        "train_info": {
            "accelerator_memory_reserved_avg": accelerator_memory_avg,
            "accelerator_memory_max": train_result.accelerator_memory_max_train,
            "accelerator_memory_reserved_99th": accelerator_memory_reserved_99th,
            "train_time": train_result.train_time,
            "file_size": file_size,
            "num_trainable_params": train_result.num_trainable_params,
            "num_total_params": train_result.num_total_params,
            "status": train_result.status.value,
            "metrics": train_result.metrics,
        },
        "meta_info": {
            "model_info": {"sha": model_sha, "created_at": model_created_at},
            "dataset_info": {"sha": dataset_sha, "created_at": dataset_created_at},
            **asdict(meta_info),
        },
    }

    log_to_console(log_data, print_fn=print)
    log_to_file(
        log_data=log_data, save_dir=save_dir, experiment_name=experiment_name, timestamp=start_date, print_fn=print_fn
    )
