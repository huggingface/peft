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
Utilities for PEFT benchmarking.
"""

import datetime
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import psutil
import torch

from peft.utils import infer_device


FILE_NAME_BENCHMARK_PARAMS = "benchmark_params.json"
FILE_NAME_DEFAULT_CONFIG = "default_benchmark_params.json"

RESULT_PATH = os.path.join(os.path.dirname(__file__), "results")
RESULT_PATH_TEMP = os.path.join(os.path.dirname(__file__), "temporary_results")
RESULT_PATH_CANCELLED = os.path.join(os.path.dirname(__file__), "cancelled_results")


class BenchmarkStatus(Enum):
    """Status of a benchmark run."""

    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RUNNING = "running"


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    experiment_name: str
    status: BenchmarkStatus

    model_id: str

    run_info: dict = field(default_factory=dict)
    generation_info: dict = field(default_factory=dict)
    meta_info: dict = field(default_factory=dict)

    def __post_init__(self):
        """Initialize structured data format."""
        device = infer_device()
        torch_accelerator_module = getattr(torch, device, torch.cuda)
        self.run_info = {
            "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
            "duration": 0.0,
            "status": self.status.value,
            "hardware": {
                "num_accelerators": torch_accelerator_module.device_count() if torch_accelerator_module.is_available() else 0,
                "accelerator_type": torch_accelerator_module.get_device_name(0) if torch_accelerator_module.is_available() else "N/A",
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
                "pytorch_version": torch.__version__,
            },
        }

        self.meta_info = {
            "model_id": self.model_id,
            "parameters": {
                "base_params": 0,
                "trainable_params": 0,
                "total_params": 0,
                "param_ratio": 0.0,
            },
            "model_size": {
                "base_model_size_mb": 0.0,
                "adapter_size_mb": 0.0,
            },
            "package_info": {
                "transformers-version": None,
                "transformers-commit-hash": None,
                "peft-version": None,
                "peft-commit-hash": None,
                "datasets-version": None,
                "datasets-commit-hash": None,
                "bitsandbytes-version": None,
                "bitsandbytes-commit-hash": None,
                "torch-version": torch.__version__,
                "torch-commit-hash": None,
            },
            "system_info": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "accelerator": torch_accelerator_module.get_device_name(0) if torch_accelerator_module.is_available() else "N/A",
            },
        }

        self.generation_info = {
            "memory": {
                "peak_accelerator_memory_mb": 0.0,
                "peak_ram_memory_mb": 0.0,
                "memory_logs": [],
            },
            "by_category": {},
            "overall": {},
        }

    def update_meta_info(self, param_counts: dict, size_info: dict, package_info: Optional[dict] = None):
        """Update model metadata information."""
        self.meta_info["parameters"].update(param_counts)
        self.meta_info["model_size"].update(size_info)
        if package_info:
            self.meta_info["package_info"].update(package_info)

    def update_generation_info(self, memory_data: Optional[dict] = None, performance_metrics: Optional[dict] = None):
        """Update generation performance information, primarily for memory and high-level performance."""
        if memory_data:
            self.generation_info["memory"].update(memory_data)
        if performance_metrics:  # For things like overall tokens/sec if calculated
            self.generation_info.update(performance_metrics)

    def add_memory_log(self, stage: str, ram_mb: float, accelerator_allocated_mb: float, accelerator_reserved_mb: float):
        """Add a memory usage log entry to generation_info."""
        self.generation_info["memory"]["memory_logs"].append(
            {
                "stage": stage,
                "ram_mb": ram_mb,
                "accelerator_allocated_mb": accelerator_allocated_mb,
                "accelerator_reserved_mb": accelerator_reserved_mb,
            }
        )

    def add_metrics_for_category(self, category: str, metrics: dict, individual_samples: list = None):
        """Add metrics for a specific prompt category under generation_info."""
        category_data = {"metrics": metrics, "samples": individual_samples if individual_samples is not None else []}
        self.generation_info["by_category"][category] = category_data

    def update_run_info(
        self,
        duration: float,
        status: BenchmarkStatus,
        error: Optional[str] = None,
        peft_config: Optional[dict] = None,
        benchmark_config: Optional[dict] = None,
    ):
        """Update run information."""
        self.run_info["duration"] = duration
        self.run_info["status"] = status.value
        if error:
            self.run_info["error"] = error
        if peft_config:
            self.run_info["peft_config"] = peft_config
        if benchmark_config:
            self.run_info["benchmark_config"] = benchmark_config

    def compute_overall_metrics(self):
        """Compute overall metrics across all categories within generation_info."""
        if not self.generation_info["by_category"]:
            return

        categories = self.generation_info["by_category"]
        key_metrics = [
            "inference_time",
            "base_inference_time",
            "inference_overhead_pct",
            "time_per_token",
            "generated_tokens",
        ]

        for metric in key_metrics:
            values = []
            for category_data in categories.values():
                if "metrics" in category_data and metric in category_data["metrics"]:
                    values.append(category_data["metrics"][metric])

            if values:
                self.generation_info["overall"][metric] = sum(values) / len(values)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        self.compute_overall_metrics()
        return {
            "run_info": self.run_info,
            "generation_info": self.generation_info,
            "meta_info": self.meta_info,
        }

    def save(self, path: Optional[str] = None):
        """Save result to JSON file."""
        if path is None:
            peft_branch = get_peft_branch()
        if self.status == BenchmarkStatus.CANCELLED:
            base_path = RESULT_PATH_CANCELLED
        elif peft_branch != "main":
            base_path = RESULT_PATH_TEMP
        elif self.status == BenchmarkStatus.SUCCESS:
            base_path = RESULT_PATH
        elif self.status == BenchmarkStatus.FAILED:
            base_path = RESULT_PATH_CANCELLED
        else:
            base_path = RESULT_PATH_TEMP

        filename = f"{self.experiment_name}.json"
        path = os.path.join(base_path, filename)

        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        return path


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking PEFT methods."""

    model_id: str

    seed: int
    num_inference_runs: int
    max_new_tokens: int

    dtype: str = "float16"
    use_4bit: bool = False
    use_8bit: bool = False

    category_generation_params: Optional[dict] = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not isinstance(self.model_id, str):
            raise ValueError(f"Invalid model_id: {self.model_id}")

        if self.seed < 0:
            raise ValueError(f"Invalid seed: {self.seed}")

        if self.num_inference_runs <= 0:
            raise ValueError(f"Invalid num_inference_runs: {self.num_inference_runs}")

        if self.max_new_tokens <= 0:
            raise ValueError(f"Invalid max_new_tokens: {self.max_new_tokens}")

    @classmethod
    def from_dict(cls, config_dict: dict) -> "BenchmarkConfig":
        """Create config from dictionary."""
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}

        return cls(**filtered_dict)

    @classmethod
    def from_json(cls, json_path: str) -> "BenchmarkConfig":
        """Load config from JSON file."""
        with open(json_path) as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        result = asdict(self)
        return result

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def merge_from_dict(self, config_dict: dict) -> None:
        """Merge settings from a dictionary into this config object.
        Keys in config_dict will override existing attributes.
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


def validate_experiment_path(path: str) -> tuple[str, "BenchmarkConfig"]:
    """Validate experiment path, load and merge configs, and return them."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Experiment path not found: {path}")

    path_parts = os.path.normpath(path).split(os.sep)

    try:
        experiments_idx = path_parts.index("experiments")
    except ValueError:
        experiment_name = os.path.basename(path.rstrip(os.sep))
    else:
        if experiments_idx + 1 < len(path_parts):
            method_name = path_parts[experiments_idx + 1]
            remaining_parts = path_parts[experiments_idx + 2 :]
            if remaining_parts:
                remaining_name = "-".join(remaining_parts)
                experiment_name = f"{method_name}--{remaining_name}"
            else:
                experiment_name = method_name
        else:
            experiment_name = os.path.basename(path.rstrip(os.sep))

    default_config_path = os.path.join(os.path.dirname(__file__), FILE_NAME_DEFAULT_CONFIG)
    experiment_benchmark_params_path = os.path.join(path, FILE_NAME_BENCHMARK_PARAMS)

    if not os.path.exists(default_config_path):
        raise FileNotFoundError(f"Default configuration file not found: {default_config_path}. This is required.")
    benchmark_config = BenchmarkConfig.from_json(default_config_path)
    print(f"Loaded default configuration from {default_config_path}")

    if os.path.exists(experiment_benchmark_params_path):
        with open(experiment_benchmark_params_path) as f:
            experiment_specific_params = json.load(f)

        benchmark_config.merge_from_dict(experiment_specific_params)
        print(f"Loaded and merged experiment-specific parameters from {experiment_benchmark_params_path}")
    else:
        print(f"No {FILE_NAME_BENCHMARK_PARAMS} found in {path}. Using only default configuration.")

    return experiment_name, benchmark_config


def get_memory_usage() -> tuple[float, float, float]:
    """Get current memory usage (RAM and accelerator)."""
    process = psutil.Process(os.getpid())
    ram_usage_bytes = process.memory_info().rss
    ram_usage_mb = ram_usage_bytes / (1024 * 1024)

    if torch.cuda.is_available():
        accelerator_allocated = torch.cuda.memory_allocated()
        accelerator_reserved = torch.cuda.memory_reserved()
        accelerator_allocated_mb = accelerator_allocated / (1024 * 1024)
        accelerator_reserved_mb = accelerator_reserved / (1024 * 1024)
    elif torch.xpu.is_available():
        accelerator_allocated = torch.xpu.memory_allocated()
        accelerator_reserved = torch.xpu.memory_reserved()
        accelerator_allocated_mb = accelerator_allocated / (1024 * 1024)
        accelerator_reserved_mb = accelerator_reserved / (1024 * 1024)
    else:
        accelerator_allocated_mb = 0.0
        accelerator_reserved_mb = 0.0

    return ram_usage_mb, accelerator_allocated_mb, accelerator_reserved_mb


def init_accelerator() -> tuple[float, float]:
    """Initialize accelerator and return initial memory usage."""
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.empty_cache()
        _, accelerator_allocated, accelerator_reserved = get_memory_usage()
    elif torch.xpu.is_available():
        torch.xpu.init()
        torch.xpu.empty_cache()
        _, accelerator_allocated, accelerator_reserved = get_memory_usage()
    else:
        accelerator_allocated = 0.0
        accelerator_reserved = 0.0
    return accelerator_allocated, accelerator_reserved


def get_model_size_mb(model: torch.nn.Module, dtype_bytes: int = 4) -> float:
    """Calculate model size in MB."""
    return sum(p.numel() * dtype_bytes for p in model.parameters()) / (1024 * 1024)


def get_peft_branch() -> str:
    repo_root = os.path.dirname(__file__)
    return subprocess.check_output("git rev-parse --abbrev-ref HEAD".split(), cwd=repo_root).decode().strip()


def log_results(
    experiment_name: str,
    benchmark_result: BenchmarkResult,
    print_fn: Callable = print,
) -> None:
    """Log benchmark results to console."""
    print_fn("\n" + "=" * 50)
    print_fn(f"Benchmark Results: {experiment_name}")
    print_fn("=" * 50)

    print_fn(f"Status: {benchmark_result.run_info.get('status', 'N/A')}")
    print_fn(f"Duration: {benchmark_result.run_info.get('duration', 0):.2f} seconds")

    if benchmark_result.run_info.get("status") != BenchmarkStatus.SUCCESS.value:
        print_fn(f"Error: {benchmark_result.run_info.get('error', 'Unknown error')}")
        print_fn("=" * 50)
        return

    print_fn("\nModel Information:")
    print_fn(f"  Base Model: {benchmark_result.meta_info.get('model_id', 'N/A')}")

    print_fn("\nParameter Counts:")
    params = benchmark_result.meta_info.get("parameters", {})
    print_fn(f"  Base Parameters: {params.get('base_params', 0):,}")
    print_fn(f"  Trainable Parameters: {params.get('trainable_params', 0):,}")
    print_fn(f"  Parameter Ratio: {params.get('param_ratio', 0):.5%}")

    print_fn("\nModel Size:")
    size_info = benchmark_result.meta_info.get("model_size", {})
    print_fn(f"  Base Model: {size_info.get('base_model_size_mb', 0):.2f} MB")
    print_fn(f"  Adapter: {size_info.get('adapter_size_mb', 0):.2f} MB")

    print_fn("\nMemory Usage (from generation_info):")
    memory_data = benchmark_result.generation_info.get("memory", {})
    print_fn(f"  Peak Accelerator Memory: {memory_data.get('peak_accelerator_memory_mb', 0):.2f} MB")
    print_fn(f"  Peak RAM Memory: {memory_data.get('peak_ram_memory_mb', 0):.2f} MB")

    print_fn("\nDetailed Metrics (from generation_info.by_category):")
    if benchmark_result.generation_info.get("by_category"):
        for category, cat_data in benchmark_result.generation_info["by_category"].items():
            print_fn(f"  Category: {category}")
            metrics = cat_data.get("metrics", {})
            print_fn(f"    Inference Time: {metrics.get('inference_time', 0):.4f} seconds")
            print_fn(f"    Base Inference Time: {metrics.get('base_inference_time', 0):.4f} seconds")
            print_fn(f"    Inference Overhead: {metrics.get('inference_overhead_pct', 0):.2f}%")
            print_fn(f"    Time Per Token: {metrics.get('time_per_token', 0):.6f} seconds/token")
            print_fn(f"    Generated Tokens: {metrics.get('generated_tokens', 0):.1f}")

            samples = cat_data.get("samples", [])
            if samples:
                print_fn(f"    Number of Samples: {len(samples)}")
                print_fn(
                    f"    Average Generated Tokens: {sum(s.get('generated_tokens', 0) for s in samples) / len(samples):.1f}"
                )
    else:
        print_fn("  No per-category metrics available.")

    benchmark_result.compute_overall_metrics()

    print_fn("\nOverall Metrics (from generation_info.overall):")
    overall = benchmark_result.generation_info.get("overall")
    if overall:
        print_fn(f"    Inference Time: {overall.get('inference_time', 0):.4f} seconds")
        print_fn(f"    Base Inference Time: {overall.get('base_inference_time', 0):.4f} seconds")
        print_fn(f"    Inference Overhead: {overall.get('inference_overhead_pct', 0):.2f}%")
        print_fn(f"    Time Per Token: {overall.get('time_per_token', 0):.6f} seconds/token")
        print_fn(f"    Generated Tokens: {overall.get('generated_tokens', 0):.1f}")
    else:
        print_fn("  No overall metrics computed.")

    print_fn("\nSaved results to:", benchmark_result.save())
    print_fn("=" * 50)
