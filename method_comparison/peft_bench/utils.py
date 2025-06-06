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
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import psutil
import torch


# Constants
FILE_NAME_BENCHMARK_PARAMS = "benchmark_params.json"
FILE_NAME_DEFAULT_CONFIG = "default_benchmark_params.json"

# Main paths for storing results
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

    # Experiment identification
    experiment_id: str
    experiment_name: str
    status: BenchmarkStatus

    # Model info
    model_id: str
    peft_method: str

    # Structure for organized results
    run_info: dict = field(default_factory=dict)
    generation_info: dict = field(default_factory=dict)
    meta_info: dict = field(default_factory=dict)

    def __post_init__(self):
        """Initialize structured data format."""
        # Default run_info
        self.run_info = {
            "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
            "duration": 0.0,
            "status": self.status.value,
            "hardware": {
                "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
                "pytorch_version": torch.__version__,
            },
        }

        # Default meta_info
        self.meta_info = {
            "model_id": self.model_id,
            "peft_method": self.peft_method,
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
        }

        self.generation_info = {
            "memory": {
                "peak_gpu_memory_mb": 0.0,
                "peak_ram_memory_mb": 0.0,
                "memory_logs": [],  # Detailed memory usage at different stages
            },
            "by_category": {},  # Will hold metrics for each prompt category (e.g., inference_time, overhead_pct)
            "overall": {},  # Overall metrics across all categories
            # training_throughput can be removed or renamed if not relevant, e.g. tokens_per_second
        }

    def update_meta_info(self, param_counts: dict, size_info: dict):
        """Update model metadata information."""
        self.meta_info["parameters"].update(param_counts)
        self.meta_info["model_size"].update(size_info)

    def update_generation_info(self, memory_data: dict = None, performance_metrics: dict = None):
        """Update generation performance information, primarily for memory and high-level performance."""
        if memory_data:
            self.generation_info["memory"].update(memory_data)
        if performance_metrics:  # For things like overall tokens/sec if calculated
            self.generation_info.update(performance_metrics)

    def add_memory_log(self, stage: str, ram_mb: float, gpu_allocated_mb: float, gpu_reserved_mb: float):
        """Add a memory usage log entry to generation_info."""
        self.generation_info["memory"]["memory_logs"].append(
            {
                "stage": stage,
                "ram_mb": ram_mb,
                "gpu_allocated_mb": gpu_allocated_mb,
                "gpu_reserved_mb": gpu_reserved_mb,
            }
        )

    def add_metrics_for_category(self, category: str, metrics: dict):
        """Add metrics for a specific prompt category under generation_info."""
        self.generation_info["by_category"][category] = metrics

    def update_run_info(self, duration: float, status: BenchmarkStatus, error: Optional[str] = None):
        """Update run information."""
        self.run_info["duration"] = duration
        self.run_info["status"] = status.value
        if error:
            self.run_info["error"] = error

    def compute_overall_metrics(self):
        """Compute overall metrics across all categories within generation_info."""
        if not self.generation_info["by_category"]:
            return

        all_metrics = self.generation_info["by_category"].values()
        metric_keys = set().union(*(d.keys() for d in all_metrics))

        for key in metric_keys:
            values = [d.get(key) for d in all_metrics if key in d]
            values = [v for v in values if v is not None]

            if values:
                if all(isinstance(v, (int, float)) for v in values):
                    self.generation_info["overall"][key] = sum(values) / len(values)
                else:
                    self.generation_info["overall"][key] = values[0]

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
            # Determine the appropriate path based on status
            if self.status == BenchmarkStatus.SUCCESS:
                # If successful, use the main result path
                base_path = RESULT_PATH
            elif self.status == BenchmarkStatus.FAILED:
                # If failed, use the cancelled path
                base_path = RESULT_PATH_CANCELLED
            elif self.status == BenchmarkStatus.CANCELLED:
                # If explicitly cancelled, use the cancelled path
                base_path = RESULT_PATH_CANCELLED
            else:
                # For running or other statuses, use temporary path
                base_path = RESULT_PATH_TEMP

            # Create the filename
            filename = f"{self.experiment_name}_{self.experiment_id}.json"
            path = os.path.join(base_path, filename)

        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save the result
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        return path


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking PEFT methods."""

    # Model configuration
    model_id: str
    peft_method: str  # Using str instead of Literal to be more flexible

    # Benchmark settings
    seed: int
    num_inference_runs: int
    max_new_tokens: int
    train_batch_size: int
    train_steps: int

    # Data settings
    num_prompt_samples: int
    reserve_output_tokens: int = 50

    # Optional settings with defaults
    dtype: str = "float16"
    use_4bit: bool = False
    use_8bit: bool = False

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

        if self.train_batch_size <= 0:
            raise ValueError(f"Invalid train_batch_size: {self.train_batch_size}")

        if self.train_steps <= 0:
            raise ValueError(f"Invalid train_steps: {self.train_steps}")

    @classmethod
    def from_dict(cls, config_dict: dict) -> "BenchmarkConfig":
        """Create config from dictionary."""
        # Extract basic configuration fields
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


def generate_experiment_id() -> str:
    """Generate a unique experiment ID."""
    return datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")


def validate_experiment_path(path: str) -> tuple[str, "BenchmarkConfig"]:
    """Validate experiment path, load and merge configs, and return them."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Experiment path not found: {path}")

    experiment_name = os.path.basename(path)

    # Define paths for default and experiment-specific config files
    default_config_path = os.path.join(os.path.dirname(__file__), FILE_NAME_DEFAULT_CONFIG)
    experiment_benchmark_params_path = os.path.join(path, FILE_NAME_BENCHMARK_PARAMS)

    # 1. Load default config - this is now mandatory
    if not os.path.exists(default_config_path):
        raise FileNotFoundError(f"Default configuration file not found: {default_config_path}. This is required.")
    benchmark_config = BenchmarkConfig.from_json(default_config_path)
    print(f"Loaded default configuration from {default_config_path}")

    # 2. Load experiment-specific benchmark_params.json if it exists
    if os.path.exists(experiment_benchmark_params_path):
        with open(experiment_benchmark_params_path) as f:
            experiment_specific_params = json.load(f)

        # 3. Merge experiment-specific params into the default config
        benchmark_config.merge_from_dict(experiment_specific_params)
        print(f"Loaded and merged experiment-specific parameters from {experiment_benchmark_params_path}")
    else:
        print(f"No {FILE_NAME_BENCHMARK_PARAMS} found in {path}. Using only default configuration.")

    return experiment_name, benchmark_config


def get_memory_usage() -> tuple[float, float, float]:
    """Get current memory usage (RAM and GPU)."""
    # Get RAM usage
    process = psutil.Process(os.getpid())
    ram_usage_bytes = process.memory_info().rss
    ram_usage_mb = ram_usage_bytes / (1024 * 1024)

    # Get GPU usage if available
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated()
        gpu_reserved = torch.cuda.memory_reserved()
        gpu_allocated_mb = gpu_allocated / (1024 * 1024)
        gpu_reserved_mb = gpu_reserved / (1024 * 1024)
    else:
        gpu_allocated_mb = 0.0
        gpu_reserved_mb = 0.0

    return ram_usage_mb, gpu_allocated_mb, gpu_reserved_mb


def init_cuda() -> tuple[float, float]:
    """Initialize CUDA and return initial memory usage."""
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.cuda.empty_cache()
        _, gpu_allocated, gpu_reserved = get_memory_usage()
        return gpu_allocated, gpu_reserved
    return 0.0, 0.0


def get_model_size_mb(model: torch.nn.Module, dtype_bytes: int = 4) -> float:
    """Calculate model size in MB."""
    return sum(p.numel() * dtype_bytes for p in model.parameters()) / (1024 * 1024)


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
        # Optionally print other sections if needed for failed runs, or just return
        print_fn("=" * 50)
        return

    print_fn("\nModel Information:")
    print_fn(f"  Base Model: {benchmark_result.meta_info.get('model_id', 'N/A')}")
    print_fn(f"  PEFT Method: {benchmark_result.meta_info.get('peft_method', 'N/A')}")

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
    print_fn(f"  Peak GPU Memory: {memory_data.get('peak_gpu_memory_mb', 0):.2f} MB")
    print_fn(f"  Peak RAM Memory: {memory_data.get('peak_ram_memory_mb', 0):.2f} MB")

    print_fn("\nDetailed Metrics (from generation_info.by_category):")
    if benchmark_result.generation_info.get("by_category"):
        for category, cat_metrics in benchmark_result.generation_info["by_category"].items():
            print_fn(f"  Category: {category}")
            print_fn(f"    Inference Time: {cat_metrics.get('inference_time', 0):.4f} seconds")
            print_fn(f"    Base Inference Time: {cat_metrics.get('base_inference_time', 0):.4f} seconds")
            print_fn(f"    Inference Overhead: {cat_metrics.get('inference_overhead_pct', 0):.2f}%")
            print_fn(f"    Time Per Token: {cat_metrics.get('time_per_token', 0):.6f} seconds/token")
            print_fn(f"    Generated Tokens: {cat_metrics.get('generated_tokens', 0):.1f}")
    else:
        print_fn("  No per-category metrics available.")

    benchmark_result.compute_overall_metrics()

    print_fn("\nOverall Metrics (from generation_info.overall):")
    if benchmark_result.generation_info.get("overall"):
        for metric_name, value in benchmark_result.generation_info["overall"].items():
            if isinstance(value, float):
                print_fn(f"  {metric_name.replace('_', ' ').title()}: {value:.4f}")
            else:
                print_fn(f"  {metric_name.replace('_', ' ').title()}: {value}")
    else:
        print_fn("  No overall metrics computed.")

    print_fn("\nSaved results to:", benchmark_result.save())
    print_fn("=" * 50)
