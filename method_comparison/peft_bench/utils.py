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
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import psutil
import torch
from peft import PeftConfig


# Constants
FILE_NAME_BENCHMARK_PARAMS = "benchmark_params.json"
FILE_NAME_DEFAULT_CONFIG = "default_config.json"

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
    run_info: dict = field(
        default_factory=dict
    )  # Basic run information (timestamp, duration, hardware)
    train_info: dict = field(default_factory=dict)  # Training metrics and performance
    meta_info: dict = field(default_factory=dict)  # Model metadata and configuration
    metrics: dict = field(default_factory=dict)  # Detailed metrics by prompt category

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

        # Default train_info
        self.train_info = {
            "training_throughput": 0.0,  # tokens/second
            "memory": {
                "peak_gpu_memory_mb": 0.0,
                "peak_ram_memory_mb": 0.0,
                "memory_logs": [],
            },
            "inference": {
                "times": {},
                "overhead": {},
            },
        }

        # Default metrics structure
        self.metrics = {
            "by_category": {},  # Will hold metrics for each prompt category
            "overall": {},  # Overall metrics across all categories
        }

    def update_meta_info(self, param_counts: dict, size_info: dict):
        """Update model metadata information."""
        self.meta_info["parameters"].update(param_counts)
        self.meta_info["model_size"].update(size_info)

    def update_train_info(
        self, memory_data: dict, inference_metrics: dict, throughput: float = 0.0
    ):
        """Update training performance information."""
        self.train_info["training_throughput"] = throughput
        self.train_info["memory"].update(memory_data)
        self.train_info["inference"].update(inference_metrics)

    def add_memory_log(
        self, stage: str, ram_mb: float, gpu_allocated_mb: float, gpu_reserved_mb: float
    ):
        """Add a memory usage log entry."""
        self.train_info["memory"]["memory_logs"].append(
            {
                "stage": stage,
                "ram_mb": ram_mb,
                "gpu_allocated_mb": gpu_allocated_mb,
                "gpu_reserved_mb": gpu_reserved_mb,
            }
        )

    def add_metrics_for_category(self, category: str, metrics: dict):
        """Add metrics for a specific prompt category."""
        self.metrics["by_category"][category] = metrics

    def update_run_info(self, duration: float, status: BenchmarkStatus):
        """Update run information."""
        self.run_info["duration"] = duration
        self.run_info["status"] = status.value

    def compute_overall_metrics(self):
        """Compute overall metrics across all categories."""
        if not self.metrics["by_category"]:
            return

        # Average metrics across all categories
        all_metrics = self.metrics["by_category"].values()
        metric_keys = set().union(*(d.keys() for d in all_metrics))

        for key in metric_keys:
            values = [d.get(key) for d in all_metrics if key in d]
            values = [v for v in values if v is not None]

            if values:
                # For numeric values, compute average
                if all(isinstance(v, (int, float)) for v in values):
                    self.metrics["overall"][key] = sum(values) / len(values)
                # For other types, just keep the first value
                else:
                    self.metrics["overall"][key] = values[0]

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        # Compute overall metrics before converting to dict
        self.compute_overall_metrics()

        # Create the structured output
        return {
            "run_info": self.run_info,
            "train_info": self.train_info,
            "meta_info": self.meta_info,
            "metrics": self.metrics,
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
    prompt_categories: list[str] = field(default_factory=list)
    use_4bit: bool = False
    use_8bit: bool = False

    # PEFT specific configurations - these are dynamically set based on the method
    peft_config_variants: list[dict] = field(default_factory=list)

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

        # Set default prompt categories if not provided
        if not self.prompt_categories:
            self.prompt_categories = ["short", "medium", "long"]

    @classmethod
    def from_dict(cls, config_dict: dict) -> "BenchmarkConfig":
        """Create config from dictionary."""
        # Extract basic configuration fields
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}

        # Extract PEFT variants if specified
        peft_config_variants = config_dict.get("peft_config_variants", [])

        # If no variants but old peft_params exists, create a variant from it
        if not peft_config_variants and "peft_params" in config_dict:
            peft_config_variants = [config_dict["peft_params"]]

        # Also check for any other parameters that might be PEFT specific
        additional_params = {
            k: v
            for k, v in config_dict.items()
            if k not in valid_keys and k != "peft_config_variants" and k != "peft_params"
        }

        # If we have additional params but no variants, create a variant
        if additional_params and not peft_config_variants:
            peft_config_variants = [additional_params]

        filtered_dict["peft_config_variants"] = peft_config_variants

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

    def get_variant_configs(self) -> list["BenchmarkConfig"]:
        """Generate separate configs for each PEFT variant."""
        if not self.peft_config_variants:
            # If no variants defined, return just this config
            return [self]

        variant_configs = []
        for idx, variant in enumerate(self.peft_config_variants):
            # Create a copy of this config
            config_dict = self.to_dict()

            # Remove the variants list
            config_dict.pop("peft_config_variants")

            # Add the specific variant params
            config_dict.update(variant)

            # Add a variant identifier
            if "variant_name" not in variant:
                config_dict["variant_name"] = f"variant_{idx+1}"

            # Create new config
            variant_configs.append(BenchmarkConfig.from_dict(config_dict))

        return variant_configs


def generate_experiment_id() -> str:
    """Generate a unique experiment ID."""
    return datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")


def validate_experiment_path(path: str) -> tuple[str, "BenchmarkConfig", Any]:
    """Validate experiment path and return configs."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Experiment path not found: {path}")

    # Get experiment name from directory
    experiment_name = os.path.basename(path)

    # Check for benchmark params file
    benchmark_params_path = os.path.join(path, FILE_NAME_BENCHMARK_PARAMS)
    default_config_path = os.path.join(os.path.dirname(__file__), FILE_NAME_DEFAULT_CONFIG)
    
    # Use benchmark_params.json if exists, otherwise use default config
    if os.path.exists(benchmark_params_path):
        benchmark_config = BenchmarkConfig.from_json(benchmark_params_path)
    elif os.path.exists(default_config_path):
        print(f"No benchmark_params.json found in {path}, using default configuration")
        benchmark_config = BenchmarkConfig.from_json(default_config_path)
    else:
        raise FileNotFoundError(f"Neither benchmark_params.json nor default_config.json found")

    # Try to load PEFT config
    try:
        peft_config = PeftConfig.from_pretrained(path)
    except Exception as e:
        raise ValueError(f"Failed to load PEFT config: {e}") from e

    return experiment_name, benchmark_config, peft_config


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


def get_trainable_parameters(model: torch.nn.Module) -> int:
    """Get number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_results(
    experiment_name: str,
    benchmark_result: BenchmarkResult,
    print_fn: Callable = print,
) -> None:
    """Log benchmark results to console."""
    # Format results for console output
    print_fn("\n" + "=" * 50)
    print_fn(f"Benchmark Results: {experiment_name}")
    print_fn("=" * 50)

    print_fn(f"Status: {benchmark_result.status.value}")
    print_fn(f"Duration: {benchmark_result.run_info.get('duration', 0):.2f} seconds")

    if benchmark_result.status != BenchmarkStatus.SUCCESS:
        print_fn(f"Error: {benchmark_result.metrics.get('error', 'Unknown error')}")
        return

    print_fn("\nModel Information:")
    print_fn(f"  Base Model: {benchmark_result.model_id}")
    print_fn(f"  PEFT Method: {benchmark_result.peft_method}")

    print_fn("\nParameter Counts:")
    params = benchmark_result.meta_info.get("parameters", {})
    print_fn(f"  Base Parameters: {params.get('base_params', 0):,}")
    print_fn(f"  Trainable Parameters: {params.get('trainable_params', 0):,}")
    print_fn(f"  Parameter Ratio: {params.get('param_ratio', 0):.5%}")

    print_fn("\nModel Size:")
    size_info = benchmark_result.meta_info.get("model_size", {})
    print_fn(f"  Base Model: {size_info.get('base_model_size_mb', 0):.2f} MB")
    print_fn(f"  Adapter: {size_info.get('adapter_size_mb', 0):.2f} MB")

    print_fn("\nMemory Usage:")
    memory_data = benchmark_result.train_info.get("memory", {})
    print_fn(f"  Peak GPU Memory: {memory_data.get('peak_gpu_memory_mb', 0):.2f} MB")
    print_fn(f"  Peak RAM Memory: {memory_data.get('peak_ram_memory_mb', 0):.2f} MB")

    print_fn("\nInference Times:")
    inference_data = benchmark_result.train_info.get("inference", {})
    for category, time_value in inference_data.get("times", {}).items():
        print_fn(f"  {category}: {time_value:.4f} seconds")

    print_fn("\nInference Overhead:")
    for category, overhead in inference_data.get("overhead", {}).items():
        print_fn(f"  {category}: {overhead:.2f}%")

    print_fn("\nSaved results to:", benchmark_result.save())
    print_fn("=" * 50)
