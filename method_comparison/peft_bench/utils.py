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
import enum
import json
import os
import platform
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import psutil  # You might need to install this: pip install psutil

# Constants
FILE_NAME_BENCHMARK_PARAMS = "benchmark_params.json"

# Main paths for storing results
RESULT_PATH = os.path.join(os.path.dirname(__file__), "results")
RESULT_PATH_TEMP = os.path.join(os.path.dirname(__file__), "temporary_results")
RESULT_PATH_CANCELLED = os.path.join(os.path.dirname(__file__), "cancelled_results")

# Make sure directories exist
os.makedirs(RESULT_PATH, exist_ok=True)
os.makedirs(RESULT_PATH_TEMP, exist_ok=True)
os.makedirs(RESULT_PATH_CANCELLED, exist_ok=True)


class BenchmarkStatus(Enum):
    """Status of a benchmark run."""
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RUNNING = "running"


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    # Metadata
    experiment_id: str
    experiment_name: str
    status: BenchmarkStatus
    
    # Model info
    model_id: str
    peft_method: str
    
    # Time tracking
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    duration: float = 0.0
    
    # Parameter counts
    base_params: int = 0
    trainable_params: int = 0
    total_params: int = 0
    param_ratio: float = 0.0
    
    # Size metrics
    base_model_size_mb: float = 0.0
    adapter_size_mb: float = 0.0
    
    # Memory metrics
    peak_gpu_memory_mb: float = 0.0
    peak_ram_memory_mb: float = 0.0
    memory_allocated_log: List[float] = field(default_factory=list)
    memory_reserved_log: List[float] = field(default_factory=list)
    
    # Performance metrics
    inference_times: Dict[str, float] = field(default_factory=dict)
    inference_overhead: Dict[str, float] = field(default_factory=dict)
    training_throughput: float = 0.0  # tokens/second
    
    # Additional metrics
    metrics: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        result_dict = asdict(self)
        # Convert enum to string
        result_dict["status"] = self.status.value
        return result_dict
    
    def save(self, path: Optional[str] = None) -> str:
        """Save benchmark result to JSON file."""
        if path is None:
            # Use default path based on status
            if self.status == BenchmarkStatus.SUCCESS:
                base_path = RESULT_PATH
            elif self.status == BenchmarkStatus.RUNNING:
                base_path = RESULT_PATH_TEMP
            else:
                base_path = RESULT_PATH_CANCELLED
                
            path = os.path.join(base_path, f"{self.experiment_name}", "benchmark_result.json")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save to file
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return path


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking PEFT methods."""
    # Model configuration
    model_id: str
    peft_method: Literal["lora", "adalora", "bone", "ia3", "prompt_tuning", "prefix_tuning", "none"]
    dtype: Literal["float32", "float16", "bfloat16", "int8", "int4"] = "float16"
    
    # Benchmark settings
    seed: int = 42
    num_inference_runs: int = 5
    max_new_tokens: int = 20
    train_batch_size: int = 4
    train_steps: int = 10
    
    # Data settings
    prompt_categories: List[str] = field(default_factory=lambda: ["short", "medium"])
    num_prompt_samples: int = 2
    reserve_output_tokens: int = 50
    
    # Optional settings
    use_4bit: bool = False
    use_8bit: bool = False
    compile_model: bool = False
    merge_adapter: bool = False
    
    # Method-specific parameters (these would be overridden by the experiment config)
    peft_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if not isinstance(self.model_id, str):
            raise ValueError(f"Invalid model_id: {self.model_id}")
        
        if self.dtype not in ["float32", "float16", "bfloat16", "int8", "int4"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")
        
        if self.peft_method not in ["lora", "adalora", "bone", "ia3", "prompt_tuning", "prefix_tuning", "none"]:
            raise ValueError(f"Invalid peft_method: {self.peft_method}")
        
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
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BenchmarkConfig":
        """Create config from dictionary."""
        # Filter out keys that are not in BenchmarkConfig
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        # Handle peft_params separately
        peft_params = {k: v for k, v in config_dict.items() if k not in valid_keys}
        filtered_dict["peft_params"] = peft_params
        
        return cls(**filtered_dict)
    
    @classmethod
    def from_json(cls, json_path: str) -> "BenchmarkConfig":
        """Load config from JSON file."""
        with open(json_path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = asdict(self)
        # Merge peft_params into the main dict for saving
        peft_params = result.pop("peft_params", {})
        result.update(peft_params)
        return result
    
    def save(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def generate_experiment_id() -> str:
    """Generate a unique experiment ID."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]


def validate_experiment_path(path: str) -> Tuple[str, "BenchmarkConfig", Any]:
    """Validate experiment path and return configs."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Experiment path not found: {path}")
    
    # Get experiment name from directory
    experiment_name = os.path.basename(path)
    
    # Check for benchmark params file
    benchmark_params_path = os.path.join(path, FILE_NAME_BENCHMARK_PARAMS)
    if not os.path.exists(benchmark_params_path):
        raise FileNotFoundError(f"Benchmark params not found: {benchmark_params_path}")
    
    # Load benchmark config
    benchmark_config = BenchmarkConfig.from_json(benchmark_params_path)
    
    # Try to load PEFT config
    try:
        from peft import PeftConfig
        peft_config = PeftConfig.from_pretrained(path)
    except Exception as e:
        raise ValueError(f"Failed to load PEFT config: {e}")
    
    return experiment_name, benchmark_config, peft_config


def get_memory_usage() -> Tuple[float, float, float]:
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


def init_cuda() -> Tuple[float, float]:
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


def time_function(fn: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """Time a function call and return result and elapsed time."""
    start_time = time.perf_counter()
    result = fn(*args, **kwargs)
    end_time = time.perf_counter()
    return result, end_time - start_time


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
    print_fn(f"Duration: {benchmark_result.duration:.2f} seconds")
    
    print_fn("\nModel Information:")
    print_fn(f"  Base Model: {benchmark_result.model_id}")
    print_fn(f"  PEFT Method: {benchmark_result.peft_method}")
    
    print_fn("\nParameter Counts:")
    print_fn(f"  Base Parameters: {benchmark_result.base_params:,}")
    print_fn(f"  Trainable Parameters: {benchmark_result.trainable_params:,}")
    print_fn(f"  Parameter Ratio: {benchmark_result.param_ratio:.5%}")
    
    print_fn("\nModel Size:")
    print_fn(f"  Base Model: {benchmark_result.base_model_size_mb:.2f} MB")
    print_fn(f"  Adapter: {benchmark_result.adapter_size_mb:.2f} MB")
    
    print_fn("\nMemory Usage:")
    print_fn(f"  Peak GPU Memory: {benchmark_result.peak_gpu_memory_mb:.2f} MB")
    print_fn(f"  Peak RAM Memory: {benchmark_result.peak_ram_memory_mb:.2f} MB")
    
    print_fn("\nPerformance Metrics:")
    print_fn(f"  Training Throughput: {benchmark_result.training_throughput:.2f} tokens/sec")
    
    print_fn("\nInference Times:")
    for category, time_value in benchmark_result.inference_times.items():
        print_fn(f"  {category}: {time_value:.4f} seconds")
    
    print_fn("\nInference Overhead:")
    for category, overhead in benchmark_result.inference_overhead.items():
        print_fn(f"  {category}: {overhead:.2f}%")
    
    print_fn("\nSaved results to:", benchmark_result.save())
    print_fn("=" * 50)



