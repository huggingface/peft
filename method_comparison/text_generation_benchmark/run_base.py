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

import argparse
import json
import os
import sys
import time

import torch
from data import prepare_benchmark_prompts
from run import measure_inference_time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from utils import (
    BenchmarkConfig,
    get_memory_usage,
    init_accelerator,
)


def run_base_model_benchmark(benchmark_config: BenchmarkConfig, print_fn=print) -> dict:
    """Run benchmark for base model only and return results."""

    print_fn(f"Running base model benchmark for: {benchmark_config.model_id}")

    print_fn("Initializing accelerator...")
    init_accelerator()

    set_seed(benchmark_config.seed)

    print_fn(f"Loading base model: {benchmark_config.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(benchmark_config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "device_map": "auto" if (torch.cuda.is_available() or torch.xpu.is_available()) else None,
    }

    if benchmark_config.dtype == "float32":
        model_kwargs["torch_dtype"] = torch.float32
    elif benchmark_config.dtype == "float16":
        model_kwargs["torch_dtype"] = torch.float16
    elif benchmark_config.dtype == "bfloat16":
        model_kwargs["torch_dtype"] = torch.bfloat16

    if benchmark_config.use_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True
        )
    elif benchmark_config.use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=model_kwargs.get("torch_dtype", torch.float16),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(benchmark_config.model_id, **model_kwargs)

    ram, accelerator_allocated, accelerator_reserved = get_memory_usage()
    print_fn(f"Memory after model load - RAM: {ram:.2f}MB, {model.device.type.upper()}: {accelerator_allocated:.2f}MB")

    print_fn("Preparing benchmark prompts...")
    prompts = prepare_benchmark_prompts(
        config=benchmark_config.to_dict(),
        tokenizer=tokenizer,
        max_input_length=None,
        seed=benchmark_config.seed,
    )

    # Measure base model inference for each prompt category
    print_fn("Measuring base model inference times...")
    base_inference_results = measure_inference_time(
        model,
        tokenizer,
        prompts,
        max_new_tokens=benchmark_config.max_new_tokens,
        num_runs=benchmark_config.num_inference_runs,
        print_fn=print_fn,
        category_generation_params=benchmark_config.category_generation_params,
    )

    result = {
        "model_id": benchmark_config.model_id,
        "benchmark_config": benchmark_config.to_dict(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inference_results": base_inference_results,
        "memory_info": {
            "ram_mb": ram,
            "accelerator_allocated_mb": accelerator_allocated,
            "accelerator_reserved_mb": accelerator_reserved,
        },
    }

    return result


def save_base_results(result: dict, model_id: str) -> str:
    """Save base model results with a filename based on model and config."""
    base_results_dir = os.path.join(os.path.dirname(__file__), "base_results")
    os.makedirs(base_results_dir, exist_ok=True)

    model_name = model_id.replace("/", "_").replace("-", "_")
    filename = f"base_{model_name}.json"
    filepath = os.path.join(base_results_dir, filename)

    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)

    return filepath


def main():
    """Main entry point for the base model benchmark runner."""
    parser = argparse.ArgumentParser(description="Run base model benchmarks")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--force", "-f", action="store_true", help="Force re-run even if results exist")
    args = parser.parse_args()

    print_fn = print if args.verbose else lambda *args, **kwargs: None

    default_config_path = os.path.join(os.path.dirname(__file__), "default_benchmark_params.json")
    benchmark_config = BenchmarkConfig.from_json(default_config_path)

    model_name = benchmark_config.model_id.replace("/", "_").replace("-", "_")
    base_results_dir = os.path.join(os.path.dirname(__file__), "base_results")
    filename = f"base_{model_name}.json"
    filepath = os.path.join(base_results_dir, filename)

    if os.path.exists(filepath) and not args.force:
        print(f"Base results already exist at: {filepath}")
        print("Use --force to re-run the benchmark")
        return 0

    print_fn(f"Running base model benchmark for: {benchmark_config.model_id}")

    result = run_base_model_benchmark(benchmark_config, print_fn=print_fn)

    saved_path = save_base_results(result, benchmark_config.model_id)
    device_type = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
    print(f"Base model results saved to: {saved_path}")

    print("\nBase Model Benchmark Summary:")
    print(f"Model: {result['model_id']}")
    print(
        f"Memory Usage - RAM: {result['memory_info']['ram_mb']:.2f}MB, {device_type.upper()}: {result['memory_info']['accelerator_allocated_mb']:.2f}MB"
    )

    print("\nInference Times by Category:")
    for category, time_val in result["inference_results"]["inference_times"].items():
        time_per_token = result["inference_results"]["time_per_token"][category]
        tokens = result["inference_results"]["generated_tokens"][category]
        print(f"  {category}: {time_val:.4f}s ({time_per_token:.6f}s/token, {tokens:.1f} tokens)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
