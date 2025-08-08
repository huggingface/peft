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
Main entry point to run the experiments. Contains general setup and the proper inference code.
"""

import argparse
import gc
import json
import os
import sys
import time
from typing import Optional

import bitsandbytes
import torch
import transformers
from data import prepare_benchmark_prompts
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from utils import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkStatus,
    get_memory_usage,
    init_accelerator,
    log_results,
    validate_experiment_path,
)

import peft
from peft import PeftConfig, get_peft_model


def load_base_results(model_id: str) -> Optional[dict]:
    """Load base model results if they exist."""
    base_results_dir = os.path.join(os.path.dirname(__file__), "base_results")
    model_name = model_id.replace("/", "_").replace("-", "_")
    filename = f"base_{model_name}.json"
    filepath = os.path.join(base_results_dir, filename)

    if os.path.exists(filepath):
        with open(filepath) as f:
            return json.load(f)
    return None


def measure_inference_time(model, tokenizer, prompts, max_new_tokens, num_runs, print_fn, category_generation_params):
    """Measure inference time for each prompt category."""
    inference_times = {}
    time_per_token = {}
    generated_tokens = {}
    individual_samples = {}

    for category, category_prompts in prompts.items():
        print_fn(f"\nMeasuring inference time for {category} prompts...")
        category_times = []
        category_tokens = []
        category_time_per_token = []
        category_samples = []

        for prompt in category_prompts:
            prompt_times = []
            prompt_tokens = []
            prompt_time_per_token = []

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            cat_max_new_tokens = category_generation_params.get(category, {}).get("max_new_tokens", max_new_tokens)

            for _ in range(num_runs):
                start_time = time.perf_counter()
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=cat_max_new_tokens,
                    min_new_tokens=cat_max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
                end_time = time.perf_counter()

                # Calculate metrics
                inference_time = end_time - start_time
                num_tokens = len(outputs[0]) - len(inputs["input_ids"][0])
                time_per_token_val = inference_time / num_tokens if num_tokens > 0 else 0

                prompt_times.append(inference_time)
                prompt_tokens.append(num_tokens)
                prompt_time_per_token.append(time_per_token_val)

            # Calculate averages for this prompt
            avg_time = sum(prompt_times) / len(prompt_times)
            avg_tokens = sum(prompt_tokens) / len(prompt_tokens)
            avg_time_per_token = sum(prompt_time_per_token) / len(prompt_time_per_token)

            sample_result = {
                "inference_time": avg_time,
                "generated_tokens": avg_tokens,
                "time_per_token": avg_time_per_token,
                "individual_runs": [
                    {"inference_time": t, "generated_tokens": tok, "time_per_token": tpt}
                    for t, tok, tpt in zip(prompt_times, prompt_tokens, prompt_time_per_token)
                ],
            }
            category_samples.append(sample_result)

            category_times.append(avg_time)
            category_tokens.append(avg_tokens)
            category_time_per_token.append(avg_time_per_token)

        if category_times:
            avg_category_time = sum(category_times) / len(category_times)
            avg_category_tokens = sum(category_tokens) / len(category_tokens)
            avg_category_time_per_token = sum(category_time_per_token) / len(category_time_per_token)

            inference_times[category] = avg_category_time
            generated_tokens[category] = avg_category_tokens
            time_per_token[category] = avg_category_time_per_token
            individual_samples[category] = category_samples

    return {
        "inference_times": inference_times,
        "time_per_token": time_per_token,
        "generated_tokens": generated_tokens,
        "individual_samples": individual_samples,
    }


def run_benchmark(
    benchmark_config: BenchmarkConfig, experiment_name: str, experiment_path: str, print_fn=print
) -> BenchmarkResult:
    """Run benchmarks for the specified PEFT method configuration."""
    result = BenchmarkResult(
        experiment_name=experiment_name,
        status=BenchmarkStatus.RUNNING,
        model_id=benchmark_config.model_id,
    )

    result.save()

    start_time = time.perf_counter()
    e_main_benchmark: Optional[Exception] = None

    try:
        print_fn("Initializing accelerator...")
        accelerator_allocated_init, accelerator_reserved_init = init_accelerator()
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
        else:
            raise ValueError(f"Unsupported dtype: {benchmark_config.dtype}")

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

        base_model = AutoModelForCausalLM.from_pretrained(benchmark_config.model_id, **model_kwargs)

        base_results = load_base_results(benchmark_config.model_id)

        print_fn("Preparing benchmark prompts...")
        prompts = prepare_benchmark_prompts(
            config=benchmark_config,
            tokenizer=tokenizer,
            max_input_length=None,
            seed=benchmark_config.seed,
        )

        if base_results:
            print_fn("Using cached base model results...")
            base_inference_times = base_results["inference_results"]
        else:
            raise FileNotFoundError(
                "No cached base results found. Please run `python run_base.py` first to generate base model results."
            )

        try:
            print_fn(f"Loading PEFT config from {experiment_path}")
            peft_config = PeftConfig.from_pretrained(experiment_path)
            print_fn(f"Loaded PEFT config: {peft_config.peft_type}, with parameters: {vars(peft_config)}")
            model = get_peft_model(base_model, peft_config)
        except Exception as exc:
            error_msg = f"Error loading PEFT config: {str(exc)}"
            print_fn(error_msg)

        del base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.xpu.is_available():
            torch.xpu.empty_cache()

        ram, accelerator_allocated, accelerator_reserved = get_memory_usage()
        result.add_memory_log("peft_model_loaded", ram, accelerator_allocated, accelerator_reserved)

        # Calculate PEFT model metrics
        trainable_params = model.get_nb_trainable_parameters()[0]
        total_params = sum(p.numel() for p in model.parameters())
        base_params = sum(p.numel() for p in model.base_model.parameters())
        dtype_bytes = 2 if benchmark_config.dtype in ["float16", "bfloat16"] else 4
        adapter_size_mb = trainable_params * dtype_bytes / (1024 * 1024)
        base_model_size_mb = base_params * dtype_bytes / (1024 * 1024)
        param_ratio = trainable_params / total_params if total_params > 0 else 0

        result.update_meta_info(
            param_counts={
                "base_params": base_params,
                "trainable_params": trainable_params,
                "total_params": total_params,
                "param_ratio": param_ratio,
            },
            size_info={"base_model_size_mb": base_model_size_mb, "adapter_size_mb": adapter_size_mb},
            package_info={
                "transformers-version": transformers.__version__,
                "peft-version": peft.__version__,
                "bitsandbytes-version": bitsandbytes.__version__ if hasattr(bitsandbytes, "__version__") else None,
            },
        )

        print_fn("Measuring PEFT model inference times...")
        peft_inference_times = measure_inference_time(
            model,
            tokenizer,
            prompts,
            max_new_tokens=benchmark_config.max_new_tokens,
            num_runs=benchmark_config.num_inference_runs,
            print_fn=print_fn,
            category_generation_params=benchmark_config.category_generation_params,
        )

        # Calculate inference overhead for each category
        inference_overhead = {
            k: (peft_inference_times["inference_times"][k] - base_inference_times["inference_times"][k])
            / base_inference_times["inference_times"][k]
            * 100
            for k in base_inference_times["inference_times"]
        }

        for category in prompts:
            category_metrics = {
                "inference_time": peft_inference_times["inference_times"][category],
                "base_inference_time": base_inference_times["inference_times"][category],
                "inference_overhead_pct": inference_overhead[category],
                "time_per_token": peft_inference_times["time_per_token"][category],
                "generated_tokens": peft_inference_times["generated_tokens"][category],
            }
            result.add_metrics_for_category(
                category, category_metrics, individual_samples=peft_inference_times["individual_samples"][category]
            )

        result.update_generation_info(
            memory_data={
                "peak_accelerator_memory_mb": max(
                    (log["accelerator_allocated_mb"] for log in result.generation_info["memory"]["memory_logs"]), default=0
                ),
                "peak_ram_memory_mb": max(
                    (log["ram_mb"] for log in result.generation_info["memory"]["memory_logs"]), default=0
                ),
            }
        )

        ram, accelerator_allocated, accelerator_reserved = get_memory_usage()
        result.add_memory_log("benchmark_complete", ram, accelerator_allocated, accelerator_reserved)

        result.status = BenchmarkStatus.SUCCESS

    except Exception as exc:
        print_fn(f"Benchmark failed with error: {exc}")
        result.status = BenchmarkStatus.FAILED
        e_main_benchmark = exc
    end_time = time.perf_counter()
    error_message = str(e_main_benchmark) if e_main_benchmark is not None else None

    peft_config_dict = peft_config.to_dict() if "peft_config" in locals() else None
    if peft_config_dict:
        for key, value in peft_config_dict.items():
            if isinstance(value, set):
                peft_config_dict[key] = list(value)

    result.update_run_info(
        duration=end_time - start_time,
        status=result.status,
        error=error_message,
        peft_config=peft_config_dict,
        benchmark_config=benchmark_config.to_dict(),
    )

    return result


def main() -> None:
    """Main entry point for the benchmark runner."""
    parser = argparse.ArgumentParser(description="Run PEFT method benchmarks")
    parser.add_argument("experiment_path", help="Path to experiment directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    print_fn = print if args.verbose else lambda *args, **kwargs: None

    experiment_path = args.experiment_path
    allowed_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    abs_experiment_path = os.path.abspath(experiment_path)
    if not abs_experiment_path.startswith(allowed_root):
        print(f"Experiment path must be inside {allowed_root}, got: {abs_experiment_path}. Skipping execution.")
        return 0
    if not os.path.exists(abs_experiment_path):
        print(f"Experiment path not found: {abs_experiment_path}. Skipping execution.")
        return 0
    experiment_path = abs_experiment_path

    experiment_name, benchmark_config = validate_experiment_path(experiment_path)

    print_fn(f"Running benchmark for experiment: {experiment_name}")

    result = run_benchmark(
        benchmark_config=benchmark_config,
        experiment_name=experiment_name,
        experiment_path=experiment_path,
        print_fn=print_fn,
    )

    log_results(experiment_name, result, print_fn=print)


if __name__ == "__main__":
    sys.exit(main())
