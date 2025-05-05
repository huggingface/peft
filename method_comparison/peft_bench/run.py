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
Main entry point to run the experiments. Contains general setup and the proper training code.
"""
import argparse
import gc
import sys
import time

import torch
from method_comparison.peft_bench.data import prepare_benchmark_prompts
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from method_comparison.peft_bench.utils import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkStatus,
    generate_experiment_id,
    get_memory_usage,
    get_model_size_mb,
    init_cuda,
    log_results,
    validate_experiment_path,
)

from peft import PeftConfig, get_peft_model


def measure_inference_time(model, tokenizer, prompts, max_new_tokens, num_runs, print_fn):
    """Measure inference time across different prompt categories."""
    inference_times = {}
    time_per_token = {}
    generated_tokens = {}

    for category, prompt_list in prompts.items():
        if not prompt_list:
            continue

        category_times = []
        category_tokens = []

        # Measure each prompt in the category
        for prompt in prompt_list:
            # Prepare input
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # Warmup run
            with torch.inference_mode():
                _ = model.generate(**inputs, max_new_tokens=max_new_tokens)

            # Measurement runs
            prompt_times = []
            prompt_tokens = []
            for i in range(num_runs):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.perf_counter()

                with torch.inference_mode():
                    output = model.generate(**inputs, max_new_tokens=max_new_tokens)

                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.perf_counter()
                
                # Calculate tokens generated (output minus input)
                num_tokens_generated = output.shape[1] - inputs.input_ids.shape[1]
                prompt_tokens.append(num_tokens_generated)
                prompt_times.append(end_time - start_time)

            # Average for this specific prompt
            avg_time = sum(prompt_times) / len(prompt_times)
            avg_tokens = sum(prompt_tokens) / len(prompt_tokens)
            
            category_times.append(avg_time)
            category_tokens.append(avg_tokens)

        # Calculate category averages
        if category_times:
            avg_category_time = sum(category_times) / len(category_times)
            avg_category_tokens = sum(category_tokens) / len(category_tokens)
            
            inference_times[category] = avg_category_time
            generated_tokens[category] = avg_category_tokens
            
            # Calculate time per token
            if avg_category_tokens > 0:
                time_per_token[category] = avg_category_time / avg_category_tokens
            else:
                time_per_token[category] = 0.0

    return {
        "inference_times": inference_times,
        "time_per_token": time_per_token,
        "generated_tokens": generated_tokens
    }


def run_benchmark(
    benchmark_config: BenchmarkConfig, experiment_name: str, experiment_path: str, print_fn=print
) -> BenchmarkResult:
    """Run benchmarks for the specified PEFT method configuration."""
    # Initialize benchmark result
    result = BenchmarkResult(
        experiment_id=generate_experiment_id(),
        experiment_name=experiment_name,
        status=BenchmarkStatus.RUNNING,
        model_id=benchmark_config.model_id,
        peft_method=benchmark_config.peft_method,
    )

    # Save initial result to show running status
    result.save()

    start_time = time.perf_counter()

    try:
        # Initialize CUDA
        print_fn("Initializing CUDA...")
        gpu_allocated_init, gpu_reserved_init = init_cuda()

        # Set random seed
        set_seed(benchmark_config.seed)

        # Load base model and tokenizer
        print_fn(f"Loading base model: {benchmark_config.model_id}")
        tokenizer = AutoTokenizer.from_pretrained(benchmark_config.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Configure model loading parameters
        model_kwargs = {
            "device_map": "auto" if torch.cuda.is_available() else None,
        }

        # Add dtype configuration
        if benchmark_config.dtype == "float32":
            model_kwargs["torch_dtype"] = torch.float32
        elif benchmark_config.dtype == "float16":
            model_kwargs["torch_dtype"] = torch.float16
        elif benchmark_config.dtype == "bfloat16":
            model_kwargs["torch_dtype"] = torch.bfloat16

        # Add quantization if needed
        if benchmark_config.use_8bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
        elif benchmark_config.use_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=model_kwargs.get("torch_dtype", torch.float16),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        # Load the base model
        base_model = AutoModelForCausalLM.from_pretrained(benchmark_config.model_id, **model_kwargs)

        # Track memory after base model load
        ram, gpu_allocated, gpu_reserved = get_memory_usage()
        result.add_memory_log("base_model_loaded", ram, gpu_allocated, gpu_reserved)

        # Calculate base model metrics
        base_params = sum(p.numel() for p in base_model.parameters())
        dtype_bytes = 2 if benchmark_config.dtype in ["float16", "bfloat16"] else 4
        base_model_size_mb = get_model_size_mb(base_model, dtype_bytes=dtype_bytes)

        # Store in result
        result.update_meta_info(
            param_counts={"base_params": base_params}, size_info={"base_model_size_mb": base_model_size_mb}
        )

        # Prepare prompts for benchmarking
        print_fn("Preparing benchmark prompts...")
        prompts = prepare_benchmark_prompts(
            config=benchmark_config.to_dict(), tokenizer=tokenizer, num_samples=benchmark_config.num_prompt_samples
        )

        # Ensure we only use the prompt categories specified in the config
        prompts = {
            category: prompt_list
            for category, prompt_list in prompts.items()
            if category in benchmark_config.prompt_categories
        }

        # Measure base model inference for each prompt category
        print_fn("Measuring base model inference times...")
        base_inference_times = measure_inference_time(
            base_model,
            tokenizer,
            prompts,
            max_new_tokens=benchmark_config.max_new_tokens,
            num_runs=benchmark_config.num_inference_runs,
            print_fn=print_fn,
        )

        # Apply PEFT method
        print_fn(f"Applying PEFT method: {benchmark_config.peft_method}")

        # Load PEFT configuration from path or create dynamically
        try:
            print_fn(f"Loading PEFT config from {experiment_path}")
            peft_config = PeftConfig.from_pretrained(experiment_path)
            print_fn(f"Loaded PEFT config: {peft_config.peft_type}, with parameters: {vars(peft_config)}")
            model = get_peft_model(base_model, peft_config)
        except Exception as e:
            error_msg = f"Error loading PEFT config: {str(e)}"
            print_fn(error_msg)
            import traceback
            print_fn(traceback.format_exc())
            raise ValueError(error_msg) from e

        # Free memory by removing base model reference
        del base_model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Track memory after PEFT application
        ram, gpu_allocated, gpu_reserved = get_memory_usage()
        result.add_memory_log("peft_model_loaded", ram, gpu_allocated, gpu_reserved)

        # Calculate PEFT model metrics
        trainable_params = model.get_nb_trainable_parameters()[0]
        total_params = sum(p.numel() for p in model.parameters())
        adapter_size_mb = trainable_params * dtype_bytes / (1024 * 1024)
        param_ratio = trainable_params / total_params if total_params > 0 else 0

        # Update result with parameter information
        result.update_meta_info(
            param_counts={
                "trainable_params": trainable_params,
                "total_params": total_params,
                "param_ratio": param_ratio,
            },
            size_info={"adapter_size_mb": adapter_size_mb},
        )

        # Measure PEFT model inference
        print_fn("Measuring PEFT model inference times...")
        peft_inference_times = measure_inference_time(
            model,
            tokenizer,
            prompts,
            max_new_tokens=benchmark_config.max_new_tokens,
            num_runs=benchmark_config.num_inference_runs,
            print_fn=print_fn,
        )

        # Calculate inference overhead for each category
        inference_overhead = {
            k: (peft_inference_times["inference_times"][k] - base_inference_times["inference_times"][k]) / base_inference_times["inference_times"][k] * 100
            for k in base_inference_times["inference_times"]
        }

        # Process metrics for each prompt category
        for category in prompts:
            category_metrics = {
                "inference_time": peft_inference_times["inference_times"][category],
                "base_inference_time": base_inference_times["inference_times"][category],
                "inference_overhead_pct": inference_overhead[category],
                "time_per_token": peft_inference_times["time_per_token"][category],
                "generated_tokens": peft_inference_times["generated_tokens"][category],
            }
            result.add_metrics_for_category(category, category_metrics)

        # Update inference metrics in the result
        result.update_train_info(
            memory_data={
                "peak_gpu_memory_mb": max(
                    log["gpu_allocated_mb"] for log in result.train_info["memory"]["memory_logs"]
                ),
                "peak_ram_memory_mb": max(log["ram_mb"] for log in result.train_info["memory"]["memory_logs"]),
            },
            inference_metrics={"times": peft_inference_times["inference_times"], "overhead": inference_overhead},
        )

        # Track final memory usage
        ram, gpu_allocated, gpu_reserved = get_memory_usage()
        result.add_memory_log("benchmark_complete", ram, gpu_allocated, gpu_reserved)

        # Set successful status
        result.status = BenchmarkStatus.SUCCESS

    except Exception as e:
        print_fn(f"Benchmark failed with error: {e}")
        result.status = BenchmarkStatus.FAILED
        result.metrics["error"] = str(e)

    # Record duration and update final status
    end_time = time.perf_counter()
    result.update_run_info(duration=end_time - start_time, status=result.status)

    return result


def main():
    """Main entry point for the benchmark runner."""
    parser = argparse.ArgumentParser(description="Run PEFT method benchmarks")
    parser.add_argument("experiment_path", help="Path to experiment directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    # Configure print function based on verbosity
    print_fn = print if args.verbose else lambda *args, **kwargs: None

    try:
        # Validate experiment path and load configs
        experiment_name, benchmark_config, peft_config = validate_experiment_path(args.experiment_path)

        print_fn(f"Running benchmark for experiment: {experiment_name}")

        # Run the benchmark
        result = run_benchmark(
            benchmark_config=benchmark_config,
            experiment_name=experiment_name,
            experiment_path=args.experiment_path,
            print_fn=print_fn,
        )

        # Log and save results
        log_results(experiment_name, result, print_fn=print)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
