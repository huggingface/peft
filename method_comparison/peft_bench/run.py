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
import datetime
import gc
import json
import os
import sys
import time
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from peft import get_peft_model, PeftModel, LoraConfig, PeftConfig
from utils import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkStatus,
    generate_experiment_id,
    get_memory_usage,
    get_model_size_mb,
    get_trainable_parameters,
    init_cuda,
    log_results,
    time_function,
    validate_experiment_path,
)
from data import prepare_benchmark_prompts, get_prompts_by_length


def measure_inference_time(
    model, 
    tokenizer, 
    prompts, 
    max_new_tokens=20, 
    num_runs=3, 
    print_fn=print
):
    """Measure inference time across different prompt categories."""
    inference_times = {}
    
    for category, prompt_list in prompts.items():
        if not prompt_list:
            continue
            
        # Use first prompt for measurement
        prompt = prompt_list[0]
        print_fn(f"Measuring inference time for {category} prompt")
        
        # Prepare input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Warmup run
        with torch.inference_mode():
            _ = model.generate(**inputs, max_new_tokens=max_new_tokens)
            
        # Measurement runs
        times = []
        for i in range(num_runs):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.perf_counter()
            
            with torch.inference_mode():
                _ = model.generate(**inputs, max_new_tokens=max_new_tokens)
                
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            
        # Record average time
        inference_times[category] = sum(times) / len(times)
        
    return inference_times


def measure_training_throughput(
    model, 
    tokenizer, 
    prompts, 
    batch_size=4, 
    num_steps=10, 
    print_fn=print
):
    """Measure training speed in tokens per second."""
    # Get training data
    all_prompts = get_prompts_by_length(prompts, "all")
    selected_prompts = all_prompts[:batch_size] * (num_steps + 2)  # Ensure enough data
    
    # Tokenize
    inputs = tokenizer(
        selected_prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    ).to(model.device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # Count tokens for throughput calculation
    total_tokens = inputs["attention_mask"].sum().item()
    
    # Warmup step
    model.train()
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Benchmark training steps
    print_fn(f"Measuring training throughput over {num_steps} steps...")
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    
    for step in range(num_steps):
        batch_start = step * batch_size
        batch_end = batch_start + batch_size
        batch = {k: v[batch_start:batch_end] for k, v in inputs.items()}
        
        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.perf_counter()
    
    # Calculate throughput
    elapsed_time = end_time - start_time
    tokens_per_second = total_tokens / elapsed_time
    
    return tokens_per_second


def run_benchmark(
    benchmark_config: BenchmarkConfig,
    experiment_name: str,
    experiment_path: str,
    print_fn=print
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
    memory_logs = []
    
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
            model_kwargs["load_in_8bit"] = True
        elif benchmark_config.use_4bit:
            model_kwargs["load_in_4bit"] = True
            
        # Load the base model
        base_model = AutoModelForCausalLM.from_pretrained(
            benchmark_config.model_id,
            **model_kwargs
        )
        
        # Track memory after base model load
        ram, gpu_allocated, gpu_reserved = get_memory_usage()
        memory_logs.append({
            "stage": "base_model_loaded",
            "ram_mb": ram,
            "gpu_allocated_mb": gpu_allocated,
            "gpu_reserved_mb": gpu_reserved
        })
        
        # Calculate base model metrics
        base_params = sum(p.numel() for p in base_model.parameters())
        dtype_bytes = 2 if benchmark_config.dtype in ["float16", "bfloat16"] else 4
        base_model_size_mb = get_model_size_mb(base_model, dtype_bytes=dtype_bytes)
        
        # Store in result
        result.base_params = base_params
        result.base_model_size_mb = base_model_size_mb
        
        # Prepare prompts for benchmarking
        print_fn("Preparing benchmark prompts...")
        prompts = prepare_benchmark_prompts(
            config=benchmark_config.to_dict(),
            tokenizer=tokenizer,
            num_samples=benchmark_config.num_prompt_samples
        )
        
        # Measure base model inference
        print_fn("Measuring base model inference times...")
        base_inference_times = measure_inference_time(
            base_model,
            tokenizer,
            prompts,
            max_new_tokens=benchmark_config.max_new_tokens,
            num_runs=benchmark_config.num_inference_runs,
            print_fn=print_fn
        )
        
        # Apply PEFT method
        print_fn(f"Applying PEFT method: {benchmark_config.peft_method}")

        # Load PEFT configuration from path - use the PeftConfig we imported at the top
        try:
            peft_config = PeftConfig.from_pretrained(experiment_path)
            model = get_peft_model(base_model, peft_config)
        except Exception as e:
            print_fn(f"Error loading PEFT config: {e}")
            raise
        
        # Free memory by removing base model reference
        del base_model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Track memory after PEFT application
        ram, gpu_allocated, gpu_reserved = get_memory_usage()
        memory_logs.append({
            "stage": "peft_model_loaded",
            "ram_mb": ram,
            "gpu_allocated_mb": gpu_allocated,
            "gpu_reserved_mb": gpu_reserved
        })
        
        # Calculate PEFT model metrics
        trainable_params = get_trainable_parameters(model)
        total_params = sum(p.numel() for p in model.parameters())
        adapter_size_mb = trainable_params * dtype_bytes / (1024 * 1024)
        
        # Update result
        result.trainable_params = trainable_params
        result.total_params = total_params
        result.param_ratio = trainable_params / total_params if total_params > 0 else 0
        result.adapter_size_mb = adapter_size_mb
        
        # Measure PEFT model inference
        print_fn("Measuring PEFT model inference times...")
        peft_inference_times = measure_inference_time(
            model,
            tokenizer,
            prompts,
            max_new_tokens=benchmark_config.max_new_tokens,
            num_runs=benchmark_config.num_inference_runs,
            print_fn=print_fn
        )
        
        # Calculate inference overhead
        inference_overhead = {
            k: (peft_inference_times[k] - base_inference_times[k]) / base_inference_times[k] * 100
            for k in base_inference_times
        }
        
        # Update result
        result.inference_times = peft_inference_times
        result.inference_overhead = inference_overhead
        
        # Measure training throughput
        print_fn("Measuring training throughput...")
        training_throughput = measure_training_throughput(
            model,
            tokenizer,
            prompts,
            batch_size=benchmark_config.train_batch_size,
            num_steps=benchmark_config.train_steps,
            print_fn=print_fn
        )
        
        # Update result
        result.training_throughput = training_throughput
        
        # Track final memory usage
        ram, gpu_allocated, gpu_reserved = get_memory_usage()
        memory_logs.append({
            "stage": "benchmark_complete",
            "ram_mb": ram,
            "gpu_allocated_mb": gpu_allocated,
            "gpu_reserved_mb": gpu_reserved
        })
        
        # Calculate peak memory usage
        peak_ram = max(log["ram_mb"] for log in memory_logs)
        peak_gpu = max(log["gpu_allocated_mb"] for log in memory_logs)
        
        result.peak_ram_memory_mb = peak_ram
        result.peak_gpu_memory_mb = peak_gpu
        result.memory_allocated_log = [log["gpu_allocated_mb"] for log in memory_logs]
        result.memory_reserved_log = [log["gpu_reserved_mb"] for log in memory_logs]
        
        # Set successful status
        result.status = BenchmarkStatus.SUCCESS
        
    except Exception as e:
        print_fn(f"Benchmark failed with error: {e}")
        result.status = BenchmarkStatus.FAILED
        result.metrics.append({"error": str(e)})
    
    # Record duration
    end_time = time.perf_counter()
    result.duration = end_time - start_time
    
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
        experiment_name, benchmark_config, _ = validate_experiment_path(args.experiment_path)
        
        print(f"Running benchmark for experiment: {experiment_name}")
        
        # Run the benchmark
        result = run_benchmark(
            benchmark_config=benchmark_config,
            experiment_name=experiment_name,
            experiment_path=args.experiment_path,
            print_fn=print_fn
        )
        
        # Log and save results
        log_results(experiment_name, result, print_fn=print)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())