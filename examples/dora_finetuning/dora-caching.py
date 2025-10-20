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
Small script to measure DoRA caching efficiency
"""

import argparse
import time
from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM

from peft import LoraConfig, get_peft_model
from peft.helpers import DoraCaching
from peft.utils import infer_device


device = infer_device()
# check for CPU
if device == "cpu":
    raise ValueError("This benchmark requires a hardware accelerator, only found CPU")
torch_accelerator_module = getattr(torch, device, torch.cuda)


@contextmanager
def timeit(logs):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    dur = end - start
    logs["time"].append(dur)


def run_benchmark(model, num_runs):
    logs = {
        "time": [],
    }

    mem_start = torch_accelerator_module.max_memory_reserved()
    for _ in range(num_runs + 1):
        with timeit(logs):
            for i in range(3):
                x = torch.randint(10, 100, (1, 50)).to(device)
                model(x)
    mem_end = torch_accelerator_module.max_memory_reserved()
    logs["memory"] = (mem_end - mem_start) / 1024**2

    # remove the first run (warm up)
    del logs["time"][0]
    return logs


def main(model_id, num_runs):
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device)
    base_memory = torch_accelerator_module.max_memory_reserved() / 1024**2

    # LORA
    config = LoraConfig(init_lora_weights=False, use_dora=False)
    model = get_peft_model(model, config)
    model.eval()
    torch_accelerator_module.reset_peak_memory_stats()
    logs_lora = run_benchmark(model, num_runs)
    avg_duration_lora = sum(logs_lora["time"]) / num_runs
    max_memory_lora = logs_lora["memory"] + base_memory

    # DORA
    del model
    torch_accelerator_module.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=device)
    base_memory = torch_accelerator_module.max_memory_reserved() / 1024**2
    config = LoraConfig(init_lora_weights=False, use_dora=True)
    model = get_peft_model(model, config)
    model.eval()

    # WITHOUT CACHING
    torch_accelerator_module.reset_peak_memory_stats()
    logs_dora_no_caching = run_benchmark(model, num_runs)
    avg_duration_no_caching = sum(logs_dora_no_caching["time"]) / num_runs
    max_memory_no_caching = logs_dora_no_caching["memory"] + base_memory

    # WITH CACHING
    torch_accelerator_module.reset_peak_memory_stats()
    with DoraCaching():
        logs_dora_caching = run_benchmark(model, num_runs)
    avg_duration_caching = sum(logs_dora_caching["time"]) / num_runs
    max_memory_caching = logs_dora_caching["memory"] + base_memory

    print(
        f"Benchmark results for model {model_id} with {num_runs} runs:\n\n"
        f"avg time LoRA:                     {avg_duration_lora:.4f} sec\n"
        f"avg time DoRA no caching:          {avg_duration_no_caching:.4f} sec\n"
        f"avg time DoRA with caching:        {avg_duration_caching:.4f} sec\n"
        f"\n"
        f"memory LoRA:                       {max_memory_lora:.2f} MB\n"
        f"memory DoRA no caching:            {max_memory_no_caching:.2f} MB\n"
        f"memory DoRA with caching:          {max_memory_caching:.2f} MB\n"
        f"\n"
        f"DoRA time overhead no caching:     {(avg_duration_no_caching - avg_duration_lora) / avg_duration_lora * 100:.2f}%\n"
        f"DoRA time overhead with caching:   {(avg_duration_caching - avg_duration_lora) / avg_duration_lora * 100:.2f}%\n"
        f"\n"
        f"DoRA memory overhead no caching:   {(max_memory_no_caching - max_memory_lora) / max_memory_lora * 100:.2f}%\n"
        f"DoRA memory overhead with caching: {(max_memory_caching - max_memory_lora) / max_memory_lora * 100:.2f}%"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark DoRA caching efficiency")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B", help="Model ID to benchmark")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of runs for the benchmark")
    args = parser.parse_args()

    main(args.model_id, args.num_runs)
