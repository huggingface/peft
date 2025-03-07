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

"""This script trains a model on a small text dataset and measures the memory consumption, as well as a few other
useful metrics.

Example:

Get help:

```bash
python train_memory.py --help
```

Train the google/gemma-2-2b model with a LoRA config json at the indicated location.

```bash
python train_memory.py "google/gemma-2-2b" --max_seq_length 256 --batch_size 1 --rank 32 --dtype bfloat16 --path_config <path-to-adapter-config.json>
```

Fully fine-tune the model (i.e. without LoRA) by setting the rank to 0:

```bash
python train_memory.py "google/gemma-2-2b" --rank 0
```

Get an estimate of the size of the hidden states by passing `--monitor_tensors`. This trains just for a single epoch. For realistic estimates, the batch size for this:

```bash
python train_memory.py "google/gemma-2-2b" --max_seq_length 256 --batch_size 32 --rank 32 --dtype bfloat16 --path_config configs/lora_rank-32_embedding-lora/ --monitor_tensors
```

"""

import argparse
import gc
import os
import sys
import tempfile
import time
import warnings
from collections import Counter
from contextlib import nullcontext
from functools import partial

import torch
from datasets import load_dataset
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.utils import CONFIG_NAME, SAFETENSORS_WEIGHTS_NAME


# suppress all warnings
warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype_to_bytes_linear = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "int4": 0.5}


def init_cuda():
    torch.manual_seed(0)
    if device == "cpu":
        return

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.manual_seed_all(0)
    # might not be necessary, but just to be sure
    nn.Linear(1, 1).to(device)


def get_data(tokenizer):
    def tokenize(samples):
        # For some reason, the max sequence length is not honored by the tokenizer, resulting in IndexErrors. Thus,
        # manually ensure that sequences are not too long.
        tokenized = tokenizer(samples["quote"])
        tokenized["input_ids"] = [input_ids[: tokenizer.model_max_length] for input_ids in tokenized["input_ids"]]
        tokenized["attention_mask"] = [
            input_ids[: tokenizer.model_max_length] for input_ids in tokenized["attention_mask"]
        ]
        return tokenized

    data = load_dataset("ybelkada/english_quotes_copy")
    data = data.map(tokenize, batched=True)
    # We need to manually remove unused columns. This is because we cannot use remove_unused_columns=True in the
    # Trainer, as this leads to errors with torch.compile. We also cannot just leave them in, as they contain
    # strings. Therefore, manually remove all unused columns.
    data = data.remove_columns(["quote", "author", "tags"])
    return data


def train(model_id, rank, dtype, monitor_tensors, max_seq_length, batch_size, max_steps, path_config):
    init_cuda()
    cuda_memory_init = torch.cuda.max_memory_allocated()
    cuda_memory_log = []

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.model_max_length = max_seq_length
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    data = get_data(tokenizer)

    if dtype == "int4":
        quant_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, quantization_config=quant_config)
        model = prepare_model_for_kbit_training(model)
    elif dtype == "int8":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, quantization_config=quant_config)
        model = prepare_model_for_kbit_training(model)
    elif dtype == "bfloat16":
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, torch_dtype=torch.bfloat16)
    elif dtype == "float16":
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, torch_dtype=torch.float16)
    elif dtype == "float32":
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device)
    else:
        raise ValueError(f"Invalid dtype: {dtype}")

    if rank > 0:
        if path_config is None:
            raise RuntimeError("LoRA rank > 0 requires a path to a LoRA config")
        if path_config.endswith(CONFIG_NAME):
            path_config = path_config.removesuffix(CONFIG_NAME)
        config = LoraConfig.from_pretrained(path_config)
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    else:
        print("Not using LoRA")

    model.config.use_cache = False
    storage = []

    def pack(x):
        storage.append(x)
        return len(storage) - 1

    def unpack(x):
        return storage[x]

    train_ctx = partial(torch.autograd.graph.saved_tensors_hooks, pack, unpack) if monitor_tensors else nullcontext

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    losses = []
    sample = 0
    tic_total = time.perf_counter()
    for i in range(0, max_steps):
        storage.clear()
        tic = time.perf_counter()
        try:
            batch = tokenizer.pad(data["train"][sample : sample + batch_size], return_tensors="pt").to(model.device)
            sample += batch_size

            # add targets
            batch["labels"] = batch["input_ids"].clone()
            optimizer.zero_grad()

            with train_ctx():
                outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            cuda_memory_log.append(torch.cuda.memory_allocated() - cuda_memory_init)
            torch.cuda.empty_cache()
            gc.collect()
            toc = time.perf_counter()
            print(f"step {i:3d} loss {loss.item():.6f} time {toc - tic:.2f}s", file=sys.stderr)
        except KeyboardInterrupt:
            print("canceled training")
            break

        if monitor_tensors:
            break

    toc_total = time.perf_counter()

    cuda_memory_final = torch.cuda.max_memory_allocated()
    cuda_memory_avg = int(sum(cuda_memory_log) / len(cuda_memory_log))
    print(f"cuda memory avg: {cuda_memory_avg // 2**20}MB")
    print(f"cuda memory max: {(cuda_memory_final - cuda_memory_init) // 2**20}MB")
    print(f"total time: {toc_total - tic_total:.2f}s")

    with tempfile.TemporaryDirectory() as tmp_dir:
        model.save_pretrained(tmp_dir)
        stat = os.stat(os.path.join(tmp_dir, SAFETENSORS_WEIGHTS_NAME))
    file_size = stat.st_size
    print(f"file size: {file_size / 2**20:.1f}MB")

    if monitor_tensors:
        dtype_counts = Counter(t.dtype for t in storage)
        shape_counts = Counter(t.shape for t in storage)
        param_shape_counts = Counter(p.shape for p in model.parameters())
        param_shape_counts_copy = dict(param_shape_counts).copy()

        # shape counts includes the params, so we need to subtract them; note that they can be transposed
        # this is an approximation
        diff_shape_counts = {}
        for shape, count in shape_counts.items():
            if shape in param_shape_counts_copy:
                diff_count = count - param_shape_counts[shape]
                if diff_count > 0:
                    diff_shape_counts[shape] = diff_count
                    param_shape_counts_copy[shape] = max(0, param_shape_counts_copy[shape] - diff_count)
            elif shape[::-1] in param_shape_counts:
                diff_count = count - param_shape_counts[shape[::-1]]
                if diff_count > 0:
                    diff_shape_counts[shape] = diff_count
                    param_shape_counts_copy[shape[::-1]] = max(0, param_shape_counts_copy[shape[::-1]] - diff_count)
            else:
                diff_shape_counts[shape] = count

        total_size = sum(t.numel() * t.element_size() for t in storage)
        total_size_mb = f"{total_size // 2**20}MB"
        diff_size = 0
        for shape, count in diff_shape_counts.items():
            diff_size += count * torch.zeros(shape).numel() * dtype_to_bytes_linear[dtype]
        param_size = total_size - diff_size

        diff_size_mb = f"{diff_size // 2**20}MB"
        param_size_mb = f"{param_size // 2**20}MB"

        print(f"Dtype counts: {dtype_counts.most_common()}")
        print(f"Total size of tensors:     {total_size_mb: >12}")
        print(f"Total size of activations: {diff_size_mb: >12}")
        print(f"Total size of parameters:  {param_size_mb: >12}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", type=str, help="Model name on Hugging Face Hub")
    parser.add_argument("--rank", type=int, default=8, help="Rank of LoRA, 0 => no LoRA, default 8")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Data type, one of float32, float16, bfloat16, int8, int4, default float32",
    )
    parser.add_argument(
        "--monitor_tensors",
        action="store_true",
        help="Monitor tensor sizes during training for a single training step, off by default",
    )
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length, default 128")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size, default 1")
    parser.add_argument("--max_steps", type=int, default=50, help="Maximum number of training steps, default 50")
    parser.add_argument("--path_config", type=str, default=None, help="Path to LoRA config")
    args = parser.parse_args()
    train(
        model_id=args.model_id,
        rank=args.rank,
        dtype=args.dtype,
        monitor_tensors=args.monitor_tensors,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        path_config=args.path_config,
    )
