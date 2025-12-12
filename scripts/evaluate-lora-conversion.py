#!/usr/bin/env python3
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

"""Script to evaluate a PEFT checkpoint converted into a LoRA on GSM8K

To run this script, first train a PEFT model on MetaMathQA as described here:

https://github.com/huggingface/peft/tree/main/method_comparison/MetaMathQA

Call the script with the `-v` (verbose) option. When that run finishes, it will save a checkpoint of that model and
print a message like this: "Saved PEFT checkpoint to ...". Use this path as the `--path` argument to this script.

Example usage:

```bash
# Convert to LoRA with rank 8 and evaluate it
python evaluate-lora-conversion.py --path /path/to/peft/checkpoint --rank 8
# Convert to LoRA with dynamic rank (50% singular value threshold) and evaluate it
python evaluate-lora-conversion.py --path /path/to/peft/checkpoint --rank 0.5
# Evaluate the original PEFT model without LoRA conversion
python evaluate-lora-conversion.py --path /path/to/peft/checkpoint
```

The script will report the evaluation accuracy, maximum CUDA memory reserved, and evaluation time for the converted LoRA
model.

"""

import argparse
import importlib.util
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM

from peft import PeftModel, convert_to_lora, get_peft_model, set_peft_model_state_dict


root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

spec = importlib.util.spec_from_file_location("data", os.path.join(root, "method_comparison", "MetaMathQA", "data.py"))
mm_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mm_data)
sys.modules["data"] = mm_data

spec = importlib.util.spec_from_file_location(
    "utils", os.path.join(root, "method_comparison", "MetaMathQA", "utils.py")
)
mm_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mm_utils)
sys.modules["utils"] = mm_utils

spec = importlib.util.spec_from_file_location("run", os.path.join(root, "method_comparison", "MetaMathQA", "run.py"))
mm_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mm_run)


def noop(*args, **kwargs):
    pass


def evaluate_model(model, tokenizer, ds_test):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    tic = time.perf_counter()
    predictions, responses = mm_run.evaluate(
        model=model,
        tokenizer=tokenizer,
        ds=ds_test,
        batch_size=50,
        generate_kwargs={"max_length": 800, "max_new_tokens": 300, "pad_token_id": tokenizer.eos_token_id},
        use_tqdm=True,
    )
    toc = time.perf_counter()
    accuracy_peft = mm_utils.get_accuracy(predictions=predictions, responses=responses)
    cuda_mem_reserved_max = torch.cuda.memory_reserved(0)
    print(f"Evaluation Accuracy: {100 * accuracy_peft:.2f}%")
    print(f"Max CUDA Memory Reserved: {cuda_mem_reserved_max / (1024**3):.2f} GB")
    print(f"Evaluation Time: {toc - tic:.0f} seconds".format(toc - tic))


def main(path_peft_model: str, rank: int | float | None) -> None:
    model_id = "meta-llama/Llama-3.2-3B"
    tokenizer = mm_utils.get_tokenizer(model_id=model_id, max_seq_length=768)
    _, _, ds_test = mm_data.get_train_valid_test_datasets(
        tokenizer=tokenizer, query_template="Question: {query} Think step by step.\nAnswer:", print_fn=noop
    )

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(0)
    model = PeftModel.from_pretrained(model, path_peft_model)
    if rank is None:
        print("Evaluating the original PEFT model without LoRA conversion...")
        model.set_adapter("default")
        model.print_trainable_parameters()
        model.eval()
        evaluate_model(model, tokenizer, ds_test)
        return

    print(f"Converting PEFT model to LoRA with rank={rank}...")
    tic = time.perf_counter()
    lora_config, lora_state_dict = convert_to_lora(model, rank=rank, progressbar=True)
    toc = time.perf_counter()
    print(f"Conversion completed in {toc - tic:.0f} seconds.".format(toc - tic))

    del model
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(0)

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    load_result = set_peft_model_state_dict(model, lora_state_dict)
    assert not load_result.unexpected_keys, (
        f"Unexpected keys when loading LoRA state dict: {load_result.unexpected_keys}"
    )

    del lora_state_dict
    model.eval()
    evaluate_model(model, tokenizer, ds_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a PEFT checkpoint converted into a LoRA on GSM8K")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the input PEFT checkpoint",
    )
    parser.add_argument(
        "--rank",
        required=False,
        default=None,
        help="Rank for the LoRA decomposition (int, float, or None for no conversion)",
    )

    args = parser.parse_args()
    if args.rank is not None:
        if "." in str(args.rank):
            args.rank = float(args.rank)
        else:
            args.rank = int(args.rank)
    main(args.path, args.rank)
