# Copyright 2024-present the HuggingFace Inc. team.
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
Example script demonstrating the time difference loading a model with a DoRA using ephemeral GPU offloading vs doing it purely on the CPU.

Example outputs:
$ python load_with_dora.py
--- Loading model ---
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:04<00:00,  1.03s/it]
--- Loading PeftModel ---
--- Done ---
Model loading time: 4.83s
PeftModel loading time: 28.14s
Use ephemeral GPU offloading: False

(Note: if this was the first time you ran the script, or if your cache was cleared, the times shown above are invalid, due to the time taken to download the model and DoRA files. Just re-run the script in this case.)

$ python load_with_dora.py --ephemeral_gpu_offload
--- Loading model ---
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:03<00:00,  1.11it/s]
--- Loading PeftModel ---
--- Done ---
Model loading time: 4.28s
PeftModel loading time: 16.59s
Use ephemeral GPU offloading: True

(Note: if this was the first time you ran the script, or if your cache was cleared, the times shown above are invalid, due to the time taken to download the model and DoRA files. Just re-run the script in this case.)
"""

import argparse
import time

from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description="Load a model with DoRA using ephemeral GPU offloading")
    parser.add_argument("--model", type=str, default="NousResearch/Hermes-2-Pro-Mistral-7B", help="Model to load")
    parser.add_argument(
        "--dora",
        type=str,
        default="peft-internal-testing/DoRA-Hermes-2-Pro-Mistral-7B",
        help="DoRA to use",
    )
    parser.add_argument("--ephemeral_gpu_offload", action="store_true", help="Use ephemeral GPU offloading")
    parser.add_argument(
        "--merge_model_path", type="str", help="Merge the model with the DoRA model and save to the given path"
    )
    args = parser.parse_args()

    peft_model_kwargs = {
        "ephemeral_gpu_offload": args.ephemeral_gpu_offload,
        "max_memory": {"cpu": "256GiB"},
        "device_map": {"": "cpu"},
    }

    # Predownload
    try:
        snapshot_download(repo_id=args.model)
    except Exception as e:
        print(f"Failed to download model: {e}")
        # We continue anyway as this might be e.g. a local directory or something
    try:
        snapshot_download(repo_id=args.dora)
    except Exception as e:
        print(f"Failed to download DoRA: {e}")
        # We continue anyway as this might be e.g. a local directory or something

    start = time.perf_counter()
    print("--- Loading model ---")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model_time = time.perf_counter() - start
    print("--- Loading PeftModel ---")
    peft_model = PeftModel.from_pretrained(model, args.dora, **peft_model_kwargs)
    print("--- Done ---")
    peft_model_time = time.perf_counter() - start

    print(f"Model loading time: {model_time:.2f}s")
    print(f"PeftModel loading time: {peft_model_time:.2f}s")
    print(f"Use ephemeral GPU offloading: {args.ephemeral_gpu_offload}")

    if args.merge_model_path is not None:
        merged_model = peft_model.merge_and_unload(progressbar=True)
        merged_model.save_pretrained(args.merge_model_path)


if __name__ == "__main__":
    main()
