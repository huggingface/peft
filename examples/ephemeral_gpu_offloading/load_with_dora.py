# Example script demonstrating the time difference loading a model with a DoRA using ephemeral GPU offloading
# vs doing it purely on the CPU.

import argparse
import time

from transformers import AutoModelForCausalLM

from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description="Load a model with DoRA using ephemeral GPU offloading")
    parser.add_argument("--model", type=str, default="NousResearch/Hermes-2-Pro-Mistral-7B", help="Model to load")
    parser.add_argument(
        "--dora",
        type=str,
        default="anamikac2708/Mistral-7B-DORA-finetuned-investopedia-Lora-Adapters",
        help="DoRA to use",
    )
    parser.add_argument("--ephemeral_gpu_offload", action="store_true", help="Use ephemeral GPU offloading")
    parser.add_argument(
        "--merge_model", action="store_true", help="Merge the model with the DoRA model and save to disk"
    )
    args = parser.parse_args()

    peft_model_kwargs = {
        "ephemeral_gpu_offload": args.ephemeral_gpu_offload,
        "max_memory": {"cpu": "256GiB"},
        "device_map": {"": "cpu"},
    }

    start = time.perf_counter()
    print("-" * 20 + " Loading model " + "-" * 20)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model_time = time.perf_counter() - start
    print("-" * 20 + " Loading PeftModel " + "-" * 20)
    peft_model = PeftModel.from_pretrained(model, args.dora, **peft_model_kwargs)
    print("-" * 20 + " Done " + "-" * 20)
    peft_model_time = time.perf_counter() - start

    print(f"Model loading time: {model_time:.2f}s")
    print(f"PeftModel loading time: {peft_model_time:.2f}s")
    print(f"Use ephemeral GPU offloading: {args.ephemeral_gpu_offload}")
    print(
        "\n(Note: if this was the first time you ran the script, or if your cache was cleared, the times shown above are invalid, due to the time taken to download the model and DoRA files. Just re-run the script in this case.)\n"
    )

    if args.merge_model:
        merged_model = peft_model.merge_and_unload(progressbar=True)
        merged_model.save_pretrained("merged_model")


if __name__ == "__main__":
    main()

"""
Example outputs:
$ python load_with_dora.py
-------------------- Loading model --------------------
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:04<00:00,  1.03s/it]
-------------------- Loading PeftModel --------------------
-------------------- Done --------------------
Model loading time: 4.83s
PeftModel loading time: 28.14s
Use ephemeral GPU offloading: False

(Note: if this was the first time you ran the script, or if your cache was cleared, the times shown above are invalid, due to the time taken to download the model and DoRA files. Just re-run the script in this case.)

$ python load_with_dora.py --ephemeral_gpu_offload
-------------------- Loading model --------------------
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:03<00:00,  1.11it/s]
-------------------- Loading PeftModel --------------------
-------------------- Done --------------------
Model loading time: 4.28s
PeftModel loading time: 16.59s
Use ephemeral GPU offloading: True

(Note: if this was the first time you ran the script, or if your cache was cleared, the times shown above are invalid, due to the time taken to download the model and DoRA files. Just re-run the script in this case.)
"""
