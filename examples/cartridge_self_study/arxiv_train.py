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
from pathlib import Path

import torch
from train_distill import DistillationCollator, DistillationTrainer, DistillJsonlDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from peft import CartridgeConfig, get_peft_model
from peft.tuners.cartridge.utils import initialize_kv_prefix_from_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model to use for both teacher and student")
    parser.add_argument("--document", type=str, required=True, help="Path to text file for KV cache initialization")
    parser.add_argument("--distill_jsonl", type=str, default="distill.jsonl")
    parser.add_argument("--output_dir", type=str, default="cartridge_adapter")
    parser.add_argument("--num_virtual_tokens", type=int, default=256)
    parser.add_argument("--num_frozen_tokens", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "mps", "cuda", "xpu"])
    parser.add_argument(
        "--max_init_length", type=int, default=2048, help="Max tokens for text initialization (truncate long docs)"
    )
    args = parser.parse_args()

    if args.device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise ValueError("Requested device 'mps' but MPS is not available.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested device 'cuda' but CUDA is not available.")

    model_dtype = torch.float16 if args.device in {"cuda", "mps"} else None
    device_map = args.device if args.device != "cpu" else None

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=model_dtype, device_map=device_map)
    model = get_peft_model(
        base_model,
        CartridgeConfig(
            task_type="CAUSAL_LM",
            num_virtual_tokens=args.num_virtual_tokens,
            num_frozen_tokens=args.num_frozen_tokens,
        ),
    )

    print(f"Initializing cartridge from document: {args.document}", flush=True)
    document_text = Path(args.document).read_text()
    initialize_kv_prefix_from_text(
        model,
        tokenizer,
        text=document_text,
        use_chat_template=False,
        max_length=args.max_init_length,
    )
    print(f"Cartridge initialized with {args.num_virtual_tokens} tokens from text", flush=True)

    ds = DistillJsonlDataset(args.distill_jsonl)
    collator = DistillationCollator(tokenizer)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=10,
        save_steps=100,
        report_to=[],
        remove_unused_columns=False,
        use_cpu=args.device == "cpu",
        dataloader_pin_memory=False,
    )

    trainer = DistillationTrainer(
        model=model,
        top_k=args.top_k,
        args=train_args,
        train_dataset=ds,
        data_collator=collator,
    )
    trainer.train()
    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
