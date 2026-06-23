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

"""Minimal ShadowPEFT fine-tuning example.

This script wraps a frozen base causal LM with a ShadowPEFT adapter, runs a few training steps on a toy dataset, saves
the adapter, reloads it and runs generation. It is intentionally small so it can serve as a starting point; swap in a
real dataset and the `Trainer` for actual fine-tuning.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftModel, ShadowConfig, get_peft_model


def parse_args():
    parser = argparse.ArgumentParser(description="ShadowPEFT fine-tuning example")
    parser.add_argument("--base_model_name_or_path", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num_shadow_layers", type=int, default=1)
    parser.add_argument("--injection_hidden_size", type=int, default=16)
    parser.add_argument("--gate_hidden_size", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--shadow_loss_weight", type=float, default=0.05)
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="./shadow-adapter")
    parser.add_argument("--shadow_only", action="store_true", help="Use shadow-only inference after training.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_name_or_path)

    config = ShadowConfig(
        num_shadow_layers=args.num_shadow_layers,
        injection_hidden_size=args.injection_hidden_size,
        gate_hidden_size=args.gate_hidden_size,
        alpha=args.alpha,
        dropout=args.dropout,
        shadow_loss_weight=args.shadow_loss_weight,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, config).to(device)
    model.print_trainable_parameters()

    # Toy training data: replace with a real dataset / transformers.Trainer for actual fine-tuning.
    texts = [
        "ShadowPEFT trains a small parallel network.",
        "The base model stays frozen during fine-tuning.",
        "Only the shadow adapter is updated.",
    ]
    batch = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    labels = batch["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    for step in range(args.num_steps):
        optimizer.zero_grad()
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=labels)
        out.loss.backward()
        optimizer.step()
        print(f"step {step}: loss={out.loss.item():.4f}")

    # Save only the shadow adapter (base weights are not stored).
    model.save_pretrained(args.output_dir)
    print(f"Saved adapter to {args.output_dir}")

    # Reload and run generation. ShadowPEFT disables the KV cache, so use_cache=False is required.
    reloaded_base = AutoModelForCausalLM.from_pretrained(args.base_model_name_or_path)
    model = PeftModel.from_pretrained(reloaded_base, args.output_dir).to(device)
    model.eval()
    if args.shadow_only:
        model.base_model.set_inference_mode("shadow_only")

    prompt = tokenizer("ShadowPEFT", return_tensors="pt").to(device)
    with torch.no_grad():
        generated = model.generate(**prompt, max_new_tokens=20, use_cache=False, do_sample=False)
    print(tokenizer.decode(generated[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
