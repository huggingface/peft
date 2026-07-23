# Copyright 2026-present the HuggingFace Inc. team.
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

"""ShadowPEFT with a *pretrained* shadow backbone.

Instead of letting ShadowPEFT build a fresh ("mirror") shadow backbone from the base config, this example initializes
the shadow backbone from a separate, (optionally smaller) pretrained model by passing its id/path as
``ShadowConfig(shadow_model=...)``. When the pretrained backbone's hidden size differs from the base model's,
ShadowPEFT automatically inserts a trainable projection to bridge the two hidden spaces.

After training, ``unload_shadow()`` returns the standalone shadow network (backbone + head), the lightweight component
that can be deployed on its own.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PeftModel, ShadowConfig, get_peft_model


def parse_args():
    parser = argparse.ArgumentParser(description="ShadowPEFT pretrained-shadow-backbone example")
    parser.add_argument("--base_model_name_or_path", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--shadow_model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Model id/path used to initialize the shadow backbone (its hidden size may differ from the base model).",
    )
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--update_hidden_size", type=int, default=None)
    parser.add_argument("--shadow_alpha", type=float, default=1.0)
    parser.add_argument("--shadow_dropout", type=float, default=0.0)
    parser.add_argument("--auxiliary_loss_weight", type=float, default=0.05)
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="./shadow-explicit-adapter")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_name_or_path)

    config = ShadowConfig(
        shadow_model=args.shadow_model,
        r=args.r,
        update_hidden_size=args.update_hidden_size,
        shadow_alpha=args.shadow_alpha,
        shadow_dropout=args.shadow_dropout,
        auxiliary_loss_weight=args.auxiliary_loss_weight,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, config).to(device)
    model.print_trainable_parameters()

    projection = model.base_model.shadow_projection["default"]
    print(f"shadow_projection: {type(projection).__name__}")

    # Toy training data: replace with a real dataset / transformers.Trainer for actual fine-tuning.
    texts = [
        "A small shadow backbone can adapt a much larger base model.",
        "A projection bridges the shadow and base hidden spaces when they differ.",
        "Only the shadow backbone and the injection/update adapters are trained.",
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

    # Save only the adapter (shadow backbone + injection/update + projection). The base model is not stored. On reload,
    # the shadow backbone architecture is rebuilt from `shadow_model` and the fine-tuned weights are restored.
    model.save_pretrained(args.output_dir)
    print(f"Saved adapter to {args.output_dir}")

    reloaded_base = AutoModelForCausalLM.from_pretrained(args.base_model_name_or_path)
    model = PeftModel.from_pretrained(reloaded_base, args.output_dir).to(device)
    model.eval()

    prompt = tokenizer("Shadow adaptation", return_tensors="pt").to(device)
    with torch.no_grad():
        generated = model.generate(**prompt, max_new_tokens=20, use_cache=False, do_sample=False)
    print(tokenizer.decode(generated[0], skip_special_tokens=True))

    # Recover the standalone shadow network (backbone + projection + head). It behaves like a normal causal LM (it
    # supports generate()), so it can be evaluated on its own and saved/pushed like any HF model. This is how you
    # measure the shadow path's own performance, independent of the base model.
    shadow = model.base_model.unload_shadow()
    shadow.eval()
    with torch.no_grad():
        shadow_generated = shadow.generate(**prompt, max_new_tokens=20, use_cache=True, do_sample=False)
    print("shadow-only generation:", tokenizer.decode(shadow_generated[0], skip_special_tokens=True))
    shadow.save_pretrained(f"{args.output_dir}-standalone-shadow")
    print(f"Saved standalone shadow model ({type(shadow).__name__}) to {args.output_dir}-standalone-shadow")


if __name__ == "__main__":
    main()
