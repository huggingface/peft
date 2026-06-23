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

"""ShadowPEFT with an *explicit* shadow model.

Instead of letting ShadowPEFT build an implicit shadow model from the base config, this example uses a separate,
(optionally smaller) model as the shadow network. When the shadow model's hidden size differs from the base model's,
ShadowPEFT automatically inserts a trainable ``shadow_hidden_projection`` to bridge the two hidden spaces.

The explicit shadow model is passed through ``get_peft_model(model, config, shadow_model=...)`` and, when loading a
saved adapter, through ``PeftModel.from_pretrained(model, path, shadow_model=...)``.

Optionally, you can package a pre-trained small model + projection + the base ``lm_head`` into a single
``AutoModelForCausalLMWithHiddenProjection`` checkpoint (the canonical distribution format for projected shadow
models) and use that as the explicit shadow model.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import (
    AutoModelForCausalLMWithHiddenProjection,
    PeftModel,
    ShadowConfig,
    get_peft_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="ShadowPEFT explicit-shadow-model example")
    parser.add_argument("--base_model_name_or_path", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--shadow_model_name_or_path",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Smaller model used as the explicit shadow network (its hidden size may differ from the base model).",
    )
    parser.add_argument(
        "--projected_shadow",
        action="store_true",
        help="Wrap the shadow model + projection + base lm_head into an AutoModelForCausalLMWithHiddenProjection "
        "(initialized with the pseudo-inverse recipe) and use that as the explicit shadow model.",
    )
    parser.add_argument("--injection_hidden_size", type=int, default=16)
    parser.add_argument("--gate_hidden_size", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--shadow_loss_weight", type=float, default=0.05)
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="./shadow-explicit-adapter")
    return parser.parse_args()


def build_explicit_shadow_model(args, base_model):
    """Return the explicit shadow model to hand to ShadowPEFT."""
    shadow_model = AutoModelForCausalLM.from_pretrained(args.shadow_model_name_or_path)

    if not args.projected_shadow:
        # Plain explicit shadow model. ShadowPEFT inserts a trainable projection automatically if the hidden sizes
        # differ between the shadow model and the base model.
        return shadow_model

    # Bundle the small backbone + a (shadow_hidden -> base_hidden) projection + the frozen base lm_head into a single
    # standalone, HF-loadable model. The projection is initialized via the pseudo-inverse recipe so the shadow path
    # starts aligned with the base vocabulary space.
    base_hidden = base_model.config.hidden_size
    shadow_hidden = shadow_model.config.hidden_size
    wrapped = AutoModelForCausalLMWithHiddenProjection.wrap(
        shadow_model=shadow_model,
        shadow_hidden_projection=torch.nn.Linear(shadow_hidden, base_hidden, bias=False),
        lm_head=base_model.lm_head,
        init_optimal_projection=True,
        reference_lm_head=shadow_model.lm_head,
    )
    # ShadowPEFT reuses the trained projection carried by this wrapper instead of randomly initializing a new one.
    return wrapped


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_name_or_path)
    shadow_model = build_explicit_shadow_model(args, base_model)

    # Note: num_shadow_layers is ignored when an explicit shadow_model is provided.
    config = ShadowConfig(
        injection_hidden_size=args.injection_hidden_size,
        gate_hidden_size=args.gate_hidden_size,
        alpha=args.alpha,
        dropout=args.dropout,
        shadow_loss_weight=args.shadow_loss_weight,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, config, shadow_model=shadow_model).to(device)
    model.print_trainable_parameters()

    proj = model.base_model.shadow_hidden_projection
    print(
        f"shadow hidden size: {model.base_model.shadow_hidden_size}, base hidden size: {model.base_model.base_hidden_size}"
    )
    print(f"shadow_hidden_projection: {type(proj).__name__}")

    # Toy training data: replace with a real dataset / transformers.Trainer for actual fine-tuning.
    texts = [
        "A small shadow model can adapt a much larger backbone.",
        "The projection bridges the shadow and base hidden spaces.",
        "Only the shadow adapter and projection are trained.",
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

    # Save only the adapter (shadow backbone + injection/update + projection). The base model is not stored.
    model.save_pretrained(args.output_dir)
    print(f"Saved adapter to {args.output_dir}")

    # Reload: an explicit shadow model must be supplied again so the architecture matches the saved weights.
    reloaded_base = AutoModelForCausalLM.from_pretrained(args.base_model_name_or_path)
    reloaded_shadow = build_explicit_shadow_model(args, reloaded_base)
    model = PeftModel.from_pretrained(reloaded_base, args.output_dir, shadow_model=reloaded_shadow).to(device)
    model.eval()

    prompt = tokenizer("Shadow adaptation", return_tensors="pt").to(device)
    with torch.no_grad():
        generated = model.generate(**prompt, max_new_tokens=20, use_cache=False, do_sample=False)
    print(tokenizer.decode(generated[0], skip_special_tokens=True))

    # Export a standalone shadow model. When hidden sizes differ, this returns an
    # AutoModelForCausalLMWithHiddenProjection bundling the trained backbone, projection and base lm_head.
    exported = model.base_model.export_shadow()
    exported.save_pretrained(f"{args.output_dir}-exported-shadow")
    print(f"Exported standalone shadow model ({type(exported).__name__}) to {args.output_dir}-exported-shadow")


if __name__ == "__main__":
    main()
