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
"""
Fine-tune a causal LM on ARC-Easy (multiple-choice QA) with Bayesian-LoRA and compare the calibration of the point
estimate with that of Monte Carlo predictions obtained with `peft.helpers.enable_probabilistic_sampling`.
"""

import argparse
import itertools
import math
import random

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import BayesianLoraConfig, LoraConfig, PeftModel, get_peft_model
from peft.helpers import enable_probabilistic_sampling
from peft.utils import infer_device


LETTERS = "ABCDE"
PROMPT = "Return the label of the correct answer for the question below.\n\nQuestion: {question}\nChoices:\n{choices}\nAnswer:"


def load_arc(split):
    """Returns a list of (prompt, number of choices, label index)."""
    examples = []
    for e in load_dataset("allenai/ai2_arc", "ARC-Easy", split=split):
        # a few questions use the labels 1-4 instead of A-D
        letters = [LETTERS[int(x) - 1] if x.isdigit() else x for x in e["choices"]["label"]]
        answer = LETTERS[int(e["answerKey"]) - 1] if e["answerKey"].isdigit() else e["answerKey"]
        choices = "\n".join(f"{letter}) {text}" for letter, text in zip(letters, e["choices"]["text"]))
        examples.append((PROMPT.format(question=e["question"], choices=choices), len(letters), LETTERS.index(answer)))
    return examples


def answer_token_ids(tokenizer):
    # id of the token of each answer letter when it follows "Answer:"
    prefix = tokenizer("Answer:", add_special_tokens=False).input_ids
    ids = []
    for letter in LETTERS:
        full = tokenizer(f"Answer: {letter}", add_special_tokens=False).input_ids
        if full[: len(prefix)] != prefix or len(full) != len(prefix) + 1:
            raise ValueError(f"The tokenizer does not encode ' {letter}' as a single token after 'Answer:'.")
        ids.append(full[-1])
    return torch.tensor(ids)


def batches(examples, batch_size, shuffle=False, rng=None):
    order = list(range(len(examples)))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), batch_size):
        chunk = [examples[j] for j in order[i : i + batch_size]]
        yield [c[0] for c in chunk], torch.tensor([c[1] for c in chunk]), torch.tensor([c[2] for c in chunk])


def answer_logits(model, tokenizer, prompts, num_choices, target_ids, device):
    """Logits over the answer letters at the last prompt position; letters that are not a choice are masked."""
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    logits = model(**enc).logits[:, -1, :][:, target_ids.to(device)].float()
    mask = torch.arange(len(LETTERS), device=device)[None, :] >= num_choices.to(device)[:, None]
    return logits.masked_fill(mask, float("-inf"))


def expected_calibration_error(probs, labels, num_bins=15):
    confidence, prediction = probs.max(1)
    correct = (prediction == labels).float()
    ece = 0.0
    edges = torch.linspace(0, 1, num_bins + 1)
    for low, high in itertools.pairwise(edges):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            ece += in_bin.float().mean() * (confidence[in_bin].mean() - correct[in_bin].mean()).abs()
    return float(ece)


@torch.no_grad()
def evaluate(model, tokenizer, examples, target_ids, device, batch_size, num_samples):
    """num_samples=0: deterministic point estimate; num_samples>0: average of num_samples sampled predictions."""
    model.eval()
    all_probs, all_labels = [], []
    for prompts, num_choices, labels in batches(examples, batch_size):
        if num_samples == 0:
            probs = answer_logits(model, tokenizer, prompts, num_choices, target_ids, device).softmax(-1)
        else:
            with enable_probabilistic_sampling(model):
                probs = torch.stack(
                    [
                        answer_logits(model, tokenizer, prompts, num_choices, target_ids, device).softmax(-1)
                        for _ in range(num_samples)
                    ]
                ).mean(0)
        all_probs.append(probs.cpu())
        all_labels.append(labels)
    probs, labels = torch.cat(all_probs), torch.cat(all_labels)
    return {
        "accuracy": 100 * (probs.argmax(1) == labels).float().mean().item(),
        "ece": 100 * expected_calibration_error(probs, labels),
        "nll": F.nll_loss(probs.clamp_min(1e-12).log(), labels).item(),
    }


def main(args):
    torch.manual_seed(args.seed)
    device = infer_device()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.padding_side = "left"  # the answer is predicted at the last position
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    target_ids = answer_token_ids(tokenizer)

    train, test = load_arc("train"), load_arc("test")
    if args.max_train_samples:
        train = train[: args.max_train_samples]
    if args.max_eval_samples:
        test = test[: args.max_eval_samples]

    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16).to(device)
    config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=args.target_modules.split(","),
        task_type="CAUSAL_LM",
        bayesian_lora_config=BayesianLoraConfig(inducing_rows=args.inducing, inducing_cols=args.inducing),
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=0.1
    )
    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    # The KL divergence is a sum over all inducing weights, the cross-entropy a mean over the batch: scaling the KL by
    # 1 / steps_per_epoch gives the per-batch share of the evidence lower bound. kl_weight < 1 tempers the posterior.
    kl_coefficient = args.kl_weight / steps_per_epoch

    rng = random.Random(args.seed)
    step = 0
    for epoch in range(args.num_epochs):
        for prompts, num_choices, labels in batches(train, args.batch_size, shuffle=True, rng=rng):
            model.train()  # the Bayesian-LoRA factors are sampled in every training forward pass
            logits = answer_logits(model, tokenizer, prompts, num_choices, target_ids, device)
            nll = F.cross_entropy(logits, labels.to(device))
            kl = model.base_model._get_bayesian_lora_loss()
            loss = nll + kl_coefficient * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step % args.logging_steps == 0:
                print(f"epoch {epoch} step {step}: nll={nll.item():.4f} kl={kl.item():.3e}")

    point = evaluate(model, tokenizer, test, target_ids, device, args.eval_batch_size, num_samples=0)
    mc = evaluate(model, tokenizer, test, target_ids, device, args.eval_batch_size, num_samples=args.num_mc_samples)
    print(f"ARC-Easy test, point estimate: {point}")
    print(f"ARC-Easy test, MC ({args.num_mc_samples} samples): {mc}")

    model.save_pretrained(args.output_dir)
    print(f"Adapter saved to {args.output_dir}")

    if args.check_reload:
        # a reloaded adapter gives the same point-estimate predictions as the trained one
        base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16).to(device)
        reloaded = PeftModel.from_pretrained(base, args.output_dir)
        reloaded_point = evaluate(reloaded, tokenizer, test, target_ids, device, args.eval_batch_size, num_samples=0)
        print(f"Reloaded adapter, point estimate: {reloaded_point}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a causal LM on ARC-Easy with Bayesian-LoRA")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", type=str, default="bayesian-lora-arc-easy")
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--inducing", type=int, default=9, help="inducing_rows = inducing_cols")
    parser.add_argument("--kl_weight", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--num_mc_samples", type=int, default=10)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--check_reload", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
