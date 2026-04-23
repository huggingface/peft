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
This script provides a simple evaluation pipeline for multiple-choice reasoning datasets
(e.g., BoolQ, HellaSwag, ARC, OpenBookQA, Winogrande) with different composition strategies.

Usage examples:
    python arrow_phi3_mini.py --strategy base --ds_name arc-challenge
    python arrow_phi3_mini.py --strategy arrow --ds_name boolq
    python arrow_phi3_mini.py --strategy gks --ds_name hswag

Key features:
- Supports three strategies:
    • "base"   → Evaluate the quantized base model directly
    • "arrow"  → Use Arrow modular routing with task-specific adapters
    • "gks"    → Use Arrow + GenKnowSub (subtracting general-domain knowledge)
- Loads evaluation datasets from the Hugging Face Hub
- Implements a batched evaluation loop that computes per-option likelihoods and selects
  the answer with the lowest average loss
- Reports simple accuracy

Implementation details:
- The base model is quantized to 4-bit using `BitsAndBytesConfig` (nf4, bf16 compute).
- For Arrow and GKS, task-specific adapters are loaded from the Hugging Face Hub:
    TahaBa/phi3-mini-clustered-flan/ts_expert_i
- Task-specific adapters were trained on 10 clusters of FLAN tasks.
- The clusters were created using Model-Based Clustering (MBC):
    1. Train a LoRA adapter for each individual task.
    2. Apply k-means clustering to group tasks based on these adapters.
    3. Train a LoRA adapter for each resulting cluster.
For more details, see the Arrow paper: https://huggingface.co/papers/2405.11157

- For GKS, general adapters are loaded from:
    TahaBa/phi3-mini-general-adapters/...
- These adapters were trained on English, French, and German Wikipedia data
  using a causal language modeling objective with (507-token context → 5-token completion) pairs.
- This setup encodes general knowledge into the LoRA space, which can then be
  subtracted from task-specific adapters during inference to isolate and purify them.
For more details, see the GenKnowSub paper: https://huggingface.co/papers/2505.10939

- `evaluate_on_multi_choice_batched` handles tokenization, masking context tokens,
  and computing per-choice log-likelihoods for fair comparison.
- Accuracy is printed at the end for the selected dataset.

This script is mainly meant for demonstration purposes and lightweight evaluation,
not full-scale benchmarking (batch size / max length can be tuned).

=======================================================================================

Results (evaluated with microsoft/Phi-3-mini-4k-instruct, 4-bit quantization):

| Dataset      | Base Acc. | Arrow Acc. | Arrow+GKS Acc. |
|--------------|-----------|------------|----------------|
| ARC-Challenge|   0.4515  |   0.5418   |     0.5585     |
| ARC-Easy     |   0.6894  |   0.8404   |     0.8473     |
| Winogrande   |   0.5769  |   0.6550   |     0.6724     |
| BoolQ        |   0.8146  |   0.8030   |     0.8247     |
| OpenBookQA   |   0.43    |   0.448    |     0.472      |
| HellaSwag    |   0.7318  |   0.7150   |     0.7376     |

Observations:
- Arrow generally improves over the base model by routing tokens to the most relevant task adapters.
- Applying GKS (general knowledge subtraction) consistently gives further gains compared to Arrow and Base.

These numbers are not meant as leaderboard results, but as a sanity check
to verify that the implementation works as expected and demonstrates
the benefits of Arrow and GenKnowSub.
"""

import argparse
import random

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from peft import ArrowConfig, create_arrow_model


MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
MODEL_MAX_LEN = 2048


def parse_args():
    parser = argparse.ArgumentParser(description="Training script with strategy selection")

    parser.add_argument(
        "--strategy",
        type=str,
        choices=["base", "arrow", "gks"],
        default="base",
        help="Training strategy to use: base, arrow, or gks",
    )
    parser.add_argument(
        "--ds_name",
        type=str,
        choices=["boolq", "hswag", "arc-easy", "arc-challenge", "oqa", "wg"],
        default="arc-challenge",
        help="Dataset to use: boolq, hswag, arc-easy, arc-challenge, oqa, wg",
    )

    return parser.parse_args()


def read_test_dataset(ds_name):
    if ds_name == "boolq":
        ds = load_dataset("google/boolq", split="validation", trust_remote_code=True)
    elif ds_name == "hswag":
        ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
    elif ds_name == "arc-challenge":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="validation", trust_remote_code=True)
    elif ds_name == "arc-easy":
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation", trust_remote_code=True)
    elif ds_name == "oqa":
        ds = load_dataset("allenai/openbookqa", split="validation", trust_remote_code=True)
    elif ds_name == "wg":
        ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation", trust_remote_code=True)
    else:
        raise f"Dataset {ds_name} is not supported yet."

    return ds


def extract_input_content(ds_name, row):
    if ds_name == "boolq":
        return f"[passage]{row['passage']}[question]{row['question']}"
    if ds_name == "hswag":
        return row["ctx"]
    if (ds_name == "arc-challenge") or (ds_name == "arc-easy"):
        return row["question"]
    if ds_name == "oqa":
        return row["question_stem"]
    if ds_name == "wg":
        return row["sentence"]


def create_multi_choice_options(row, ds_name):
    options_texts = []
    content = extract_input_content(ds_name, row)
    if ds_name == "boolq":
        choices = ["true", "false"]
    if ds_name == "hswag":
        choices = row["endings"]
    if (ds_name == "arc-challenge") or (ds_name == "arc-easy"):
        choices = row["choices"]["text"]
    if ds_name == "wg":
        choices = [row["option1"], row["option2"]]
    if ds_name == "oqa":
        choices = row["choices"]["text"]

    for choice in choices:
        options_texts.append(f"<|user|>\n{content}<|end|>\n<|assistant|>{choice}<|end|>\n")

    return options_texts


def extract_multi_choice_target_index(row, ds_name):
    if ds_name == "boolq":
        return 0 if row["answer"] is True else 1
    if ds_name == "hswag":
        return int(row["label"])
    if (ds_name == "arc-challenge") or (ds_name == "arc-easy"):
        return row["choices"]["label"].index(row["answerKey"])
    if ds_name == "wg":
        return int(row["answer"]) - 1
    if ds_name == "oqa":
        return row["choices"]["label"].index(row["answerKey"])


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)


def compute_loglike_loss(logits, labels, reduction="none"):
    bs = logits.size(0)
    vocab_size = logits.size(-1)
    labels = labels.squeeze(-1)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction=reduction)
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)

    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)

    # reshape back
    if reduction == "none":
        loss = loss.view((bs, -1))
        non_zero_loss = (loss != 0).sum(dim=-1)
        non_zero_loss[non_zero_loss == 0] = 1
        loss = loss.sum(dim=-1) / non_zero_loss

    return loss.float()  # Convert to float32 before returning


def evaluate_on_multi_choice_batched(
    eval_dataset, model, tokenizer, ds_name, labels, predictions, args, batch_size=32, max_length=512, device="auto"
):
    # Local import to mirror your original function
    model.eval()

    if device == "auto":
        device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
    else:
        device = torch.device(device)

    for start in tqdm(
        range(0, len(eval_dataset), batch_size), total=(len(eval_dataset) + batch_size - 1) // batch_size
    ):
        rows = [eval_dataset[i] for i in range(start, min(start + batch_size, len(eval_dataset)))]

        # Build the flattened option texts for this batch
        all_texts = []
        options_per_sample = []  # number of options for each sample
        ctx_lens_per_option = []  # context length replicated per option

        for row in rows:
            # options: ["<|user|>...<|assistant|>choiceA<|end|>", ...]
            options = create_multi_choice_options(row, ds_name)
            options_per_sample.append(len(options))

            # compute context length once per sample (align with your -1 shift)
            content = extract_input_content(ds_name, row)
            context_prompt = f"<|user|>\n{content}<|end|>\n<|assistant|>"
            ctx_len = len(tokenizer.encode(context_prompt)) - 1

            all_texts.extend(options)
            ctx_lens_per_option.extend([ctx_len] * len(options))

            # collect gold label
            labels.append(extract_multi_choice_target_index(row, ds_name))

        # Tokenize all options in one go
        tokenized = tokenizer(
            all_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        # Create masked labels: ignore context and padding
        masked_labels = tokenized["input_ids"].clone()
        for i, ctx_len in enumerate(ctx_lens_per_option):
            masked_labels[i, :ctx_len] = -100
        masked_labels[tokenized["attention_mask"] == 0] = -100

        with torch.no_grad():
            logits = model(input_ids=tokenized["input_ids"], attention_mask=tokenized["attention_mask"]).logits
            # per-sequence losses
            losses = compute_loglike_loss(logits, masked_labels, reduction="none").detach().cpu()

        # Reduce per sample (argmin across its options)
        idx = 0
        for n_opt in options_per_sample:
            pred = torch.argmin(losses[idx : idx + n_opt]).item()
            predictions.append(pred)
            idx += n_opt

    print(
        f"Accuracy for dataset {args.ds_name} and strategy {args.strategy} is: {accuracy_score(labels, predictions)}"
    )


if __name__ == "__main__":
    args = parse_args()
    print(f"Selected strategy: {args.strategy}")
    print(f"Dataset name: {args.ds_name}")

    # Loading the tokeniser
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        padding_side="right",
        model_max_length=MODEL_MAX_LEN,
    )

    # Quantisation config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    # Loading the model
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
    )

    # Loading the test dataset
    test_dataset = read_test_dataset(args.ds_name)
    print(f"{args.ds_name} is loaded with size: {len(test_dataset)}.")

    labels, predictions = [], []
    if args.strategy == "base":
        # Batch-wise inference
        with torch.no_grad():
            evaluate_on_multi_choice_batched(
                test_dataset,
                base_model,
                tokenizer,
                args.ds_name,
                labels,
                predictions,
                args,
                batch_size=64,  # tune this
                max_length=512,  # tune if options are long
                device="auto",
            )
    else:
        general_adapter_paths = []
        if args.strategy == "gks":
            arrow_config = ArrowConfig(
                top_k=3,
                router_temperature=1.0,
                use_gks=True,
            )
            # General adapter paths from the hub
            general_adapter_paths = [
                "TahaBa/phi3-mini-general-adapters/cluster0_batch16_prop1.0_langen/checkpoint-17",
                "TahaBa/phi3-mini-general-adapters/cluster0_batch16_prop1.0_langfr/checkpoint-35",
                "TahaBa/phi3-mini-general-adapters/cluster0_batch16_prop1.0_langger/checkpoint-17",
            ]
        else:
            arrow_config = ArrowConfig(
                top_k=3,
                router_temperature=1.0,
            )

        # Task-specific adapter paths from the hub
        task_specific_adapter_paths = [f"TahaBa/phi3-mini-clustered-flan/ts_expert_{i}" for i in range(10)]

        # Creating the Arrow model
        model = create_arrow_model(
            base_model=base_model,
            task_specific_adapter_paths=task_specific_adapter_paths,
            general_adapter_paths=general_adapter_paths,
            arrow_config=arrow_config,
        )

        # Batch-wise inference
        with torch.no_grad():
            evaluate_on_multi_choice_batched(
                test_dataset,
                model,
                tokenizer,
                args.ds_name,
                labels,
                predictions,
                args,
                batch_size=32,  # tune this
                max_length=512,  # tune if options are long
                device="auto",
            )
