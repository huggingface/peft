#!/usr/bin/env python
"""Evaluate a base model (no training) on a given dataset.

Usage:
    python scripts/eval_base_model.py <model_id> <dataset_name> [--eval-size 500]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import load_rl_datasets
from run import evaluate_pass_at_1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("dataset_name")
    parser.add_argument("--eval-size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading {args.model_id}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.eval()

    print(f"Loading dataset: {args.dataset_name}...", flush=True)
    _, test_ds = load_rl_datasets(
        dataset_name=args.dataset_name,
        dataset_config="main",
        train_split="train",
        test_split="test",
        train_subset_size=0,
        eval_subset_size=args.eval_size,
        seed=args.seed,
    )
    print(f"Test dataset: {len(test_ds)} samples", flush=True)

    print("Running evaluation...", flush=True)
    pass_at_1 = evaluate_pass_at_1(
        model=model,
        tokenizer=tokenizer,
        dataset=test_ds,
        max_completion_length=args.max_completion_length,
        batch_size=args.batch_size,
    )
    print(f"\nBASE MODEL pass@1: {pass_at_1:.3f} ({len(test_ds)} samples)")
    print(f"Model: {args.model_id}")
    print(f"Dataset: {args.dataset_name}")


if __name__ == "__main__":
    main()
