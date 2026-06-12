#!/usr/bin/env python
"""Re-evaluate existing checkpoints with more test samples.

Usage:
    python scripts/reeval.py [--eval-size 500] [--filter PATTERN] [--batch-size 8]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent dir so we can import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data import load_rl_datasets
from reward import extract_boxed, safe_grade
from run import _load_eval_model, _remap_adapter_keys, evaluate_pass_at_1
from utils import load_train_config

from peft import PeftConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--filter", type=str, default="")
    parser.add_argument("--force", action="store_true", help="Re-eval even if already done at this size")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
    results_dirs = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "temporary_results"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "results"),
    ]

    # Find all result files
    result_files = []
    for rdir in results_dirs:
        result_files.extend(glob.glob(os.path.join(rdir, "*.json")))

    for result_path in sorted(result_files):
        with open(result_path) as f:
            result = json.load(f)

        experiment_name = result["run_info"]["experiment_name"]
        status = result["train_info"]["status"]

        if status != "success":
            continue
        if args.filter and args.filter not in experiment_name:
            continue

        # Check if already evaluated at this size
        existing_eval = result.get("rl_eval_info", {})
        if not args.force and existing_eval.get("num_eval_samples") == args.eval_size:
            print(f"[SKIP] {experiment_name} (already eval'd at {args.eval_size} samples)")
            continue

        # Find checkpoint
        ckpt_name = experiment_name.replace("/", "--")
        eval_ckpt = os.path.join(checkpoint_base, ckpt_name, "eval_checkpoint")
        if not os.path.exists(eval_ckpt):
            print(f"[SKIP] {experiment_name} (no checkpoint at {eval_ckpt})")
            continue

        # Load config
        exp_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments", experiment_name)
        cfg = load_train_config(exp_path)

        # Load peft config
        adapter_path = os.path.join(exp_path, "adapter_config.json")
        peft_cfg = None
        if os.path.exists(adapter_path):
            with open(adapter_path) as f:
                cfg_dict = json.load(f)
            peft_cfg = PeftConfig.from_peft_type(**cfg_dict)

        print(f"[EVAL] {experiment_name} ({args.eval_size} samples)...", end=" ", flush=True)

        try:
            # Load test dataset
            _, test_ds = load_rl_datasets(
                dataset_name=cfg.dataset_name,
                dataset_config=cfg.dataset_config,
                train_split=cfg.dataset_train_split,
                test_split=cfg.dataset_test_split,
                train_subset_size=0,
                eval_subset_size=args.eval_size,
                seed=cfg.seed,
            )

            tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = _load_eval_model(
                model_id=cfg.model_id,
                checkpoint_dir=eval_ckpt,
                peft_cfg=peft_cfg,
                dtype_name=cfg.dtype,
            )

            pass_at_1 = evaluate_pass_at_1(
                model=model,
                tokenizer=tokenizer,
                dataset=test_ds,
                max_completion_length=cfg.max_completion_length,
                batch_size=args.batch_size,
            )

            print(f"pass@1={pass_at_1:.3f} ({len(test_ds)} samples)")

            # Update result
            result["rl_eval_info"] = {
                "test_pass_at_1": pass_at_1,
                "num_eval_samples": len(test_ds),
            }
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)

        except Exception as e:
            print(f"FAILED: {e}")
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
