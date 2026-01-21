#!/usr/bin/env python
# coding: utf-8

"""
Uni-LoRA GLUE head_lr sweep script (reproducible, dataloader-style)

Features:
- model_name from command line: roberta-base / roberta-large
- task from command line: cola / sst2 / mrpc / qnli / rte / stsb
- max_length:
    - roberta-base  -> 512
    - roberta-large -> 128
- num_epochs follows the table (base/large × task)
- theta_d_lr fixed to 5e-3
- head_lr from command line
- warmup_ratio fixed to 0.06, linear schedule
- track best validation metric (task-dependent) across epochs
- save best result to JSON
"""

import os
import json
import argparse
import random
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

import evaluate
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from peft import get_peft_model
from peft import UniLoraConfig, PeftType


# =========================
# Reproducibility
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# GLUE config
# =========================
GLUE_TASKS = ["cola", "sst2", "mrpc", "qnli", "rte", "stsb"]

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
}

# max seq length rule
MAX_LENGTH = {
    "roberta-base": 512,
    "roberta-large": 128,
}

# epochs from your table
EPOCHS = {
    "roberta-base": {
        "sst2": 60,
        "mrpc": 30,
        "cola": 80,
        "qnli": 25,
        "rte": 160,
        "stsb": 80,
    },
    "roberta-large": {
        "sst2": 20,
        "mrpc": 40,
        "cola": 40,
        "qnli": 20,
        "rte": 40,
        "stsb": 40,
    },
}

# best-tracking metric per task (as you used in table logic)
TASK_TO_METRIC = {
    "cola": "matthews_correlation",
    "sst2": "accuracy",
    "mrpc": "accuracy",   # table通常用accuracy（如你要f1可改这里）
    "qnli": "accuracy",
    "rte": "accuracy",
    "stsb": "pearson",
}


def parse_args():
    parser = argparse.ArgumentParser()

    # === 基础模型设置（默认 roberta-base） ===
    parser.add_argument(
        "--model_name",
        type=str,
        default="roberta-base",
        choices=["roberta-base", "roberta-large"],
        help="Base model type"
    )

    # === GLUE task 设置（默认 SST-2 或你喜欢的一个）===
    parser.add_argument(
        "--task",
        type=str,
        default="mrpc",
        choices=GLUE_TASKS,
        help="GLUE task name"
    )

    # === 学习率默认值（示例设为 1e-3）===
    parser.add_argument(
        "--head_lr",
        type=float,
        default=1e-3,
        help="Learning rate for the classification head"
    )

    # === 随机种子（默认 42）===
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    # === 输出目录 ===
    parser.add_argument(
        "--out_dir", "--output_dir",
        dest="out_dir",
        type=str,
        default="results_glue",
        help="Output directory"
    )

    # === UniLoRA 相关超参（保持你原来的默认值）===
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--theta_d_length", type=int, default=23040)
    parser.add_argument("--init_theta_d_bound", type=float, default=0.02)
    parser.add_argument("--unilora_dropout", type=float, default=0.0)

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    model_name = args.model_name
    task = args.task

    if model_name not in MAX_LENGTH:
        raise ValueError(f"Unsupported model_name: {model_name}")
    if task not in GLUE_TASKS:
        raise ValueError(f"Unsupported task: {task}")

    batch_size = 32
    max_length = MAX_LENGTH[model_name]
    num_epochs = EPOCHS[model_name][task]

    theta_d_lr = 5e-3
    warmup_ratio = 0.06

    # UniLoRA params
    rank = args.rank
    theta_d_length = args.theta_d_length
    proj_seed = args.seed
    init_theta_d_bound = args.init_theta_d_bound

    device = "cuda" if torch.cuda.is_available() else "cpu"

    metric_name = TASK_TO_METRIC[task]

    print("=" * 80)
    print("Run config:")
    print(f"  model_name        = {model_name}")
    print(f"  task              = {task}")
    print(f"  seed              = {args.seed}")
    print(f"  head_lr           = {args.head_lr}")
    print(f"  theta_d_lr        = {theta_d_lr} (fixed)")
    print(f"  batch_size        = {batch_size}")
    print(f"  max_length        = {max_length}")
    print(f"  num_epochs        = {num_epochs}")
    print(f"  warmup_ratio      = {warmup_ratio}")
    print(f"  metric_for_best   = {metric_name}")
    print(f"  out_dir           = {args.out_dir}")
    print("=" * 80)

    # =========================
    # Data
    # =========================
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        # RoBERTa usually has pad_token_id, but keep safe like your script
        tokenizer.pad_token_id = tokenizer.eos_token_id

    datasets = load_dataset("glue", task)

    s1_key, s2_key = TASK_TO_KEYS[task]

    def tokenize_fn(examples):
        if s2_key is None:
            return tokenizer(
                examples[s1_key],
                truncation=True,
                padding="max_length",   # ✅ 强制 pad
                max_length=max_length,
            )
        return tokenizer(
            examples[s1_key],
            examples[s2_key],
            truncation=True,
            padding="max_length",       # ✅ 强制 pad
            max_length=max_length,
        )

    # remove ONLY text columns + idx (keep label)
    remove_cols = []
    for col in ["idx", s1_key, s2_key]:
        if col is not None and col in datasets["train"].column_names:
            remove_cols.append(col)

    datasets = datasets.map(
        tokenize_fn,
        batched=True,
        remove_columns=remove_cols,
    )

    # HF model expects "labels"
    if "label" in datasets["train"].column_names:
        datasets = datasets.rename_column("label", "labels")

    def collate_fn(examples):
        return tokenizer.pad(examples, return_tensors="pt")

    train_loader = DataLoader(
        datasets["train"],
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        datasets["validation"],
        shuffle=False,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )

    # =========================
    # Model + UniLoRA
    # =========================
    num_labels = 1 if task == "stsb" else 2
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        return_dict=True,
    )

    # NOTE: Keep the same target_modules/modules_to_save pattern as your working CoLA script.
    peft_config = UniLoraConfig(
        task_type="SEQ_CLS",
        peft_type=PeftType.UNILORA,
        r=rank,
        theta_d_length=theta_d_length,
        proj_seed=proj_seed,
        init_theta_d_bound=init_theta_d_bound,
        unilora_dropout=args.unilora_dropout,
        target_modules=[
            "query", "key", "value",
            "output.dense", "intermediate.dense",
        ],
    )

    # model = get_peft_model(base_model, peft_config)
    # model.to(device)

    # =========================
    # Optimizer & Scheduler
    # =========================
    
   


    # =========================
    # Train / Eval (best tracking)
    # =========================
    best_score = -1e18
    best_epoch = -1
    best_metric = None

    # === build one real batch from train_loader ===
    # model.eval()   # 不需要训练模式
    batch = next(iter(train_loader))

# 把 batch 放到正确 device
    batch = {k: v.to(device) for k, v in batch.items()}

# 配置 inputs（只保留 forward 需要的字段）
    inputs = {}
    for k in ["input_ids", "attention_mask", "token_type_ids", "labels"]:
        if k in batch:
            inputs[k] = batch[k]

    # === BASE 输出（参考值）===
    base_model.to(device)
    base_model.eval()
    with torch.no_grad():
        base_out = base_model(**inputs).logits.clone()

    # === PEFT model ===
    model = get_peft_model(base_model, peft_config)
    model.to(device)
    model.eval()

    # === PEFT输出（未禁用 adapter 情况）===
    with torch.no_grad():
        peft_out = model(**inputs).logits

    # === mismatch 检测 ===
    diff = (peft_out - base_out).abs().max().item()
    print(f"[UniLoRA mismatch test] max abs diff = {diff:.8f}")

    ok = torch.allclose(peft_out, base_out)
    print(f"[UniLoRA mismatch test] allclose base vs peft = {ok}")


    # for epoch in range(num_epochs):
    #     # ---- train ----
    #     model.train()
    #     for batch in tqdm(train_loader, desc=f"Train epoch {epoch}", leave=False):
    #         batch = {k: v.to(device) for k, v in batch.items()}
    #         out = model(**batch)
    #         loss = out.loss
    #         loss.backward()
    #         optimizer.step()
    #         scheduler.step()
    #         optimizer.zero_grad()

    #     # ---- eval ----
    #     model.eval()
    #     metric = evaluate.load("glue", task)

    #     for batch in eval_loader:
    #         batch = {k: v.to(device) for k, v in batch.items()}
    #         with torch.no_grad():
    #             logits = model(**batch).logits

    #         if task == "stsb":
    #             # regression: predictions must be float
    #             preds = logits.squeeze(-1).detach().cpu().numpy().astype(float)
    #             refs = batch["labels"].detach().cpu().numpy().astype(float)
    #             metric.add_batch(predictions=preds, references=refs)
    #         else:
    #             preds = logits.argmax(dim=-1)
    #             metric.add_batch(predictions=preds, references=batch["labels"])

    #     eval_metric = metric.compute()
    #     score = eval_metric[metric_name]

    #     print(
    #         f"[model={model_name} task={task} lr={args.head_lr:.0e} seed={args.seed}] "
    #         f"epoch {epoch}: {eval_metric}"
    #     )

    #     if score > best_score:
    #         best_score = score
    #         best_epoch = epoch
    #         best_metric = eval_metric

    # =========================
    # Save best result
    # =========================
    # os.makedirs(args.out_dir, exist_ok=True)

    # result = {
    #     "task": task,
    #     "model": model_name,
    #     "head_lr": args.head_lr,
    #     "theta_d_lr": theta_d_lr,
    #     "seed": args.seed,
    #     "proj_seed": proj_seed,
    #     "batch_size": batch_size,
    #     "max_length": max_length,
    #     "num_epochs": num_epochs,
    #     "warmup_ratio": warmup_ratio,
    #     "metric_for_best": metric_name,
    #     "best_score": best_score,
    #     "best_epoch": best_epoch,
    #     "best_metric": best_metric,
    #     "rank": rank,
    #     "theta_d_length": theta_d_length,
    #     "init_theta_d_bound": init_theta_d_bound,
    #     "unilora_dropout": args.unilora_dropout,
    # }

    # out_path = os.path.join(
    #     args.out_dir,
    #     f"{task}_{model_name}_lr_{args.head_lr:.0e}_seed_{args.seed}.json",
    # )

    # with open(out_path, "w") as f:
    #     json.dump(result, f, indent=2)

    # print("Saved best result to:", out_path)

    # adapter_ckpt_dir = os.path.join(args.out_dir, "adapter_ckpt")
    # model.save_pretrained(adapter_ckpt_dir)
    # print("Adapter saved to:", adapter_ckpt_dir)

    # # ========= 加载并捕获 warnings 来复现 pytest 的行为 =========
    # from peft import PeftModel
    # import warnings

    # print("=== Try to reload adapter and capture warnings ===")
    # base2 = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # with warnings.catch_warnings(record=True) as recs:
    #     model2 = PeftModel.from_pretrained(base2, adapter_ckpt_dir)

    # for w in recs:
    #     print("[warning]", str(w.message))


if __name__ == "__main__":
    main()