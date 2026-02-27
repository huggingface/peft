#!/usr/bin/env python3
"""
Transformer Engine ESM2 LoRA fine-tuning example using Hugging Face Trainer.

This script demonstrates:
1. Loading a Transformer Engine-based ESM2 token classification model
2. Applying LoRA adapters with PEFT
3. Building a synthetic protein token-classification dataset in code
4. Training with the Hugging Face Trainer (no DDP setup required)
"""

import os


# TE-backed models are incompatible with Trainer's DataParallel wrapping.
# Pin to a single GPU before torch is imported so torch.cuda.device_count() == 1.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, TaskType, get_peft_model


SS3_ID2LABEL = {0: "H", 1: "E", 2: "C"}
SS3_LABEL2ID = {label: idx for idx, label in SS3_ID2LABEL.items()}

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
HELIX_AA = set("AELMQKRH")
BETA_AA = set("VIFYWT")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Transformer Engine ESM2 LoRA fine-tuning with Hugging Face Trainer")

    parser.add_argument(
        "--base_model",
        type=str,
        default="nvidia/esm2_t6_8M_UR50D",
        help="Transformer Engine ESM2 model name or path",
    )
    parser.add_argument("--output_dir", type=str, default="./esm2_lora_output", help="Output directory")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--num_train_samples", type=int, default=256, help="Number of synthetic training samples")
    parser.add_argument("--num_eval_samples", type=int, default=64, help="Number of synthetic evaluation samples")
    parser.add_argument("--min_seq_len", type=int, default=32, help="Minimum sequence length")
    parser.add_argument("--max_seq_len", type=int, default=96, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging frequency (steps)")
    parser.add_argument("--eval_steps", type=int, default=25, help="Evaluation frequency (steps)")
    parser.add_argument("--save_steps", type=int, default=25, help="Save frequency (steps)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=False,
        help="Allow loading model code from the Hub. Required for models like nvidia/esm2_*.",
    )
    parser.add_argument(
        "--train_parquet",
        type=str,
        default=None,
        help="Path to a training parquet file with Sequence and Secondary_structure columns. "
        "When provided, the synthetic dataset is not generated.",
    )
    parser.add_argument(
        "--val_parquet",
        type=str,
        default=None,
        help="Path to a validation parquet file with the same schema as --train_parquet.",
    )

    args = parser.parse_args()

    if bool(args.train_parquet) != bool(args.val_parquet):
        parser.error("--train_parquet and --val_parquet must both be provided or both omitted.")

    return args


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_synthetic_sequences(num_samples: int, min_len: int, max_len: int):
    """Generate random amino-acid sequences."""
    sequences = []
    for _ in range(num_samples):
        length = random.randint(min_len, max_len)
        sequences.append("".join(random.choices(AMINO_ACIDS, k=length)))
    return sequences


def ss_char_to_label(char: str) -> int:
    """Map a single secondary structure character to a label id (H/E/C)."""
    return SS3_LABEL2ID.get(char, SS3_LABEL2ID["C"])


def tokenize_and_align_labels(sequences, label_strings, tokenizer, max_length: int):
    """Tokenize protein sequences and align per-residue label strings to token positions."""
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for sequence, label_str in zip(sequences, label_strings):
        encoded = tokenizer(
            sequence,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        labels = [-100] * len(input_ids)
        usable_len = min(len(sequence), len(label_str), len(input_ids) - 2)
        for idx in range(usable_len):
            labels[idx + 1] = ss_char_to_label(label_str[idx])

        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(labels)

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels,
    }


def load_parquet_dataset(train_path: str, val_path: str, tokenizer, max_length: int):
    """Load train/val parquet files and return tokenized Datasets."""
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    train_dataset = Dataset.from_pandas(train_df).map(
        lambda ex: tokenize_and_align_labels(ex["Sequence"], ex["Secondary_structure"], tokenizer, max_length),
        batched=True,
        remove_columns=train_df.columns.tolist(),
    )
    val_dataset = Dataset.from_pandas(val_df).map(
        lambda ex: tokenize_and_align_labels(ex["Sequence"], ex["Secondary_structure"], tokenizer, max_length),
        batched=True,
        remove_columns=val_df.columns.tolist(),
    )
    return train_dataset, val_dataset


def compute_metrics(eval_pred):
    """Compute token accuracy while ignoring -100 labels."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    mask = labels != -100
    correct = (predictions == labels) & mask
    total_tokens = mask.sum()
    accuracy = float(correct.sum() / total_tokens) if total_tokens > 0 else 0.0
    return {"token_accuracy": accuracy}


def residue_to_ss_char(aa: str) -> str:
    """Map an amino acid to a synthetic secondary structure character (H/E/C)."""
    if aa in HELIX_AA:
        return "H"
    if aa in BETA_AA:
        return "E"
    return "C"


def sequence_to_synthetic_labels(sequence: str) -> str:
    """Derive a synthetic per-residue label string from an amino acid sequence."""
    return "".join(residue_to_ss_char(aa) for aa in sequence)


def make_synthetic_dataset(
    tokenizer,
    num_train_samples: int,
    num_eval_samples: int,
    min_seq_len: int,
    max_seq_len: int,
    max_length: int,
):
    """Create a synthetic train/eval dataset for token classification."""
    train_sequences = build_synthetic_sequences(num_train_samples, min_seq_len, max_seq_len)
    eval_sequences = build_synthetic_sequences(num_eval_samples, min_seq_len, max_seq_len)

    train_labels = [sequence_to_synthetic_labels(s) for s in train_sequences]
    eval_labels = [sequence_to_synthetic_labels(s) for s in eval_sequences]

    train_dataset = Dataset.from_dict({"sequence": train_sequences, "labels_str": train_labels}).map(
        lambda ex: tokenize_and_align_labels(ex["sequence"], ex["labels_str"], tokenizer, max_length),
        batched=True,
        remove_columns=["sequence", "labels_str"],
    )
    eval_dataset = Dataset.from_dict({"sequence": eval_sequences, "labels_str": eval_labels}).map(
        lambda ex: tokenize_and_align_labels(ex["sequence"], ex["labels_str"], tokenizer, max_length),
        batched=True,
        remove_columns=["sequence", "labels_str"],
    )
    return train_dataset, eval_dataset


def main():
    args = parse_args()
    set_seed(args.seed)

    if not args.trust_remote_code and "esm2" in args.base_model.lower():
        raise ValueError(
            f"Model '{args.base_model}' requires remote code execution. "
            "Re-run with --trust_remote_code to confirm you trust this model's code."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model_dtype = torch.bfloat16 if use_bf16 else torch.float32

    print(f"Loading tokenizer and model from: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)

    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    config.num_labels = 3
    config.id2label = SS3_ID2LABEL
    config.label2id = SS3_LABEL2ID

    model = AutoModelForTokenClassification.from_pretrained(
        args.base_model,
        config=config,
        trust_remote_code=args.trust_remote_code,
        dtype=model_dtype,
    )

    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["layernorm_qkv"],
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if args.train_parquet and args.val_parquet:
        print(f"Loading parquet datasets: train={args.train_parquet}, val={args.val_parquet}")
        train_dataset, eval_dataset = load_parquet_dataset(
            train_path=args.train_parquet,
            val_path=args.val_parquet,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )
    else:
        print("Building synthetic dataset...")
        train_dataset, eval_dataset = make_synthetic_dataset(
            tokenizer=tokenizer,
            num_train_samples=args.num_train_samples,
            num_eval_samples=args.num_eval_samples,
            min_seq_len=args.min_seq_len,
            max_seq_len=args.max_seq_len,
            max_length=args.max_length,
        )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="token_accuracy",
        greater_is_better=True,
        report_to="none",
        remove_unused_columns=False,
        bf16=use_bf16,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    final_metrics = trainer.evaluate()
    print(f"Final evaluation metrics: {final_metrics}")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter and tokenizer to: {args.output_dir}")


if __name__ == "__main__":
    main()
