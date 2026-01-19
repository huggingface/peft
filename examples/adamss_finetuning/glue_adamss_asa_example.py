"""
GLUE Task Fine-tuning with AdaMSS and ASA

This script demonstrates how to fine-tune RoBERTa on GLUE tasks using AdaMSS
with Adaptive Subspace Allocation (ASA) for efficient parameter updates.

Example usage:
    # CoLA with RoBERTa-base, 100 epochs
    python glue_adamss_asa_example.py --dataset_name cola --num_epochs 100 --seed 0

    # With ASA enabled (K: 10→5)
    python glue_adamss_asa_example.py --dataset_name cola --num_epochs 100 --use_asa --asa_target_subspaces 5

    # MRPC with RoBERTa-large
    python glue_adamss_asa_example.py --dataset_name mrpc --model_name_or_path roberta-large --num_epochs 10

Requirements:
    pip install peft transformers datasets torch evaluate scikit-learn
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

import torch
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    EvalPrediction,
)
from torch.optim import AdamW

from peft import AdamssConfig, get_peft_model, AdamssASACallback


# Table 19: Hyperparameters for GLUE tasks (from paper)
HYPERPARAMS = {
    "roberta-base": {
        "sst2": {"lr": 0.001, "head_lr": 0.005, "wd": 0.0005},
        "mrpc": {"lr": 0.01, "head_lr": 0.0005, "wd": 0.0},
        "cola": {"lr": 0.001, "head_lr": 0.005, "wd": 0.005},
        "qnli": {"lr": 0.001, "head_lr": 0.005, "wd": 0.005},
        "rte": {"lr": 0.0005, "head_lr": 0.005, "wd": 0.005},
        "stsb": {"lr": 0.001, "head_lr": 0.005, "wd": 0.005},
    },
    "roberta-large": {
        "sst2": {"lr": 0.001, "head_lr": 0.0005, "wd": 0.0},
        "mrpc": {"lr": 0.001, "head_lr": 0.00005, "wd": 0.005},
        "cola": {"lr": 0.005, "head_lr": 0.0005, "wd": 0.1},
        "qnli": {"lr": 0.0005, "head_lr": 0.05, "wd": 0.005},
        "rte": {"lr": 0.005, "head_lr": 0.005, "wd": 0.5},
        "stsb": {"lr": 0.001, "head_lr": 0.0005, "wd": 0.0005},
    },
}

TASK_METRICS = {
    "cola": "matthews_correlation",
    "stsb": "pearson",
    "mrpc": "accuracy",
    "qqp": "accuracy",
    "sst2": "accuracy",
    "qnli": "accuracy",
    "rte": "accuracy",
}


@dataclass
class AdaMSSTrainingArguments:
    """Arguments for AdaMSS training on GLUE tasks."""
    
    # Model and Dataset
    model_name_or_path: str = field(
        default="roberta-base",
        metadata={"help": "Model identifier: roberta-base or roberta-large"}
    )
    dataset_name: str = field(
        default="cola",
        metadata={"help": "GLUE task: cola, mrpc, sst2, qnli, rte, stsb"}
    )
    
    # AdaMSS Configuration
    adamss_r: int = field(default=100, metadata={"help": "SVD rank"})
    adamss_k: int = field(default=10, metadata={"help": "Number of subspaces (K)"})
    adamss_ri: int = field(default=1, metadata={"help": "Subspace rank (rk), use 1 for NLU"})
    
    # ASA Configuration
    use_asa: bool = field(default=False, metadata={"help": "Enable Adaptive Subspace Allocation"})
    asa_target_subspaces: int = field(default=5, metadata={"help": "Target active subspaces for ASA"})
    asa_init_warmup: int = field(default=5, metadata={"help": "ASA init warmup in EPOCHS"})
    asa_final_warmup: int = field(default=95, metadata={"help": "ASA final warmup in EPOCHS"})
    asa_mask_interval: int = field(default=10, metadata={"help": "ASA mask interval in EPOCHS"})
    asa_importance_beta: float = field(default=0.85, metadata={"help": "EMA coefficient for importance"})
    asa_uncertainty_beta: float = field(default=0.85, metadata={"help": "EMA coefficient for uncertainty"})
    asa_schedule_exponent: float = field(default=3.0, metadata={"help": "ASA schedule exponent"})
    
    # Training Configuration
    num_epochs: int = field(default=100, metadata={"help": "Number of training epochs"})
    batch_size: int = field(default=32, metadata={"help": "Batch size per device"})
    max_length: int = field(default=512, metadata={"help": "Maximum sequence length"})
    warmup_ratio: float = field(default=0.06, metadata={"help": "Warmup ratio"})
    
    # Other
    seed: int = field(default=0, metadata={"help": "Random seed"})
    output_dir: str = field(default="./output", metadata={"help": "Output directory"})
    cache_dir: str = field(default="./cache", metadata={"help": "Cache directory"})


def get_dataset(args: AdaMSSTrainingArguments, tokenizer):
    """Load and tokenize GLUE dataset."""
    # Load dataset
    raw_datasets = load_dataset("glue", args.dataset_name, cache_dir=args.cache_dir)
    
    # Get sentence keys
    sentence_keys = {
        "cola": ("sentence", None),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
    }
    sentence1_key, sentence2_key = sentence_keys[args.dataset_name]
    
    # Tokenize
    def preprocess_function(examples):
        texts = (
            (examples[sentence1_key],) if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, truncation=True, max_length=args.max_length, padding="max_length")
        result["labels"] = examples["label"]
        return result
    
    # Remove original text columns but keep label
    columns_to_remove = [col for col in raw_datasets["train"].column_names if col != "label"]
    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=columns_to_remove,
        desc="Tokenizing dataset",
    )
    
    train_ds = tokenized_datasets["train"]
    val_ds = tokenized_datasets["validation"]
    test_ds = tokenized_datasets["validation"]  # Use validation as test for GLUE
    
    return train_ds, val_ds, test_ds


def main():
    # Parse arguments
    parser = HfArgumentParser(AdaMSSTrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Get model short name
    model_short = "roberta-large" if "large" in args.model_name_or_path else "roberta-base"
    
    # Get hyperparameters from Table 19
    if model_short in HYPERPARAMS and args.dataset_name in HYPERPARAMS[model_short]:
        hp = HYPERPARAMS[model_short][args.dataset_name]
    else:
        hp = {"lr": 0.001, "head_lr": 0.005, "wd": 0.005}
        print(f"Using default hyperparameters for {model_short}/{args.dataset_name}")
    
    # Print configuration
    print("=" * 80)
    print(f"AdaMSS {'with ASA' if args.use_asa else 'without ASA'} - GLUE: {args.dataset_name.upper()}")
    print("=" * 80)
    print(f"  Model: {model_short}")
    print(f"  AdaMSS: r={args.adamss_r}, K={args.adamss_k}, ri={args.adamss_ri}")
    if args.use_asa:
        print(f"  ASA: K={args.adamss_k} → target={args.asa_target_subspaces}")
    print(f"  Hyperparameters (Table 19): lr={hp['lr']}, head_lr={hp['head_lr']}, wd={hp['wd']}")
    print(f"  Training: {args.num_epochs} epochs, batch_size={args.batch_size}, seed={args.seed}")
    print("=" * 80 + "\n")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    
    # Load dataset
    print(f"Loading {args.dataset_name} dataset...")
    train_ds, val_ds, test_ds = get_dataset(args, tokenizer)
    
    # Determine task type
    is_regression = args.dataset_name == "stsb"
    if not is_regression:
        label_list = train_ds.features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1
    
    print(f"Dataset loaded - Task: {'regression' if is_regression else 'classification'}\n")
    
    # Load model
    print(f"Loading {model_short}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=args.cache_dir,
    )
    
    # Convert epoch-based ASA parameters to step-based (before config creation)
    steps_per_epoch = len(train_ds) // args.batch_size
    if len(train_ds) % args.batch_size != 0:
        steps_per_epoch += 1
    
    asa_init_warmup_steps = args.asa_init_warmup * steps_per_epoch
    asa_final_warmup_steps = args.asa_final_warmup * steps_per_epoch
    asa_mask_interval_steps = args.asa_mask_interval * steps_per_epoch
    
    # Apply AdaMSS
    print("\nApplying AdaMSS...")
    config = AdamssConfig(
        r=args.adamss_r,
        num_subspaces=args.adamss_k,
        subspace_rank=args.adamss_ri,
        target_modules=["query", "value"],
        use_asa=args.use_asa,
        asa_target_subspaces=args.asa_target_subspaces if args.use_asa else None,
        init_warmup=asa_init_warmup_steps if args.use_asa else None,
        final_warmup=asa_final_warmup_steps if args.use_asa else None,
        mask_interval=asa_mask_interval_steps if args.use_asa else None,
        asa_importance_beta=args.asa_importance_beta if args.use_asa else None,
        asa_uncertainty_beta=args.asa_uncertainty_beta if args.use_asa else None,
        asa_schedule_exponent=args.asa_schedule_exponent if args.use_asa else None,
        modules_to_save=["classifier"],
    )
    
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    # Print detailed parameter breakdown (same logic as exec_adamss_peft_glue.py)
    print("\n[Detailed Parameter Breakdown]")
    head_params = [p for n, p in model.named_parameters() if ("classifier" in n or "score" in n) and p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if ("classifier" not in n and "score" not in n) and p.requires_grad]
    head_count = sum(p.numel() for p in head_params)
    adapter_count = sum(p.numel() for p in other_params)
    print(f"Classifier Head Params: {head_count:,}")
    print(f"AdaMSS Adapter Params:  {adapter_count:,}")
    print(f"Total Trainable Params: {head_count + adapter_count:,}")
    
    # Debug: print parameter names to verify
    if adapter_count == 0:
        print("\nWARNING: No AdaMSS parameters found!")
        print("All trainable parameter names:")
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f"  {n}: {p.numel():,} params")
    
    # GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"\n[GPU Memory - Before Training]")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # Setup ASA callback
    callbacks = []
    if args.use_asa:
        print("\nSetting up ASA callback...")
        
        print(f"\n[ASA Configuration]")
        print(f"Dataset size: {len(train_ds)}, Batch size: {args.batch_size}")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total training steps: {steps_per_epoch * args.num_epochs}")
        print(f"ASA warmup (epochs → steps):")
        print(f"  init_warmup: {args.asa_init_warmup} epochs → {asa_init_warmup_steps} steps")
        print(f"  final_warmup: {args.asa_final_warmup} epochs → {asa_final_warmup_steps} steps")
        print(f"  mask_interval: {args.asa_mask_interval} epochs → {asa_mask_interval_steps} steps\n")
        
        asa_callback = AdamssASACallback(verbose=True)
        callbacks.append(asa_callback)
    
    # Training configuration
    print(f"\n[Training Configuration]")
    print(f"Dataset size: {len(train_ds)}, Batch size: {args.batch_size}")
    steps_per_epoch = len(train_ds) // args.batch_size
    if len(train_ds) % args.batch_size != 0:
        steps_per_epoch += 1
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {steps_per_epoch * args.num_epochs}")
    
    # Metrics
    metric = evaluate.load("glue", args.dataset_name)
    
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        return result
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=hp['lr'],
        weight_decay=hp['wd'],
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=TASK_METRICS.get(args.dataset_name, "accuracy"),
        greater_is_better=True,
        logging_steps=100,
        seed=args.seed,
        report_to="none",
    )
    
    # Custom optimizer with different LR for head
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if ("classifier" in n or "score" in n) and p.requires_grad],
            "lr": hp['head_lr'],
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if ("classifier" not in n and "score" not in n) and p.requires_grad],
            "lr": hp['lr'],
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=hp['wd'])
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
        callbacks=callbacks,
    )
    
    # Train
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    train_result = trainer.train()
    
    # GPU memory stats
    if torch.cuda.is_available():
        print(f"\n[GPU Memory - Peak During Training]")
        print(f"Peak Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        print(f"Peak Reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
    
    # Print best metric
    if trainer.state.best_metric is not None:
        metric_name = TASK_METRICS.get(args.dataset_name, "accuracy")
        print(f"\n[Best Model Info]")
        print(f"Best {metric_name}: {trainer.state.best_metric:.4f}")
    
    # Final evaluation on validation set
    print("\n" + "="*80)
    print("Final evaluation on validation set...")
    print("="*80 + "\n")
    
    final_metrics = trainer.evaluate(val_ds)
    print(f"\nFinal Validation Results: {final_metrics}")
    
    # Save model
    trainer.save_model()
    print(f"\nModel saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
