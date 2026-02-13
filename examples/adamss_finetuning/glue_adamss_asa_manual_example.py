"""
GLUE Task Fine-tuning with AdaMSS and Manual ASA

This script demonstrates how to manually call update_and_allocate() for ASA
instead of using AdamssASACallback. This approach is useful for custom training loops.

Note:
    This is an alternative to using AdamssASACallback. Choose ONE approach:
    - Use AdamssASACallback (recommended, see glue_adamss_asa_example.py)
    - Use manual update_and_allocate() (this script, for custom control)
    DO NOT use both together!

Example usage:
    # CoLA with RoBERTa-base and manual ASA
    python glue_adamss_asa_manual_example.py \
        --dataset_name cola \
        --use_asa \
        --asa_target_subspaces 5 \
        --num_epochs 100 \
        --batch_size 32 \
        --warmup_ratio 0.06 \
        --seed 0 \
        --output_dir ./output/cola_asa_manual

Requirements:
    pip install peft transformers datasets torch evaluate scikit-learn
"""

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
    set_seed,
)

from peft import AdaMSSConfig, get_peft_model


class CustomTrainerWithManualASA(Trainer):
    """
    Custom Trainer that manually calls update_and_allocate() for ASA.
    
    This demonstrates the manual approach as an alternative to using AdamssASACallback.
    The update_and_allocate() method is called after optimizer.step() but before
    zero_grad() to compute importance scores from gradients.
    """
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override training_step to add manual ASA update.
        
        Training step sequence:
        1. Forward pass
        2. Backward pass (gradients computed)
        3. Optimizer step (parameters updated)
        4. >>> Manual ASA update (importance scoring & masking) <<<
        5. Zero gradients
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        # Forward & backward pass
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        self.accelerator.backward(loss)
        
        # 🔑 Key: Manual ASA update after backward, before zero_grad
        # This is where update_and_allocate() inspects gradients and applies masking
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'update_and_allocate'):
            # Only call if we're actually doing optimizer step (not accumulating)
            if (self.state.global_step + 1) % self.args.gradient_accumulation_steps == 0:
                model.base_model.update_and_allocate(self.state.global_step)
        
        return loss.detach()


@dataclass
class AdaMSSArguments:
    """Arguments for AdaMSS configuration."""
    
    # Basic AdaMSS parameters
    adamss_r: int = field(
        default=100,
        metadata={"help": "SVD decomposition rank (R in paper)."}
    )
    adamss_k: int = field(
        default=10,
        metadata={"help": "Number of subspaces (K in paper)."}
    )
    adamss_ri: int = field(
        default=1,
        metadata={"help": "Subspace rank (rk in paper), typically 1 for NLU."}
    )
    
    # Training configuration
    num_epochs: int = field(
        default=100,
        metadata={"help": "Number of training epochs."}
    )
    batch_size: int = field(
        default=32,
        metadata={"help": "Batch size per device."}
    )
    warmup_ratio: float = field(
        default=0.06,
        metadata={"help": "Warmup ratio."}
    )
    seed: int = field(
        default=0,
        metadata={"help": "Random seed."}
    )
    output_dir: str = field(
        default="./output",
        metadata={"help": "Output directory."}
    )
    
    # ASA parameters
    use_asa: bool = field(
        default=False,
        metadata={"help": "Enable Adaptive Subspace Allocation (manual mode)."}
    )
    asa_target_subspaces: int = field(
        default=5,
        metadata={"help": "Target number of active subspaces when ASA is enabled."}
    )
    asa_init_warmup: int = field(
        default=5,
        metadata={"help": "ASA warmup EPOCHS before starting masking."}
    )
    asa_final_warmup: int = field(
        default=95,
        metadata={"help": "ASA EPOCHS to reach target active subspaces."}
    )
    asa_mask_interval: int = field(
        default=10,
        metadata={"help": "EPOCHS between ASA updates."}
    )
    asa_importance_beta: float = field(
        default=0.85,
        metadata={"help": "EMA coefficient for importance."}
    )
    asa_uncertainty_beta: float = field(
        default=0.85,
        metadata={"help": "EMA coefficient for uncertainty."}
    )
    asa_schedule_exponent: float = field(
        default=3.0,
        metadata={"help": "ASA schedule exponent."}
    )


@dataclass
class DataArguments:
    """Arguments for dataset configuration."""
    
    dataset_name: str = field(
        default="cola",
        metadata={"help": "GLUE task name (cola, mrpc, qnli, rte, stsb, sst2)."}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length."}
    )


# Hyperparameters from Table 19 in the paper
HYPERPARAMS = {
    "roberta-large": {
        "cola": {"lr": 0.005, "head_lr": 0.0005, "wd": 0.1},
        "mrpc": {"lr": 0.001, "head_lr": 0.00005, "wd": 0.005},
        "qnli": {"lr": 0.0005, "head_lr": 0.05, "wd": 0.005},
        "rte": {"lr": 0.005, "head_lr": 0.005, "wd": 0.5},
        "stsb": {"lr": 0.001, "head_lr": 0.0005, "wd": 0.0005},
        "sst2": {"lr": 0.001, "head_lr": 0.0005, "wd": 0.0},
    },
    "roberta-base": {
        "cola": {"lr": 0.001, "head_lr": 0.005, "wd": 0.005},
        "mrpc": {"lr": 0.01, "head_lr": 0.0005, "wd": 0.0},
        "qnli": {"lr": 0.001, "head_lr": 0.005, "wd": 0.005},
        "rte": {"lr": 0.0005, "head_lr": 0.005, "wd": 0.005},
        "stsb": {"lr": 0.001, "head_lr": 0.005, "wd": 0.005},
        "sst2": {"lr": 0.001, "head_lr": 0.005, "wd": 0.0005},
    }
}

# Metrics for each task
TASK_METRICS = {
    "cola": "matthews_correlation",
    "stsb": "pearson",
    "mrpc": "accuracy",
    "qqp": "accuracy",
    "sst2": "accuracy",
    "qnli": "accuracy",
    "rte": "accuracy",
}


def main():
    # Parse arguments
    parser = HfArgumentParser((DataArguments, AdaMSSArguments))
    data_args, adamss_args = parser.parse_args_into_dataclasses()
    
    # Set seed
    set_seed(adamss_args.seed)
    
    # Extract model name from output_dir or use default
    if "roberta-large" in str(adamss_args.output_dir).lower():
        model_name = "roberta-large"
    else:
        model_name = "roberta-base"
    
    print("=" * 80)
    print(f"AdaMSS with MANUAL ASA - GLUE Task: {data_args.dataset_name.upper()}")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"AdaMSS: r={adamss_args.adamss_r}, K={adamss_args.adamss_k}, ri={adamss_args.adamss_ri}")
    
    # Get hyperparameters
    if model_name in HYPERPARAMS and data_args.dataset_name in HYPERPARAMS[model_name]:
        hp = HYPERPARAMS[model_name][data_args.dataset_name]
        print(f"Hyperparameters (Table 19): lr={hp['lr']}, head_lr={hp['head_lr']}, wd={hp['wd']}")
    else:
        hp = {"lr": 0.001, "head_lr": 0.005, "wd": 0.005}
        print(f"Using default hyperparameters: {hp}")
    
    print(f"Training: {adamss_args.num_epochs} epochs, batch_size={adamss_args.batch_size}, seed={adamss_args.seed}")
    
    if adamss_args.use_asa:
        print(f"Manual ASA Mode: Target {adamss_args.asa_target_subspaces}/{adamss_args.adamss_k} subspaces")
        print(f"     Warmup epochs {adamss_args.asa_init_warmup} → {adamss_args.asa_final_warmup}")
        print(f"     Using update_and_allocate() instead of AdamssASACallback")
    
    # Load dataset
    print(f"\nLoading {data_args.dataset_name} dataset...")
    dataset = load_dataset("glue", data_args.dataset_name)
    
    # Get task info
    is_regression = data_args.dataset_name == "stsb"
    if not is_regression:
        label_list = dataset["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1
    
    print(f"Dataset loaded - Task type: {'regression' if is_regression else 'classification'}")
    
    # Load tokenizer and model
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    
    # Tokenize dataset
    def preprocess_function(examples):
        # Handle different GLUE tasks
        if data_args.dataset_name in ["mrpc", "stsb", "qqp"]:
            texts = (examples["sentence1"], examples["sentence2"])
        elif data_args.dataset_name == "qnli":
            texts = (examples["question"], examples["sentence"])
        elif data_args.dataset_name == "rte":
            texts = (examples["sentence1"], examples["sentence2"])
        else:  # cola, sst2, etc.
            texts = (examples["sentence"],)
        
        result = tokenizer(*texts, truncation=True, max_length=data_args.max_length, padding="max_length")
        result["labels"] = examples["label"]
        return result
    
    print("Tokenizing dataset...")
    # Remove all columns except label
    columns_to_remove = [col for col in dataset["train"].column_names if col != "label"]
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=columns_to_remove,
    )
    
    train_ds = tokenized_datasets["train"]
    val_ds = tokenized_datasets["validation"]
    test_key = "test" if "test" in tokenized_datasets else "validation"
    test_ds = tokenized_datasets[test_key]
    
    # Create TrainingArguments manually (not parsed to avoid conflicts)
    training_args = TrainingArguments(
        output_dir=adamss_args.output_dir,
        num_train_epochs=adamss_args.num_epochs,
        per_device_train_batch_size=adamss_args.batch_size,
        per_device_eval_batch_size=adamss_args.batch_size,
        learning_rate=hp['lr'],
        weight_decay=hp['wd'],
        warmup_ratio=adamss_args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=TASK_METRICS.get(data_args.dataset_name, "accuracy"),
        greater_is_better=True,
        logging_steps=100,
        logging_strategy="steps",
        seed=adamss_args.seed,
        report_to="none",
    )
    
    # Configure AdaMSS with ASA parameters stored in config
    print("\nApplying AdaMSS...")
    
    # Convert epoch-based parameters to step-based for config
    steps_per_epoch = len(train_ds) // adamss_args.batch_size
    if len(train_ds) % adamss_args.batch_size != 0:
        steps_per_epoch += 1
    total_steps = adamss_args.num_epochs * steps_per_epoch
    
    print(f"\n[Training Configuration]")
    print(f"Dataset size: {len(train_ds)}")
    print(f"Batch size: {adamss_args.batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {adamss_args.num_epochs} epochs × {steps_per_epoch} steps = {total_steps} steps")
    
    asa_init_warmup_steps = adamss_args.asa_init_warmup * steps_per_epoch
    asa_final_warmup_steps = adamss_args.asa_final_warmup * steps_per_epoch
    asa_mask_interval_steps = adamss_args.asa_mask_interval * steps_per_epoch
    
    if adamss_args.use_asa:
        print(f"\n[ASA Configuration (Epoch → Step Conversion)]")
        print(f"  init warmup: {adamss_args.asa_init_warmup} epochs → {asa_init_warmup_steps} steps")
        print(f"  final warmup: {adamss_args.asa_final_warmup} epochs → {asa_final_warmup_steps} steps")
        print(f"  mask interval: {adamss_args.asa_mask_interval} epochs → {asa_mask_interval_steps} steps")
    
    config = AdaMSSConfig(
        r=adamss_args.adamss_r,
        num_subspaces=adamss_args.adamss_k,
        subspace_rank=adamss_args.adamss_ri,
        target_modules=["query", "value"],
        use_asa=adamss_args.use_asa,
        asa_target_subspaces=adamss_args.asa_target_subspaces if adamss_args.use_asa else None,
        # Store step-based ASA parameters in config
        init_warmup=asa_init_warmup_steps if adamss_args.use_asa else None,
        final_warmup=asa_final_warmup_steps if adamss_args.use_asa else None,
        mask_interval=asa_mask_interval_steps if adamss_args.use_asa else None,
        asa_importance_beta=adamss_args.asa_importance_beta if adamss_args.use_asa else None,
        asa_uncertainty_beta=adamss_args.asa_uncertainty_beta if adamss_args.use_asa else None,
        asa_schedule_exponent=adamss_args.asa_schedule_exponent if adamss_args.use_asa else None,
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
    
    # GPU memory monitoring
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"\n[GPU Memory - Before Training]")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # Metrics
    metric = evaluate.load("glue", data_args.dataset_name)
    
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        return result
    
    # Create custom optimizer with different LR for head
    from torch.optim import AdamW
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if ("classifier" in n or "score" in n) and p.requires_grad],
            "lr": hp['head_lr'],
        },
        {
            "params": [p for n, p in model.named_parameters() if ("classifier" not in n and "score" not in n) and p.requires_grad],
            "lr": hp['lr'],
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=hp['wd'])
    
    # Create trainer with custom class that calls update_and_allocate()
    # Note: NO callbacks here - we're using manual approach
    trainer = CustomTrainerWithManualASA(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
    )
    
    # Train
    print("\n" + "="*80)
    print("Starting training...")
    if adamss_args.use_asa:
        print("Manual ASA: update_and_allocate() will be called in training_step")
    print("="*80 + "\n")
    
    train_result = trainer.train()
    
    # GPU memory stats
    if torch.cuda.is_available():
        print(f"\n[GPU Memory - Peak During Training]")
        print(f"Peak Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        print(f"Peak Reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
    
    # Print best metric
    if trainer.state.best_metric is not None:
        metric_name = TASK_METRICS.get(data_args.dataset_name, "accuracy")
        print(f"\n[Best Model Info]")
        print(f"Best {metric_name}: {trainer.state.best_metric:.4f}")
    
    # Evaluate on validation set (use val_ds, not test_ds to avoid label issues)
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
