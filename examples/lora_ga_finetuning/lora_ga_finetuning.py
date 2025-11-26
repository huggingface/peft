#!/usr/bin/env python3
"""
Example script demonstrating LoRA-GA (Low-Rank Adaptation with Gradient Approximation) fine-tuning.

LoRA-GA improves upon standard LoRA by using gradient information during initialization,
achieving 2-4x faster convergence while maintaining the same final performance.

This example shows:
1. How to estimate gradients for LoRA-GA initialization
2. How to use the LoraGAContext for proper initialization
3. How to save initial and final adapter states for delta computation
4. Training with standard Hugging Face Trainer
"""

import argparse
import os

import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from peft import LoraGAConfig, get_peft_model
from peft.utils import LoraGAContext, estimate_gradient, save_loraga_model_final, save_loraga_model_init


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA-GA fine-tuning example")

    # Model arguments
    parser.add_argument("--base_model", type=str, default="gpt2", help="Base model name or path")
    parser.add_argument("--output_dir", type=str, default="./lora_ga_output", help="Output directory")

    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1", help="Dataset configuration")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")

    # LoRA-GA configuration
    parser.add_argument("--r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--direction", type=str, default="ArB2r",
                       choices=["ArBr", "A2rBr", "ArB2r", "random"],
                       help="Direction strategy for LoRA-GA initialization")
    parser.add_argument("--scale", type=str, default="stable",
                       choices=["stable", "weight_svd", "gd_scale", "unit"],
                       help="Scaling strategy for LoRA-GA initialization")
    parser.add_argument("--stable_gamma", type=int, default=16, help="Gamma for stable scaling")

    # Gradient estimation arguments
    parser.add_argument("--grad_estimate_iters", type=int, default=64,
                       help="Number of iterations for gradient estimation")
    parser.add_argument("--grad_estimate_batch_size", type=int, default=2,
                       help="Batch size for gradient estimation")

    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation steps")

    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cuda/cpu)")

    return parser.parse_args()


def prepare_dataset(dataset_name, dataset_config, tokenizer, max_length):
    """Load and prepare the dataset."""
    print(f"\nLoading dataset: {dataset_name}/{dataset_config}")
    dataset = load_dataset(dataset_name, dataset_config)

    def tokenize_function(examples):
        # For causal language modeling, we tokenize and set labels = input_ids
        result = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        result["labels"] = result["input_ids"].clone()
        return result

    # Tokenize the dataset
    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing"
    )

    return tokenized_datasets


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Setup device
    if args.device == "auto":
        device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer and model
    print(f"\nLoading model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.base_model)

    # Prepare dataset
    tokenized_datasets = prepare_dataset(
        args.dataset_name,
        args.dataset_config,
        tokenizer,
        args.max_length
    )

    # Create LoRA-GA configuration
    print("\nCreating LoRA-GA configuration...")
    peft_config = LoraGAConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["c_attn"],  # For GPT-2; adjust based on your model
        direction=args.direction,
        scale=args.scale,
        stable_gamma=args.stable_gamma,
        bias="none",
        task_type="CAUSAL_LM",
    )
    print(f"  Direction: {args.direction}")
    print(f"  Scale: {args.scale}")
    print(f"  Rank: {args.r}, Alpha: {args.lora_alpha}")

    # ===== GRADIENT ESTIMATION PHASE =====
    print("\n" + "="*70)
    print("GRADIENT ESTIMATION PHASE")
    print("="*70)
    print(f"Estimating gradients over {args.grad_estimate_iters} iterations...")
    print("This allows LoRA-GA to initialize adapters aligned with full fine-tuning.")

    # Prepare gradient estimation dataloader
    train_dataset = tokenized_datasets["train"]
    grad_dataloader = DataLoader(
        train_dataset,
        batch_size=args.grad_estimate_batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
    )

    # Initialize Accelerator for gradient estimation
    accelerator = Accelerator()
    model_for_grad = accelerator.prepare(model)
    grad_dataloader = accelerator.prepare(grad_dataloader)

    # Estimate gradients
    named_grad = estimate_gradient(
        model_for_grad,
        grad_dataloader,
        accelerator,
        iters=args.grad_estimate_iters,
        quant_flag=False,
    )

    print(f"✓ Gradient estimation complete! Estimated gradients for {len(named_grad)} modules.")

    # ===== MODEL INITIALIZATION PHASE =====
    print("\n" + "="*70)
    print("LORA-GA INITIALIZATION PHASE")
    print("="*70)
    print("Initializing LoRA adapters with gradient information...")

    # Create PEFT model with LoRA-GA initialization
    with LoraGAContext(model, named_grad):
        peft_model = get_peft_model(model, peft_config)

    # Print trainable parameters
    peft_model.print_trainable_parameters()

    # Save initial adapter state (required for delta computation)
    print("\nSaving initial adapter state...")
    save_loraga_model_init(peft_model, args.output_dir)

    # ===== TRAINING PHASE =====
    print("\n" + "="*70)
    print("TRAINING PHASE")
    print("="*70)
    print("Starting training with LoRA-GA initialized adapters...")
    print("LoRA-GA achieves 2-4x faster convergence compared to random initialization!\n")

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        seed=args.seed,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Create Trainer
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    # ===== SAVING PHASE =====
    print("\n" + "="*70)
    print("SAVING PHASE")
    print("="*70)
    print("Computing adapter delta (final - initial) and saving...")

    # Save final adapter state with delta computation
    save_loraga_model_final(peft_model, args.output_dir)

    print(f"\n✓ Training complete! Model saved to: {args.output_dir}")
    print("\nLoRA-GA specific files saved:")
    print(f"  - adapter_model_init.safetensors: Initial adapter state")
    print(f"  - adapter_model.safetensors: Adapter delta (final - initial)")
    print(f"  - adapter_config.json: Configuration file")

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
    print("\nYou can now use the trained adapter with:")
    print(f"  from peft import PeftModel")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{args.base_model}')")
    print(f"  model = PeftModel.from_pretrained(model, '{args.output_dir}')")


if __name__ == "__main__":
    main()
