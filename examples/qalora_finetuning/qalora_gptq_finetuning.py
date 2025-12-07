#!/usr/bin/env python3
"""
Training script for fine-tuning language models with QALoRA using GPTQ quantization.
This script supports cached quantization to avoid repeating expensive quantization processes.
"""

import argparse
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPTQConfig,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model


def load_or_quantize_model(
    base_model: str, tokenizer, bits: int = 4, cache_dir: str = "./quantized_models"
) -> AutoModelForCausalLM:
    """
    Load a pre-quantized model from cache or quantize and cache a new one.
    Automatically detects if the model is already GPTQ-quantized.

    Args:
        base_model: Model identifier or path
        tokenizer: Tokenizer for the model
        bits: Bit-width for quantization (default: 4)
        cache_dir: Directory to store quantized models

    Returns:
        The loaded (quantized) model
    """
    # First, check if the model is already GPTQ-quantized by trying to load it
    print(f"Checking if {base_model} is already GPTQ-quantized...")
    try:
        # Try to load the model and check if it has GPTQ quantization
        test_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            dtype=torch.float16,
            trust_remote_code=True,  # Some GPTQ models might need this
        )

        # Check if the model has GPTQ quantization attributes
        has_gptq = False
        for module in test_model.modules():
            if hasattr(module, "qweight") or hasattr(module, "qzeros") or "gptq" in str(type(module)).lower():
                has_gptq = True
                break

        if has_gptq:
            print(f"✅ Model {base_model} is already GPTQ-quantized. Using directly.")
            return test_model
        else:
            print(f"Model {base_model} is not GPTQ-quantized. Will quantize it.")
            # Clean up the test model to free memory
            del test_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.xpu.is_available():
                torch.xpu.empty_cache()

    except Exception as e:
        print(f"Could not load model {base_model} directly: {e}")
        print("Will attempt to quantize it...")

    # If we get here, the model needs to be quantized
    os.makedirs(cache_dir, exist_ok=True)
    model_id = base_model.replace("/", "_").replace("\\", "_")  # Handle Windows paths too
    quantized_model_path = os.path.join(cache_dir, f"{model_id}_gptq_{bits}bit")

    # Check if we already have a cached quantized version
    if os.path.exists(quantized_model_path) and os.path.exists(os.path.join(quantized_model_path, "config.json")):
        print(f"Loading pre-quantized model from cache: {quantized_model_path}")
        return AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto")

    print(f"Quantizing model and saving to cache: {quantized_model_path}")

    # Configure GPTQ for first-time quantization
    gptq_config = GPTQConfig(
        bits=bits,
        dataset="c4",
        tokenizer=tokenizer,
        group_size=128,
        desc_act=False,
        sym=False,
    )

    # Load and quantize the model
    model = AutoModelForCausalLM.from_pretrained(
        base_model, device_map="auto", quantization_config=gptq_config, dtype=torch.float16
    )

    # Save the quantized model to cache
    print(f"Saving quantized model to {quantized_model_path}")
    model.save_pretrained(quantized_model_path)
    tokenizer.save_pretrained(quantized_model_path)

    return model


def tokenize_and_preprocess(examples, tokenizer, max_length: int = 128):
    """
    Tokenize text data and prepare it for language modeling.

    Args:
        examples: Dataset examples with 'text' field
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length

    Returns:
        Processed examples with input_ids and labels
    """
    # Tokenize the text with truncation and padding
    tokenized_output = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)

    # Preprocess labels (set pad tokens to -100 for loss masking)
    labels = tokenized_output["input_ids"].copy()
    labels = [[-100 if token == tokenizer.pad_token_id else token for token in seq] for seq in labels]
    tokenized_output["labels"] = labels

    return tokenized_output


def train_model(
    base_model: str,
    data_path: str,
    data_split: str,
    output_dir: str,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    cutoff_len: int,
    use_qalora: bool,
    eval_step: int,
    save_step: int,
    device: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: str,
    push_to_hub: bool,
    qalora_group_size: int,
    bits: int,
) -> None:
    """
    Train a model with QALoRA and GPTQ quantization.

    Args:
        base_model: Base model to fine-tune
        data_path: Dataset path
        output_dir: Directory to save model outputs
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        cutoff_len: Maximum sequence length
        val_set_size: Validation set size
        use_dora: Whether to use DoRA
        use_qalora: Whether to use QALoRA
        quantize: Whether to use quantization
        eval_step: Steps between evaluations
        save_step: Steps between saving checkpoints
        device: Device to use (cuda:0, xpu:0, etc.)
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout rate
        lora_target_modules: Target modules for LoRA
        push_to_hub: Whether to push to Hugging Face Hub
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    hf_token = os.getenv("HF_TOKEN")
    device = torch.device(device)
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load or quantize model
    model = load_or_quantize_model(base_model, tokenizer, bits=bits)

    # Configure LoRA
    target_modules = (
        lora_target_modules.split(",")
        if lora_target_modules
        else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    print("use_qalora", use_qalora)
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        use_qalora=use_qalora,
        qalora_group_size=qalora_group_size,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )

    # Get PEFT model with adapters
    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # Move model to device if not already there
    if not hasattr(model, "device") or model.device.type != device.type:
        model = model.to(device)

    # Load and prepare dataset
    dataset = load_dataset(data_path, data_split)

    tokenized_datasets = {
        "train": dataset["train"].map(
            lambda x: tokenize_and_preprocess(x, tokenizer, max_length=cutoff_len),
            batched=True,
            remove_columns=["text"],
            load_from_cache_file=True,
        ),
        "test": dataset["test"].map(
            lambda x: tokenize_and_preprocess(x, tokenizer, max_length=cutoff_len),
            batched=True,
            remove_columns=["text"],
            load_from_cache_file=True,
        ),
    }

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=eval_step,
        save_steps=save_step,
        save_total_limit=2,
        push_to_hub=push_to_hub,
        gradient_accumulation_steps=16,
        fp16=True,
        learning_rate=learning_rate,
        hub_token=hf_token,
        label_names=["labels"],
    )

    # Clear accelerator cache to free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.xpu.is_available():
        torch.xpu.empty_cache()

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
    )

    # Start training
    print("\nStarting training...")
    trainer.train()

    # Save the final model
    if push_to_hub:
        trainer.push_to_hub(commit_message="Fine-tuned model with QALoRA")

    # Always save locally
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nTraining complete. Model saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LLMs with QALoRA and GPTQ quantization")

    # Model and dataset parameters
    parser.add_argument("--base_model", type=str, default="TheBloke/Llama-2-7b-GPTQ", help="Base model path or name")
    parser.add_argument(
        "--data_path", type=str, default="timdettmers/openassistant-guanaco", help="Dataset path or name"
    )
    parser.add_argument("--data_split", type=str, default="", help="Dataset path or name")

    parser.add_argument(
        "--output_dir", type=str, default="./qalora_output", help="Output directory for the fine-tuned model"
    )
    parser.add_argument("--bits", type=int, default=4, help="Init quantization bits")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--cutoff_len", type=int, default=128, help="Max sequence length")

    # Adapter configuration
    parser.add_argument("--use_qalora", action="store_true", help="Apply QALoRA")
    parser.add_argument("--qalora_group_size", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument(
        "--lora_target_modules", type=str, default=None, help="Comma-separated list of target modules for LoRA"
    )

    # Training process options
    parser.add_argument("--eval_step", type=int, default=100, help="Evaluation step interval")
    parser.add_argument("--save_step", type=int, default=500, help="Save step interval")
    parser.add_argument("--device", type=str, default="auto", help="Device to use for training")

    # Hugging Face Hub options
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the model to Hugging Face Hub")

    args = parser.parse_args()

    device = args.device
    if args.device == "auto":
        device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"

    # If use_qalora isn't explicitly set in args but passed to train_model
    if not args.use_qalora:
        args.use_qalora = True  # Default to True as in the original code

    train_model(
        base_model=args.base_model,
        data_path=args.data_path,
        data_split=args.data_split,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        cutoff_len=args.cutoff_len,
        use_qalora=args.use_qalora,
        eval_step=args.eval_step,
        save_step=args.save_step,
        device=device,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        push_to_hub=args.push_to_hub,
        qalora_group_size=args.qalora_group_size,
        bits=args.bits,
    )
