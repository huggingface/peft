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


def check_adapter_gradients(model) -> None:
    """
    Verify that QALoRA adapters are properly initialized with correct gradient settings.
    Checks if adapter parameters are trainable while base model weights are frozen.

    Args:
        model: The PEFT model with QALoRA adapters
    """
    print("\n=== QALoRA Implementation Verification ===")

    total_params = 0
    trainable_params = 0
    qalora_modules = 0

    # Check trainable parameters
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    # Check for QALoRA modules
    for name, module in model.named_modules():
        if hasattr(module, "qalora_group_size"):
            qalora_modules += 1
            adapter_name = next(iter(module.qalora_group_size.keys()), None)
            if adapter_name:
                print(f"\nQALoRA module found: {name}")
                print(f"  - Group size: {module.qalora_group_size[adapter_name]}")
                print(f"  - Scaling factor: {module.qalora_scaling_factor.get(adapter_name, 'N/A')}")

                # Check adapter matrices
                if hasattr(module, "lora_A") and adapter_name in module.lora_A:
                    a_shape = module.lora_A[adapter_name].weight.shape
                    a_grad = module.lora_A[adapter_name].weight.requires_grad
                    print(f"  - lora_A shape: {a_shape}, requires_grad: {a_grad}")

                if hasattr(module, "lora_B") and adapter_name in module.lora_B:
                    b_shape = module.lora_B[adapter_name].weight.shape
                    b_grad = module.lora_B[adapter_name].weight.requires_grad
                    print(f"  - lora_B shape: {b_shape}, requires_grad: {b_grad}")

                # Check base weights
                if hasattr(module, "weight"):
                    base_grad = module.weight.requires_grad
                    print(f"  - Base weight requires_grad: {base_grad} (should be False)")

    # Print summary
    print("\nParameter summary:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)")
    print(f"  - QALoRA modules found: {qalora_modules}")

    if qalora_modules > 0 and trainable_params > 0:
        print("✅ QALoRA implementation verification: PASSED")
    else:
        print("❌ QALoRA implementation verification: FAILED")


def print_parameter_memory(model, optimizer_name: str = "adam") -> None:
    """
    Print detailed memory usage information for model parameters.

    Args:
        model: The model to analyze
        optimizer_name: The optimizer type to estimate memory for ("adam" or "sgd")
    """
    total_param_mem = 0
    total_grad_mem = 0
    total_opt_mem = 0

    print(f"{'Name':50} {'Shape':20} {'Dtype':10} {'Param MB':10} {'Grad MB':10} {'Requires Grad'}")
    print("-" * 110)

    for name, param in model.named_parameters():
        param_size = param.numel() * param.element_size() / 1024**2  # in MB
        grad_size = param_size if param.requires_grad else 0
        opt_size = 0

        if param.requires_grad:
            if optimizer_name.lower() == "adam":
                opt_size = param_size * 2  # Adam keeps 2 extra states (m, v)
            elif optimizer_name.lower() == "sgd":
                opt_size = param_size  # SGD with momentum keeps 1 extra state

        print(
            f"{name:50} {str(list(param.shape)):20} {str(param.dtype):10} "
            f"{param_size:10.2f} {grad_size:10.2f} {str(param.requires_grad):>12}"
        )

        total_param_mem += param_size
        total_grad_mem += grad_size
        total_opt_mem += opt_size

    print("-" * 110)
    print(f"{'TOTAL':50} {'':20} {'':10} {total_param_mem:10.2f} {total_grad_mem:10.2f}")
    print(
        f"Parameter dtype is per-column above. "
        f"\nTotal parameter memory: {total_param_mem:.2f} MB"
        f"\nTotal gradient memory (training): {total_grad_mem:.2f} MB"
        f"\nTotal optimizer state memory (training): {total_opt_mem:.2f} MB"
        f"\nTotal memory for training: {total_param_mem + total_grad_mem + total_opt_mem:.2f} MB"
        f"\nTotal memory for inference: {total_param_mem:.2f} MB"
    )


def load_or_quantize_model(
    base_model: str, tokenizer, bits: int = 4, cache_dir: str = "./quantized_models"
) -> AutoModelForCausalLM:
    """
    Load a pre-quantized model from cache or quantize and cache a new one.

    Args:
        base_model: Model identifier or path
        tokenizer: Tokenizer for the model
        bits: Bit-width for quantization (default: 4)
        cache_dir: Directory to store quantized models

    Returns:
        The loaded (quantized) model
    """
    os.makedirs(cache_dir, exist_ok=True)
    model_id = base_model.replace("/", "_")
    quantized_model_path = os.path.join(cache_dir, f"{model_id}_gptq_{bits}bit")

    # Check if the quantized model already exists in cache
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
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", quantization_config=gptq_config)

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
    output_dir: str,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    cutoff_len: int,
    val_set_size: int,
    use_dora: bool,
    use_qalora: bool,
    quantize: bool,
    eval_step: int,
    save_step: int,
    device: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: str,
    hub_model_id: str,
    push_to_hub: bool,
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
        device: Device to use (cuda:0, etc.)
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout rate
        lora_target_modules: Target modules for LoRA
        hub_model_id: Hugging Face Hub model ID
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
    model = load_or_quantize_model(base_model, tokenizer)

    # Configure LoRA
    target_modules = (
        lora_target_modules.split(",")
        if lora_target_modules
        else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    print("use_qalora", use_qalora)
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        use_dora=use_dora,
        use_qalora=use_qalora,
        qalora_group_size=8,  # Explicitly set group size for QALoRA
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )

    # Get PEFT model with adapters
    model = get_peft_model(model, lora_config)
    model.save_pretrained(output_dir)

    model.print_trainable_parameters()
    check_adapter_gradients(model)

    # Move model to device if not already there
    if device.type != "cuda" or not hasattr(model, "device") or model.device.type != "cuda":
        model = model.to(device)

    # Load and prepare dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    tokenized_datasets = {
        "train": dataset["train"].map(
            lambda x: tokenize_and_preprocess(x, tokenizer, max_length=cutoff_len),
            batched=True,
            remove_columns=["text"],
        ),
        "test": dataset["test"].map(
            lambda x: tokenize_and_preprocess(x, tokenizer, max_length=cutoff_len),
            batched=True,
            remove_columns=["text"],
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
        hub_model_id=hub_model_id,
        gradient_accumulation_steps=16,
        fp16=True,
        learning_rate=learning_rate,
        hub_token=hf_token,
        label_names=["labels"],
    )

    # Clear CUDA cache to free memory
    torch.cuda.empty_cache()

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
    )

    # Print memory usage before training
    print("\nMemory usage before training:")
    print_parameter_memory(model)

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
    parser.add_argument("--base_model", type=str, default="HuggingFaceTB/SmolLM2-135M", help="Base model path or name")
    parser.add_argument("--data_path", type=str, default="wikitext", help="Dataset path or name")
    parser.add_argument(
        "--output_dir", type=str, default="./qalora_output", help="Output directory for the fine-tuned model"
    )

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--cutoff_len", type=int, default=128, help="Max sequence length")
    parser.add_argument("--val_set_size", type=int, default=500, help="Validation set size")

    # Adapter configuration
    parser.add_argument("--use_dora", action="store_true", help="Apply DoRA")
    parser.add_argument("--use_qalora", action="store_true", help="Apply QALoRA")
    parser.add_argument("--quantize", action="store_true", help="Use GPTQ quantization")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument(
        "--lora_target_modules", type=str, default=None, help="Comma-separated list of target modules for LoRA"
    )

    # Training process options
    parser.add_argument("--eval_step", type=int, default=100, help="Evaluation step interval")
    parser.add_argument("--save_step", type=int, default=500, help="Save step interval")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")

    # Hugging Face Hub options
    parser.add_argument("--hub_model_id", type=str, default=None, help="Repository name to push the model on the Hub")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the model to Hugging Face Hub")

    args = parser.parse_args()

    # If use_qalora isn't explicitly set in args but passed to train_model
    if not args.use_qalora:
        args.use_qalora = True  # Default to True as in the original code

    train_model(
        base_model=args.base_model,
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        cutoff_len=args.cutoff_len,
        val_set_size=args.val_set_size,
        use_dora=args.use_dora,
        use_qalora=args.use_qalora,
        quantize=args.quantize,
        eval_step=args.eval_step,
        save_step=args.save_step,
        device=args.device,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        hub_model_id=args.hub_model_id,
        push_to_hub=args.push_to_hub,
    )
