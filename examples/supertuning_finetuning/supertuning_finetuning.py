# This script is based on examples/delora_finetuning/delora_finetuning.py
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from peft import SupertuningConfig, get_peft_model


def train_model(
    base_model: str,
    data_path: str,
    output_dir: str,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    cutoff_len: int,
    eval_step: int,
    save_step: int,
    device: str,
    sparsity: float,
    select_top: bool,
    rank: int,
    lora_alpha: int,
    target_modules: str,
    hub_model_id: str,
    push_to_hub: bool,
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    hf_token = os.getenv("HF_TOKEN")

    # Setup device
    device = torch.device(device)
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)

    # Compute dtype
    device_type = device.type
    device_module = getattr(torch, device_type, torch.cuda)
    bf16_supported = device_module.is_available() and device_module.is_bf16_supported()
    dtype = torch.bfloat16 if bf16_supported else torch.float32

    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)

    # Super-Tuning config. Leave `r=None` for pure Super (sparse support only); set `r` to a positive
    # integer to train the "Supra" hybrid (sparse support + a LoRA-style low-rank adapter on top).
    peft_config = SupertuningConfig(
        sparsity=sparsity,
        select_top=select_top,
        r=rank,
        lora_alpha=lora_alpha if rank is not None else None,
        target_modules=(target_modules.split(",") if target_modules else None),
    )

    # Wrap the base model with the Super-Tuning config
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    model.to(device)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the dataset
    dataset = load_dataset(data_path)

    def tokenize_function(examples):
        inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=cutoff_len)
        inputs["labels"] = inputs["input_ids"].copy()  # labels for a language-modeling task
        return inputs

    # Tokenize the dataset and prepare for training
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

    # Data collator to dynamically pad the batched examples
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Total number of training steps, used for warmup
    max_steps = int((len(dataset) // batch_size) * num_epochs)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=int(max_steps * 0.1),  # 10% of total training steps
        weight_decay=0.0,
        logging_steps=eval_step,
        save_steps=save_step,
        save_total_limit=2,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        gradient_accumulation_steps=16,
        learning_rate=learning_rate,
        hub_token=hf_token,
        label_names=["labels"],
    )

    # Clear accelerator cache to free memory
    device_module.empty_cache()

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
    )

    # Start training
    trainer.train()

    # Save and push the trained model and tokenizer
    if push_to_hub:
        trainer.push_to_hub(commit_message="Fine-tuned model")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune a model with Super-Tuning / Supra")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B", help="Base model path or name")
    parser.add_argument(
        "--data_path", type=str, default="timdettmers/openassistant-guanaco", help="Dataset path or name"
    )
    parser.add_argument(
        "--output_dir", type=str, default="path/to/output", help="Output directory for the fine-tuned model"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--cutoff_len", type=int, default=512, help="Cutoff length for tokenization")
    parser.add_argument("--eval_step", type=int, default=10, help="Logging step interval")
    parser.add_argument("--save_step", type=int, default=100, help="Save step interval")
    parser.add_argument("--device", type=str, default="auto", help="Device to use for training")
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.99,
        help="Target sparsity in [0.0, 1.0); 0.99 = 1%% of weight entries are trainable",
    )
    parser.add_argument(
        "--select_top",
        action="store_true",
        default=True,
        help="Keep the largest-magnitude entries as the trainable support (paper's Super/Supra)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="LoRA rank for the Supra hybrid. Leave unset for pure Super (sparse support only)",
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=None, help="LoRA scaling for Supra mode; defaults to 2*rank when unset"
    )
    parser.add_argument("--target_modules", type=str, default=None, help="Comma-separated list of target modules")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default="path/to/repo",
        help="Repository name to push the model to on the Hugging Face Hub",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the model to the Hugging Face Hub")
    args = parser.parse_args()

    if args.device == "auto":
        args.device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"

    train_model(
        base_model=args.base_model,
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        cutoff_len=args.cutoff_len,
        eval_step=args.eval_step,
        save_step=args.save_step,
        device=args.device,
        sparsity=args.sparsity,
        select_top=args.select_top,
        rank=args.rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        hub_model_id=args.hub_model_id,
        push_to_hub=args.push_to_hub,
    )
