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

from peft import PacaConfig, get_peft_model


def train_model(
    base_model: str,
    data_path: str,
    output_dir: str,
    batch_size: int,
    num_epochs: int,
    max_steps: int,
    learning_rate: float,
    cutoff_len: int,
    eval_step: int,
    save_step: int,
    device: str,
    rank: int,
    paca_alpha: int,
    target_modules: str,
    random_seed: int,
    hub_model_id: str,
    push_to_hub: bool,
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    hf_token = os.getenv("HF_TOKEN")

    device = torch.device(device)
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)

    device_type = device.type
    device_module = getattr(torch, device_type, torch.cuda)
    bf16_supported = device_module.is_available() and device_module.is_bf16_supported()
    dtype = torch.bfloat16 if bf16_supported else torch.float32

    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype, token=hf_token)

    # PaCA trains `rank` randomly selected input columns of every targeted linear layer
    peft_config = PacaConfig(
        r=rank,
        paca_alpha=paca_alpha,
        target_modules=(target_modules.split(",") if target_modules else "all-linear"),
        random_seed=random_seed,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    model.to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(data_path)

    def tokenize_function(examples):
        inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=cutoff_len)
        inputs["labels"] = inputs["input_ids"].copy()
        return inputs

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        weight_decay=0.0,
        logging_steps=eval_step,
        save_steps=save_step,
        save_total_limit=2,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        hub_token=hf_token,
        label_names=["labels"],
        bf16=bf16_supported,
    )

    device_module.empty_cache()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("test"),
        data_collator=data_collator,
    )
    trainer.train()

    if push_to_hub:
        trainer.push_to_hub(commit_message="Fine-tuned model")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune a causal LM with PaCA")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-3B", help="Base model path or name")
    parser.add_argument(
        "--data_path", type=str, default="timdettmers/openassistant-guanaco", help="Dataset path or name"
    )
    parser.add_argument(
        "--output_dir", type=str, default="path/to/output", help="Output directory for the fine-tuned model"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="If > 0, overrides num_epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--cutoff_len", type=int, default=512, help="Cutoff length for tokenization")
    parser.add_argument("--eval_step", type=int, default=10, help="Logging step interval")
    parser.add_argument("--save_step", type=int, default=100, help="Save step interval")
    parser.add_argument("--device", type=str, default="auto", help="Device to use for training")
    parser.add_argument("--rank", type=int, default=16, help="Number of fine-tuned input columns per layer")
    parser.add_argument("--paca_alpha", type=int, default=16, help="PaCA alpha, the update is scaled by alpha / rank")
    parser.add_argument(
        "--target_modules",
        type=str,
        default=None,
        help="Comma-separated list of target modules for PaCA (default: all linear layers except the output layer)",
    )
    parser.add_argument("--random_seed", type=int, default=0, help="Seed for selecting the partial connections")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default="path/to/repo",
        help="Repository name to push the model on the Hugging Face Hub",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the model to Hugging Face Hub")
    args = parser.parse_args()

    if args.device == "auto":
        args.device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"

    train_model(
        base_model=args.base_model,
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        cutoff_len=args.cutoff_len,
        eval_step=args.eval_step,
        save_step=args.save_step,
        device=args.device,
        rank=args.rank,
        paca_alpha=args.paca_alpha,
        target_modules=args.target_modules,
        random_seed=args.random_seed,
        hub_model_id=args.hub_model_id,
        push_to_hub=args.push_to_hub,
    )
