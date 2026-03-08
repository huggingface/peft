# This script is based on examples/lily_finetuning/lily_finetuning.py
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

from peft import PeanutConfig, get_peft_model


def train_model(
    base_model: str,
    data_path: str,
    output_dir: str,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    cutoff_len: int,
    val_set_size: int,
    eval_step: int,
    save_step: int,
    device: str,
    peanut_r: int,
    peanut_depth: int,
    peanut_scaling: float,
    peanut_act_fn: str,
    peanut_target_modules: str,
    peanut_init_weights: bool,
    hub_model_id: str,
    push_to_hub: bool,
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    hf_token = os.getenv("HF_TOKEN")

    # Setup device
    if device == "auto":
        device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
    else:
        device = torch.device(device)
    print(f"Using device: {device}")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)

    model = AutoModelForCausalLM.from_pretrained(base_model, token=hf_token)

    # PEANuT config for the PEFT model
    peanut_config = PeanutConfig(
        r=peanut_r,
        depth=peanut_depth,
        scaling=peanut_scaling,
        act_fn=peanut_act_fn,
        init_weights=peanut_init_weights,
        target_modules=(
            peanut_target_modules.split(",")
            if peanut_target_modules
            else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        ),
    )

    # get the peft model with PEANuT config
    model = get_peft_model(model, peanut_config)
    model.print_trainable_parameters()

    model.to(device)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the dataset
    dataset = load_dataset(data_path)

    def tokenize_function(examples):
        inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=cutoff_len)
        inputs["labels"] = inputs["input_ids"].copy()  # setting labels for a language modeling task
        return inputs

    # Tokenize the dataset and prepare for training
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

    # Data collator to dynamically pad the batched examples
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=eval_step,
        save_steps=save_step,
        save_total_limit=2,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        gradient_accumulation_steps=16,
        fp16=True,
        learning_rate=learning_rate,
        hub_token=hf_token,
    )

    # Clear device cache to free memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.xpu.is_available():
        torch.xpu.empty_cache()

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
    )

    # Start model training
    trainer.train()

    # Save and push the trained model and tokenizer
    if push_to_hub:
        trainer.push_to_hub(commit_message="Fine-tuned model with PEANuT")

    # Save the model and tokenizer locally
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune LLaMA with PEANuT and PEFT")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-3B", help="Base model path or name")
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
    parser.add_argument("--val_set_size", type=int, default=500, help="Validation set size")
    parser.add_argument("--eval_step", type=int, default=10, help="Evaluation step interval")
    parser.add_argument("--save_step", type=int, default=100, help="Save step interval")
    parser.add_argument("--device", type=str, default="auto", help="Device to use for training")
    parser.add_argument("--peanut_r", type=int, default=32, help="PEANuT rank")
    parser.add_argument(
        "--peanut_depth",
        type=int,
        default=2,
        help="Total number of PEANuT transforms including A and B (must be even and >= 2)",
    )
    parser.add_argument(
        "--peanut_scaling", type=float, default=1.0, help="PEANuT scaling factor applied to adapter output"
    )
    parser.add_argument(
        "--peanut_act_fn",
        type=str,
        default="relu",
        help="Activation used inside PEANuT neural tweakers (must be a valid transformers ACT2FN key)",
    )
    parser.add_argument(
        "--peanut_target_modules", type=str, default=None, help="Comma-separated list of target modules for PEANuT"
    )
    parser.add_argument(
        "--peanut_init_weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use PEANuT default init: zero-init B and Kaiming-init the other layers",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default="path/to/repo",
        help="Repository name to push the model on the Hugging Face Hub",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the model to Hugging Face Hub")
    args = parser.parse_args()

    train_model(
        base_model=args.base_model,
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        cutoff_len=args.cutoff_len,
        val_set_size=args.val_set_size,
        eval_step=args.eval_step,
        save_step=args.save_step,
        device=args.device,
        peanut_r=args.peanut_r,
        peanut_depth=args.peanut_depth,
        peanut_scaling=args.peanut_scaling,
        peanut_act_fn=args.peanut_act_fn,
        peanut_target_modules=args.peanut_target_modules,
        peanut_init_weights=args.peanut_init_weights,
        hub_model_id=args.hub_model_id,
        push_to_hub=args.push_to_hub,
    )
