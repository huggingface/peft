import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


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
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    hf_token = os.getenv("HF_TOKEN")

    # Setup device
    device = torch.device(device)
    print(f"Using device: {device}")

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)

    # QDoRA (quantized dora): IF YOU WANNA QUANTIZE THE MODEL
    if quantize:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            token=hf_token,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=(
                    torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
                ),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
        )
        # setup for quantized training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model, token=hf_token)
    # LoRa config for the PEFT model
    lora_config = LoraConfig(
        use_dora=use_dora,  # to use Dora OR compare to Lora just set the --use_dora
        r=lora_r,  # Rank of matrix
        lora_alpha=lora_alpha,
        target_modules=(
            lora_target_modules.split(",")
            if lora_target_modules
            else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        ),
        lora_dropout=lora_dropout,
        bias="none",
    )

    # get the peft model with LoRa config
    model = get_peft_model(model, lora_config)

    model.to(device)  # MODEL TO GPU/CUDA
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
    data_collator = DataCollatorWithPadding(tokenizer)

    # Define training arguments
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
    )

    # Clear CUDA cache to free memory
    torch.cuda.empty_cache()

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
        # Push the main model to the hub
        trainer.push_to_hub(commit_message="Fine-tuned model")

    # Save the model and tokenizer locally
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune LLaMA with DoRA and PEFT")
    parser.add_argument("--base_model", type=str, default="huggyllama/llama-7b", help="Base model path or name")
    parser.add_argument(
        "--data_path", type=str, default="timdettmers/openassistant-guanaco", help="Dataset path or name"
    )
    parser.add_argument(
        "--output_dir", type=str, default="path/to/output", help="Output directory for the fine-tuned model"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--cutoff_len", type=int, default=512, help="Cutoff length for tokenization")
    parser.add_argument("--val_set_size", type=int, default=500, help="Validation set size")
    parser.add_argument("--use_dora", action="store_true", help="Apply Dora")
    parser.add_argument("--quantize", action="store_true", help="Use quantization")
    parser.add_argument("--eval_step", type=int, default=10, help="Evaluation step interval")
    parser.add_argument("--save_step", type=int, default=100, help="Save step interval")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument(
        "--lora_target_modules", type=str, default=None, help="Comma-separated list of target modules for LoRA"
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
        use_dora=args.use_dora,
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
