# Copyright 2025-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.optimizers import create_lorafa_optimizer


def train_model(
    base_model_name_or_path: str,
    dataset_name_or_path: str,
    output_dir: str,
    batch_size: int,
    num_epochs: int,
    lr: float,
    cutoff_len: int,
    quantize: bool,
    eval_step: int,
    save_step: int,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: Optional[str],
    lorafa: bool,
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    is_bf16_supported = False
    device_map = "cpu"
    if torch.cuda.is_available():
        is_bf16_supported = torch.cuda.is_bf16_supported()
        device_map = "cuda"
    elif torch.xpu.is_available():
        is_bf16_supported = torch.xpu.is_bf16_supported()
        device_map = "xpu"
    compute_dtype = torch.bfloat16 if is_bf16_supported else torch.float16

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)

    # load model
    if quantize:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
            ),
            dtype=compute_dtype,
            device_map=device_map,
        )
        # setup for quantized training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path, dtype=compute_dtype, device_map=device_map
        )

    # LoRA config for the PEFT model
    if lora_target_modules is not None:
        if lora_target_modules == "all-linear":
            target_modules = "all-linear"
        else:
            target_modules = lora_target_modules.split(",")
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )

    # get the peft model with LoRA config
    model = get_peft_model(model, lora_config)

    tokenizer.pad_token = tokenizer.eos_token

    # Load the dataset
    dataset = load_dataset(dataset_name_or_path)

    def tokenize_function(examples):
        inputs = tokenizer(examples["query"], padding="max_length", truncation=True, max_length=cutoff_len)
        outputs = tokenizer(examples["response"], padding="max_length", truncation=True, max_length=cutoff_len)
        inputs["labels"] = outputs["input_ids"].copy()
        return inputs

    # Tokenize the dataset and prepare for training
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
    dataset = tokenized_datasets["train"].train_test_split(test_size=0.1, shuffle=True, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

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
        logging_dir="./logs",
        logging_steps=eval_step,
        save_steps=save_step,
        save_total_limit=2,
        gradient_accumulation_steps=1,
        bf16=True if compute_dtype == torch.bfloat16 else False,
        fp16=True if compute_dtype == torch.float16 else False,
        learning_rate=lr,
    )

    # Here we initialize the LoRA-FA Optimizer
    # After this, all adapter A will be fixed, only adapter B will be trainable
    if lorafa:
        optimizer = create_lorafa_optimizer(
            model=model, r=lora_rank, lora_alpha=lora_alpha, lr=lr, weight_decay=training_args.weight_decay
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            optimizers=(optimizer, None),
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

    # Start model training
    trainer.train()

    # Save the model and tokenizer locally
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Meta-Llama-3-8B-Instruct with LoRA-FA and PEFT")
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Base model name or path",
    )
    parser.add_argument(
        "--dataset_name_or_path", type=str, default="meta-math/MetaMathQA-40K", help="Dataset name or path"
    )
    parser.add_argument("--output_dir", type=str, help="Output directory for the fine-tuned model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=7e-5, help="Learning rate")
    parser.add_argument("--cutoff_len", type=int, default=1024, help="Cutoff length for tokenization")
    parser.add_argument("--quantize", action="store_true", help="Use quantization")
    parser.add_argument("--eval_step", type=int, default=10, help="Evaluation step interval")
    parser.add_argument("--save_step", type=int, default=100, help="Save step interval")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument(
        "--lora_target_modules", type=str, default=None, help="Comma-separated list of target modules for LoRA"
    )
    parser.add_argument("--lorafa", action="store_true", help="Use LoRA-FA Optimizer")

    args = parser.parse_args()

    train_model(
        base_model_name_or_path=args.base_model_name_or_path,
        dataset_name_or_path=args.dataset_name_or_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        cutoff_len=args.cutoff_len,
        quantize=args.quantize,
        eval_step=args.eval_step,
        save_step=args.save_step,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        lorafa=args.lorafa,
    )
