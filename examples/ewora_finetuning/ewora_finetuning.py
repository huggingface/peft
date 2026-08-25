# This script reproduces the mixed-task training setup from the EWoRA paper
# (https://aclanthology.org/2025.findings-ijcnlp.108/): Mistral 7B finetuned on a concatenation of
# Magicoder-Evol-Instruct-110K (code), MetaMathQA (math), and HellaSwag + Winogrande (general reasoning).
import argparse
import os

import torch
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from peft import EworaConfig, get_peft_model


def create_prompt(sample):
    # Instruction/response formatting used in the paper's training runs. The field names cover the four
    # mixed-setting datasets: Magicoder (instruction/response), MetaMathQA (query/response),
    # HellaSwag (ctx/endings+label) and Winogrande (sentence/option+answer).
    bos_token = "<s>"
    eos_token = "</s>"

    if "instruction" in sample and sample["instruction"] is not None:
        instruction = sample["instruction"]
    elif "query" in sample and sample["query"] is not None:
        instruction = sample["query"]
    elif "ctx" in sample and sample["ctx"] is not None:
        instruction = sample["ctx"]
    elif "sentence" in sample and sample["sentence"] is not None:
        instruction = sample["sentence"]

    if "response" in sample and sample["response"] is not None:
        response = sample["response"]
    elif "endings" in sample and sample["endings"] is not None:
        response = sample["endings"][int(sample["label"])]
    elif "answer" in sample and sample["answer"] is not None:
        response = sample["option" + sample["answer"]]

    full_prompt = ""
    full_prompt += bos_token
    full_prompt += "### Instruction:"
    full_prompt += "\n" + instruction
    full_prompt += "\n\n### Response:"
    full_prompt += "\n" + response
    full_prompt += eos_token

    return full_prompt


def train_model(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load and concatenate the training datasets (simple concatenation, shuffled at train time)
    train_datasets = []
    for dataset_name in args.datasets.split(","):
        if dataset_name == "allenai/winogrande":
            dataset = load_dataset(dataset_name, "winogrande_xl", trust_remote_code=True)["train"]
        else:
            dataset = load_dataset(dataset_name, trust_remote_code=True)["train"]
        train_datasets.append(dataset)
    train_dataset = concatenate_datasets(train_datasets)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        use_cache=False,
        dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Paper configuration: total rank 32 partitioned into num_experts=4 experts. Note that `r` is the rank
    # of each individual expert, so the paper's total rank corresponds to r=8, num_experts=4.
    peft_config = EworaConfig(
        r=args.rank,
        num_experts=args.num_experts,
        ewora_dropout=args.ewora_dropout,
        task_type="CAUSAL_LM",
        target_modules=args.target_modules.split(","),
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        logging_steps=50,
        save_strategy="epoch",
        optim="adamw_torch",
        max_length=args.max_seq_length,
        packing=True,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": args.min_lr_rate},
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # must be False for DDP
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        formatting_func=create_prompt,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune a causal LM with EWoRA on a mixed-task corpus")
    parser.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-v0.1", help="Base model path or name")
    parser.add_argument(
        "--datasets",
        type=str,
        default="ise-uiuc/Magicoder-Evol-Instruct-110K,meta-math/MetaMathQA,allenai/winogrande,Rowan/hellaswag",
        help="Comma-separated list of training datasets to concatenate",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/ewora_mixed", help="Output directory for the fine-tuned model"
    )
    parser.add_argument("--rank", type=int, default=8, help="Rank of each EWoRA expert adapter")
    parser.add_argument("--num_experts", type=int, default=4, help="Number of EWoRA experts")
    parser.add_argument("--ewora_dropout", type=float, default=0.1, help="EWoRA dropout rate")
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated list of target modules for EWoRA",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--min_lr_rate", type=float, default=0.1, help="Ratio of final learning rate to initial")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of warmup steps in the LR scheduler")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Optimizer weight decay")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum norm for gradient clipping")
    parser.add_argument(
        "--gradient_checkpointing", action="store_true", help="Use gradient checkpointing to save memory"
    )
    args = parser.parse_args()

    train_model(args)
