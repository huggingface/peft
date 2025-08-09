#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QLoRA training script (PEFT + Hugging Face Transformers + FlashAttention 2 + Liger-Kernel)
for causal language modeling on the Mielikki/Erebus-87k dataset, using the `body` column as text.

Quick start (recommended Python >= 3.10, CUDA 11.8/12.x with recent PyTorch):

pip install -U "transformers>=4.41.0" "peft>=0.10.0" "bitsandbytes>=0.43.1" datasets accelerate
pip install -U liger-kernel
# Install FlashAttention 2 prebuilt wheels (requires matching CUDA / PyTorch):
pip install flash-attn --no-build-isolation

Example run (Mistral 7B, with QLoRA + FA2 + Liger-Kernel):
python examples/causal_language_modeling/train_qlora_flash_liger_mielikki.py \
  --model_name_or_path mistralai/Mistral-7B-v0.3 \
  --output_dir ./outputs/mistral7b-qlora-erebus87k \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_seq_length 1024 \
  --learning_rate 2e-4 \
  --num_train_epochs 2 \
  --use_liger true \
  --attn_impl flash_attention_2 \
  --bf16 true

# FSDP (multi-GPU) example (experimental with 4-bit QLoRA):
# - Run with torchrun/accelerate on N GPUs
# - When using FSDP, the script disables device_map to let FSDP manage sharding
# - Adjust --fsdp_transformer_layer_cls_to_wrap to your architecture
torchrun --nproc_per_node=2 examples/causal_language_modeling/train_qlora_flash_liger_mielikki.py \
  --model_name_or_path mistralai/Mistral-7B-v0.3 \
  --output_dir ./outputs/mistral7b-qlora-erebus87k-fsdp \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_seq_length 1024 \
  --learning_rate 2e-4 \
  --num_train_epochs 2 \
  --use_liger true \
  --attn_impl flash_attention_2 \
  --bf16 true \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap MistralDecoderLayer

Notes:
- If FlashAttention 2 is not available, the script falls back to PyTorch SDPA automatically.
- Liger-Kernel is enabled by default with --use_liger true (works for many architectures like LLaMA/Mistral/Qwen/Gemma).
- The script trains adapters only (LoRA) and saves the PEFT adapter in output_dir.
- The dataset column used is `body` by default (per the dataset card).
"""

import argparse
import os
import math
import logging
from typing import List, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# Try to import Liger-Kernel AutoModel wrapper
_HAS_LIGER = False
try:
    from liger_kernel.transformers import AutoLigerKernelForCausalLM  # type: ignore

    _HAS_LIGER = True
except Exception:
    _HAS_LIGER = False

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def count_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total, 100 * trainable / total


def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA + FA2 + Liger-Kernel causal LM finetuning on Erebus-87k")
    parser.add_argument("--model_name_or_path", type=str, default="mistralai/Mistral-7B-v0.3")
    parser.add_argument("--dataset_name", type=str, default="Mielikki/Erebus-87k")
    parser.add_argument("--text_column", type=str, default="body")
    parser.add_argument("--output_dir", type=str, required=True)

    # Training hyperparameters
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")

    # Sequence / packing
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--packing", type=lambda x: str(x).lower() == "true", default=True)

    # Precision / attention kernels
    parser.add_argument("--bf16", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--fp16", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument(
        "--attn_impl", type=str, default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"]
    )  # fallback to sdpa if FA2 missing

    # Liger-Kernel
    parser.add_argument(
        "--use_liger",
        type=lambda x: str(x).lower() == "true",
        default=True,
        help="Use Liger-Kernel optimized kernels if available",
    )

    # LoRA config
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs="*",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        help="LoRA target modules; adjust per-architecture if needed",
    )

    # FSDP (experimental with 4-bit QLoRA)
    parser.add_argument("--fsdp", type=str, default="", help="FSDP config string, e.g. 'full_shard auto_wrap'")
    parser.add_argument("--fsdp_min_num_params", type=int, default=1_000_000)
    parser.add_argument(
        "--fsdp_transformer_layer_cls_to_wrap",
        type=str,
        nargs="*",
        default=["MistralDecoderLayer", "LlamaDecoderLayer", "Qwen2DecoderLayer", "GemmaDecoderLayer"],
    )
    parser.add_argument("--fsdp_cpu_offload", type=lambda x: str(x).lower() == "true", default=False)

    # Eval / logging
    parser.add_argument(
        "--val_size", type=float, default=0.01, help="Fraction for validation split; 0.0 disables eval"
    )
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=3)
    # Weights & Biases
    parser.add_argument(
        "--report_to", type=str, default="wandb", choices=["none", "wandb"], help="Where to report logs"
    )
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name")
    parser.add_argument(
        "--wandb_project", type=str, default=None, help="W&B project name; defaults to 'huggingface' if not set"
    )
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (team) name")
    parser.add_argument("--wandb_group", type=str, default=None, help="W&B group name for runs")
    parser.add_argument("--wandb_tags", type=str, nargs="*", default=None, help="W&B tags for the run")
    parser.add_argument(
        "--wandb_watch", type=str, default="false", choices=["false", "gradients", "all"], help="W&B watch setting"
    )
    parser.add_argument(
        "--wandb_log_model",
        type=str,
        default="false",
        choices=["false", "end", "checkpoint"],
        help="Log model checkpoints to W&B Artifacts",
    )
    parser.add_argument(
        "--wandb_mode", type=str, default=None, choices=["online", "offline", "disabled"], help="W&B mode override"
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    return parser.parse_args()


def get_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_and_prepare_datasets(
    dataset_name: str, text_column: str, tokenizer: AutoTokenizer, max_seq_length: int, packing: bool, val_size: float
):
    logger.info(f"Loading dataset: {dataset_name}")
    raw = load_dataset(dataset_name, split="train")

    # Filter out empty / None bodies
    def _filter_nonempty(example):
        txt = example.get(text_column, None)
        return isinstance(txt, str) and len(txt.strip()) > 0

    raw = raw.filter(_filter_nonempty)

    logger.info("Tokenizing dataset ...")

    def tokenize_function(examples):
        texts = examples[text_column]
        return tokenizer(texts, truncation=not packing, max_length=max_seq_length, padding=True)

    tokenized = raw.map(
        tokenize_function, batched=True, remove_columns=[c for c in raw.column_names if c != text_column]
    )

    if packing:
        logger.info("Packing sequences by concatenation and chunking ...")

        # Concatenate all texts then chunk into blocks of max_seq_length
        def group_texts(examples):
            # Concatenate lists of tokens
            concatenated = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated["input_ids"])
            # Drop the remainder
            total_length = (total_length // max_seq_length) * max_seq_length
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        tokenized = tokenized.map(group_texts, batched=True)
    else:
        tokenized = tokenized.map(lambda e: {"labels": e["input_ids"]})

    if val_size and val_size > 0.0:
        logger.info(f"Splitting train/validation with val_size={val_size}")
        split = tokenized.train_test_split(test_size=val_size, seed=42)
        return split["train"], split["test"]
    else:
        logger.info("val_size is 0.0; disabling evaluation and using all data for training")
        return tokenized, None


def build_model(args, tokenizer):
    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    attn_impl = args.attn_impl
    # Try FA2; if not installed, fallback to sdpa
    if attn_impl == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except Exception:
            logger.warning("flash-attn not available; falling back to SDPA (attn_implementation='sdpa').")
            attn_impl = "sdpa"

    use_liger = args.use_liger and _HAS_LIGER
    if args.use_liger and not _HAS_LIGER:
        logger.warning(
            "--use_liger was set but liger-kernel is not installed or import failed. Falling back to HF model."
        )

    if args.fsdp:
        logger.warning(
            "FSDP is enabled; device_map will be disabled to let FSDP manage sharding. FSDP + 4-bit QLoRA is experimental."
        )

    common_kwargs = dict(
        quantization_config=bnb_config,
        device_map=None if args.fsdp is None or len(args.fsdp) == 0 else "auto",
        attn_implementation=attn_impl,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )

    if use_liger:
        logger.info("Loading model with Liger-Kernel AutoModel wrapper ...")
        model = AutoLigerKernelForCausalLM.from_pretrained(args.model_name_or_path, **common_kwargs)
    else:
        logger.info("Loading standard HF AutoModelForCausalLM ...")
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **common_kwargs)

    # Ensure pad token id is set
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False  # required for gradient checkpointing
    model.gradient_checkpointing_enable()

    # Apply LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable, total, pct = count_trainable_parameters(model)
    logger.info(f"Trainable params: {trainable} / {total} ({pct:.2f}%)")
    return model


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tokenizer = get_tokenizer(args.model_name_or_path)

    train_dataset, eval_dataset = load_and_prepare_datasets(
        args.dataset_name,
        args.text_column,
        tokenizer,
        args.max_seq_length,
        args.packing,
        args.val_size,
    )

    model = build_model(args, tokenizer)

    # Setup Weights & Biases if requested
    wandb_run = None
    if args.report_to == "wandb":
        # Set common env vars so HF WandbCallback picks them up
        if args.wandb_project:
            os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity
        if args.wandb_watch:
            os.environ["WANDB_WATCH"] = args.wandb_watch
        if args.wandb_log_model:
            os.environ["WANDB_LOG_MODEL"] = args.wandb_log_model
        if args.wandb_mode == "offline":
            os.environ["WANDB_MODE"] = "offline"
        elif args.wandb_mode == "disabled":
            os.environ["WANDB_DISABLED"] = "true"
        # Optional manual init to customize more fields
        try:
            import wandb  # type: ignore

            wandb_kwargs = {}
            if args.wandb_project:
                wandb_kwargs["project"] = args.wandb_project
            if args.run_name:
                wandb_kwargs["name"] = args.run_name
            if args.wandb_entity:
                wandb_kwargs["entity"] = args.wandb_entity
            if args.wandb_group:
                wandb_kwargs["group"] = args.wandb_group
            if args.wandb_tags:
                wandb_kwargs["tags"] = args.wandb_tags
            if wandb_kwargs:
                wandb_run = wandb.init(**wandb_kwargs)
        except Exception as e:
            logger.warning(f"wandb not available or failed to initialize: {e}. Falling back to 'none'.")
            args.report_to = "none"

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Pick a BNB-aware optimizer for QLoRA
    optim = "adamw_bnb_8bit"

    eval_strategy = "steps" if eval_dataset is not None else "no"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy=eval_strategy,
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16 and not args.bf16,
        optim=optim,
        report_to=[args.report_to] if isinstance(args.report_to, str) else args.report_to,
        run_name=args.run_name,
        seed=args.seed,
        torch_compile=True,
        torch_compile_mode="max-autotune",
        gradient_checkpointing=True,
        fsdp=args.fsdp if args.fsdp else "",
        fsdp_config={
            "min_num_params": args.fsdp_min_num_params,
            "transformer_layer_cls_to_wrap": args.fsdp_transformer_layer_cls_to_wrap,
            "cpu_offload": args.fsdp_cpu_offload,
        }
        if args.fsdp
        else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Starting training ...")
    trainer.train()

    logger.info("Saving adapter (PEFT) ...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub:
        repo_id = args.hub_model_id if args.hub_model_id else os.path.basename(os.path.abspath(args.output_dir))
        logger.info(f"Pushing adapter to Hub: {repo_id}")
        trainer.model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)

    logger.info("Done.")

    # Finish W&B run if we started it manually
    try:
        if "wandb_run" in locals() and wandb_run is not None:
            import wandb  # type: ignore

            wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    main()
