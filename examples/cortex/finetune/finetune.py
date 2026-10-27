# Copyright 2026-present the HuggingFace Inc. team.
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

"""Fine-tune a dense or MoE causal LM with Cortex and save the adapter."""

import argparse
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import CortexConfig, get_peft_model


# This example is also runnable directly from either subdirectory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chat_utils import tokenize_assistant_turn
from training_utils import warmup_kwargs


def training_messages(row: dict) -> list[dict] | None:
    """Validate conversational JSONL, or use plain text for language modeling."""
    if row.get("messages") is not None:
        messages = row["messages"]
    elif isinstance(row.get("text"), str) and row["text"].strip():
        return None
    else:
        raise ValueError("Expected a messages list or nonempty text.")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("A chat record needs prompt messages and a final assistant answer.")
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise TypeError("Every chat message needs a role and string content.")
        if message.get("role") not in {"system", "developer", "user", "assistant"}:
            raise ValueError("This example accepts system, developer, user, and assistant messages.")
    if messages[-2]["role"] != "user" or messages[-1]["role"] != "assistant":
        raise ValueError("Chat records must end with a user question and an assistant answer.")
    if not messages[-1]["content"].strip():
        raise ValueError("The final assistant answer must not be empty.")
    return deepcopy(messages)


def prepare_training_record(
    row: dict,
    tokenizer,
    model_type: str,
    *,
    max_length: int,
    train_reasoning: bool = True,
    reasoning_effort: str = "medium",
    statistics: dict | None = None,
) -> dict[str, list[int]]:
    """Right-truncate exact native tokens and supervise the final assistant turn."""
    if max_length < 2:
        raise ValueError("The training length must retain at least two tokens.")
    messages = training_messages(row)
    if messages is None:
        full_ids = tokenizer(row["text"], truncation=False)["input_ids"]
        answer_start = 0
    else:
        full_ids, answer_start = tokenize_assistant_turn(
            tokenizer,
            messages,
            model_type,
            include_reasoning=train_reasoning,
            reasoning_effort=reasoning_effort,
        )
    ids = full_ids[:max_length]
    if statistics is not None:
        statistics["truncated_records"] += int(len(full_ids) > len(ids))
        statistics["records"] += 1
    if len(ids) < 2 or len(ids) <= answer_start:
        if statistics is not None:
            statistics["records_without_targets"] += 1
        raise ValueError(
            "Right truncation leaves no assistant tokens to supervise "
            f"(original length={len(full_ids)}, answer starts at={answer_start}, max_length={max_length})."
        )
    labels = [-100] * answer_start + ids[answer_start:]
    if statistics is not None:
        statistics["supervised_tokens"] += sum(label != -100 for label in labels[1:])
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--revision", help="Pin the base model and tokenizer to a Hub revision")
    parser.add_argument("--train-file", type=Path, required=True, help="JSONL with messages or text")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cortex-adapter"))
    parser.add_argument("--merge-output", type=Path, help="Optionally export a native merged model")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--init-weights", choices=["random", "sample"], default="sample")
    parser.add_argument("--moe-target", nargs="+", choices=["shared", "routed"], default=["shared"])
    parser.add_argument("--public-weighting", choices=["unweighted", "routing"], default="unweighted")
    parser.add_argument("--adapter-bias", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--adapter-bias-init", choices=["uniform", "zeros"], default="uniform")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument(
        "--experts-implementation",
        choices=["eager", "grouped_mm"],
        help="Override the native MoE experts backend",
    )
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps", type=int, default=-1, help="Override the epoch budget; -1 uses epochs")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-train-length", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-scheduler-type", choices=["cosine", "linear", "constant"], default="cosine")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-reasoning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the final assistant's reasoning in CE; --no-train-reasoning omits it",
    )
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high"], default="medium")
    args = parser.parse_args(argv)
    if min(args.max_train_length, args.batch_size, args.gradient_accumulation_steps) < 1:
        parser.error("Lengths, batch size, and gradient accumulation must be positive.")
    if args.max_steps != -1 and args.max_steps < 1:
        parser.error("--max-steps must be positive or -1 to use epochs.")
    if not math.isfinite(args.epochs) or args.epochs <= 0:
        parser.error("--epochs must be finite and positive.")
    if not 0 <= args.warmup_ratio < 1 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("Warmup ratio must be in [0, 1); weight decay must be finite and nonnegative.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be finite and positive.")
    if args.k < 1:
        parser.error("--k must be positive.")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.output_dir.exists():
        raise ValueError("Choose a new --output-dir for each training run.")
    if args.merge_output is not None and args.merge_output.exists():
        raise ValueError("Choose a new --merge-output directory.")
    if Path(args.model_name_or_path).is_dir():
        args.model_name_or_path = str(Path(args.model_name_or_path).resolve())
    set_seed(args.seed)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        **warmup_kwargs(args.warmup_ratio),
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        optim="adamw_torch",
        bf16=args.dtype == "bfloat16",
        use_cpu=args.use_cpu,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=args.seed,
    )
    if training_args.world_size != 1 or training_args.n_gpu > 1:
        raise ValueError("This example uses one device. Select one accelerator, for example CUDA_VISIBLE_DEVICES=0.")
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dataset = load_dataset("json", data_files=str(args.train_file), split="train")

    if not len(dataset):
        raise ValueError("The training file must contain at least one record.")
    config = AutoConfig.from_pretrained(args.model_name_or_path, revision=args.revision)
    if config.model_type == "gpt_oss" and getattr(config, "quantization_config", None):
        raise ValueError("Use a dequantized BF16 GPT-OSS checkpoint for this example, not the MXFP4 Hub weights.")
    chat_mode = training_messages(dataset[0]) is not None
    if any((training_messages(row) is not None) != chat_mode for row in dataset):
        raise ValueError("Use one input format per file: chat records or plain text.")

    preprocessing = {"records": 0, "truncated_records": 0, "records_without_targets": 0, "supervised_tokens": 0}
    dataset = dataset.map(
        lambda row: prepare_training_record(
            row,
            tokenizer,
            config.model_type,
            max_length=args.max_train_length,
            train_reasoning=args.train_reasoning,
            reasoning_effort=args.reasoning_effort,
            statistics=preprocessing,
        ),
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
    )
    print(json.dumps({"preprocessing": preprocessing}, sort_keys=True))
    load_kwargs = {
        "dtype": dtype,
        "use_kernels": False,
    }
    if args.experts_implementation is not None:
        load_kwargs["experts_implementation"] = args.experts_implementation
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, revision=args.revision, **load_kwargs)
    base_model.config.use_cache = False
    cortex_config = CortexConfig(
        k=args.k,
        init_weights=args.init_weights,
        moe_target=args.moe_target,
        public_weighting=args.public_weighting,
        adapter_bias=args.adapter_bias,
        adapter_bias_init=args.adapter_bias_init,
        task_type="CAUSAL_LM",
        revision=args.revision,
    )
    model = get_peft_model(base_model, cortex_config)
    model.print_trainable_parameters()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    trainer.train()
    model = trainer.accelerator.unwrap_model(trainer.model_wrapped, keep_fp32_wrapper=False)
    model.save_pretrained(args.output_dir, save_embedding_layers=False)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adapter and tokenizer saved to {args.output_dir}.")

    if args.merge_output is not None:
        merged = model.merge_and_unload(safe_merge=True)
        merged.save_pretrained(args.merge_output)
        tokenizer.save_pretrained(args.merge_output)
        print(f"Native model saved to {args.merge_output}.")


if __name__ == "__main__":
    main()
