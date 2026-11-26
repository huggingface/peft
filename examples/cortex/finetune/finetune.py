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

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import CortexConfig, get_peft_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--train-file", required=True, help="JSONL file with a 'text' or a 'messages' field per line")
    parser.add_argument("--output-dir", default="outputs/cortex-adapter")
    parser.add_argument("--k", type=int, default=64, help="Number of channels added to each FFN")
    parser.add_argument("--init-weights", choices=["random", "sample"], default="sample")
    parser.add_argument("--moe-target", choices=["shared", "routed", "both"], default="shared")
    parser.add_argument("--cortex-bias", choices=["auto", "none"], default="auto")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="Overrides --epochs if positive")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(row):
        # Chat records are rendered with the chat template of the model, which already adds the special tokens.
        if row.get("messages") is not None:
            text = tokenizer.apply_chat_template(row["messages"], tokenize=False)
            encoding = tokenizer(text, add_special_tokens=False, truncation=True, max_length=args.max_length)
        else:
            encoding = tokenizer(row["text"], truncation=True, max_length=args.max_length)

        # The loss is computed on all tokens.
        encoding["labels"] = list(encoding["input_ids"])
        return encoding

    dataset = load_dataset("json", data_files=args.train_file, split="train")
    dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, dtype=getattr(torch, args.dtype))
    config = CortexConfig(
        k=args.k,
        init_weights=args.init_weights,
        moe_target=args.moe_target,
        cortex_bias=args.cortex_bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        bf16=args.dtype == "bfloat16",
        use_cpu=args.use_cpu,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )

    # Pad the labels with -100, which the loss ignores. `DataCollatorForLanguageModeling` would instead ignore all
    # tokens that are equal to the pad token, i.e. also the EOS tokens if the EOS token is used for padding.
    data_collator = DataCollatorForSeq2Seq(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    trainer.train()

    # Saves the adapter and the tokenizer, only on the main process when training on multiple GPUs.
    trainer.save_model()


if __name__ == "__main__":
    main()
