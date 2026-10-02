# Copyright 2023-present the HuggingFace Inc. team.
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

import argparse
import json
import os
from dataclasses import dataclass

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import FineGatesConfig, FineGatesTrainerMixin, TaskType, get_peft_model


class FineGatesTrainer(FineGatesTrainerMixin, Trainer):
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        head_learning_rate = getattr(self.args, "head_learning_rate", self.args.learning_rate)
        gates_learning_rate = getattr(self.args, "gates_learning_rate", self.args.learning_rate)
        weight_decay = self.args.weight_decay

        optimizer_grouped_parameters = [
            {
                "params": [
                    parameter
                    for name, parameter in self.model.named_parameters()
                    if parameter.requires_grad and ("finegates_" not in name)
                ],
                "lr": head_learning_rate,
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    parameter
                    for name, parameter in self.model.named_parameters()
                    if parameter.requires_grad and ("finegates_" in name)
                ],
                "lr": gates_learning_rate,
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=head_learning_rate)
        return self.optimizer

    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)

        base_model = getattr(self.model, "base_model", None)
        if (base_model is not None) and hasattr(base_model, "get_finegates_compression_stats"):
            compression = base_model.get_finegates_compression_stats()["total"]
            metrics["eval_compressed_params"] = compression["pruned_params"]
            metrics["eval_param_sparsity"] = compression["param_sparsity"]
            self.log(
                {
                    "eval_compressed_params": compression["pruned_params"],
                    "eval_param_sparsity": compression["param_sparsity"],
                }
            )

        return metrics


@dataclass
class ScriptArgs:
    model_name_or_path: str
    task_name: str
    output_dir: str
    target_modules: str
    target_sparsity: float
    sparsity_loss_weight: float
    learning_rate: float
    head_learning_rate: float | None
    gates_learning_rate: float | None
    weight_decay: float
    num_train_epochs: float
    warmup_steps: int
    max_grad_norm: float
    lr_scheduler_type: str
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    seed: int


def parse_args() -> ScriptArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
    parser.add_argument("--task_name", type=str, default="mrpc")
    parser.add_argument("--output_dir", type=str, default="./finegates-output")
    parser.add_argument(
        "--target_modules",
        type=str,
        default="query,key,value,attention.output.dense,intermediate.dense,output.dense",
    )
    parser.add_argument("--target_sparsity", type=float, default=0.4)
    parser.add_argument("--sparsity_loss_weight", type=float, default=50.0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--head_learning_rate", type=float, default=None)
    parser.add_argument("--gates_learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=float, default=10.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=0.0)
    parser.add_argument("--lr_scheduler_type", type=str, default="constant")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1948)
    parsed = parser.parse_args()
    return ScriptArgs(**vars(parsed))


def main():
    args = parse_args()
    set_seed(args.seed)
    head_learning_rate = args.head_learning_rate if args.head_learning_rate is not None else args.learning_rate
    gates_learning_rate = args.gates_learning_rate if args.gates_learning_rate is not None else args.learning_rate
    dataset = load_dataset("glue", args.task_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    metric = evaluate.load("glue", args.task_name)

    sentence1_key, sentence2_key = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }[args.task_name]

    def preprocess_function(examples):
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        return tokenizer(*texts, truncation=True)

    encoded = dataset.map(preprocess_function, batched=True)
    num_labels = 1 if args.task_name == "stsb" else len(set(dataset["train"]["label"]))
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=num_labels)

    peft_config = FineGatesConfig(
        task_type=TaskType.SEQ_CLS,
        target_modules=[module.strip() for module in args.target_modules.split(",") if module.strip()],
        modules_to_save=["classifier"],
        bias="finegates_only",
        target_sparsity=args.target_sparsity,
        sparsity_loss_weight=args.sparsity_loss_weight,
        gate_init_mean=1.0,
        gate_init_std=0.1,
        gate_noise_std=0.1,
    )
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=head_learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        evaluation_strategy="epoch",
        save_strategy="no",
        report_to=[],
    )
    training_args.head_learning_rate = head_learning_rate
    training_args.gates_learning_rate = gates_learning_rate

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if args.task_name == "stsb":
            predictions = predictions[:, 0]
        else:
            predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = FineGatesTrainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation_matched" if args.task_name == "mnli" else "validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.save_model(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
        json.dump(eval_metrics, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
