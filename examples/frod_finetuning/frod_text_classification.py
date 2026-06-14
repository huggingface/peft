# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from dataclasses import dataclass, field

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from peft import FrodConfig, TaskType, get_peft_model


@dataclass
class FrodTextArguments:
    model_name_or_path: str = field(
        default="google-bert/bert-base-uncased",
        metadata={"help": "Model checkpoint used for sequence classification."},
    )
    dataset_name: str = field(default="nyu-mll/glue", metadata={"help": "Dataset name or local dataset path."})
    task_name: str = field(default="sst2", metadata={"help": "Dataset configuration name."})
    target_modules: list[str] = field(
        default_factory=lambda: ["query", "value"],
        metadata={"help": "Module names to replace with FRoD adapters."},
    )
    sparse_rate: float = field(
        default=0.02,
        metadata={"help": "Fraction of off-diagonal entries trained in the sparse FRoD matrix."},
    )
    frod_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout probability applied before the FRoD adapter branch."},
    )
    frod_lambda_l_lr: float = field(
        default=2e-2,
        metadata={"help": "Learning rate for the trainable diagonal FRoD coefficients."},
    )
    frod_lambda_s_lr: float = field(
        default=2e-3,
        metadata={"help": "Learning rate for the trainable sparse FRoD coefficients."},
    )
    classifier_lr: float = field(default=1e-2, metadata={"help": "Learning rate for the classification head."})
    runtime_offload_base_weight: bool = field(
        default=False,
        metadata={"help": "Move target base weights to CPU during active FRoD forward passes to reduce GPU memory."},
    )


@dataclass
class FrodTextTrainingArguments(TrainingArguments):
    output_dir: str = "bert-base-uncased-frod-sst2"
    learning_rate: float = 2e-2
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    num_train_epochs: float = 1
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "accuracy"
    report_to: str = "none"


def main():
    parser = HfArgumentParser((FrodTextArguments, FrodTextTrainingArguments))
    frod_args, training_args = parser.parse_args_into_dataclasses()

    dataset = load_dataset(frod_args.dataset_name, frod_args.task_name)
    tokenizer = AutoTokenizer.from_pretrained(frod_args.model_name_or_path)

    def preprocess(batch):
        return tokenizer(batch["sentence"], truncation=True)

    tokenized = dataset.map(preprocess, batched=True)
    tokenized = tokenized.rename_column("label", "labels")

    model = AutoModelForSequenceClassification.from_pretrained(frod_args.model_name_or_path, num_labels=2)
    peft_config = FrodConfig(
        task_type=TaskType.SEQ_CLS,
        target_modules=frod_args.target_modules,
        modules_to_save=["classifier"],
        frod_dropout=frod_args.frod_dropout,
        sparse_rate=frod_args.sparse_rate,
        runtime_offload_base_weight=frod_args.runtime_offload_base_weight,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=-1)
        return {"accuracy": (predictions == eval_pred.label_ids).mean().item()}

    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for n, p in model.named_parameters() if "frod_lambda_l" in n],
                "lr": frod_args.frod_lambda_l_lr,
            },
            {
                "params": [p for n, p in model.named_parameters() if "frod_lambda_s_values" in n],
                "lr": frod_args.frod_lambda_s_lr,
            },
            {"params": [p for n, p in model.named_parameters() if "classifier" in n], "lr": frod_args.classifier_lr},
        ]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
    )
    trainer.train()
    trainer.evaluate()
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
