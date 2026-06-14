# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from peft import FrodConfig, get_peft_model


@dataclass
class FrodImageArguments:
    model_name_or_path: str = field(
        default="openai/clip-vit-base-patch32",
        metadata={"help": "Model checkpoint used for image classification."},
    )
    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Optional local Stanford Cars dataset directory containing the parquet data files."},
    )
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        metadata={"help": "Module names to replace with FRoD adapters."},
    )
    sparse_rate: float = field(
        default=0.01,
        metadata={"help": "Fraction of off-diagonal entries trained in the sparse FRoD matrix."},
    )
    frod_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout probability applied before the FRoD adapter branch."},
    )
    frod_lambda_l_lr: float = field(
        default=5e-4,
        metadata={"help": "Learning rate for the trainable diagonal FRoD coefficients."},
    )
    frod_lambda_s_lr: float = field(
        default=5e-5,
        metadata={"help": "Learning rate for the trainable sparse FRoD coefficients."},
    )
    classifier_lr: float = field(default=1e-4, metadata={"help": "Learning rate for the classification head."})
    projection_prng_key: int = field(default=3, metadata={"help": "Random seed used for FRoD projection masks."})
    runtime_offload_base_weight: bool = field(
        default=False,
        metadata={"help": "Move target base weights to CPU during active FRoD forward passes to reduce GPU memory."},
    )


@dataclass
class FrodImageTrainingArguments(TrainingArguments):
    output_dir: str = "clip-vit-base-patch32-frod-stanford-cars"
    learning_rate: float = 5e-4
    per_device_train_batch_size: int = 64
    per_device_eval_batch_size: int = 64
    num_train_epochs: float = 3
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "accuracy"
    lr_scheduler_type: str = "constant"
    remove_unused_columns: bool = False
    report_to: str = "none"


def main():
    parser = HfArgumentParser((FrodImageArguments, FrodImageTrainingArguments))
    frod_args, training_args = parser.parse_args_into_dataclasses()

    if frod_args.data_dir:
        data_files = {
            "train": [
                os.path.join(frod_args.data_dir, "data", "train-00000-of-00002.parquet"),
                os.path.join(frod_args.data_dir, "data", "train-00001-of-00002.parquet"),
            ],
            "test": [
                os.path.join(frod_args.data_dir, "data", "test-00000-of-00002.parquet"),
                os.path.join(frod_args.data_dir, "data", "test-00001-of-00002.parquet"),
            ],
        }
    else:
        data_files = {
            "train": [
                "hf://datasets/tanganke/stanford_cars/data/train-00000-of-00002.parquet",
                "hf://datasets/tanganke/stanford_cars/data/train-00001-of-00002.parquet",
            ],
            "test": [
                "hf://datasets/tanganke/stanford_cars/data/test-00000-of-00002.parquet",
                "hf://datasets/tanganke/stanford_cars/data/test-00001-of-00002.parquet",
            ],
        }

    dataset = load_dataset("parquet", data_files=data_files)
    train_split = dataset["train"]
    eval_split = dataset["test"]
    image_processor = AutoImageProcessor.from_pretrained(frod_args.model_name_or_path)
    label_feature = train_split.features["label"]
    label_names = (
        label_feature.names if hasattr(label_feature, "names") else [str(i) for i in sorted(set(train_split["label"]))]
    )
    id2label = dict(enumerate(label_names))
    label2id = {name: idx for idx, name in id2label.items()}

    model = AutoModelForImageClassification.from_pretrained(
        frod_args.model_name_or_path,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    peft_config = FrodConfig(
        target_modules=frod_args.target_modules,
        modules_to_save=["classifier"],
        frod_dropout=frod_args.frod_dropout,
        sparse_rate=frod_args.sparse_rate,
        projection_prng_key=frod_args.projection_prng_key,
        runtime_offload_base_weight=frod_args.runtime_offload_base_weight,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def transform(batch):
        images = [image.convert("RGB") for image in batch["image"]]
        inputs = image_processor(images, return_tensors="pt")
        inputs["labels"] = batch["label"]
        return inputs

    train_dataset = train_split.with_transform(transform)
    eval_dataset = eval_split.with_transform(transform)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["labels"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

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
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
    )
    trainer.train()
    trainer.evaluate()
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
