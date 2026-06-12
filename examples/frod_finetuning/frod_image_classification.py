# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import os

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments

from peft import FrodConfig, get_peft_model


MODEL_NAME = os.environ.get("FROD_IMAGE_MODEL_NAME", "openai/clip-vit-base-patch32")
OUTPUT_DIR = os.environ.get("FROD_IMAGE_OUTPUT_DIR", "clip-vit-base-patch32-frod-stanford-cars")
DATA_DIR = os.environ.get("FROD_STANFORD_CARS_DATA_DIR")
NUM_TRAIN_EPOCHS = int(os.environ.get("FROD_IMAGE_NUM_TRAIN_EPOCHS", "3"))
TRAIN_BATCH_SIZE = int(os.environ.get("FROD_IMAGE_TRAIN_BATCH_SIZE", "64"))
EVAL_BATCH_SIZE = int(os.environ.get("FROD_IMAGE_EVAL_BATCH_SIZE", "64"))
SPARSE_RATE = float(os.environ.get("FROD_IMAGE_SPARSE_RATE", "0.01"))
FROD_LAMBDA_L_LR = float(os.environ.get("FROD_IMAGE_LAMBDA_L_LR", "5e-4"))
FROD_LAMBDA_S_LR = float(os.environ.get("FROD_IMAGE_LAMBDA_S_LR", "5e-5"))
CLASSIFIER_LR = float(os.environ.get("FROD_IMAGE_CLASSIFIER_LR", "1e-4"))
CLIP_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]


def main():
    if DATA_DIR:
        data_files = {
            "train": [
                os.path.join(DATA_DIR, "data", "train-00000-of-00002.parquet"),
                os.path.join(DATA_DIR, "data", "train-00001-of-00002.parquet"),
            ],
            "test": [
                os.path.join(DATA_DIR, "data", "test-00000-of-00002.parquet"),
                os.path.join(DATA_DIR, "data", "test-00001-of-00002.parquet"),
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
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    label_feature = train_split.features["label"]
    label_names = (
        label_feature.names if hasattr(label_feature, "names") else [str(i) for i in sorted(set(train_split["label"]))]
    )
    id2label = dict(enumerate(label_names))
    label2id = {name: idx for idx, name in id2label.items()}

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    peft_config = FrodConfig(
        target_modules=CLIP_TARGET_MODULES,
        modules_to_save=["classifier"],
        frod_dropout=0.0,
        sparse_rate=SPARSE_RATE,
        projection_prng_key=3,
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
            {"params": [p for n, p in model.named_parameters() if "frod_lambda_l" in n], "lr": FROD_LAMBDA_L_LR},
            {
                "params": [p for n, p in model.named_parameters() if "frod_lambda_s_values" in n],
                "lr": FROD_LAMBDA_S_LR,
            },
            {"params": [p for n, p in model.named_parameters() if "classifier" in n], "lr": CLASSIFIER_LR},
        ]
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=FROD_LAMBDA_L_LR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        lr_scheduler_type="constant",
        remove_unused_columns=False,
        report_to="none",
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
    model.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
