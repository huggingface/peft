# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from peft import FRODConfig, TaskType, get_peft_model


MODEL_NAME = "google-bert/bert-base-uncased"
DATASET_NAME = "nyu-mll/glue"
TASK_NAME = "sst2"
OUTPUT_DIR = "bert-base-uncased-frod-sst2"
FROD_LAMBDA_L_LR = 2e-2
FROD_LAMBDA_S_LR = 2e-3
CLASSIFIER_LR = 1e-2


def main():
    dataset = load_dataset(DATASET_NAME, TASK_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(batch):
        return tokenizer(batch["sentence"], truncation=True)

    tokenized = dataset.map(preprocess, batched=True)
    tokenized = tokenized.rename_column("label", "labels")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    peft_config = FRODConfig(
        task_type=TaskType.SEQ_CLS,
        target_modules=["query", "value"],
        modules_to_save=["classifier"],
        frod_dropout=0.0,
        sparse_rate=0.02,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

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
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
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
    model.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
