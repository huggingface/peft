import argparse

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
)

from peft import LoraConfig, TaskType, get_peft_model


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a BERT-based model on Relation Extraction using PEFT LoRA")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-base-uncased",
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate for LoRA fine-tuning",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=5,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/relation_extraction_peft",
        help="Output directory where model predictions and checkpoints will be written",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading dataset 'joelniklaus/sem_eval_2010_task_8' from parquet converter...")
    # Load dataset from the parquet branch to avoid remote script execution deprecation issues
    dataset = load_dataset("joelniklaus/sem_eval_2010_task_8", revision="refs/convert/parquet")

    # Get class labels
    features = dataset["train"].features
    label_list = features["relation"].names
    num_labels = len(label_list)
    id2label = dict(enumerate(label_list))
    label2id = {label: i for i, label in enumerate(label_list)}

    print(f"Relations classes: {label_list}")
    print(f"Number of classes: {num_labels}")

    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # For Relation Extraction, the nominals are marked with tags <e1>, </e1>, <e2>, </e2>.
    # Adding these tags as special tokens ensures they are tokenized as individual units,
    # and their representations can be adapted during training.
    special_tokens_dict = {"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>"]}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} special tokens to the tokenizer.")

    # Preprocessing function
    def preprocess_function(examples):
        # Tokenize the texts
        result = tokenizer(examples["sentence"], truncation=True, max_length=128)
        # Add labels
        result["label"] = examples["relation"]
        return result

    print("Tokenizing and preprocessing dataset splits...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # Load pretrained model
    print(f"Loading base sequence classification model: {args.model_name_or_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # Resize model token embeddings to accommodate the newly added special tokens
    model.resize_token_embeddings(len(tokenizer))
    print("Resized model token embeddings to match tokenizer vocabulary.")

    # Configure LoRA for Sequence Classification
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
    )

    print("Wrapping model with PEFT (LoRA)...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load metrics
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        acc = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
        f1_macro = f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]
        f1_micro = f1_metric.compute(predictions=predictions, references=labels, average="micro")["f1"]
        return {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
        }

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=100,
        seed=args.seed,
    )

    # Initialize data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],  # semeval test split is standard for eval
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Start training
    print("Starting fine-tuning with PEFT...")
    trainer.train()

    # Save model and tokenizer
    print(f"Saving PEFT adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Fine-tuning completed successfully!")


if __name__ == "__main__":
    main()
