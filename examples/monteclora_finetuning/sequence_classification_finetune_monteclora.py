import argparse
import os

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

# Assuming MonteCLoRA is available in your local installed PEFT version
from peft import MonteCLoraConfig, TaskType, get_peft_model


# ----------------------------------------------------------------------------
# 1. Custom Trainer Definition
# ----------------------------------------------------------------------------
# We subclass Trainer to inject the Variational Loss (KLD + Entropy)
# into the training loop.
class MonteCLoRATrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Overrides the default compute_loss to add MonteCLoRA variational regularization.
        """
        # 1. Compute standard task loss (CrossEntropy)
        if return_outputs:
            task_loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        else:
            task_loss = super().compute_loss(model, inputs, return_outputs=False)
            outputs = None

        # 2. Aggregating Variational Loss from MonteCLoRA layers
        var_loss_sum = 0.0
        num_monte_layers = 0

        # Iterate over modules to find MonteCLoRA layers (Linear or Sampler)
        # We use string matching to avoid import errors if specific classes aren't global
        for name, module in model.named_modules():
            if module.__class__.__name__ in ["MonteCLoRASampler", "MonteCLoRALinear"]:
                if hasattr(module, "get_variational_loss"):
                    l1, l2 = module.get_variational_loss()
                    var_loss_sum += l1 + l2
                    num_monte_layers += 1

        # 3. Normalize and combine
        regularization_loss = 0.0
        if num_monte_layers > 0:
            regularization_loss = var_loss_sum / num_monte_layers

        # Total loss = Task Loss + (KLD + Entropy)
        total_loss = task_loss + regularization_loss

        return (total_loss, outputs) if return_outputs else total_loss


# ----------------------------------------------------------------------------
# 2. Metrics Helper
# ----------------------------------------------------------------------------
# GLUE/MRPC uses Accuracy and F1 score
metric = evaluate.load("glue", "mrpc")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


# ----------------------------------------------------------------------------
# 3. Main Training Function
# ----------------------------------------------------------------------------
def train_model(
    base_model: str,
    output_dir: str,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    max_length: int,
    device: str,
    rank: int,
    lora_alpha: int,
    target_modules: str,
    n_samples: int,
    push_to_hub: bool,
    hub_model_id: str,
):
    hf_token = os.getenv("HF_TOKEN", "")

    # --- Device Setup ---
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)

    # --- Load Dataset (GLUE MRPC) ---
    # MRPC is a classification task (Is sentence B a paraphrase of sentence A?)
    dataset = load_dataset("glue", "mrpc")

    def tokenize_function(examples):
        return tokenizer(
            examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True, max_length=max_length
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Remove raw text columns to avoid Trainer warnings
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    # --- Load Base Model ---
    # num_labels=2 because MRPC is binary classification
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2, token=hf_token)

    # --- PEFT Configuration (MonteCLoRA) ---
    # Note: Using n_samples to control Monte Carlo iterations
    peft_config = MonteCLoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules.split(",") if target_modules else ["query", "value"],
        monteclora_n=n_samples,
        bias="none",
    )

    # Wrap model with PEFT
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    print(model)
    model.to(device)

    # --- Training Setup ---
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",  # Evaluate at end of every epoch
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",  # Optimize for F1 score
        logging_steps=10,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        hub_token=hf_token,
        remove_unused_columns=False,  # Important for PEFT sometimes
    )

    # Use our CUSTOM MonteCLoRATrainer
    trainer = MonteCLoRATrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],  # MRPC standard validation split
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting Training...")
    trainer.train()

    # --- Evaluation ---
    print("Evaluating...")
    eval_results = trainer.evaluate()
    print(f"Evaluation Results: {eval_results}")

    # --- Save & Push ---
    if push_to_hub:
        trainer.push_to_hub()

    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")


# ----------------------------------------------------------------------------
# 4. Entry Point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa on MRPC with MonteCLoRA")

    parser.add_argument("--base_model", type=str, default="roberta-base", help="Base model name")
    parser.add_argument("--output_dir", type=str, default="./monteclora-roberta-mrpc", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (per device)")
    parser.add_argument("--num_epochs", type=int, default=5, help="Training epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Learning rate"
    )  # Higher LR for PEFT is common
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")

    # MonteCLoRA specific args
    parser.add_argument("--rank", type=int, default=8, help="LoRA Rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA Alpha")
    parser.add_argument("--target_modules", type=str, default="query,value", help="Modules to apply adapter to")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of MC samples")

    parser.add_argument("--push_to_hub", action="store_true", help="Push to HF Hub")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Hub Repo ID")

    args = parser.parse_args()

    train_model(
        base_model=args.base_model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        device=args.device,
        rank=args.rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        n_samples=args.n_samples,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )
