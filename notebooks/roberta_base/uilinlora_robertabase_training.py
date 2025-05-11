import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch, evaluate, argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, BitsAndBytesConfig, Trainer
)
from peft import UILinLoRAConfig, get_peft_model, TaskType

# ---------------------------  custom trainer  --------------------------- #
class UILinLoRATrainer(Trainer):
    def __init__(self, *args, head_lr=1e-3, adapter_lr=4e-3, **kw):
        super().__init__(*args, **kw)
        self.head_lr, self.adapter_lr = head_lr, adapter_lr

    def create_optimizer(self):                       # two learning rates
        if self.optimizer is None:
            head, adapter = [], []
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    (head if "classifier" in n else adapter).append(p)
            groups = [{"params": head,    "lr": self.head_lr},
                      {"params": adapter, "lr": self.adapter_lr}]
            self.optimizer = torch.optim.AdamW(groups)
        return self.optimizer

# ---------------------------  helpers  --------------------------- #
def prepare_sst2_dataset(tokenizer, max_len=128):
    ds = load_dataset("glue", "sst2")
    ds = ds.map(
        lambda ex: tokenizer(ex["sentence"],
                             truncation=True,
                             padding="max_length",
                             max_length=max_len),
        batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return ds

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return accuracy.compute(predictions=preds, references=labels)

# ---------------------------  main  --------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.set_printoptions(threshold=float("inf"))
    global accuracy; accuracy = evaluate.load("accuracy")

    base_id = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(base_id, use_fast=True)

    base = AutoModelForSequenceClassification.from_pretrained(
        base_id, num_labels=2, device_map="auto"
    )

    uilinlora_cfg = UILinLoRAConfig(
            target_modules=["query", "value"],
            rank=128,
            uilinlora_alpha=1.0,
            uilinlora_dropout=0.0,
            fan_in_fan_out=False,
            init_uilinlora_weights=True,
            task_type=TaskType.SEQ_CLS)
    model = get_peft_model(base, uilinlora_cfg)
    model.classifier.requires_grad_(True)   # make head trainable
    model.config.pad_token_id = tokenizer.pad_token_id

    data = prepare_sst2_dataset(tokenizer)

    train_args = TrainingArguments(
        output_dir="uilinlora-roberta-base-sst2",
        per_device_train_batch_size=64,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        warmup_ratio=0.06,
        lr_scheduler_type="linear",
        logging_steps=50,
        save_total_limit=1,
        seed=args.seed,
    )

    trainer = UILinLoRATrainer(
        model=model,
        args=train_args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        head_lr=4e-3,
        adapter_lr=4e-3
    )

    trainer.train()
    print("Best-epoch accuracy:",
          trainer.evaluate()["eval_accuracy"])

if __name__ == "__main__":
    main()
