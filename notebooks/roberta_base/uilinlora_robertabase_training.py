import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch, evaluate, argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, BitsAndBytesConfig, Trainer
)
from peft import UILinLoRAConfig, get_peft_model, TaskType
from clearml import Task



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
def prepare_dataset(tokenizer, max_len=128, task="sst2"):
    ds = load_dataset("glue", task)
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
    return eval_metrics.compute(predictions=preds, references=labels)

def get_eval_metric_type(task):
    if task == "sst2":
        return "accuracy"
    elif task == "cola":
        return "matthews_correlation"
    elif task == "mrpc":
        return "f1"
    else:
        raise ValueError(f"Unsupported task: {task}")


def get_task_type(task):
    if task == "sst2":
        return TaskType.SEQ_CLS
    elif task == "cola":
        return TaskType.SEQ_CLS
    elif task == "mrpc":
        return TaskType.SEQ_CLS
    

def print_trainable_params(model):
    total = 0
    trainable = 0
    for name, param in model.named_parameters():
        if "classifier" in name:
            continue
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    print(f"Trainable parameters (excluding classifier): {trainable:,}")
    print(f"Total parameters (excluding classifier): {total:,}")
    print(f"Trainable %: {100 * trainable / total:.8f}%")


# ---------------------------  main  --------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--task",   type=str, default="sst2")
    parser.add_argument("--rank",   type=int, default=128)
    parser.add_argument("--head_lr",   type=float, default=4e-3)
    parser.add_argument("--adapter_lr",   type=float, default=4e-3)
    parser.add_argument("--batch_size",   type=int, default=64)
    parser.add_argument("--max_len", type=int, default=128)
    args = parser.parse_args()

    task = Task.init(
        project_name="GLUE benchmark", 
        task_name=f"UILinLoRA tuner - {args.task}"
    )
    task.connect(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.set_printoptions(threshold=float("inf"))
    eval_metric_type = get_eval_metric_type(args.task)
    global eval_metrics; eval_metrics = evaluate.load(eval_metric_type)

    base_id = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(base_id, use_fast=True)

    base = AutoModelForSequenceClassification.from_pretrained(
        base_id, num_labels=2, device_map="auto"
    )

    uilinlora_cfg = UILinLoRAConfig(
            target_modules=["query", "value"],
            rank=args.rank,
            uilinlora_alpha=1.0,
            uilinlora_dropout=0.0,
            fan_in_fan_out=False,
            init_uilinlora_weights=True,
            task_type=get_task_type(args.task))
    model = get_peft_model(base, uilinlora_cfg)
    model.classifier.requires_grad_(True)   # make head trainable
    model.config.pad_token_id = tokenizer.pad_token_id
    print_trainable_params(model)

    data = prepare_dataset(tokenizer, task=args.task, max_len=args.max_len)

    train_args = TrainingArguments(
        output_dir=f"uilinlora-roberta-base-{args.task}",
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=eval_metric_type,
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
        head_lr=args.head_lr,
        adapter_lr=args.adapter_lr
    )

    trainer.train()
    print(f"Best-epoch {eval_metric_type}:", trainer.evaluate()[f"eval_{eval_metric_type}"])


if __name__ == "__main__":
    main()
