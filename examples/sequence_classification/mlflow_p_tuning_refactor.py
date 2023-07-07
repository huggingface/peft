import argparse
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PromptEncoderConfig,
)
from datasets import load_dataset
import evaluate
import torch
import numpy as np
import mlflow

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script description")
    parser.add_argument("--model", type=str, default="roberta-large", help="Model name or path")
    parser.add_argument("--task", type=str, default="mrpc", help="Task name")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output dir")

    args = parser.parse_args()

    # mlflow init
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    model_name_or_path = args.model
    task = args.task
    num_epochs = args.num_epochs
    lr = args.lr
    batch_size = args.batch_size
    output = args.output_dir

    dataset = load_dataset("glue", task)
    metric = evaluate.load("glue", task)

    if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def tokenize_function(examples):
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
        return outputs

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence1", "sentence2"],
    )

    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    peft_config = PromptEncoderConfig(task_type="SEQ_CLS", num_virtual_tokens=20, encoder_hidden_size=128)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)
    model = get_peft_model(model, peft_config)
    training_args = TrainingArguments(
        output_dir=output,
        learning_rate=lr,
per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # mlflow initial
    experiment_id = mlflow.create_experiment('p_tuning_seq_cls-{}'.format(model_name_or_path))
    experiment = mlflow.get_experiment(experiment_id)
    mlflow_runner = mlflow.start_run(run_name=model_name_or_path, experiment_id=experiment.experiment_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    with mlflow_runner:
        train_results = trainer.train()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_model()
        trainer.save_state()

        mlflow.log_metric('loss', train_results.metrics["train_loss"])
        mlflow.log_metric('runtime', train_results.metrics["train_runtime"])
        mlflow.log_metric('samples_per_second', train_results.metrics["train_samples_per_second"])
        mlflow.log_metric('steps_per_second', train_results.metrics["train_steps_per_second"])
        mlflow.end_run()

        # Evaluate the model on the validation set
        trainer.evaluate(tokenized_datasets["validation"])
