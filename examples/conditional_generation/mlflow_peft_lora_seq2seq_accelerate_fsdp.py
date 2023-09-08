import os

import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup

from peft import LoraConfig, TaskType, get_peft_model
from peft.utils.other import fsdp_auto_wrap_policy
import mlflow
import time


def get_num_parameters(model):
  num_params = 0
  for param in model.parameters():
    num_params += param.numel()
  # in million
  num_params /= 10**6
  return num_params

def main():
    accelerator = Accelerator()
    model_name_or_path = "t5-base"
    batch_size = 8
    text_column = "sentence"
    label_column = "label"
    max_length = 64
    lr = 1e-3
    num_epochs = 1
    base_path = "temp/data/FinancialPhraseBank-v1.0"

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    model = get_peft_model(model, peft_config)
    accelerator.print(model.print_trainable_parameters())

    dataset = load_dataset(
        "json",
        data_files={
            "train": os.path.join(base_path, "financial_phrase_bank_train.jsonl"),
            "validation": os.path.join(base_path, "financial_phrase_bank_val.jsonl"),
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[label_column]
        model_inputs = tokenizer(
            inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        labels = tokenizer(targets, max_length=2, padding="max_length", truncation=True, return_tensors="pt")
        labels = labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs

    # mlflow initial
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if (not mlflow_uri):
        mlflow_uri = "http://127.0.0.1:5001"
        mlflow.set_tracking_uri(mlflow_uri)

    experiment_id = mlflow.create_experiment('conditional_generation-{}'.format(model_name_or_path))
    experiment = mlflow.get_experiment(experiment_id)
    mlflow_runner = mlflow.start_run(run_name=model_name_or_path, experiment_id=experiment.experiment_id)
    num_params = get_num_parameters(model)
    mlflow.log_param('num_params', num_params)

    with accelerator.main_process_first():
        processed_datasets = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    if getattr(accelerator.state, "fsdp_plugin", None) is not None:
        accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)

    model, train_dataloader, eval_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, eval_dataloader, optimizer, lr_scheduler
    )
    accelerator.print(model)

    with mlflow_runner:
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            start_time = time.time()  # Start time for the epoch
            for step, batch in enumerate(tqdm(train_dataloader)):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            end_time = time.time()  # End time for the epoch

            # Calculate metrics
            epoch_runtime = end_time - start_time
            samples_per_second = len(train_dataloader) / epoch_runtime
            steps_per_second = len(train_dataloader) / epoch_runtime
            avg_loss = total_loss / len(train_dataloader)

            # Log metrics for the epoch
            mlflow.log_metric('loss', avg_loss)
            mlflow.log_metric('total_loss', total_loss)
            mlflow.log_metric('train_runtime', epoch_runtime)
            mlflow.log_metric('train_samples_per_second', samples_per_second)
            mlflow.log_metric('train_steps_per_second', steps_per_second)

            model.eval()
            eval_loss = 0
            eval_preds = []
            for step, batch in enumerate(tqdm(eval_dataloader)):
                with torch.no_grad():
                    outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()
                preds = accelerator.gather_for_metrics(torch.argmax(outputs.logits, -1)).detach().cpu().numpy()
                eval_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))
            eval_epoch_loss = eval_loss / len(eval_dataloader)
            eval_ppl = torch.exp(eval_epoch_loss)
            train_epoch_loss = total_loss / len(train_dataloader)
            train_ppl = torch.exp(train_epoch_loss)
            accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

            correct = 0
            total = 0
            for pred, true in zip(eval_preds, dataset["validation"][label_column]):
                if pred.strip() == true.strip():
                    correct += 1
                total += 1
            accuracy = correct / total * 100
            accelerator.print(f"{accuracy=}")
            accelerator.print(f"{eval_preds[:10]=}")
            accelerator.print(f"{dataset['validation'][label_column][:10]=}")
            
            accelerator.wait_for_everyone()
        mlflow.end_run()
        


if __name__ == "__main__":
    main()
