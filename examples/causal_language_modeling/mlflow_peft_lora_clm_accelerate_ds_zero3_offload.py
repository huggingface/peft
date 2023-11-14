import gc
import os
import sys
import threading
import argparse

import numpy as np
import psutil
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
    set_seed,
)

from peft import LoraConfig, TaskType, get_peft_model
import mlflow
import time

# mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
# if (not mlflow_uri):
#     mlflow_uri = "http://127.0.0.1:5001"
#     mlflow.set_tracking_uri(mlflow_uri)

def get_num_parameters(model):
  num_params = 0
  for param in model.parameters():
    num_params += param.numel()
  # in million
  num_params /= 10**6
  return num_params

def levenshtein_distance(str1, str2):
    # TC: O(N^2)
    # SC: O(N^2)
    if str1 == str2:
        return 0
    num_rows = len(str1) + 1
    num_cols = len(str2) + 1
    dp_matrix = np.empty((num_rows, num_cols))
    dp_matrix[0, :] = range(num_cols)
    dp_matrix[:, 0] = range(num_rows)

    for i in range(1, num_rows):
        for j in range(1, num_cols):
            if str1[i - 1] == str2[j - 1]:
                dp_matrix[i, j] = dp_matrix[i - 1, j - 1]
            else:
                dp_matrix[i, j] = min(dp_matrix[i - 1, j - 1], dp_matrix[i - 1, j], dp_matrix[i, j - 1]) + 1

    return dp_matrix[num_rows - 1, num_cols - 1]


def get_closest_label(eval_pred, classes):
    min_id = sys.maxsize
    min_edit_distance = sys.maxsize
    for i, class_label in enumerate(classes):
        edit_distance = levenshtein_distance(eval_pred.strip(), class_label)
        if edit_distance < min_edit_distance:
            min_id = i
            min_edit_distance = edit_distance
    return classes[min_id]


# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2**20)


# This context manager is used to track the peak memory usage of the process
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")


def main(args):
    accelerator = Accelerator(mixed_precision=args.amp)
    model_name_or_path = args.model_name_or_path
    dataset_name = args.dataset_name
    text_column = args.text_column
    label_column = args.label_column
    lr = args.lr
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    seed = args.seed
    checkpoint_dir = args.checkpoint_dir
    max_length = args.max_length
    do_test = args.do_test
    cache_dir = args.cache_dir
    set_seed(seed)
    

    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    dataset = load_dataset("ought/raft", dataset_name, cache_dir=cache_dir)
    classes = [k.replace("_", " ") for k in dataset["train"].features["Label"].names]
    dataset = dataset.map(
        lambda x: {"text_label": [classes[label] for label in x["Label"]]},
        batched=True,
        num_proc=1,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    def preprocess_function(examples):
        batch_size = len(examples[text_column])
        inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
        targets = [str(x) for x in examples[label_column]]
        model_inputs = tokenizer(inputs)
        labels = tokenizer(targets)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]
            labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def test_preprocess_function(examples):
        batch_size = len(examples[text_column])
        inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
        model_inputs = tokenizer(inputs)
        # print(model_inputs)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
                "attention_mask"
            ][i]
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
            model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        return model_inputs

    with accelerator.main_process_first():
        processed_datasets = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset",
        )
    accelerator.wait_for_everyone()

    train_dataset = processed_datasets["train"]

    # mlflow initial
    # experiment_id = mlflow.create_experiment('casual-language-modeling-{}'.format(model_name_or_path))
    # experiment = mlflow.get_experiment(experiment_id)
    # mlflow_runner = mlflow.start_run(run_name=model_name_or_path, experiment_id=experiment.experiment_id)

    with accelerator.main_process_first():
        processed_datasets = dataset.map(
            test_preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
    eval_dataset = processed_datasets["train"]
    test_dataset = processed_datasets["test"]

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
    )

    print(next(iter(train_dataloader)))

    # creating model
    
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, pad_token_id=tokenizer.eos_token_id, trust_remote_code=True)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # optimizer
    if args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    # lr scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )
# Load checkpoint
    if args.load_checkpoint == "True":
        if os.path.exists(checkpoint_dir):
            checkpoint = torch.load(checkpoint_dir)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
        else:
            pass

    model, train_dataloader, eval_dataloader, test_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, train_dataloader, eval_dataloader, test_dataloader, optimizer, lr_scheduler
    )
    accelerator.print(model)
    accelerator.print(f"  Num examples = {len(train_dataset)}")
    accelerator.print(f"  Num Epochs = {num_epochs}")
    accelerator.print(f"  Num batch sizes = {batch_size}")
    is_ds_zero_3 = False
    if getattr(accelerator.state, "deepspeed_plugin", None):
        is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3


    # with mlflow_runner:
    mlflow.start_run()

    num_params = get_num_parameters(model)
    mlflow.log_param('num_params', num_params)
    
    elapsed = 0
    epoch_runtime_list = []
    for epoch in range(num_epochs):
        start_time = time.time()  # Start time for the epoch
        with TorchTracemalloc() as tracemalloc:
            model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader)):
                start_time_step = time.time()
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                elapsed += time.time() - start_time_step
                total_steps = epoch * len(train_dataloader) + step + 1
                if total_steps % args.log_interval == 0:
                    thoughput = total_steps * args.batch_size / elapsed
                    mlflow.log_metric('throughput', thoughput, step=total_steps)
                    mlflow.log_metric('loss', loss, step=total_steps)
                    mlflow.log_metric('lr', lr_scheduler.get_last_lr()[0], step=total_steps)
                
            end_time = time.time()  # End time for the epoch

            # Calculate metrics
            epoch_runtime = end_time - start_time
            epoch_runtime_list.append(epoch_runtime)

            # Log metrics for the epoch
            mlflow.log_metric('epoch_time', epoch_runtime, step=epoch)        
        
        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print("GPU Memory before entering the train : {}".format(b2mb(tracemalloc.begin)))
        accelerator.print("GPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.used))
        accelerator.print("GPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.peaked))
        accelerator.print(
            "GPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.peaked + b2mb(tracemalloc.begin)
            )
        )

        accelerator.print("CPU Memory before entering the train : {}".format(b2mb(tracemalloc.cpu_begin)))
        accelerator.print("CPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.cpu_used))
        accelerator.print("CPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.cpu_peaked))
        accelerator.print(
            "CPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
            )
        )
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")

        model.eval()
        eval_preds = []
        with TorchTracemalloc() as tracemalloc:
            for _, batch in enumerate(tqdm(eval_dataloader)):
                batch = {k: v for k, v in batch.items() if k != "labels"}
                with torch.no_grad():
                    outputs = accelerator.unwrap_model(model).generate(
                        **batch, synced_gpus=is_ds_zero_3, max_new_tokens=10
                    )  # synced_gpus=True for DS-stage 3
                outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id)
                preds = accelerator.gather_for_metrics(outputs)
                preds = preds[:, max_length:].detach().cpu().numpy()
                eval_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))

        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print("GPU Memory before entering the eval : {}".format(b2mb(tracemalloc.begin)))
        accelerator.print("GPU Memory consumed at the end of the eval (end-begin): {}".format(tracemalloc.used))
        accelerator.print("GPU Peak Memory consumed during the eval (max-begin): {}".format(tracemalloc.peaked))
        accelerator.print(
            "GPU Total Peak Memory consumed during the eval (max): {}".format(
                tracemalloc.peaked + b2mb(tracemalloc.begin)
            )
        )

        accelerator.print("CPU Memory before entering the eval : {}".format(b2mb(tracemalloc.cpu_begin)))
        accelerator.print("CPU Memory consumed at the end of the eval (end-begin): {}".format(tracemalloc.cpu_used))
        accelerator.print("CPU Peak Memory consumed during the eval (max-begin): {}".format(tracemalloc.cpu_peaked))
        accelerator.print(
            "CPU Total Peak Memory consumed during the eval (max): {}".format(
                tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
            )
        )

        correct = 0
        total = 0
        assert len(eval_preds) == len(
            dataset["train"][label_column]
        ), f"{len(eval_preds)} != {len(dataset['train'][label_column])}"
        for pred, true in zip(eval_preds, dataset["train"][label_column]):
            if pred.strip() == true.strip():
                correct += 1
            total += 1
        accuracy = correct / total * 100
        accelerator.print(f"{accuracy=}")
        accelerator.print(f"{eval_preds[:10]=}")
        accelerator.print(f"{dataset['train'][label_column][:10]=}")
    
    avg_throughput = len(train_dataloader) * args.batch_size*num_epochs / sum(epoch_runtime_list)
    mlflow.log_metric('avg_throughput', avg_throughput)
    mlflow.end_run()
    if do_test:
        model.eval()
        test_preds = []
        for _, batch in enumerate(tqdm(test_dataloader)):
            batch = {k: v for k, v in batch.items() if k != "labels"}
            with torch.no_grad():
                outputs = accelerator.unwrap_model(model).generate(
                    **batch, synced_gpus=is_ds_zero_3, max_new_tokens=10
                )  # synced_gpus=True for DS-stage 3
            outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id)
            preds = accelerator.gather(outputs)
            preds = preds[:, max_length:].detach().cpu().numpy()
            test_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))

        test_preds_cleaned = []
        for _, pred in enumerate(test_preds):
            test_preds_cleaned.append(get_closest_label(pred, classes))

        test_df = dataset["test"].to_pandas()
        assert len(test_preds_cleaned) == len(test_df), f"{len(test_preds_cleaned)} != {len(test_df)}"
        test_df[label_column] = test_preds_cleaned
        test_df["text_labels_orig"] = test_preds
        accelerator.print(test_df[[text_column, label_column]].sample(20))

        pred_df = test_df[["ID", label_column]]
        pred_df.columns = ["ID", "Label"]

        os.makedirs(f"data/{dataset_name}", exist_ok=True)
        pred_df.to_csv(f"data/{dataset_name}/predictions.csv", index=False)

    accelerator.wait_for_everyone()
    if args.save_checkpoint == "True":
        if epoch == num_epochs - 1:
            os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)
            torch.save({
            'epoch': num_epochs - 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
            }, checkpoint_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Indexing elasticsearch documents.')
    parser.add_argument('--model_name_or_path', type=str, default='facebook/opt-125m', help='Path to pretrained model or model identifier from huggingface.co/models.')
    parser.add_argument('--dataset_name', type=str, default='twitter_complaints', help='The name of the Dataset (from the HuggingFace hub) to train on.')
    parser.add_argument('--text_column', type=str, default="Tweet text", help='text column.')
    parser.add_argument('--label_column', type=str, default="text_label", help='label column.')
    parser.add_argument('--lr', type=float, default=3e-3, help='learning rate .')
    parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs .')
    parser.add_argument('--batch_size', type=int, default=900, help='training batch size .')
    parser.add_argument('--seed', type=int, default=42, help='A seed for reproducible training.')
    parser.add_argument('--max_length', type=int, default=64, help='model max length.')
    parser.add_argument('--do_test', type=bool, default=False, help='do test.')
    parser.add_argument('--log_interval', type=int, default=1, help='log interval.')
    parser.add_argument('--cache_dir', type=str, default=None, help='Directory to read/write data.')
    parser.add_argument("--amp", type=str, choices=["bf16", "fp16", "no"], default="fp16", help="Choose AMP mode")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Choose the optimization computation method")
    parser.add_argument('--checkpoint_dir', type=str, default='/nas/test_case_set_{set#}/a100/{benchmark}/{task}/{model_name}/10epoch', help='Directory to save checkpoints.')
    parser.add_argument('--load_checkpoint', type=str, default="False", help='Load checkpoint or not.')
    parser.add_argument('--save_checkpoint', type=str, default="False", help='Save checkpoint or not.')
    args = parser.parse_args()
    main(args)
