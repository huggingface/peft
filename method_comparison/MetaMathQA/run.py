# Copyright 2025-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Main entry point to run the experiments. Contains general setup and the proper training code.
"""

import argparse
import datetime as dt
import gc
import json
import os
import random
import sys
import textwrap
import time
from contextlib import nullcontext
from typing import Any, Literal, Optional

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import GenerationConfig, get_cosine_schedule_with_warmup, set_seed
from utils import (
    FILE_NAME_TRAIN_PARAMS,
    TrainResult,
    TrainStatus,
    get_accuracy,
    get_base_model_info,
    get_dataset_info,
    get_model,
    get_tokenizer,
    get_train_config,
    init_cuda,
    log_results,
    validate_experiment_path,
)

from data import (
    get_train_valid_test_datasets,
)
from peft import AdaLoraConfig, PeftConfig


# # suppress all warnings
# warnings.filterwarnings("ignore") # FIXME?

dtype_to_bytes_linear = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "int4": 0.5}
# if lr scheduler with warmup is used, the ratio of warmup steps to total steps
WARMUP_STEP_RATIO = 0.1


def evaluate(model, tokenizer, ds, batch_size, generate_kwargs, use_tqdm=False) -> tuple[list[str], list[str]]:
    # filter out None values so that we don't depend on setting correct defaults in the config
    generation_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}
    generation_config = GenerationConfig(**generate_kwargs)
    with torch.inference_mode():
        predictions = []
        responses = []
        pbar = range(0, len(ds), batch_size)
        if use_tqdm:
            pbar = tqdm(pbar)
        for j in pbar:
            sliced = ds[j : j + batch_size]
            responses += sliced.pop("response")
            batch = tokenizer.pad(sliced, return_tensors="pt", padding_side="left").to(model.device)
            outputs = model.generate(**batch, generation_config=generation_config, pad_token_id=tokenizer.eos_token_id)
            predictions += tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return predictions, responses


class DummyScheduler:
    # if no lr scheduler is being used
    def __init__(self, lr):
        self.lr = lr

    def get_last_lr(self):
        return [self.lr]

    def step(self):
        pass


class DummyGradScaler:
    # if no mixed precision is being used
    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass


def train(
    *,
    model: nn.Module,
    max_steps: int,
    batch_size: int,
    batch_size_eval: int,
    tokenizer: Any,
    cuda_memory_init: int,
    eval_steps: int,
    generation_kwargs: dict[str, Any],
    grad_norm_clip: float,
    optimizer_kwargs: dict[str, Any],
    query_template: str,
    lr_scheduler_arg: Optional[Literal["cosine"]],
    use_amp: bool,
    is_adalora: bool,
) -> TrainResult:
    cuda_memory_allocated_log = []
    cuda_memory_reserved_log = []
    losses = []
    durations = []
    metrics = []
    sample = 0  # keep count of the current sample
    total_samples = 0  # total number of samples over all epochs
    total_tokens = []  # total number of tokens over all epochs
    if use_amp:
        grad_scaler: GradScaler | DummyGradScaler = GradScaler()
        autocast_ctx: type[autocast] | type[nullcontext[None]] = autocast
    else:
        grad_scaler = DummyGradScaler()
        autocast_ctx = nullcontext
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    if lr_scheduler_arg == "cosine":
        warmup_steps = int(WARMUP_STEP_RATIO * max_steps)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps
        )
    elif lr_scheduler_arg is None:
        lr_scheduler = DummyScheduler(optimizer_kwargs["lr"])
    else:
        raise ValueError(f"Invalid lr_scheduler argument: {lr_scheduler_arg}")

    status = TrainStatus.FAILED
    tic_train = time.perf_counter()
    eval_time = 0.0

    ds_train, ds_valid, ds_test = get_train_valid_test_datasets(
        tokenizer=tokenizer, query_template=query_template, print_fn=print_verbose
    )
    try:
        pbar = tqdm(range(1, max_steps + 1))
        for step in pbar:
            tic = time.perf_counter()

            # create the batch
            sliced = ds_train[sample : sample + batch_size]
            tokens_per_sample = [len(i) for i in sliced["input_ids"]]
            total_tokens.append(sum(tokens_per_sample) + len(tokens_per_sample))  # add EOS token
            del sliced["response"]
            batch = tokenizer.pad(sliced, return_tensors="pt").to(model.device)
            actual_batch_size = len(batch["input_ids"])
            total_samples += actual_batch_size
            sample += batch_size
            if sample >= len(ds_train):  # new epoch
                sample = 0

            # add labels, they are automatically shifted by transformers
            labels = batch["input_ids"].clone()
            # We want to ignore the padding tokens except for the first EOS token; if we don't ignore them, the loss
            # will be dominated by padding tokens; if we ignore all, the model will not learn to predict the EOS token.
            # TODO: Note that the longest sequence in the batch won't have any PAD/EOS token at the end, this is fine if
            # the batch size is > 1 but should still be fixed eventually.
            for i, num_tokens in enumerate(tokens_per_sample):
                labels[i, num_tokens + 1 :] = -100
            batch["labels"] = labels
            num_items_in_batch = batch["attention_mask"].sum().item()

            # train step
            optimizer.zero_grad()
            with autocast_ctx():
                outputs = model(**batch, num_items_in_batch=num_items_in_batch)
                loss = outputs.loss
            grad_scaler.scale(loss).backward()
            if grad_norm_clip:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            lr_scheduler.step()

            if is_adalora:
                model.base_model.update_and_allocate(step)

            losses.append(loss.item())
            pbar.set_postfix({"loss": loss.item()})
            cuda_memory_allocated_log.append(torch.cuda.memory_allocated() - cuda_memory_init)
            cuda_memory_reserved_log.append(torch.cuda.memory_reserved() - cuda_memory_init)
            toc = time.perf_counter()
            durations.append(toc - tic)

            # every couple of steps, evaluate; this can be slow due to generation
            if step % eval_steps == 0:
                tic_eval = time.perf_counter()
                loss_avg = sum(losses[-eval_steps:]) / eval_steps
                memory_allocated_avg = sum(cuda_memory_allocated_log[-eval_steps:]) / eval_steps
                memory_reserved_avg = sum(cuda_memory_reserved_log[-eval_steps:]) / eval_steps
                token_sum = sum(total_tokens[-eval_steps:])
                dur_train = sum(durations[-eval_steps:])
                tokens_per_sec = token_sum / dur_train

                model.eval()
                predictions, responses = evaluate(
                    model=model,
                    tokenizer=tokenizer,
                    ds=ds_valid,
                    batch_size=batch_size_eval,
                    generate_kwargs={**generation_kwargs},
                )

                example = random.choice(predictions)
                example = textwrap.shorten(example, width=750)
                example = textwrap.indent(example, "    ")
                print_verbose(f"\nExample prediction:\n{example}\n")

                accuracy = get_accuracy(predictions=predictions, responses=responses)
                metrics.append(
                    {"step": step, "valid accuracy": accuracy, "train loss": loss_avg, "train samples": total_samples}
                )
                model.train()
                toc_eval = time.perf_counter()
                eval_time += toc_eval - tic_eval
                elapsed = time.perf_counter() - tic_train

                log_dict = {
                    "step": f"{step:5d}",
                    "samples": f"{total_samples:7d}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                    "loss avg": f"{loss_avg:.4f}",
                    "valid acc": f"{accuracy:.3f}",
                    "train time": f"{dur_train:.2f}s",
                    "eval time": f"{toc_eval - tic_eval:.2f}s",
                    "tokens / sec": f"{tokens_per_sec:.0f}",
                    "mem allocated avg": f"{memory_allocated_avg:.0f}",
                    "mem reserved avg": f"{memory_reserved_avg:.0f}",
                    "elapsed time": f"{elapsed // 60:.0f}min {elapsed % 60:.0f}s",
                }
                print_verbose(json.dumps(log_dict))

            # # TODO is this needed?
            torch.cuda.empty_cache()
            gc.collect()

        print_verbose(f"Training finished after {max_steps} steps, evaluation on test set follows.")
        # test set evaluation
        model.eval()
        predictions, responses = evaluate(
            model=model,
            tokenizer=tokenizer,
            ds=ds_test,
            batch_size=batch_size_eval,
            generate_kwargs={**generation_kwargs, "pad_token_id": tokenizer.eos_token_id},
            use_tqdm=len(ds_test) > 100,
        )
        accuracy = get_accuracy(predictions=predictions, responses=responses)
        metrics.append(
            {
                "step": step,
                "test accuracy": accuracy,
                "train loss": sum(losses[-eval_steps:]) / eval_steps,
                "train samples": total_samples,
                "train total tokens": sum(total_tokens),
            }
        )
        print_verbose(f"Test accuracy: {accuracy:.3f}")

    except KeyboardInterrupt:
        print_verbose("canceled training")
        status = TrainStatus.CANCELED
    except torch.OutOfMemoryError:
        # ouch, still let's try to log some results
        print_verbose("out of memory error encountered")
        status = TrainStatus.CANCELED

    toc_train = time.perf_counter()
    train_time = toc_train - tic_train - eval_time

    if status != TrainStatus.CANCELED:
        status = TrainStatus.SUCCESS
    train_result = TrainResult(
        status=status,
        train_time=train_time,
        cuda_memory_reserved_log=cuda_memory_reserved_log,
        losses=losses,
        metrics=metrics,
    )
    return train_result


def main(*, path_experiment: str, experiment_name: str) -> None:
    tic_total = time.perf_counter()
    start_date = dt.datetime.now(tz=dt.timezone.utc).replace(microsecond=0).isoformat()

    # load configs
    peft_config = PeftConfig.from_pretrained(path_experiment)
    path_train_config = os.path.join(path_experiment, FILE_NAME_TRAIN_PARAMS)
    train_config = get_train_config(path_train_config)
    set_seed(train_config.seed)

    # initialize objects
    cuda_memory_init = init_cuda()
    tokenizer = get_tokenizer(model_id=train_config.model_id, max_seq_length=train_config.max_seq_length)

    model_info = get_base_model_info(train_config.model_id)
    metamath_info = get_dataset_info("meta-math/MetaMathQA")
    gsm8k_info = get_dataset_info("openai/gsm8k")
    model = get_model(
        model_id=train_config.model_id,
        dtype=train_config.dtype,
        compile=train_config.compile,
        attn_implementation=train_config.attn_implementation,
        peft_config=peft_config,
    )
    print_verbose(model)
    num_trainable_params, num_params = model.get_nb_trainable_parameters()
    print_verbose(
        f"trainable params: {num_trainable_params:,d} || all params: {num_params:,d} || "
        f"trainable: {100 * num_trainable_params / num_params:.4f}%"
    )

    # train model
    try:
        train_result = train(
            model=model,
            max_steps=train_config.max_steps,
            batch_size=train_config.batch_size,
            batch_size_eval=train_config.batch_size_eval,
            tokenizer=tokenizer,
            cuda_memory_init=cuda_memory_init,
            eval_steps=train_config.eval_steps,
            generation_kwargs=train_config.generation_kwargs,
            grad_norm_clip=train_config.grad_norm_clip,
            optimizer_kwargs=train_config.optimizer_kwargs,
            query_template=train_config.query_template,
            lr_scheduler_arg=train_config.lr_scheduler,
            use_amp=train_config.use_amp,
            is_adalora=isinstance(peft_config, AdaLoraConfig),
        )
    except Exception as e:
        print_verbose(f"Training failed with error: {e}")
        raise

    if train_result.status == TrainStatus.FAILED:
        print_verbose("Training failed, not logging results")
        sys.exit(1)

    time_total = time.perf_counter() - tic_total

    # log results: print and save to file
    log_results(
        experiment_name=experiment_name,
        train_result=train_result,
        cuda_memory_init=cuda_memory_init,
        time_total=time_total,
        model=model,
        model_info=model_info,
        datasets_info={"metamath": metamath_info, "gsm8k": gsm8k_info},
        start_date=start_date,
        train_config=train_config,
        peft_config=peft_config,
        print_fn=print_verbose,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("path_experiment", type=str, help="Path to the experiment directory")
    args = parser.parse_args()

    experiment_name = validate_experiment_path(args.path_experiment)

    if args.verbose:

        def print_verbose(*args, **kwargs) -> None:
            kwargs["file"] = sys.stderr
            print(*args, **kwargs)
    else:

        def print_verbose(*args, **kwargs) -> None:
            pass

    main(
        path_experiment=args.path_experiment,
        experiment_name=experiment_name,
    )
