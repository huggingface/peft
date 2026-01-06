#!/usr/bin/env python
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
# Modified for AdaLoRA/DoRA (AdaDoRA) support.
"""Finetuning models for sequence classification on GLUE with AdaLoRA/DoRA."""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
import transformers
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from peft import AdaLoraConfig, PeftType, get_peft_model


logger = logging.getLogger(__name__)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


@dataclass
class DataTrainingArguments:
    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(default=128)
    overwrite_cache: bool = field(default=False)
    pad_to_max_length: bool = field(default=True)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys:
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)


@dataclass
class AdaLoraArguments:
    """Arguments for AdaLoRA configuration."""

    peft_type: str = field(default="ADALORA", metadata={"help": "PEFT type: ADALORA or LORA"})
    use_dora: bool = field(default=False, metadata={"help": "Enable DoRA (Weight-Decomposed Low-Rank Adaptation)"})
    target_r: int = field(default=8, metadata={"help": "Target rank after pruning"})
    init_r: int = field(default=12, metadata={"help": "Initial rank (should be > target_r)"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha scaling factor"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout rate"})
    tinit: int = field(default=200, metadata={"help": "Initial warmup steps (no pruning)"})
    tfinal: int = field(default=1000, metadata={"help": "Final steps (fixed rank fine-tuning)"})
    deltaT: int = field(default=10, metadata={"help": "Interval between rank allocations"})
    target_modules: Optional[str] = field(
        default="query,value",
        metadata={"help": "Comma-separated target modules (e.g., 'query,value')"},
    )


class AdaLoraCallback(TrainerCallback):
    """Callback to update AdaLoRA rank allocation during training."""

    def __init__(self, model):
        self.model = model

    def on_step_end(self, args, state, control, **kwargs):
        # update_and_allocate must be called after backward but we call it here
        # in practice you might want to integrate this more carefully
        if hasattr(self.model, "base_model") and hasattr(self.model.base_model, "update_and_allocate"):
            self.model.base_model.update_and_allocate(state.global_step)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, AdaLoraArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, adalora_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, adalora_args, training_args = parser.parse_args_into_dataclasses()

    # setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(
        f"AdaLoRA parameters: use_dora={adalora_args.use_dora}, target_r={adalora_args.target_r}, init_r={adalora_args.init_r}"
    )

    set_seed(training_args.seed)

    # load dataset
    raw_datasets = load_dataset("nyu-mll/glue", data_args.task_name, cache_dir=model_args.cache_dir)

    # labels
    is_regression = data_args.task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    # load model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # calculate total steps for AdaLoRA
    num_train_samples = len(raw_datasets["train"])
    if data_args.max_train_samples:
        num_train_samples = min(num_train_samples, data_args.max_train_samples)

    steps_per_epoch = num_train_samples // training_args.per_device_train_batch_size
    total_steps = steps_per_epoch * int(training_args.num_train_epochs)

    # setup AdaLoRA config
    target_modules = adalora_args.target_modules.split(",") if adalora_args.target_modules else ["query", "value"]

    peft_config = AdaLoraConfig(
        peft_type=PeftType.ADALORA,
        task_type="SEQ_CLS",
        r=adalora_args.init_r,
        lora_alpha=adalora_args.lora_alpha,
        lora_dropout=adalora_args.lora_dropout,
        target_modules=target_modules,
        target_r=adalora_args.target_r,
        init_r=adalora_args.init_r,
        tinit=adalora_args.tinit,
        tfinal=adalora_args.tfinal,
        deltaT=adalora_args.deltaT,
        total_step=total_steps,
        use_dora=adalora_args.use_dora,
    )

    logger.info(f"PEFT Config: {peft_config}")
    logger.info(f"Total training steps: {total_steps}")

    # apply PEFT
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # preprocessing
    sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    padding = "max_length" if data_args.pad_to_max_length else False
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        return tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    train_dataset = raw_datasets["train"]
    if data_args.max_train_samples:
        train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))

    eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
    if data_args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))

    # metrics
    metric = evaluate.load("glue", data_args.task_name, cache_dir=model_args.cache_dir)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # data collator
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # trainer with AdaLoRA callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[AdaLoraCallback(model)],
    )

    # training
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("Done!")


if __name__ == "__main__":
    main()
