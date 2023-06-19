import argparse
import gc
import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from random import randint
from typing import Any, Dict, List, Union

# datasets imports
import datasets

# metric imports
import evaluate
import numpy as np
import torch
import transformers
import wandb

# accelerate imports
from accelerate import Accelerator, dispatch_model
from accelerate.logging import get_logger
from datasets import Audio, DatasetDict, IterableDatasetDict, interleave_datasets, load_dataset

# hf imports
from huggingface_hub import Repository
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    SchedulerType,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    get_scheduler,
    set_seed,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers.utils import get_full_repo_name

# peft imports
from peft import AdaLoraConfig, LoraConfig, PeftModel, get_peft_model


logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Whisper Fine-Tuning with AdaLora")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument("--language", type=str, help="Language to use for training; e.g., 'Hindi' ", required=True)
    parser.add_argument("--language_abbr", type=str, help="Language to use for training; e.g., 'hi' ", required=True)
    parser.add_argument(
        "--task", type=str, default="transcribe", help="Task to use for training; e.g., 'transcribe' ", required=False
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mozilla-foundation/common_voice_11_0",
        help="Dataset to use for training; e.g., 'whisper' ",
        required=False,
    )
    parser.add_argument(
        "--dataset_in_streaming_mode",
        action="store_true",
        help="Whether to use streaming mode for the dataset.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="lowercase the transcribed text before tokenizing"
    )
    parser.add_argument(
        "--do_remove_punctuation", action="store_true", help="remove punctuation from the transcribed text"
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--max_audio_input_length", type=float, default=30.0, help="Maximum audio length in seconds.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=5000,
        help="Number of samples to prefetch in the streaming mode.",
    )
    parser.add_argument(
        "--dataloader_pin_memory",
        action="store_true",
        help="Whether or not to pin memory for the DataLoader.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--load_best_model",
        action="store_true",
        help="Whether to load the best model at the end of training",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--evaluation_steps",
        type=int,
        default=500,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )

    # lora/adalora specific args
    parser.add_argument(
        "--use_peft",
        action="store_true",
        help="Whether to use PEFT",
    )
    parser.add_argument(
        "--use_adalora",
        action="store_true",
        help="Whether to use AdaLoRA or LoRA. If set, uses AdaLoRA instead of the default LoRA.",
    )
    parser.add_argument(
        "--init_r",
        type=int,
        default=12,
        help="Initial AdaLoRA rank",
    )
    parser.add_argument(
        "--target_r",
        type=int,
        default=4,
        help="Target AdaLoRA rank",
    )
    parser.add_argument(
        "--tinit",
        type=int,
        default=200,
        help="number of warmup steps for AdaLoRA wherein no pruning is performed",
    )
    parser.add_argument(
        "--tfinal",
        type=int,
        default=1000,
        help=" fix the resulting budget distribution and fine-tune the model for tfinal steps when using AdaLoRA ",
    )
    parser.add_argument(
        "--delta_t",
        type=int,
        default=10,
        help="interval of steps for AdaLoRA to update rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LORA alpha",
    )
    parser.add_argument(
        "--r",
        type=int,
        default=8,
        help="LORA rank",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LORA dropout",
    )
    parser.add_argument(
        "--orth_reg_weight",
        type=float,
        default=0.5,
        help="Orthogonal regularization weight",
    )
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="Whether to use debug mode",
    )

    args = parser.parse_args()

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def load_streaming_dataset(dataset_name, dataset_config_name, split, **kwargs):
    if "+" in split:
        # load multiple splits separated by the `+` symbol *with* streaming mode
        dataset_splits = [
            load_dataset(dataset_name, dataset_config_name, split=split_name, streaming=True, **kwargs)
            for split_name in split.split("+")
        ]
        # interleave multiple splits to form one dataset
        interleaved_dataset = interleave_datasets(dataset_splits)
        return interleaved_dataset
    else:
        # load a single split *with* streaming mode
        dataset = load_dataset(dataset_name, dataset_config_name, split=split, streaming=True, **kwargs)
        return dataset


def prepare_dataset_wrapper(do_lower_case, do_remove_punctuation, processor, normalizer):
    def prepare_dataset(batch):
        # load and (possibly) resample audio data to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        # compute input length of audio sample in seconds
        batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]

        # optional pre-processing steps
        transcription = batch["sentence"]
        if do_lower_case:
            transcription = transcription.lower()
        if do_remove_punctuation:
            transcription = normalizer(transcription).strip()

        # encode target text to label ids
        batch["labels"] = processor.tokenizer(transcription).input_ids
        return batch

    return prepare_dataset


def save_model_hook(models, weights, output_dir):
    for model in models:
        model.save_pretrained(output_dir)
        # make sure to pop weight so that corresponding model is not saved again
        weights.pop()


def load_model_hook(models, input_dir):
    while len(models) > 0:
        model = models.pop()
        # pop models so that they are not loaded again
        PeftModel.from_pretrained(model.base_model.model, input_dir)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def get_audio_length_processor(max_input_length):
    def is_audio_in_length_range(length):
        return length < max_input_length

    return is_audio_in_length_range


def evaluation_loop(model, eval_dataloader, processor, normalizer, metric, forced_decoder_ids, accelerator):
    model.eval()
    predictions = []
    references = []
    normalized_predictions = []
    normalized_references = []
    for _, batch in enumerate(tqdm(eval_dataloader)):
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                generated_tokens = (
                    model.generate(
                        input_features=batch["input_features"],
                        forced_decoder_ids=forced_decoder_ids,
                        max_new_tokens=255,
                    )
                    .cpu()
                    .numpy()
                )
                labels = batch["labels"].cpu().numpy()
                labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
                decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
                predictions.extend(decoded_preds)
                references.extend(decoded_labels)
                normalized_predictions.extend([normalizer(pred).strip() for pred in decoded_preds])
                normalized_references.extend([normalizer(label).strip() for label in decoded_labels])
            del generated_tokens, labels, batch
        gc.collect()
    wer = 100 * metric.compute(predictions=predictions, references=references)
    normalized_wer = 100 * metric.compute(predictions=normalized_predictions, references=normalized_references)
    eval_metrics = {"eval/wer": wer, "eval/normalized_wer": normalized_wer}
    if accelerator.get_tracker("wandb"):
        sample_size = min(len(predictions), 256)
        ids = [randint(0, len(predictions) - 1) for p in range(0, sample_size)]
        sample_predictions = [predictions[i] for i in ids]
        sample_references = [references[i] for i in ids]
        sample_normalized_predictions = [normalized_predictions[i] for i in ids]
        sample_normalized_references = [normalized_references[i] for i in ids]
        table_rows = [
            list(r)
            for r in zip(
                sample_predictions, sample_references, sample_normalized_predictions, sample_normalized_references
            )
        ]
        eval_metrics["eval_samples"] = wandb.Table(
            columns=["predictions", "references", "normalized_predictions", "normalized_references"],
            rows=table_rows,
        )
    return eval_metrics


def main():
    args = parse_args()

    # initialize accelerator
    accelerator = (
        Accelerator(
            log_with=args.report_to,
            project_dir=args.output_dir,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        if args.with_tracking
        else Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # load dataset either in streaming mode or not
    processor = WhisperProcessor.from_pretrained(args.model_name_or_path, language=args.language, task=args.task)
    normalizer = BasicTextNormalizer()
    prepare_dataset = prepare_dataset_wrapper(args.do_lower_case, args.do_remove_punctuation, processor, normalizer)
    is_audio_in_length_range = get_audio_length_processor(args.max_audio_input_length)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    if args.dataset_in_streaming_mode:
        raw_datasets = IterableDatasetDict()
        loading_method = load_streaming_dataset
    else:
        raw_datasets = DatasetDict()
        loading_method = load_dataset

    if args.debug_mode:
        train_split = "train[:100]"
        test_split = "test[:10]"
    else:
        train_split = "train+validation"
        test_split = "test"

    raw_datasets["train"] = loading_method(
        args.dataset_name, args.language_abbr, split=train_split, use_auth_token=True
    )
    raw_datasets["test"] = loading_method(args.dataset_name, args.language_abbr, split=test_split, use_auth_token=True)
    raw_datasets = raw_datasets.cast_column("audio", Audio(sampling_rate=16000))

    logger.info("Dataset loaded: %s", raw_datasets)
    logger.info(f'{raw_datasets["train"][0]}')

    vectorized_datasets = raw_datasets.map(
        prepare_dataset,
        remove_columns=list(next(iter(raw_datasets.values())).features),
        num_proc=args.preprocessing_num_workers,
    ).with_format("torch")

    if args.dataset_in_streaming_mode:
        vectorized_datasets["train"] = vectorized_datasets["train"].shuffle(
            buffer_size=args.buffer_size,
            seed=args.seed,
        )

    # filter out audio files that are too long from the training set
    is_audio_in_length_range = get_audio_length_processor(args.max_audio_input_length)
    vectorized_datasets["train"] = vectorized_datasets["train"].filter(
        is_audio_in_length_range, input_columns=["input_length"]
    )

    # get dataloaders
    train_dataloader = DataLoader(
        vectorized_datasets["train"],
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,
    )
    eval_dataloader = DataLoader(
        vectorized_datasets["test"],
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=args.dataloader_pin_memory,
    )

    # metric
    metric = evaluate.load("wer")

    # model
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path, load_in_8bit=True)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    if len(set(model.hf_device_map.values()).intersection({"cpu", "disk"})) > 0:
        raise ValueError("Training on CPU or disk is not supported.")
    if len(set(model.hf_device_map.values())) > 1:
        device_map = model.hf_device_map.copy()
        # required because `labels` are on main execution device (0) while the output of `proj_out` is on other device.
        # So, this leads to device mismatch error when calculation cross-entropy between logits and labels.
        # Won't arise during inference as `labels` aren't supplied during that time
        # instead of changing device of one of the tied modules, I have to do this for all tied modules
        # else the execution device of remaining tied modules isn't changed
        device_map["model.decoder.embed_tokens"] = model._hf_hook.execution_device
        device_map["model.decoder.embed_positions"] = model._hf_hook.execution_device
        device_map["proj_out"] = model._hf_hook.execution_device
        dispatch_model(model, device_map=device_map)

    # preparing peft model
    if args.use_peft:
        from peft import prepare_model_for_int8_training

        model = prepare_model_for_int8_training(model)

        # as Whisper model uses Conv layer in encoder, checkpointing disables grad computation
        # to avoid this, make the inputs trainable
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

        # wrapping model with adalora tuner
        if args.use_adalora:
            config = AdaLoraConfig(
                init_r=args.init_r,
                target_r=args.target_r,
                beta1=0.85,
                beta2=0.85,
                tinit=args.tinit,
                tfinal=args.tfinal,
                deltaT=args.delta_t,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
                orth_reg_weight=args.orth_reg_weight,
            )
        else:
            config = LoraConfig(
                r=args.r,
                lora_alpha=args.lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=args.lora_dropout,
            )

        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.max_train_steps is None:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # scheduler
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    accelerator.print(model)

    # Note here that the max steps is adjusted by the accelerator's num_processes
    args.max_train_steps = math.ceil(args.max_train_steps / accelerator.num_processes)
    if args.use_peft and args.use_adalora:
        model.base_model.peft_config["default"].total_step = args.max_train_steps
        # model.base_model.peft_config.total_step = args.max_train_steps

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        run_name = f"run-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers(
            "Whisper PEFT Fine-Tuning", config=experiment_config, init_kwargs={"wandb": {"name": run_name}}
        )

    # saving and loading checkpoints for resuming training
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    global_step = 0
    starting_epoch = 0
    best_metric = None
    resume_step = 0
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        path = os.path.basename(args.resume_from_checkpoint)
        training_difference = os.path.splitext(path)[0]
        global_step = resume_step = int(training_difference.replace("step_", ""))
        starting_epoch = resume_step // len(train_dataloader)
        resume_step -= starting_epoch * len(train_dataloader)

    # We need to adjust the progress bar to the current step
    progress_bar.update(resume_step)
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
            running_loss = 0
        for step, batch in enumerate(accelerator.skip_first_batches(train_dataloader, num_batches=resume_step)):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()

                # Update the importance of low-rank matrices
                # and allocate the budget accordingly.
                # This is only needed for AdaLora.
                # Note that this requires parameter gradients.
                # Hence being called before optimizer.zero_grad().
                if args.use_peft and args.use_adalora:
                    model.update_and_allocate(global_step)

                optimizer.zero_grad()
                global_step += 1
                progress_bar.update(1)

            if args.with_tracking:
                step_loss = accelerator.reduce(loss.detach().clone()).item()
                total_loss += step_loss
                running_loss += step_loss

            if global_step % args.checkpointing_steps == 0:
                output_dir = os.path.join(args.output_dir, f"step_{global_step}")
                accelerator.save_state(output_dir)

            if global_step % args.logging_steps == 0:
                if args.with_tracking:
                    accelerator.log({"train/running_loss": running_loss / args.logging_steps}, step=global_step)
                    running_loss = 0

            if global_step % args.evaluation_steps == 0:
                eval_metrics = evaluation_loop(
                    model, eval_dataloader, processor, normalizer, metric, forced_decoder_ids, accelerator
                )
                if args.with_tracking:
                    logger.info(f"Step {global_step} eval metrics: {eval_metrics}")
                    accelerator.log(eval_metrics, step=global_step)
                if best_metric is None or eval_metrics["eval/wer"] < best_metric:
                    best_metric = eval_metrics["eval/wer"]
                    accelerator.save_state(os.path.join(args.output_dir, "best_checkpoint"))
                model.train()

            if global_step >= args.max_train_steps:
                break

        if args.with_tracking:
            train_epoch_loss = total_loss / (step + 1)
            logger.info(f"Epoch {epoch} train loss: {train_epoch_loss}")
            accelerator.log({"epoch/train_loss": train_epoch_loss}, step=epoch)

        if args.push_to_hub and epoch <= args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir, is_main_process=accelerator.is_main_process)
            # evaluate the model at the end of training
            eval_metrics = evaluation_loop(
                model, eval_dataloader, processor, normalizer, metric, forced_decoder_ids, accelerator
            )
            if args.with_tracking:
                logger.info(f"Step {global_step} eval metrics: {eval_metrics}")
                accelerator.log(eval_metrics, step=global_step)
            if best_metric is None or eval_metrics["eval/wer"] < best_metric:
                best_metric = eval_metrics["eval/wer"]
                accelerator.save_state(os.path.join(args.output_dir, "best_checkpoint"))

            if accelerator.is_main_process:
                processor.tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )

    if args.load_best_model:
        # load the best model
        accelerator.load_state(os.path.join(args.output_dir, "best_checkpoint"))
        model.resize_modules_by_rank_pattern(model.peft_config["default"].rank_pattern, "default")
        eval_metrics = evaluation_loop(
            model, eval_dataloader, processor, normalizer, metric, forced_decoder_ids, accelerator
        )
        if args.with_tracking:
            best_metrics = {"best_" + k: v for k, v in eval_metrics.items()}
            accelerator.log(best_metrics, step=global_step)

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(args.output_dir, is_main_process=accelerator.is_main_process)
    if accelerator.is_main_process:
        processor.tokenizer.save_pretrained(args.output_dir)
        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

    with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
        eval_metrics.pop("eval_samples")
        json.dump(eval_metrics, f)


if __name__ == "__main__":
    main()
