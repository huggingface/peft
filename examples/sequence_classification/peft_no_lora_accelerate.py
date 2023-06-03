import argparse

import evaluate
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from peft import (
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    get_peft_model,
)
from peft.utils.other import fsdp_auto_wrap_policy


def parse_args():
    parser = argparse.ArgumentParser(description="PEFT a transformers model on a sequence classification task")
    parser.add_argument(
        "--num_virtual_tokens",
        type=int,
        default=20,
        help="num_virtual_tokens if the number of virtual tokens used in prompt/prefix/P tuning.",
    )
    parser.add_argument(
        "--encoder_hidden_size",
        type=int,
        default=128,
        help="encoder_hidden_size if the encoder hidden size used in P tuninig/Prefix tuning.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
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
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--peft_type",
        type=str,
        default="p_tuning",
        help="The PEFT type to use.",
        choices=["p_tuning", "prefix_tuning", "prompt_tuning"],
    )
    args = parser.parse_args()

    assert args.output_dir is not None, "Need an `output_dir` to store the finetune model and verify."

    return args


def main():
    args = parse_args()
    ddp_scaler = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_scaler])

    task = "mrpc"

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if args.peft_type == "p_tuning":
        peft_config = PromptEncoderConfig(
            task_type="SEQ_CLS",
            num_virtual_tokens=args.num_virtual_tokens,
            encoder_hidden_size=args.encoder_hidden_size,
        )
    elif args.peft_type == "prefix_tuning":
        peft_config = PrefixTuningConfig(
            task_type="SEQ_CLS",
            num_virtual_tokens=args.num_virtual_tokens,
            encoder_hidden_size=args.encoder_hidden_size,
        )
    else:
        peft_config = PromptTuningConfig(task_type="SEQ_CLS", num_virtual_tokens=args.num_virtual_tokens)

    tokenizer_kwargs = {}

    if any(k in args.model_name_or_path for k in ("gpt", "opt", "bloom")):
        tokenizer_kwargs["padding_side"] = "left"
    else:
        tokenizer_kwargs["padding_side"] = "right"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    datasets = load_dataset("glue", task)
    metric = evaluate.load("glue", task)

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
        return outputs

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    with accelerator.main_process_first():
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["idx", "sentence1", "sentence2"],
        )

    # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # transformers library
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # Instantiate dataloaders.
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"],
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.per_device_eval_batch_size,
    )

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if getattr(accelerator.state, "fsdp_plugin", None) is not None:
        accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)
        model = accelerator.prepare(model)

    optimizer = AdamW(params=model.parameters(), lr=args.learning_rate)

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_train_epochs),
    )

    if getattr(accelerator.state, "fsdp_plugin", None) is not None:
        train_dataloader, eval_dataloader, optimizer, lr_scheduler = accelerator.prepare(
            train_dataloader, eval_dataloader, optimizer, lr_scheduler
        )
    else:
        model, train_dataloader, eval_dataloader, optimizer, lr_scheduler = accelerator.prepare(
            model, train_dataloader, eval_dataloader, optimizer, lr_scheduler
        )

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        samples_seen = 0
        for step, batch in enumerate(tqdm(eval_dataloader)):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather((predictions, batch["labels"]))
            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                    references = references[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )
        eval_metric = metric.compute()
        accelerator.print(f"epoch {epoch}:", eval_metric)

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(args.output_dir, state_dict=accelerator.get_state_dict(model))
    if accelerator.is_main_process:
        tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
