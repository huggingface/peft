import argparse
import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    VBLoRAConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
)


import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm

batch_size = 32
model_name_or_path = "roberta-large"
task = "mrpc"
peft_type = PeftType.LORA
device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
num_epochs = 20
peft_config = VBLoRAConfig(
    task_type="SEQ_CLS", 
    r=4,
    topk=2,
    target_modules=['key', 'value', 'query', 'output.dense', 'intermediate.dense'],
    num_vectors=128,
    vector_length=256,
    save_only_topk_weights=True, # Set to True to reduce storage space. Note that the saved parameters cannot be used to resume training from checkpoints.
    vblora_dropout=0.,
)
lr = 3e-4


if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

datasets = load_dataset("glue", task)
metric = evaluate.load("glue", task)


def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default)
    outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=None)
    return outputs


tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "sentence1", "sentence2"],
)

# We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
# transformers library
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")


def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")


# Instantiate dataloaders.
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
)
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)
model = get_peft_model(model, peft_config)
# model.print_trainable_parameters()
# model
model.merge_and_unload()