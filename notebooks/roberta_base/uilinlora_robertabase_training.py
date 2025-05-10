import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, BitsAndBytesConfig
from datasets import load_dataset
from peft import UILinLoRAConfig, get_peft_model
import torch

torch.set_printoptions(threshold=torch.inf)  # Display all elements
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BASE_ID = "roberta-base"
tok  = AutoTokenizer.from_pretrained(BASE_ID, use_fast=True)

quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True)
# quant_config = BitsAndBytesConfig(load_in_8bit=True)
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_ID,
    num_labels=2,
    # quantization_config=quant_config,
    device_map="auto")

uilinlora_cfg = UILinLoRAConfig(
        target_modules=["classifier.dense"],
        uilinlora_alpha=1.0,
        uilinlora_dropout=0.0,
        fan_in_fan_out=False,
        init_uilinlora_weights=True)
model = get_peft_model(base_model, uilinlora_cfg)

model.config.pad_token_id = tok.pad_token_id

# ---------- data ----------
raw_ds = load_dataset("glue", "sst2")

def tokenize(batch):
    natural   = tok(batch["sentence"], add_special_tokens=True)
    true_lens = [len(ids) for ids in natural["input_ids"]]

    padded = tok(
        batch["sentence"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

    padded["real_length"] = true_lens
    return padded

tokenized_ds = raw_ds.map(
    tokenize,
    batched=True,
    remove_columns=["sentence", "idx"]
)

# rename + set Torch format
tokenized_ds = tokenized_ds.rename_column("label", "labels")
tokenized_ds.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels", "real_length"],
)

# ---------- stats ----------
max_len = max(tokenized_ds["train"]["real_length"])
print(f"Longest raw sentence: {max_len} tokens")


# ---------- data ----------
raw_datasets = load_dataset("glue", "sst2")
def tokenize_function(example):
    return tok(example["sentence"], truncation=True, padding="max_length", max_length=100)

# Tokenize the entire dataset
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
del raw_datasets
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


# ---------- trainer ----------
args = TrainingArguments(
        output_dir="uilinlora-roberta-base-sst2",
        per_device_train_batch_size=32,
        num_train_epochs=3,
        learning_rate=3e-3,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50)


trainer = Trainer(model=model,
                  args=args,
                  train_dataset=tokenized_datasets["train"],
                  eval_dataset=tokenized_datasets["validation"])


trainer.train()


# after trainer.train()
adapter_dir = "uilinlora_adapter"
model.save_pretrained(adapter_dir, safe_serialization=True)  # adapter only
tok.save_pretrained(adapter_dir)                             # optional, for easy reload