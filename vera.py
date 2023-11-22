import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    VeraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
)

import evaluate
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
    AutoConfig,
)
from tqdm import tqdm

experiment_configs = {
    "sst2": dict(
        batch_size=256, num_epochs=60, vera_lr=4e-3, head_lr=4e-4, max_length=128, d_initial=0.1, r=512, dropout=0.0
    ),
    "mrpc": dict(
        batch_size=256, num_epochs=30, vera_lr=2e-2, head_lr=1e-2, max_length=128, d_initial=0.1, r=512, dropout=0.0
    ),
    "cola": dict(
        batch_size=256, num_epochs=80, vera_lr=2e-2, head_lr=1e-2, max_length=128, d_initial=0.1, r=512, dropout=0.0
    ),
    "qnli": dict(
        batch_size=256, num_epochs=25, vera_lr=2e-3, head_lr=4e-3, max_length=128, d_initial=0.1, r=512, dropout=0.0
    ),
    "rte": dict(
        batch_size=256, num_epochs=80, vera_lr=2e-2, head_lr=4e-3, max_length=128, d_initial=0.1, r=512, dropout=0.0
    ),
    "stsb": dict(
        batch_size=256, num_epochs=40, vera_lr=2e-2, head_lr=1e-2, max_length=128, d_initial=0.1, r=512, dropout=0.0
    ),
}

task = "mrpc"

# hparams
batch_size = experiment_configs[task]["batch_size"]
model_name_or_path = "roberta-base"
peft_type = PeftType.VERA
device = "cuda"
num_epochs = experiment_configs[task]["num_epochs"]
dropout = experiment_configs[task]["dropout"]

vera_lr = experiment_configs[task]["vera_lr"]
head_lr = experiment_configs[task]["head_lr"]

max_length = experiment_configs[task]["max_length"]

d_initial = experiment_configs[task]["d_initial"]
r = experiment_configs[task]["r"]

# Data prep
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
    if task in ["cola"]:
        outputs = tokenizer(examples["sentence"], truncation=True, max_length=max_length)
    elif task in ["qnli"]:
        outputs = tokenizer(examples["question"], examples["sentence"], truncation=True, max_length=max_length)
    else:
        outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=max_length)
    return outputs


if task in ["cola"]:
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence"],
    )
elif task in ["qnli"]:
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence", "question"],
    )
else:
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
test_dataloader = DataLoader(tokenized_datasets["test"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size)


best_val_accuracies = []
test_accuracies = []

# for key in [1234]:
# for key in [1234, 123456789, 1234567890]:
for key in [1234, 123456789, 1234567890, 0xFFFF, 0, 1, 666]:
    best_accuracy = 0.0
    peft_config = VeraConfig(
        task_type="SEQ_CLS",
        inference_mode=False,
        r=r,
        vera_dropout=dropout,
        projection_prng_key=key,
        d_initial=d_initial,
        target_modules=["RobertaSelfAttention"]
    )

    if task in ["stsb"]:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, return_dict=True, max_length=max_length, num_labels=1
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, return_dict=True, max_length=max_length
        )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    optimizer = AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if "vera_lambda_" in n], "lr": vera_lr},
            {"params": [p for n, p in model.named_parameters() if "classifier" in n], "lr": head_lr},
        ]
    )

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch.to(device)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch.to(device)
            with torch.no_grad():
                predictions = model(**batch).logits

            if task not in ["stsb"]:
                predictions = predictions.argmax(dim=-1)
            predictions, references = predictions, batch["labels"]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        print(f"epoch {epoch}:", eval_metric)

        if list(eval_metric.values())[0] > best_accuracy:
            best_accuracy = list(eval_metric.values())[0]

    print(f"best validation result: {best_accuracy}")
    best_val_accuracies.append(best_accuracy)

    for step, batch in enumerate(tqdm(test_dataloader)):
        batch.to(device)
        if batch["labels"][0] == -1:
            break
        with torch.no_grad():
            predictions = model(**batch).logits

        if task not in ["stsb"]:
            predictions = predictions.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    if batch["labels"][0] == -1:
        test_accuracies.append(None)
        continue
    test_metric = metric.compute()
    print(f"test result: {test_metric}")

    test_accuracies.append(list(test_metric.values())[0])

# best I saw was about 0.875 pre fixing scaling
# bs64 [0.8504901960784313, 0.8578431372549019, 0.8553921568627451, 0.8602941176470589, 0.8553921568627451, 0.8676470588235294, 0.8676470588235294]
# bs32 [0.8627450980392157, 0.8553921568627451, 0.8627450980392157, 0.8602941176470589, 0.8651960784313726, 0.875, 0.8676470588235294]
# bs32+drop0.1 [0.8627450980392157, 0.8651960784313726, 0.8725490196078431, 0.8627450980392157, 0.8651960784313726, 0.8651960784313726, 0.8553921568627451]


# post fix
# [0.8921568627450981, 0.9019607843137255, 0.8700980392156863, 0.8848039215686274, 0.8823529411764706, 0.8872549019607843, 0.8872549019607843]
print("per seed best validation scores:", best_val_accuracies)
print("per seed test scores:", test_accuracies)
