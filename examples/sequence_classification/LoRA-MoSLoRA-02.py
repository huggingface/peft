import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.join(os.getcwd(), "src/"))
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
)

import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm import tqdm
from transformers import RobertaConfig

import fire

def run(seed=0):

    torch.manual_seed(seed)

    print("using seed:", seed)
    
    model_name_or_path = "roberta-large"
    task = "qnli"
    peft_type = PeftType.LORA
    device = "cuda:2"
    num_epochs = 10

    key1, key2 = "sentence1", "sentence2"
    val_key = "validation"
    if task == "mrpc": #3668/408/1725
        batch_size = 32
    elif task == "qnli": # 104743/5463/5463
        key1, key2 = "question", "sentence"
        batch_size = 8
    else:
        raise NotImplementedError

    peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1, use_moslora=True)
    lr = 1e-4

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
        outputs = tokenizer(examples[key1], examples[key2], truncation=True, max_length=None)
        return outputs


    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", key1, key2],
    )

    # We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
    # transformers library
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")


    def collate_fn(examples):
        return tokenizer.pad(examples, padding=True, max_length=512, return_tensors="pt")



    # Instantiate dataloaders.
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
    eval_dataloader = DataLoader(
        tokenized_datasets[val_key], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
    )

    config = RobertaConfig.from_pretrained(model_name_or_path)
    config.return_dict=True
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)

    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    optimizer = AdamW(params=model.parameters(), lr=lr, weight_decay=0)

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
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = predictions, batch["labels"]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        print(f"epoch {epoch}:", eval_metric)

if __name__ == "__main__":
    fire.Fire(run)