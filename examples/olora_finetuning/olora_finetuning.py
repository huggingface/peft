# Copyright 2023-present the HuggingFace Inc. team.
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


from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from peft import (
    LoraConfig,
    get_peft_model,
)


def train(
    base_model: str = "path/to/model",
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "olora",
    batch_size: int = 16,
    micro_batch_size: int = 4,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 16,
    quantize: bool = False,
    eval_step: int = 100,
    save_step: int = 100,
    device_map: str = "auto",
    lora_r: int = 32,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = None,
    init_lora_weights="olora",
):
    gradient_accumulation_steps = batch_size // micro_batch_size
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        if quantize
        else None,
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(example):
        full_prompt = generate_prompt(example)
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights=init_lora_weights,
    )
    model = get_peft_model(model, config)

    data = load_dataset(data_path)

    train_val = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
    train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=100,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=eval_step,
            save_steps=save_step,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    trainer.train()
    model.save_pretrained(output_dir)


def generate_prompt(example):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
            ### Instruction:
            {example["instruction"]}
            ### Response:
            {example["output"]}"""


if __name__ == "__main__":
    torch.manual_seed(42)
    fire.Fire(train)
