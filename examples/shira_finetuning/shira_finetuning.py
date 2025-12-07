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


import os
from typing import Optional

import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from peft import (
    PeftModel,
    ShiraConfig,
    get_peft_model,
)


def train(
    base_model: str = "path/to/model",
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "shira",
    batch_size: int = 16,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 16,
    eval_step: int = 100,
    save_step: int = 100,
    device_map: str = "auto",
    shira_r: int = 32,
    shira_target_modules: list[str] = None,
    dtype: str = "float16",
    seed: Optional[int] = None,
    use_custom_random_mask_function_with_custom_kwargs: Optional[bool] = False,
):
    # Set device_map to the right place when enabling DDP.
    world_size = int(os.environ.get("WORLD_SIZE", 0)) or int(os.environ.get("PMI_SIZE", 0))
    if world_size > 1 and device_map != "cpu":
        from accelerate import Accelerator

        device_map = {"": Accelerator().process_index}
    # Set seed
    if seed is not None:
        set_seed(seed)
    model_kwargs = {"dtype": getattr(torch, dtype), "device_map": device_map}
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    # For some tokenizer with no pad token like llama
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    def custom_random_mask_function_with_custom_kwargs(custom_arg):
        def mask_fn(base_layer, r):
            """
            This mask function is similar to the random_mask provided in src/peft/tuners/shira/mask_functions.py except the seed is derived from custom_kwargs.
            Please use this as an example to create your own custom sparse masks that may use custom_kwargs. Remember, for a pretrained weight with shape m, n,
            mask_fn must return only one mask (shape: m, n) which must be binary 0 or 1 with num_shira_parameters = r(m+n) for linear layers. Device and dtype
            of mask must be same as base layer's weight's device and dtype.
            """
            new_seed = custom_arg
            shape = base_layer.weight.shape
            num_shira_weights = r * (shape[0] + shape[1])
            random_generator = torch.Generator()
            random_generator.manual_seed(new_seed)

            idx = (torch.randperm(base_layer.weight.numel(), generator=random_generator)[:num_shira_weights]).to(
                base_layer.weight.device
            )
            val = torch.ones_like(idx.type(base_layer.weight.dtype))
            mask = torch.zeros_like(base_layer.weight.view(1, -1))
            mask = mask.scatter_(1, idx.unsqueeze(0), val.unsqueeze(0)).view(shape)

            return mask

        return mask_fn

    mask_type = "random" if not use_custom_random_mask_function_with_custom_kwargs else "custom"
    config = ShiraConfig(
        r=shira_r,
        mask_type=mask_type,
        target_modules=shira_target_modules,
        task_type="CAUSAL_LM",
    )
    if use_custom_random_mask_function_with_custom_kwargs:
        custom_arg = 120
        custom_mask_fn = custom_random_mask_function_with_custom_kwargs(custom_arg)
        config.mask_fn = custom_mask_fn

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
            per_device_train_batch_size=batch_size,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            logging_steps=100,
            optim="adamw_torch",
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=eval_step,
            save_steps=save_step,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if world_size > 1 else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    trainer.train()
    model.save_pretrained(output_dir)

    # Delete the model and load it again from the checkpoint.
    del model
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    model = PeftModel.from_pretrained(model, output_dir)


def generate_prompt(example):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
            ### Instruction:
            {example["instruction"]}
            ### Response:
            {example["output"]}"""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="path/to/model")
    parser.add_argument("--data_path", type=str, default="yahma/alpaca-cleaned")
    parser.add_argument("--output_dir", type=str, default="shira")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--cutoff_len", type=int, default=256)
    parser.add_argument("--val_set_size", type=int, default=16)
    parser.add_argument("--eval_step", type=int, default=100)
    parser.add_argument("--save_step", type=int, default=100)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--shira_r", type=int, default=32)
    parser.add_argument("--shira_target_modules", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_custom_random_mask_function_with_custom_kwargs", action="store_true")

    args = parser.parse_args()

    train(
        base_model=args.base_model,
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        cutoff_len=args.cutoff_len,
        val_set_size=args.val_set_size,
        eval_step=args.eval_step,
        save_step=args.save_step,
        device_map=args.device_map,
        shira_r=args.shira_r,
        shira_target_modules=args.shira_target_modules,
        dtype=args.dtype,
        seed=args.seed,
        use_custom_random_mask_function_with_custom_kwargs=args.use_custom_random_mask_function_with_custom_kwargs,
    )
