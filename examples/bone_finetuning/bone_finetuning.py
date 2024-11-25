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

import os
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import SFTConfig, SFTTrainer

from peft import BoneConfig, get_peft_model


@dataclass
class ScriptArguments(SFTConfig):
    # model configs
    base_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The name or path of the fp32/16 base model."}
    )
    bits: str = field(default="bf16", metadata={"help": "(`['bf16', 'fp16', fp32]`)"})
    init_weights: Literal[True, "bat"] = field(
        default=True,
        metadata={
            "help": ("True -> Bone; `bat` -> Bat"),
        },
    )
    bone_r: int = field(default=16)
    merge_and_save: bool = field(default=False)
    # dataset configs
    data_path: str = field(default="imdb", metadata={"help": "Path to the training data."})
    dataset_split: str = field(default="train[:1%]", metadata={"help": "(`['train', 'test', 'eval']`):"})
    dataset_field: list[str] = field(default=None, metadata={"help": "Fields of dataset input and output."})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
print(script_args)

print(f"Load pre-processed residual model in {script_args.bits} bits.")
if script_args.bits in ["nf4", "fp4", "int8"]:
    print("Bone currently does not support quantization.")

elif script_args.base_model_name_or_path is not None:
    print(f"No available pre-processed model, manually initialize a Bone using {script_args.base_model_name_or_path}.")
    model = AutoModelForCausalLM.from_pretrained(
        script_args.base_model_name_or_path,
        torch_dtype=(
            torch.float16
            if script_args.bits == "fp16"
            else (torch.bfloat16 if script_args.bits == "bf16" else torch.float32)
        ),
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.base_model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    lora_config = BoneConfig(
        r=script_args.bone_r,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)

print(peft_model)
peft_model.print_trainable_parameters()

print(f"Training Bone with trl on the {script_args.data_path}[{script_args.dataset_split}] dataset.")
dataset = load_dataset(script_args.data_path, split=script_args.dataset_split)
dataset = dataset.map(
    lambda example: {
        "text": f"### USER: {example[script_args.dataset_field[0]]}\n### ASSISTANT: {example[script_args.dataset_field[1]]}"
    }
)

trainer = SFTTrainer(
    model=peft_model,
    args=script_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
trainer.train()
trainer.save_state()

peft_model.save_pretrained(
    os.path.join(script_args.output_dir, "bone_ft"),
)

if script_args.merge_and_save:
    model = peft_model.merge_and_unload()
    model.save_pretrained(os.path.join(script_args.output_dir, "bone_merged"))
    tokenizer.save_pretrained(os.path.join(script_args.output_dir, "bone_merged"))
