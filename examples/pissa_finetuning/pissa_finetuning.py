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
from typing import List, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments
from trl import SFTTrainer

from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training


@dataclass
class TrainingArguments(TrainingArguments):
    # model configs
    base_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The name or path of the fp32/16 base model."}
    )
    residual_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name or path of the fp32/16 residual model. (`['fxmeng/pissa-llama-2-7b-r16-alpha-16']`)"
        },
    )
    bits: str = field(default="fp32", metadata={"help": "(`['fp4', 'nf4', 'int8', 'bf16', 'fp16', fp32]`)"})
    init_lora_weights: str = field(default="pissa", metadata={"help": "(`['gaussian', 'pissa', 'pissa_niter_4']`)"})
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0)
    convert_pissa_to_lora: bool = field(default=False)
    merge_and_save: bool = field(default=False)
    # dataset configs
    data_path: str = field(default="imdb", metadata={"help": "Path to the training data."})
    dataset_split: str = field(default="train[:1%]", metadata={"help": "(`['train', 'test', 'eval']`):"})
    dataset_field: List[str] = field(default=None, metadata={"help": "Fields of dataset input and output."})
    max_seq_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


parser = HfArgumentParser(TrainingArguments)
script_args = parser.parse_args_into_dataclasses()[0]
print(script_args)

print(f"Load pre-processed residual model in {script_args.bits} bits.")
if script_args.bits in ["nf4", "fp4", "int8"]:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=(script_args.bits == "nf4" or script_args.bits == "fp4"),
        load_in_8bit=script_args.bits == "int8",
        bnb_4bit_quant_type=script_args.bits,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    res_model = AutoModelForCausalLM.from_pretrained(
        script_args.residual_model_name_or_path, quantization_config=quantization_config, low_cpu_mem_usage=True
    )
    res_model = prepare_model_for_kbit_training(res_model)
    print("Wrapping the residual model with PiSSA.")
    peft_model = PeftModel.from_pretrained(
        res_model, script_args.residual_model_name_or_path, subfolder="pissa_init", is_trainable=True
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.residual_model_name_or_path)

elif script_args.residual_model_name_or_path is not None:
    res_model = AutoModelForCausalLM.from_pretrained(
        script_args.residual_model_name_or_path,
        torch_dtype=(
            torch.float16
            if script_args.bits == "fp16"
            else (torch.bfloat16 if script_args.bits == "bf16" else torch.float32)
        ),
        device_map="auto",
    )
    print("Wrapping the residual model with PiSSA.")
    peft_model = PeftModel.from_pretrained(
        res_model, script_args.residual_model_name_or_path, subfolder="pissa_init", is_trainable=True
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.residual_model_name_or_path)

elif script_args.base_model_name_or_path is not None:
    print(
        f"No available pre-processed model, manually initialize a PiSSA using {script_args.base_model_name_or_path}."
    )
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
    lora_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        init_lora_weights=script_args.init_lora_weights,
        lora_dropout=script_args.lora_dropout,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)

print(peft_model)
peft_model.print_trainable_parameters()

print(f"Training PiSSA with trl on the {script_args.data_path}[{script_args.dataset_split}] dataset.")
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
    dataset_text_field="text",
    max_seq_length=script_args.max_seq_length,
    tokenizer=tokenizer,
)
trainer.train()
trainer.save_state()
############################## Upon training completion, convert and save PiSSA in LoRA format ##############################
if script_args.convert_pissa_to_lora:
    peft_model.save_pretrained(
        os.path.join(script_args.output_dir, "pissa_lora"),
        convert_pissa_to_lora=os.path.join(script_args.residual_model_name_or_path, "pissa_init"),
    )
else:
    peft_model.save_pretrained(
        os.path.join(script_args.output_dir, "pissa_ft"),
    )

if script_args.merge_and_save:
    model = peft_model.merge_and_unload()
    model.save_pretrained(os.path.join(script_args.output_dir, "pissa_merged"))
    tokenizer.save_pretrained(os.path.join(script_args.output_dir, "pissa_merged"))
