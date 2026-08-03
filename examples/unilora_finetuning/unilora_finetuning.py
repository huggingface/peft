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
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import SFTConfig, SFTTrainer

from peft import get_peft_model


@dataclass
class ScriptArguments(SFTConfig):
    base_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The name or path of the fp32/fp16/bf16 base model."}
    )
    bits: str = field(default="bf16", metadata={"help": "Model dtype to load: bf16, fp16, or fp32."})
    unilora_r: int = field(default=32, metadata={"help": "Rank of the UniLoRA adapter."})
    theta_d_length: int = field(default=256, metadata={"help": "Length of the shared UniLoRA theta_d vector bank."})
    proj_seed: int = field(default=42, metadata={"help": "Seed used for deterministic UniLoRA projection indices."})
    unilora_dropout: float = field(default=0.0, metadata={"help": "Dropout probability for UniLoRA layers."})
    init_weights: bool = field(default=True, metadata={"help": "Whether to apply UniLoRA-specific initialization."})
    target_modules: Optional[list[str]] = field(
        default_factory=lambda: ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        metadata={"help": "Target module names for UniLoRA adapters."},
    )
    merge_and_save: bool = field(
        default=False, metadata={"help": "Merge the adapter into the base model and save it."}
    )
    data_path: str = field(default="imdb", metadata={"help": "Path or Hub id of the training dataset."})
    dataset_split: str = field(default="train[:1%]", metadata={"help": "Dataset split to train on."})
    dataset_field: Optional[list[str]] = field(
        default=None, metadata={"help": "Input and output field names for instruction data."}
    )


def get_dtype(bits: str) -> torch.dtype:
    if bits == "fp16":
        return torch.float16
    if bits == "bf16":
        return torch.bfloat16
    if bits == "fp32":
        return torch.float32
    raise ValueError("UniLoRA example supports only bf16, fp16, and fp32 model loading.")


def main() -> None:
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    print(script_args)

    if script_args.base_model_name_or_path is None:
        raise ValueError("Please pass --base_model_name_or_path to load a base model.")

    from peft import UniLoraConfig

    model = AutoModelForCausalLM.from_pretrained(
        script_args.base_model_name_or_path,
        dtype=get_dtype(script_args.bits),
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.base_model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    config = UniLoraConfig(
        r=script_args.unilora_r,
        theta_d_length=script_args.theta_d_length,
        proj_seed=script_args.proj_seed,
        target_modules=script_args.target_modules,
        unilora_dropout=script_args.unilora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        init_weights=script_args.init_weights,
    )
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()

    dataset = load_dataset(script_args.data_path, split=script_args.dataset_split)
    if script_args.dataset_field:
        dataset = dataset.map(
            lambda example: {
                "text": (
                    f"### USER: {example[script_args.dataset_field[0]]}\n"
                    f"### ASSISTANT: {example[script_args.dataset_field[1]]}"
                )
            }
        )

    trainer = SFTTrainer(
        model=peft_model,
        args=script_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_state()

    adapter_dir = os.path.join(script_args.output_dir, "unilora_ft")
    peft_model.save_pretrained(adapter_dir)

    if script_args.merge_and_save:
        merged_model = peft_model.merge_and_unload()
        merged_dir = os.path.join(script_args.output_dir, "unilora_merged")
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)


if __name__ == "__main__":
    main()
