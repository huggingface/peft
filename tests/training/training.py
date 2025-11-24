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

"""
This is a simple example of training a model with QLoRA.
"""

import argparse
import os
import tempfile
from typing import Literal

import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model


def print_if_process_zero(*args, **kwargs):
    PartialState().print(*args, **kwargs)


def main(model_id: str, quant: Literal["4bit", "8bit"] | None, target_modules: list[str] | None):
    if target_modules == ["all-linear"]:
        target_modules = "all-linear"

    print_if_process_zero("=" * 50)
    print_if_process_zero(f"{model_id=}, {quant=}, {target_modules=}")
    print_if_process_zero("=" * 50)

    data = load_dataset("ybelkada/english_quotes_copy")

    is_fsdp = "FSDP_VERSION" in os.environ
    if quant == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_type="bfloat16",
            bnb_4bit_quant_storage="bfloat16",
            bnb_4bit_use_double_quant=True,
        )
    elif quant == "8bit":
        if is_fsdp:
            raise ValueError("QLoRA with 8bit bnb is not supported for FSDP.")
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quant is None:
        quant_config = None
    else:
        raise ValueError(f"Unsupported quantization: {quant}, expected one of '4bit', '8bit', or None")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant_config, dtype=torch.bfloat16, device_map={"": PartialState().process_index}
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    print_if_process_zero(model)
    if PartialState().is_local_main_process:
        model.print_trainable_parameters()

    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = Trainer(
            model=model,
            train_dataset=data["train"],
            optimizer_cls_and_kwargs=(torch.optim.SGD, {"lr": 2e-4}),
            # FSDP with AdamW:
            # > RuntimeError: output with shape [] doesn't match the broadcast shape [1]
            args=TrainingArguments(
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
                warmup_steps=2,
                max_steps=15,
                learning_rate=2e-4,
                bf16=True,
                logging_steps=5,
                output_dir=tmp_dir,
            ),
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        trainer.train()

        if trainer.is_fsdp_enabled:
            trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        trainer.save_model(tmp_dir)

        # some checks
        if PartialState().is_local_main_process:
            files = os.listdir(tmp_dir)
            assert "adapter_model.safetensors" in files
            assert "adapter_config.json" in files

        final_log = trainer.state.log_history[-1]
        assert final_log["train_loss"] < 10.0, f"Final loss is too high: {final_log['loss']}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=False, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--quant", type=str, choices=["4bit", "8bit"], required=False, default=None)
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        required=False,
        default=None,
        help="List of target modules for LoRA adaptation",
    )
    args = parser.parse_args()
    main(model_id=args.model_id, quant=args.quant, target_modules=args.target_modules)
