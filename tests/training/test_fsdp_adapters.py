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
Script to test FSDP adapter operations (disable_adapters, set_adapter, etc.) in a distributed environment.

This script is designed to be run with `accelerate launch` to properly test FSDP behavior across multiple GPUs.

Usage:
    accelerate launch --config_file tests/training/fsdp_config.yaml tests/training/test_fsdp_adapters.py
    accelerate launch --config_file tests/training/fsdp_config.yaml tests/training/test_fsdp_adapters.py --test disable_adapters
    accelerate launch --config_file tests/training/fsdp_config.yaml tests/training/test_fsdp_adapters.py --test set_adapter
"""

import argparse
import tempfile

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


def test_disable_adapters(model_id: str, quant: str | None):
    """Test that disable_adapters() works correctly with FSDP."""
    print_if_process_zero("=" * 50)
    print_if_process_zero(f"Testing disable_adapters with {model_id=}, {quant=}")
    print_if_process_zero("=" * 50)

    if quant == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_type="bfloat16",
            bnb_4bit_quant_storage="bfloat16",
            bnb_4bit_use_double_quant=True,
        )
    else:
        quant_config = None

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map={"": PartialState().process_index},
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    print_if_process_zero(model)
    if PartialState().is_local_main_process:
        model.print_trainable_parameters()

    data = load_dataset("ybelkada/english_quotes_copy")
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = Trainer(
            model=model,
            train_dataset=data["train"],
            optimizer_cls_and_kwargs=(torch.optim.SGD, {"lr": 2e-4}),
            args=TrainingArguments(
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
                warmup_steps=2,
                max_steps=5,
                learning_rate=2e-4,
                bf16=True,
                logging_steps=1,
                output_dir=tmp_dir,
            ),
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        # Train for a few steps first
        trainer.train()

        # Test disable_adapters - should not raise
        print_if_process_zero("Testing disable_adapters()...")
        model.disable_adapters()
        print_if_process_zero("disable_adapters() succeeded!")

        # Test enable_adapters - should not raise
        print_if_process_zero("Testing enable_adapters()...")
        model.enable_adapters()
        print_if_process_zero("enable_adapters() succeeded!")

        # Test context manager - should not raise
        print_if_process_zero("Testing disable_adapter() context manager...")
        with model.disable_adapter():
            pass
        print_if_process_zero("Context manager succeeded!")

        # Train a few more steps after re-enabling
        trainer.train()

        print_if_process_zero("All disable_adapters tests passed!")


def test_set_adapter(model_id: str, quant: str | None):
    """Test that set_adapter() works correctly with FSDP."""
    print_if_process_zero("=" * 50)
    print_if_process_zero(f"Testing set_adapter with {model_id=}, {quant=}")
    print_if_process_zero("=" * 50)

    if quant == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_type="bfloat16",
            bnb_4bit_quant_storage="bfloat16",
            bnb_4bit_use_double_quant=True,
        )
    else:
        quant_config = None

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map={"": PartialState().process_index},
    )

    # Create first adapter
    peft_config1 = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config1, adapter_name="adapter1")

    # Add second adapter
    peft_config2 = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.add_adapter("adapter2", peft_config2)

    print_if_process_zero(model)
    if PartialState().is_local_main_process:
        model.print_trainable_parameters()

    data = load_dataset("ybelkada/english_quotes_copy")
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = Trainer(
            model=model,
            train_dataset=data["train"],
            optimizer_cls_and_kwargs=(torch.optim.SGD, {"lr": 2e-4}),
            args=TrainingArguments(
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
                warmup_steps=2,
                max_steps=5,
                learning_rate=2e-4,
                bf16=True,
                logging_steps=1,
                output_dir=tmp_dir,
            ),
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )

        # Train with adapter1
        trainer.train()

        # Test set_adapter - should not raise
        print_if_process_zero("Testing set_adapter('adapter2')...")
        model.set_adapter("adapter2")
        print_if_process_zero("set_adapter('adapter2') succeeded!")

        # Test switching back
        print_if_process_zero("Testing set_adapter('adapter1')...")
        model.set_adapter("adapter1")
        print_if_process_zero("set_adapter('adapter1') succeeded!")

        # Test with list of adapters
        print_if_process_zero("Testing set_adapter(['adapter1', 'adapter2'])...")
        model.set_adapter(["adapter1", "adapter2"])
        print_if_process_zero("set_adapter(['adapter1', 'adapter2']) succeeded!")

        print_if_process_zero("All set_adapter tests passed!")


def main(test_name: str, model_id: str, quant: str | None):
    if test_name == "disable_adapters":
        test_disable_adapters(model_id, quant)
    elif test_name == "set_adapter":
        test_set_adapter(model_id, quant)
    elif test_name == "all":
        test_disable_adapters(model_id, quant)
        test_set_adapter(model_id, quant)
    else:
        raise ValueError(f"Unknown test: {test_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=False, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--quant", type=str, choices=["4bit"], required=False, default=None)
    parser.add_argument(
        "--test",
        type=str,
        choices=["disable_adapters", "set_adapter", "all"],
        required=False,
        default="all",
        help="Which test to run",
    )
    args = parser.parse_args()
    main(test_name=args.test, model_id=args.model_id, quant=args.quant)
