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

This script is designed to be run with `accelerate launch` to properly test FSDP behavior while running one pass with
autograd and another with adapters being disabled.

Usage:
    accelerate launch --config_file tests/training/fsdp_config.yaml tests/training/adapters.py
"""

import argparse
import tempfile

import torch
from accelerate import PartialState
from datasets import load_dataset
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from peft import LoraConfig, get_peft_model


def get_base_model_weights(peft_model):
    """Extract base model weights (non-LoRA weights)."""
    base_weights = {}
    for name, param in peft_model.named_parameters():
        if "lora" not in name.lower() and "modules_to_save" not in name:
            base_weights[name] = param.detach().clone()
    return base_weights


def get_adapter_weights(peft_model, adapter_name):
    """Extract weights for a specific adapter."""
    adapter_weights = {}
    for name, param in peft_model.named_parameters():
        if adapter_name in name:
            adapter_weights[name] = param.detach().clone()
    return adapter_weights


def verify_weights_unchanged(initial_weights, final_weights, weight_type):
    """Verify that weights have not changed during training."""
    for name in initial_weights:
        if name not in final_weights:
            raise AssertionError(f"{weight_type} weight missing after training: {name}")
        torch.testing.assert_close(
            initial_weights[name].to(device=final_weights[name].device, dtype=final_weights[name].dtype),
            final_weights[name],
        )


class Model(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            modules_to_save=["lm_head"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.peft_model = get_peft_model(model, peft_config)

        # Second adapter config (will remain disabled/unused throughout training)
        peft_config_second = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            modules_to_save=["lm_head"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.peft_model.add_adapter("second_adapter", peft_config_second)

        self.peft_model.set_adapter("default")
        self.peft_model.to(torch.bfloat16)

        self.peft_model.set_requires_grad("default", requires_grad=True)
        self.peft_model.set_requires_grad("second_adapter", requires_grad=False)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        out1 = self.peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        with self.peft_model.disable_adapter():
            out2 = self.peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        combined_loss = out1.loss + out2.loss
        return (combined_loss,)


def test_training(model_id: str):
    state = PartialState()
    torch.manual_seed(42)
    model = Model(model_id)

    initial_base_weights = get_base_model_weights(model.peft_model)
    initial_second_adapter_weights = get_adapter_weights(model.peft_model, "second_adapter")

    if state.is_main_process:
        print(f"Number of base model weight tensors: {len(initial_base_weights)}")
        print(f"Number of second_adapter weight tensors: {len(initial_second_adapter_weights)}")

    data = load_dataset("ybelkada/english_quotes_copy")
    data = data.map(lambda samples: model.tokenizer(samples["quote"]), batched=True)

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
            data_collator=DataCollatorForLanguageModeling(model.tokenizer, mlm=False),
        )
        trainer.train()
    with FSDP.summon_full_params(trainer.model):
        final_base_weights = get_base_model_weights(model.peft_model)
        final_second_adapter_weights = get_adapter_weights(model.peft_model, "second_adapter")

    # Test to make sure that through this FSDP setup the base weights remain unchanged
    # (i.e. adapter training doesn't somehow influence the base weights)
    verify_weights_unchanged(initial_base_weights, final_base_weights, "Base model")
    verify_weights_unchanged(initial_second_adapter_weights, final_second_adapter_weights, "second_adapter")


def main(model_id: str):
    test_training(model_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=False, default="Qwen/Qwen3-0.6B")
    args = parser.parse_args()
    main(model_id=args.model_id)
