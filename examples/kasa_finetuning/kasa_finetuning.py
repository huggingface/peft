# Copyright 2026-present the HuggingFace Inc. team.
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
"""Minimal KaSA fine-tuning example.

Mirrors `examples/mica_finetuning/mica_finetuning.py` in spirit but with the KaSA-specific knobs only. KaSA truncates
the `r` smallest singular components of the frozen base weight via a one-time SVD and parametrizes the trainable
update with a learnable diagonal of singular values (`lora_diag`) inserted between the LoRA A and B factors.

The KaSA paper trains with two auxiliary regularizers (an L2 penalty on the singular values and an orthogonal
regularization on the adapter factors). PEFT cannot inject them into the training loop automatically, so this example
subclasses the trainer and adds `get_kasa_regularization_loss` to the task loss.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import SFTConfig, SFTTrainer

from peft import KasaConfig, LoraConfig, get_kasa_regularization_loss, get_peft_model


@dataclass
class ScriptArguments(SFTConfig):
    base_model_name_or_path: Optional[str] = field(default=None, metadata={"help": "Name or path of the base model."})
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.0)
    kasa_beta: float = field(default=1e-4, metadata={"help": "Coefficient for the singular-value L2 regularizer."})
    kasa_gamma: float = field(default=1e-3, metadata={"help": "Coefficient for the orthogonal regularizer."})
    target_modules: Optional[str] = field(
        default="q_proj,v_proj",
        metadata={"help": "Comma-separated module names to adapt with KaSA."},
    )
    data_path: str = field(default="imdb", metadata={"help": "HF dataset path."})
    dataset_split: str = field(default="train[:1%]")
    dataset_text_field: str = field(default="text")


class KasaSFTTrainer(SFTTrainer):
    """SFTTrainer that adds the KaSA auxiliary regularization to the task loss."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
        if return_outputs:
            loss, outputs = result
            return loss + get_kasa_regularization_loss(model), outputs
        return result + get_kasa_regularization_loss(model)


def train():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    model = AutoModelForCausalLM.from_pretrained(args.base_model_name_or_path, dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    lora_config = LoraConfig(
        kasa_config=KasaConfig(beta=args.kasa_beta, gamma=args.kasa_gamma),
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[m.strip() for m in args.target_modules.split(",")],
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    dataset = load_dataset(args.data_path, split=args.dataset_split)
    trainer = KasaSFTTrainer(
        model=peft_model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    peft_model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    train()
