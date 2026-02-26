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

import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import SFTConfig, SFTTrainer

from peft import PsoftConfig, get_peft_model


@dataclass
class ScriptArguments(SFTConfig):
    # --- model ---
    base_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The name or path of the fp32/16 base model."}
    )
    bits: str = field(
        default="fp32",
        metadata={"help": "Precision to load the base model. Choices: ['bf16', 'fp16', 'fp32']."},
    )

    # --- PSOFT ---
    r: int = field(default=32, metadata={"help": "Rank (r): dimension of trainable R."})
    psoft_alpha: int = field(default=32, metadata={"help": "Scaling factor (typically set to r)."})
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"],
        metadata={"help": "Target module names, e.g. ['q_proj','k_proj','v_proj','o_proj', ...]."},
    )

    # SVD / init
    ab_svd_init: str = field(
        default="psoft_init",
        metadata={"help": "Principal-subspace init identifier (e.g. 'psoft_init')."},
    )
    psoft_svd: str = field(
        default="full",
        metadata={"help": "SVD method. Typical choices: ['full', 'lowrank']."},
    )
    psoft_svd_lowrank_niter: Optional[int] = field(
        default=None,
        metadata={"help": "If psoft_svd='lowrank', number of iterations for lowrank SVD (optional)."},
    )

    # Orth / relaxation
    psoft_orth: bool = field(default=True, metadata={"help": "Use orthogonal R (Cayley parameterization)."})
    psoft_mag_a: bool = field(default=True, metadata={"help": "Enable tunable vector alpha (relaxed mode)."})
    psoft_mag_b: bool = field(default=True, metadata={"help": "Enable tunable vector beta (relaxed mode)."})

    # Cayley–Neumann approximation
    use_cayley_neumann: bool = field(default=False, metadata={"help": "Enable Cayley-Neumann approximation."})
    num_cayley_neumann_terms: int = field(default=5, metadata={"help": "Number of Neumann series terms."})
    cayley_neumann_eps: Optional[float] = field(
        default=None, metadata={"help": "Optional eps for numerical stability."}
    )

    # --- data ---
    data_path: str = field(default="imdb", metadata={"help": "Dataset name/path for training."})
    dataset_split: str = field(default="train[:1%]", metadata={"help": "Dataset split, e.g. 'train[:1%]'."})
    dataset_field: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": (
                "Fields used to build SFT text. "
                "If provided, will build: '### USER: <field0>\\n### ASSISTANT: <field1>'. "
                "If None, must already have a 'text' column."
            )
        },
    )


def _dtype_from_bits(bits: str) -> torch.dtype:
    bits = bits.lower()
    if bits == "bf16":
        return torch.bfloat16
    if bits == "fp16":
        return torch.float16
    if bits == "fp32":
        return torch.float32
    raise ValueError(f"Unknown bits={bits}. Use one of: bf16, fp16, fp32.")


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    print(script_args)

    if script_args.base_model_name_or_path is None:
        raise ValueError("--base_model_name_or_path is required.")

    # PSOFT does NOT support quantized layers (nf4/int8/etc.).
    # We only allow fp16/bf16/fp32 here to avoid accidental quantized loading.
    if script_args.bits.lower() not in {"bf16", "fp16", "fp32"}:
        raise ValueError("PSOFT example only supports bits in ['bf16','fp16','fp32'] (no quantization).")

    torch_dtype = _dtype_from_bits(script_args.bits)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.base_model_name_or_path,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.base_model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Build PSOFT config
    psoft_kwargs = {
        "r": script_args.r,
        "psoft_alpha": script_args.psoft_alpha,
        "target_modules": script_args.target_modules,
        "ab_svd_init": script_args.ab_svd_init,
        "psoft_svd": script_args.psoft_svd,
        "psoft_orth": script_args.psoft_orth,
        "psoft_mag_a": script_args.psoft_mag_a,
        "psoft_mag_b": script_args.psoft_mag_b,
        "use_cayley_neumann": script_args.use_cayley_neumann,
        "num_cayley_neumann_terms": script_args.num_cayley_neumann_terms,
        "cayley_neumann_eps": script_args.cayley_neumann_eps,
        "task_type": "CAUSAL_LM",
    }
    # Only pass lowrank_niter when user sets it (and typically when psoft_svd='lowrank')
    if script_args.psoft_svd_lowrank_niter is not None:
        psoft_kwargs["psoft_svd_lowrank_niter"] = script_args.psoft_svd_lowrank_niter

    peft_config = PsoftConfig(**psoft_kwargs)
    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    # Load dataset
    dataset = load_dataset(script_args.data_path, split=script_args.dataset_split)

    # Ensure a "text" field for SFTTrainer
    if script_args.dataset_field is not None:
        if len(script_args.dataset_field) != 2:
            raise ValueError("dataset_field must be a list of exactly 2 field names: [input_field, output_field].")

        in_f, out_f = script_args.dataset_field[0], script_args.dataset_field[1]

        def to_sft_text(example):
            return {"text": f"### USER: {example[in_f]}\n### ASSISTANT: {example[out_f]}"}

        dataset = dataset.map(to_sft_text)
    else:
        if "text" not in dataset.column_names:
            raise ValueError("dataset_field is None but dataset has no 'text' column. Provide dataset_field.")

    # Train
    trainer = SFTTrainer(
        model=model,
        args=script_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_state()

    # Save adapter (PSOFT)
    os.makedirs(script_args.output_dir, exist_ok=True)
    model.save_pretrained(os.path.join(script_args.output_dir, "psoft_ft"))
    tokenizer.save_pretrained(os.path.join(script_args.output_dir, "psoft_ft"))


if __name__ == "__main__":
    main()
