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

"""Shared adapter configuration and training records for the Cortex examples."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import torch

from peft import CortexConfig, LoraConfig


def warmup_kwargs(ratio: float) -> dict:
    """Use the warmup-ratio spelling supported by the installed Transformers."""
    from transformers import TrainingArguments

    if "warmup_ratio" in TrainingArguments.__dataclass_fields__:
        return {"warmup_ratio": ratio}
    return {"warmup_steps": ratio}


def resolve_deepspeed(value, *, batch_size: int, gradient_accumulation_steps: int):
    """Resolve the single-device ZeRO-0 preset, a config path, or disabled mode."""
    if value is None or value == "none":
        return None
    if value == "zero0":
        return {
            "train_micro_batch_size_per_gpu": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "train_batch_size": batch_size * gradient_accumulation_steps,
            "gradient_clipping": 1.0,
            "bf16": {"enabled": "auto"},
            "fp16": {"enabled": False},
            "zero_optimization": {"stage": 0},
            "zero_allow_untested_optimizer": True,
        }
    config = deepcopy(value) if isinstance(value, dict) else json.loads(Path(value).read_text())
    if config.get("zero_optimization", {}).get("stage", 0) != 0:
        raise ValueError("This single-device example supports DeepSpeed ZeRO-0 only, including custom configs.")
    return config


def build_adapter_config(args, model_config, revision=None):
    """Select whole FFNs for Cortex or MLP projections/parameters for LoRA."""
    if args.backend == "cortex":
        targets = args.moe_target
        if isinstance(targets, list) and len(targets) == 1:
            targets = targets[0]
        return CortexConfig(
            k=args.k,
            init_weights=args.init_weights,
            adapter_bias=getattr(args, "adapter_bias", None),
            adapter_bias_init=getattr(args, "adapter_bias_init", "uniform"),
            moe_target=targets,
            public_weighting=getattr(args, "public_weighting", "unweighted"),
            task_type="CAUSAL_LM",
            revision=revision,
        )
    if args.backend != "lora":
        raise ValueError("backend must be 'cortex' or 'lora'.")
    targets = {"target_modules": ["gate_proj", "up_proj", "down_proj"]}
    if model_config.model_type == "gpt_oss":
        # Each packed expert gets independent low-rank factors. PEFT handles the
        # native [experts, in, out] layout through the experts' is_transposed flag.
        targets = {
            "target_modules": [],
            "target_parameters": ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
        }
    return LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        revision=revision,
        **targets,
    )


def initialize_adapter(model, args, model_config) -> None:
    """Apply the research GPT-OSS LoRA initializer; Cortex uses its tuner defaults.

    PEFT's ``gaussian`` option uses standard deviation 1/r; the research expert
    LoRA uses 0.02 for A and zero for B. This only affects a new adapter, never
    an adapter loaded from a checkpoint.
    """
    if args.backend != "lora" or model_config.model_type != "gpt_oss":
        return
    from peft.tuners.lora.layer import ParamWrapper

    wrappers = [module for module in model.modules() if isinstance(module, ParamWrapper)]
    if not wrappers:
        raise ValueError("No GPT-OSS expert LoRA parameters were adapted.")
    with torch.no_grad():
        for module in wrappers:
            if module.parameter_name not in {"gate_up_proj", "down_proj"}:
                raise ValueError("This GPT-OSS example adapts only the expert gate/up and down parameters.")
            for adapter in module.lora_A:
                torch.nn.init.normal_(module.lora_A[adapter].weight, std=0.02)
                torch.nn.init.zeros_(module.lora_B[adapter].weight)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_training_run(args, result, *, objective, data, loss_history, trainable_parameters, preprocessing=None):
    """Write one training record alongside the adapter for shared evaluation."""
    record = {
        "settings": vars(args),
        "objective": objective,
        "data": data,
        "optimizer_steps": result.global_step,
        "train_loss": result.training_loss,
        "trainable_parameters": trainable_parameters,
        "losses": loss_history,
    }
    if preprocessing is not None:
        record["preprocessing"] = preprocessing
    path = args.output_dir / "run.json"
    path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
