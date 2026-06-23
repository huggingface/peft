# Copyright 2024-present the HuggingFace Inc. team.
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

"""Compare LoRA and ShadowPEFT on the same task with the same training budget.

Three task types are supported, each with a single, clear metric:

- ``clm``:   causal language modeling, reported as eval loss / perplexity.
- ``cls``:   sequence classification, reported as eval accuracy.
- ``gsm8k``: grade-school math (SFT), reported as exact-match accuracy of the generated final answer.

For each requested method the script builds a PEFT model from a fresh copy of the base model, trains it with the
Hugging Face `Trainer`, evaluates it, and finally prints a side-by-side comparison (trainable parameters, training
time, and the task metric).

Examples:
    python run_benchmark.py --task cls   --model_name Qwen/Qwen3-0.6B --methods lora shadow
    python run_benchmark.py --task clm   --model_name Qwen/Qwen3-0.6B --methods lora shadow --max_steps 200
    python run_benchmark.py --task gsm8k --model_name Qwen/Qwen3-0.6B --methods lora shadow --bf16
    # train only (FSDP), then eval separately on one GPU for faster GSM8K generation:
    accelerate launch --config_file fsdp_config.yaml run_benchmark.py --mode train --task gsm8k --methods lora --bf16
    CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --mode eval --task gsm8k --methods lora --bf16
    # explicit (optionally pre-trained) shadow model:
    python run_benchmark.py --task gsm8k --methods shadow --shadow_model_name Qwen/Qwen3-0.6B
    # projected shadow model (small backbone + trained hidden projection) on a larger base:
    python run_benchmark.py --task gsm8k --model_name Qwen/Qwen3-8B --methods shadow \
        --shadow_model_name shadow-llm/Qwen3-0.6B-H8B --bf16
    # FSDP (recommended for large models; Trainer applies PEFT's auto-wrap policy automatically):
    accelerate launch --config_file fsdp_config.yaml run_benchmark.py --task cls --methods lora --bf16
"""

import argparse
import contextlib
import itertools
import json
import math
import os
import re
import time

# RTX 4000 series: PartialState/NCCL fails on single-GPU runs unless these are set early.
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from accelerate import PartialState
from accelerate.utils import broadcast as accel_broadcast
from datasets import load_dataset
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_utils import EvalLoopOutput

try:
    from transformers.trainer_optimizer import is_optimizer_factory
except ImportError:

    def is_optimizer_factory(_optimizer_cls_or_factory):
        return False

from peft import (
    AutoModelForCausalLMWithHiddenProjection,
    LoraConfig,
    PeftConfig,
    PeftModel,
    ShadowConfig,
    get_peft_model,
)
from peft.utils.peft_types import PeftType


# Sensible per-task defaults so the script runs out of the box.
TASK_DEFAULTS = {
    "clm": {"dataset_name": "Salesforce/wikitext", "dataset_config": "wikitext-2-raw-v1", "text_column": "text"},
    "cls": {"dataset_name": "SetFit/CR", "dataset_config": None, "text_column": "text", "label_column": "label"},
    "gsm8k": {"dataset_name": "openai/gsm8k"},
}

DEFAULT_GSM8K_GENERATION_TOKENS = 256

_GSM8K_ANSWER_LINE = re.compile(r"####\s*([^\n\r]+)")
_GSM8K_NUMBER = re.compile(r"[-+]?\d[\d,]*\.?\d*")
_THINK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark LoRA vs ShadowPEFT")
    parser.add_argument("--task", choices=("clm", "cls", "gsm8k"), default="cls")
    parser.add_argument("--methods", nargs="+", choices=("lora", "shadow"), default=["lora", "shadow"])
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")

    parser.add_argument(
        "--mode",
        choices=("train", "eval", "both"),
        default="both",
        help="train: train and save adapters; eval: load saved adapters and evaluate; both: train then eval.",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default=None,
        help="Directory with per-method adapter checkpoints for --mode eval (default: output_dir).",
    )

    # Data
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--text_column", type=str, default=None)
    parser.add_argument("--label_column", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default=None, help="Defaults to 'validation' (clm) or 'test'.")
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--block_size", type=int, default=256, help="Block size for causal LM grouping.")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument(
        "--training_samples",
        type=int,
        default=None,
        help="Smoke-test shortcut: limit both train and eval splits to N samples each.",
    )

    # GSM8K-specific (following run_shadow_peft.py)
    parser.add_argument("--gsm8k_subset", choices=("main", "socratic"), default="main")
    parser.add_argument("--gsm8k_answer_mode", choices=("thinking", "final"), default="thinking")
    parser.add_argument("--generation_max_length", type=int, default=None, help="Max new tokens for GSM8K eval.")
    parser.add_argument(
        "--gsm8k_max_print_predictions",
        type=int,
        default=None,
        help="Cap GSM8K prediction logging during eval (default: print all evaluated samples).",
    )
    parser.add_argument(
        "--gsm8k_eval_loss",
        action="store_true",
        help="Also run a teacher-forced eval loss pass (slower; accuracy still uses generation).",
    )
    parser.add_argument(
        "--gsm8k_distributed_eval",
        action="store_true",
        help="Run generation eval on every rank (default: rank 0 only when using multiple processes).",
    )

    # Training (shared by both methods for a fair comparison)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save Trainer checkpoints every N steps and write a PEFT adapter in each checkpoint (<=0 disables).",
    )
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--fsdp",
        type=str,
        default=None,
        help='Enable FSDP via TrainingArguments, e.g. "full_shard auto_wrap". '
        "Prefer `accelerate launch --config_file fsdp_config.yaml` instead.",
    )
    parser.add_argument(
        "--fsdp_use_orig_params",
        action="store_true",
        help="Set FSDP use_orig_params=True (required for ShadowPEFT; also set in fsdp_config.yaml).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="benchmark_outputs")

    # LoRA
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])

    # ShadowPEFT
    parser.add_argument(
        "--shadow_model_name",
        type=str,
        default=None,
        help="Optional explicit (pre-trained) shadow model. If unset, an implicit shadow model is built from the base.",
    )
    parser.add_argument("--shadow_layers", type=int, default=1)
    parser.add_argument("--injection_hidden_size", type=int, default=16)
    parser.add_argument("--gate_hidden_size", type=int, default=8)
    parser.add_argument("--shadow_intermediate_size", type=int, default=256)
    parser.add_argument("--shadow_num_attention_heads", type=int, default=None)
    parser.add_argument("--shadow_alpha", type=float, default=0.1)
    parser.add_argument("--shadow_dropout", type=float, default=0.2)
    parser.add_argument("--shadow_loss_weight", type=float, default=0.05)
    parser.add_argument(
        "--shadow_lr_scale",
        type=float,
        default=1.0,
        help="LR multiplier for the shadow backbone (lr = learning_rate * scale).",
    )
    parser.add_argument(
        "--shadow_inference_mode",
        choices=("base_shadow", "shadow_only", "both"),
        default="both",
        help="ShadowPEFT inference mode for eval: base+shadow (default path), shadow-only, or run both.",
    )

    args = parser.parse_args()
    defaults = TASK_DEFAULTS[args.task]
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    if args.eval_split is None:
        args.eval_split = "validation" if args.task == "clm" else "test"
    if args.training_samples is not None:
        args.max_train_samples = args.training_samples
        args.max_eval_samples = args.training_samples
    return args


def is_main_process():
    """True on the main process (always True for single-process / non-distributed runs)."""
    return PartialState().is_main_process


def method_dir(args, method):
    root = args.adapter_dir or args.output_dir
    if args.adapter_dir and os.path.isfile(os.path.join(root, "adapter_config.json")):
        return root
    return os.path.join(root, method)


def save_train_summary(method, args, summary):
    if not is_main_process():
        return
    path = os.path.join(method_dir(args, method), "train_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


def load_train_summary(method, args):
    path = os.path.join(method_dir(args, method), "train_summary.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {}


def resolve_checkpoint_base_model(method, args):
    """Use the base model recorded in the adapter checkpoint (falls back to CLI / train_summary)."""
    adapter_path = method_dir(args, method)
    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_model_name = peft_config.base_model_name_or_path
    if not base_model_name:
        base_model_name = load_train_summary(method, args).get("model_name")
    base_model_name = base_model_name or args.model_name
    if base_model_name != args.model_name:
        main_print(
            f"Eval for {method}: using checkpoint base model {base_model_name} "
            f"(ignoring --model_name {args.model_name})."
        )
    return base_model_name


def build_fsdp_config(args, method=None):
    if not args.fsdp:
        return None
    use_orig_params = args.fsdp_use_orig_params or method == "shadow"
    return {"fsdp_use_orig_params": use_orig_params}


def main_print(*print_args, **print_kwargs):
    if is_main_process():
        print(*print_args, **print_kwargs)


def build_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _maybe_truncate(dataset, max_samples):
    if max_samples is not None and max_samples < len(dataset):
        return dataset.select(range(max_samples))
    return dataset


# --------------------------------------------------------------------------------------- clm / cls


def prepare_clm_data(args, tokenizer):
    """Tokenize a text dataset and group it into fixed-size blocks for causal language modeling."""
    raw_train = load_dataset(args.dataset_name, args.dataset_config, split=args.train_split)
    raw_eval = load_dataset(args.dataset_name, args.dataset_config, split=args.eval_split)

    def tokenize(batch):
        return tokenizer(batch[args.text_column])

    columns = raw_train.column_names
    tokenized_train = raw_train.map(tokenize, batched=True, remove_columns=columns)
    tokenized_eval = raw_eval.map(tokenize, batched=True, remove_columns=columns)

    block_size = args.block_size

    def group_texts(examples):
        concatenated = {k: list(itertools.chain.from_iterable(examples[k])) for k in examples}
        total_length = (len(concatenated["input_ids"]) // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    train = tokenized_train.map(group_texts, batched=True)
    eval_ds = tokenized_eval.map(group_texts, batched=True)
    train = _maybe_truncate(train, args.max_train_samples)
    eval_ds = _maybe_truncate(eval_ds, args.max_eval_samples)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return train, eval_ds, collator, None, None


def prepare_cls_data(args, tokenizer):
    """Tokenize a text-classification dataset and return an accuracy `compute_metrics`."""
    raw_train = load_dataset(args.dataset_name, args.dataset_config, split=args.train_split)
    raw_eval = load_dataset(args.dataset_name, args.dataset_config, split=args.eval_split)
    num_labels = len(set(raw_train[args.label_column]))

    def tokenize(batch):
        out = tokenizer(batch[args.text_column], truncation=True, max_length=args.max_seq_length)
        out["labels"] = batch[args.label_column]
        return out

    columns = raw_train.column_names
    train = raw_train.map(tokenize, batched=True, remove_columns=columns)
    eval_ds = raw_eval.map(tokenize, batched=True, remove_columns=columns)
    train = _maybe_truncate(train, args.max_train_samples)
    eval_ds = _maybe_truncate(eval_ds, args.max_eval_samples)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = np.argmax(logits, axis=-1)
        return {"accuracy": float((preds == labels).mean())}

    return train, eval_ds, collator, compute_metrics, num_labels


# --------------------------------------------------------------------------------------- gsm8k


def extract_gsm8k_final_answer(text):
    """Extract and normalize the final GSM8K answer (the number after '####', else the first number)."""
    if not text:
        return ""
    match = _GSM8K_ANSWER_LINE.search(text)
    candidate = match.group(1).strip() if match else text.strip()
    num = _GSM8K_NUMBER.search(candidate)
    return num.group(0).replace(",", "").strip() if num else candidate


def _clean_generated_text(text):
    text = _THINK_PATTERN.sub(" ", text)
    text = _HTML_TAG_PATTERN.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _apply_chat_template(tokenizer, user_content):
    messages = [{"role": "user", "content": user_content}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Tokenizer has no chat template: fall back to the raw prompt.
        return user_content + " "


class GSM8KCollator:
    """Pad GSM8K features (right-padding) and carry prompt-only inputs + gold answers for generation eval."""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    @staticmethod
    def _pad(seqs, pad_value):
        max_len = max(len(s) for s in seqs)
        return torch.tensor([s + [pad_value] * (max_len - len(s)) for s in seqs], dtype=torch.long)

    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        attention = [f.get("attention_mask", [1] * len(f["input_ids"])) for f in features]
        labels = [f["labels"] for f in features]
        prompt_ids = [f["prompt_input_ids"] for f in features]
        prompt_attn = [f.get("prompt_attention_mask", [1] * len(f["prompt_input_ids"])) for f in features]
        return {
            "input_ids": self._pad(input_ids, self.pad_id),
            "attention_mask": self._pad(attention, 0),
            "labels": self._pad(labels, -100),
            "prompt_input_ids": self._pad(prompt_ids, self.pad_id),
            "prompt_attention_mask": self._pad(prompt_attn, 0),
            "gold_answers": [f["gold_answer"] for f in features],
        }


def prepare_gsm8k_data(args, tokenizer):
    """Format openai/gsm8k for SFT with prompt-masked labels (following run_shadow_peft.py)."""
    raw_train = load_dataset(args.dataset_name, args.gsm8k_subset, split=args.train_split)
    raw_eval = load_dataset(args.dataset_name, args.gsm8k_subset, split=args.eval_split)

    def format_example(example):
        question = example["question"]
        answer = example["answer"]
        gold = extract_gsm8k_final_answer(answer)
        if args.gsm8k_answer_mode == "thinking":
            user_content = f"Question: {question}\nAnswer:"
            target_text = answer
        else:
            user_content = f"Question: {question}\nGive only the final answer as a number.\nAnswer:"
            target_text = gold

        prompt_text = _apply_chat_template(tokenizer, user_content)
        full_text = prompt_text + target_text + (tokenizer.eos_token or "")
        if not full_text.endswith("\n"):
            full_text += "\n"

        tokenized = tokenizer(full_text, truncation=True, max_length=args.max_seq_length)
        prompt_tokenized = tokenizer(prompt_text, truncation=True, max_length=args.max_seq_length)
        prompt_length = len(prompt_tokenized["input_ids"])

        # Supervise only the answer tokens; mask the prompt.
        tokenized["labels"] = [-100] * prompt_length + tokenized["input_ids"][prompt_length:]
        tokenized["prompt_input_ids"] = prompt_tokenized["input_ids"]
        tokenized["prompt_attention_mask"] = prompt_tokenized["attention_mask"]
        tokenized["gold_answer"] = gold
        return tokenized

    train = raw_train.map(format_example, remove_columns=raw_train.column_names)
    eval_ds = raw_eval.map(format_example, remove_columns=raw_eval.column_names)
    train = _maybe_truncate(train, args.max_train_samples)
    eval_ds = _maybe_truncate(eval_ds, args.max_eval_samples)
    return train, eval_ds, GSM8KCollator(tokenizer), None, None


def _left_pad_prompts(prompt_ids, prompt_mask, pad_id):
    """Left-pad variable-length prompts so a batch can be generated in one call."""
    lengths = prompt_mask.sum(dim=1)
    max_len = int(lengths.max().item())
    batch_size = prompt_ids.shape[0]
    device = prompt_ids.device
    padded_ids = torch.full((batch_size, max_len), pad_id, dtype=prompt_ids.dtype, device=device)
    padded_mask = torch.zeros((batch_size, max_len), dtype=prompt_mask.dtype, device=device)
    for i in range(batch_size):
        length = int(lengths[i].item())
        padded_ids[i, max_len - length :] = prompt_ids[i, :length]
        padded_mask[i, max_len - length :] = 1
    return padded_ids, padded_mask, max_len


def is_shadow_backbone_param(name: str) -> bool:
    """True for trainable parameters owned by the shadow backbone (not injection/update/projection)."""
    return name.startswith("base_model.shadow_model.")


def build_shadow_optimizer_groups(opt_model, learning_rate, weight_decay, shadow_lr_scale, decay_parameters):
    shadow_tuner = getattr(opt_model, "base_model", None)
    is_explicit = bool(getattr(shadow_tuner, "_explicit_shadow_model", False))
    shadow_lr = learning_rate * shadow_lr_scale

    buckets = {
        ("shadow", True): [],
        ("shadow", False): [],
        ("other", True): [],
        ("other", False): [],
    }
    for name, param in opt_model.named_parameters():
        if not param.requires_grad:
            continue
        bucket = "shadow" if is_shadow_backbone_param(name) else "other"
        use_decay = name in decay_parameters
        buckets[(bucket, use_decay)].append(param)

    groups = []
    for (bucket, use_decay), params in buckets.items():
        if not params:
            continue
        groups.append(
            {
                "params": params,
                "weight_decay": weight_decay if use_decay else 0.0,
                "lr": shadow_lr if bucket == "shadow" else learning_rate,
            }
        )
    return groups, is_explicit, shadow_lr


class ShadowOptimizerTrainer(Trainer):
    """Trainer that applies a separate LR scale to the shadow backbone."""

    def __init__(self, *args, use_shadow_lr_scales=False, shadow_lr_scale=1.0, **kwargs):
        self.use_shadow_lr_scales = use_shadow_lr_scales
        self.shadow_lr_scale = shadow_lr_scale
        self._shadow_loss_log_sum = 0.0
        self._shadow_loss_log_count = 0
        super().__init__(*args, **kwargs)

    def create_optimizer(self, model=None):
        if not self.use_shadow_lr_scales:
            return super().create_optimizer(model)

        opt_model = self.model if model is None else model
        if self.optimizer is not None:
            return self.optimizer

        decay_parameters = self.get_decay_parameter_names(opt_model)
        optimizer_grouped_parameters, is_explicit, shadow_lr = build_shadow_optimizer_groups(
            opt_model,
            self.args.learning_rate,
            self.args.weight_decay,
            self.shadow_lr_scale,
            decay_parameters,
        )
        main_print(
            f"ShadowPEFT optimizer: {'explicit' if is_explicit else 'implicit'} shadow backbone "
            f"lr={shadow_lr:g} (scale={self.shadow_lr_scale}), other trainable params lr={self.args.learning_rate:g}"
        )

        if self.optimizer_cls_and_kwargs is not None:
            optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
        else:
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

        if is_optimizer_factory(optimizer_cls):
            self.optimizer = optimizer_cls()(opt_model, **optimizer_kwargs)
        else:
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def _record_shadow_loss(self, outputs):
        shadow_loss = getattr(outputs, "shadow_loss", None)
        if shadow_loss is None or not getattr(self, "is_in_train", False):
            return
        value = shadow_loss.detach().float()
        if value.numel() != 1:
            value = value.mean()
        if hasattr(self, "accelerator"):
            value = self.accelerator.reduce(value, reduction="mean")
        self._shadow_loss_log_sum += value.item()
        self._shadow_loss_log_count += 1

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        self._record_shadow_loss(outputs)
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if "loss" in logs and self._shadow_loss_log_count:
            logs = dict(logs)
            logs["shadow_loss"] = self._shadow_loss_log_sum / self._shadow_loss_log_count
            self._shadow_loss_log_sum = 0.0
            self._shadow_loss_log_count = 0
        return super().log(logs, *args, **kwargs)


class GSM8KTrainer(ShadowOptimizerTrainer):
    """Trainer that evaluates GSM8K by generating answers and computing exact-match accuracy."""

    def __init__(
        self,
        *args,
        gen_max_new_tokens=DEFAULT_GSM8K_GENERATION_TOKENS,
        max_print_predictions=None,
        compute_eval_loss=False,
        distributed_eval=False,
        use_shadow_lr_scales=False,
        shadow_lr_scale=1.0,
        **kwargs,
    ):
        self.gen_max_new_tokens = gen_max_new_tokens
        self.max_print_predictions = max_print_predictions
        self.compute_eval_loss = compute_eval_loss
        self.distributed_eval = distributed_eval
        self.gsm8k_inference_mode = "base_shadow"
        super().__init__(
            *args,
            use_shadow_lr_scales=use_shadow_lr_scales,
            shadow_lr_scale=shadow_lr_scale,
            **kwargs,
        )

    _EVAL_ONLY_KEYS = ("prompt_input_ids", "prompt_attention_mask", "gold_answers")

    def _gsm8k_eval_tag(self):
        return "shadow_only" if self.gsm8k_inference_mode == "shadow_only" else "base_shadow"

    def _maybe_print_prediction(self, sample_idx, pred, gold, completion):
        if not self.accelerator.is_main_process:
            return
        if self.max_print_predictions is not None and sample_idx >= self.max_print_predictions:
            return
        gold_str = str(gold).strip()
        match = pred and pred == gold_str
        status = "correct" if match else "wrong"
        completion_preview = _clean_generated_text(completion)
        if len(completion_preview) > 120:
            completion_preview = completion_preview[:117] + "..."
        print(
            f"GSM8K eval [{self._gsm8k_eval_tag()}] [{sample_idx + 1}]: pred={pred!r} gold={gold_str!r} ({status})"
            f" | completion: {completion_preview}"
        )

    def _generation_model(self):
        """Unwrapped model for generation; FSDP summon is applied separately when needed."""
        return self.accelerator.unwrap_model(self.model)

    def _fsdp_module_for_summon(self):
        if self.is_fsdp_enabled and isinstance(self.model_wrapped, FullyShardedDataParallel):
            return self.model_wrapped
        return None

    def _generation_use_cache(self):
        """ShadowPEFT disables KV cache; LoRA and other methods can use it for faster eval."""
        peft_model = self._generation_model()
        peft_config = getattr(peft_model, "peft_config", None)
        if peft_config:
            return all(config.peft_type != PeftType.SHADOW for config in peft_config.values())
        return True

    def _configure_generation_cache(self, gen_model, use_cache):
        if not use_cache:
            return
        if hasattr(gen_model, "config") and gen_model.config is not None:
            gen_model.config.use_cache = True
        generation_config = getattr(gen_model, "generation_config", None)
        if generation_config is not None:
            generation_config.use_cache = True

    def _synced_gpus_for_generate(self):
        # Never sync token steps across ranks unless ZeRO-3 requires it; FSDP uses summon instead.
        return is_deepspeed_zero3_enabled()

    @contextlib.contextmanager
    def _fsdp_summon_context(self, fsdp_module):
        if fsdp_module is not None:
            with FullyShardedDataParallel.summon_full_params(fsdp_module, writeback=False):
                yield
        else:
            yield

    @contextlib.contextmanager
    def _mixed_precision_context(self):
        if self.accelerator.mixed_precision != "no":
            with self.accelerator.autocast():
                yield
        else:
            yield

    def _main_process_eval_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            pin_memory=self.args.dataloader_pin_memory,
            num_workers=self.args.dataloader_num_workers,
        )

    def _log_eval_settings(self, use_cache, synced_gpus, num_samples):
        if not self.accelerator.is_main_process:
            return
        mode = "distributed" if self.distributed_eval and self.accelerator.num_processes > 1 else "main-process"
        fsdp = self.is_fsdp_enabled
        print(
            f"GSM8K generation eval ({self._gsm8k_eval_tag()}, {mode}, fsdp={fsdp}): "
            f"{num_samples} samples, batch_size={self.args.per_device_eval_batch_size}, "
            f"max_new_tokens={self.gen_max_new_tokens}, use_cache={use_cache}, synced_gpus={synced_gpus}"
        )
        if not use_cache:
            print(
                "ShadowPEFT eval disables KV cache, so generation reruns the full sequence each token and is much "
                "slower. For faster checks use --gsm8k_answer_mode final --generation_max_length 32."
            )

    def _generate_batch(
        self, gen_model, prompt_ids, prompt_mask, gold_answers, pad_id, eos_id, use_cache, synced_gpus, sample_idx
    ):
        gen_ids, gen_mask, prompt_len = _left_pad_prompts(prompt_ids, prompt_mask, pad_id)
        generated = gen_model.generate(
            input_ids=gen_ids,
            attention_mask=gen_mask,
            max_new_tokens=self.gen_max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            use_cache=use_cache,
            synced_gpus=synced_gpus,
        )
        batch_flags = []
        tokenizer = self.processing_class
        for i in range(generated.shape[0]):
            completion = tokenizer.decode(generated[i, prompt_len:], skip_special_tokens=True)
            pred = extract_gsm8k_final_answer(_clean_generated_text(completion))
            gold = gold_answers[i]
            self._maybe_print_prediction(sample_idx + i, pred, gold, completion)
            batch_flags.append(1.0 if (pred and pred == str(gold).strip()) else 0.0)
        return batch_flags

    def _run_generation_eval(self, dataloader):
        gen_model = self._generation_model()
        use_cache = self._generation_use_cache()
        self._configure_generation_cache(gen_model, use_cache)
        synced_gpus = self._synced_gpus_for_generate()
        fsdp_module = self._fsdp_module_for_summon()
        tokenizer = self.processing_class
        pad_id = tokenizer.pad_token_id
        eos_id = tokenizer.eos_token_id
        device = self.accelerator.device

        num_samples = len(dataloader.dataset) if hasattr(dataloader, "dataset") else None
        self._log_eval_settings(use_cache, synced_gpus, num_samples)

        correct_flags = []
        sample_idx = 0
        batch_idx = 0
        with torch.inference_mode(), self._fsdp_summon_context(fsdp_module), self._mixed_precision_context():
            for inputs in dataloader:
                batch_start = time.perf_counter()
                inputs = self._prepare_inputs(inputs)
                prompt_ids = inputs["prompt_input_ids"]
                prompt_mask = inputs["prompt_attention_mask"]
                gold_answers = inputs["gold_answers"]

                batch_flags = self._generate_batch(
                    gen_model,
                    prompt_ids,
                    prompt_mask,
                    gold_answers,
                    pad_id,
                    eos_id,
                    use_cache,
                    synced_gpus,
                    sample_idx,
                )
                sample_idx += len(batch_flags)
                batch_idx += 1

                if self.distributed_eval and self.accelerator.num_processes > 1:
                    flags = self.accelerator.gather_for_metrics(torch.tensor(batch_flags, device=device))
                    correct_flags.append(flags)
                else:
                    correct_flags.extend(batch_flags)

                if self.accelerator.is_main_process:
                    elapsed = time.perf_counter() - batch_start
                    print(
                        f"GSM8K eval [{self._gsm8k_eval_tag()}] batch {batch_idx}: "
                        f"{len(batch_flags)} sample(s) in {elapsed:.1f}s "
                        f"({elapsed / max(len(batch_flags), 1):.1f}s/sample)"
                    )

        if self.distributed_eval and self.accelerator.num_processes > 1:
            flags = torch.cat(correct_flags) if correct_flags else torch.zeros(0, device=device)
            total = int(flags.numel())
            correct = int(flags.sum().item())
        else:
            total = len(correct_flags)
            correct = int(sum(correct_flags))
        return correct, total

    def _run_teacher_forced_eval_loss(self, dataloader):
        model = self.model
        device = self.accelerator.device
        loss_sum = torch.zeros(1, device=device)
        loss_count = torch.zeros(1, device=device)
        shadow_loss_sum = torch.zeros(1, device=device)
        shadow_loss_count = torch.zeros(1, device=device)
        for inputs in dataloader:
            inputs = self._prepare_inputs(inputs)
            inputs = {k: v for k, v in inputs.items() if k not in self._EVAL_ONLY_KEYS}
            with torch.inference_mode(), self._mixed_precision_context():
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["labels"],
                )
            if getattr(outputs, "loss", None) is not None:
                loss_sum += outputs.loss.detach().float()
                loss_count += 1
            if getattr(outputs, "shadow_loss", None) is not None:
                shadow_loss_sum += outputs.shadow_loss.detach().float()
                shadow_loss_count += 1
        loss_sum = self.accelerator.reduce(loss_sum, reduction="sum")
        loss_count = self.accelerator.reduce(loss_count, reduction="sum")
        shadow_loss_sum = self.accelerator.reduce(shadow_loss_sum, reduction="sum")
        shadow_loss_count = self.accelerator.reduce(shadow_loss_count, reduction="sum")
        avg_loss = float((loss_sum / loss_count).item()) if loss_count.item() > 0 else None
        avg_shadow_loss = (
            float((shadow_loss_sum / shadow_loss_count).item()) if shadow_loss_count.item() > 0 else None
        )
        return avg_loss, avg_shadow_loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = {k: v for k, v in inputs.items() if k not in self._EVAL_ONLY_KEYS}
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    def _ensure_model_prepared_for_eval(self):
        if len(self.accelerator._models) == 0:
            self.model = self.accelerator.prepare_model(self.model, evaluation_mode=True)

    def evaluation_loop(
        self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"
    ):
        self._ensure_model_prepared_for_eval()
        self.model.eval()
        device = self.accelerator.device
        use_main_only = not self.distributed_eval and self.accelerator.num_processes > 1

        if use_main_only:
            if self.accelerator.is_main_process:
                eval_dataloader = self._main_process_eval_dataloader()
                if self.compute_eval_loss:
                    avg_loss, avg_shadow_loss = self._run_teacher_forced_eval_loss(eval_dataloader)
                else:
                    avg_loss, avg_shadow_loss = None, None
                correct, total = self._run_generation_eval(eval_dataloader)
                stats = torch.tensor(
                    [
                        correct,
                        total,
                        avg_loss if avg_loss is not None else -1.0,
                        avg_shadow_loss if avg_shadow_loss is not None else -1.0,
                    ],
                    device=device,
                    dtype=torch.float32,
                )
            else:
                stats = torch.zeros(4, device=device, dtype=torch.float32)
            stats = accel_broadcast(stats, from_process=0)
            self.accelerator.wait_for_everyone()
            correct = int(stats[0].item())
            total = int(stats[1].item())
            avg_loss = None if stats[2].item() < 0 else float(stats[2].item())
            avg_shadow_loss = None if stats[3].item() < 0 else float(stats[3].item())
        else:
            if self.compute_eval_loss:
                avg_loss, avg_shadow_loss = self._run_teacher_forced_eval_loss(dataloader)
            else:
                avg_loss, avg_shadow_loss = None, None
            correct, total = self._run_generation_eval(dataloader)

        metrics = {
            f"{metric_key_prefix}_loss": avg_loss if self.compute_eval_loss else None,
            f"{metric_key_prefix}_shadow_loss": avg_shadow_loss if self.compute_eval_loss else None,
            f"{metric_key_prefix}_accuracy": correct / total if total else 0.0,
            f"{metric_key_prefix}_samples": total,
        }
        if self.accelerator.is_main_process:
            print(
                f"GSM8K {self._gsm8k_eval_tag()} accuracy: {metrics[f'{metric_key_prefix}_accuracy']:.4f} "
                f"({correct}/{total})"
            )
        return EvalLoopOutput(predictions=None, label_ids=None, metrics=metrics, num_samples=total)


# --------------------------------------------------------------------------------------- model / run


def _resolve_dtype(args):
    return torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)


def load_base_model(args, num_labels, model_name=None):
    model_name = model_name or args.model_name
    dtype = _resolve_dtype(args)
    if args.task == "cls":
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = model.config.eos_token_id
    return model


def load_shadow_model(args, shadow_model_name=None):
    """Load an explicit shadow model (a fresh copy each call, since get_peft_model mutates in place).

    Supports both a plain causal LM and a projected shadow model (``AutoModelForCausalLMWithHiddenProjection``, e.g.
    ``shadow-llm/Qwen3-0.6B-H8B``) that bundles a small backbone + a trained hidden-size projection aligned to a larger
    base model. ShadowPEFT reuses that trained projection instead of randomly initializing one.
    """
    shadow_model_name = shadow_model_name or args.shadow_model_name
    if not shadow_model_name:
        return None
    dtype = _resolve_dtype(args)
    try:
        model_type = getattr(AutoConfig.from_pretrained(shadow_model_name), "model_type", None)
    except Exception:
        model_type = None
    if model_type == "causal_lm_with_hidden_projection":
        # Keep the backbone trainable; the projection/lm_head freezing follows the checkpoint's defaults.
        return AutoModelForCausalLMWithHiddenProjection.from_pretrained(
            shadow_model_name, dtype=dtype, freeze_backbone=False
        )
    return AutoModelForCausalLM.from_pretrained(shadow_model_name, dtype=dtype)


def apply_peft(base_model, method, args, shadow_model=None):
    task_type = "SEQ_CLS" if args.task == "cls" else "CAUSAL_LM"
    if method == "lora":
        config = LoraConfig(
            task_type=task_type,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules,
        )
        return get_peft_model(base_model, config)

    config = ShadowConfig(
        task_type=task_type,
        num_shadow_layers=args.shadow_layers,
        injection_hidden_size=args.injection_hidden_size,
        gate_hidden_size=args.gate_hidden_size,
        shadow_intermediate_size=args.shadow_intermediate_size,
        shadow_num_attention_heads=args.shadow_num_attention_heads,
        alpha=args.shadow_alpha,
        dropout=args.shadow_dropout,
        shadow_loss_weight=args.shadow_loss_weight,
    )
    return get_peft_model(base_model, config, shadow_model=shadow_model)


def resolve_shadow_model_name(method, args):
    """Return explicit shadow model path for eval (CLI flag or train_summary.json)."""
    if args.shadow_model_name:
        return args.shadow_model_name
    shadow_model_name = load_train_summary(method, args).get("shadow_model_name")
    if shadow_model_name:
        main_print(f"Eval for {method}: using shadow model {shadow_model_name} from train_summary.json.")
    return shadow_model_name


def list_adapter_weight_keys(adapter_path):
    """Return tensor names stored in an adapter checkpoint, if present."""
    for filename in ("adapter_model.safetensors", "adapter_model.bin"):
        weights_path = os.path.join(adapter_path, filename)
        if not os.path.isfile(weights_path):
            continue
        if filename.endswith(".safetensors"):
            from safetensors import safe_open

            with safe_open(weights_path, framework="pt") as f:
                return list(f.keys())
        return list(torch.load(weights_path, map_location="cpu", weights_only=True).keys())
    return []


def report_shadow_checkpoint_restore(adapter_path, shadow_model_name):
    """Explain which shadow weights come from the adapter vs a fresh shadow_model_name load."""
    keys = list_adapter_weight_keys(adapter_path)
    if not keys:
        return

    shadow_backbone = [k for k in keys if ".shadow_model." in k]
    adapter_modules = [
        k
        for k in keys
        if any(part in k for part in ("shadow_injection", "shadow_update", "shadow_hidden_projection"))
    ]
    main_print(
        f"Adapter checkpoint: {len(keys)} tensors "
        f"(shadow_backbone={len(shadow_backbone)}, injection/update/projection={len(adapter_modules)})"
    )

    if shadow_backbone:
        main_print(f"Fine-tuned shadow backbone weights will be loaded from the adapter ({len(shadow_backbone)} tensors).")
        if shadow_model_name:
            main_print(
                f"  Explicit shadow architecture is built from {shadow_model_name!r}, then adapter weights overwrite "
                "matching tensors."
            )
        return

    if shadow_model_name:
        main_print(
            "WARNING: adapter has no fine-tuned shadow backbone weights. Eval uses the pretrained shadow from "
            f"{shadow_model_name!r}; only injection/update/projection tensors saved during training are applied."
        )
    else:
        main_print(
            "WARNING: adapter has no shadow backbone weights. Eval rebuilds a fresh implicit shadow network; only "
            f"the {len(keys)} saved adapter tensor(s) are restored."
        )


def normalize_fsdp_key(name):
    """Normalize FSDP wrapper segments out of parameter/module names."""
    if name.startswith("module."):
        name = name.removeprefix("module.")
    if name.startswith("_fsdp_wrapped_module."):
        name = name.removeprefix("_fsdp_wrapped_module.")
    return name.replace("._fsdp_wrapped_module", "")


def is_shadow_trainable_name(name):
    return any(
        marker in name
        for marker in (
            ".shadow_model.",
            ".shadow_injection_model.",
            ".shadow_update_model.",
            ".shadow_hidden_projection.",
            ".shadow_lm_head.",
            ".shadow_classifier_head.",
        )
    )


def collect_fsdp_trainable_peft_state_dict(trainer):
    """Gather only trainable ShadowPEFT tensors from FSDP modules, avoiding the frozen base model."""
    fsdp_model = trainer.model_wrapped if isinstance(trainer.model_wrapped, FullyShardedDataParallel) else trainer.model
    all_trainable = {
        normalize_fsdp_key(name)
        for name, param in fsdp_model.named_parameters()
        if param.requires_grad and ".modules_to_save." not in name and ".original_module." not in name
    }
    state_dict = {}

    fsdp_module_names = [
        (normalize_fsdp_key(name), module)
        for name, module in fsdp_model.named_modules()
        if isinstance(module, FullyShardedDataParallel)
    ]
    nested_prefixes = [prefix for prefix, _ in fsdp_module_names if prefix]

    # Summon each trainable shadow FSDP unit independently. This gathers the 0.6B shadow adapter pieces without
    # materializing the frozen 8B base model on rank 0.
    for prefix, module in fsdp_module_names:
        if prefix:
            names_for_module = {
                name for name in all_trainable if name == prefix or name.startswith(prefix + ".")
            }
        else:
            # Root FSDP owns trainable tensors not handled by nested FSDP modules (e.g. injection Parameters).
            names_for_module = {
                name
                for name in all_trainable
                if not any(name == nested or name.startswith(nested + ".") for nested in nested_prefixes)
            }
        if not names_for_module:
            continue
        if prefix and not is_shadow_trainable_name(prefix + "."):
            continue
        with FullyShardedDataParallel.summon_full_params(
            module,
            recurse=bool(prefix),
            writeback=False,
            rank0_only=False,
            offload_to_cpu=True,
        ):
            if not is_main_process():
                continue
            for name, param in module.named_parameters(recurse=False):
                normalized = normalize_fsdp_key(name)
                if prefix and normalized:
                    normalized = f"{prefix}.{normalized}"
                elif prefix:
                    normalized = prefix
                if normalized in state_dict or normalized not in all_trainable:
                    continue
                if normalized in names_for_module:
                    state_dict[normalized] = param.detach().cpu().clone()

    if is_main_process():
        # Collect trainable tensors that are not FSDP-wrapped (e.g. injection Parameter tensors).
        for name, param in fsdp_model.named_parameters():
            normalized = normalize_fsdp_key(name)
            if normalized in state_dict or normalized not in all_trainable:
                continue
            if is_shadow_trainable_name(normalized):
                state_dict[normalized] = param.detach().cpu().clone()

        missing = sorted(all_trainable.difference(state_dict))
        if missing:
            preview = "\n".join(f"  - {name}" for name in missing[:20])
            raise RuntimeError(
                f"Failed to gather {len(missing)} trainable ShadowPEFT tensor(s) from FSDP; refusing to save a "
                f"partial adapter checkpoint.\n{preview}"
            )
        main_print(f"Collected {len(state_dict)} trainable ShadowPEFT tensors for adapter save.")

    return state_dict


def save_trained_adapter(trainer, method, args, output_dir=None, report=True):
    """Save adapter weights; for FSDP gather full CPU params, then PEFT writes only trainable tensors."""
    output_dir = output_dir or method_dir(args, method)
    if trainer.is_fsdp_enabled:
        trainer.accelerator.wait_for_everyone()
        if is_main_process():
            main_print(f"Saving PEFT adapter with FSDP full params offloaded to CPU: {output_dir}")
        fsdp_model = (
            trainer.model_wrapped if isinstance(trainer.model_wrapped, FullyShardedDataParallel) else trainer.model
        )
        with FullyShardedDataParallel.summon_full_params(
            fsdp_model,
            writeback=False,
            rank0_only=True,
            offload_to_cpu=True,
        ):
            if is_main_process():
                unwrapped = trainer.accelerator.unwrap_model(trainer.model)
                unwrapped.save_pretrained(output_dir)
        trainer.accelerator.wait_for_everyone()
    else:
        trainer.save_model(output_dir)

    if report and method == "shadow":
        report_shadow_checkpoint_restore(output_dir, args.shadow_model_name)


class PeftAdapterCheckpointCallback(TrainerCallback):
    """Write a PEFT adapter into each Trainer checkpoint directory."""

    def __init__(self, method, benchmark_args):
        self.method = method
        self.benchmark_args = benchmark_args
        self.trainer = None

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_save(self, args, state, control, **kwargs):
        if self.trainer is None:
            return control
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        save_trained_adapter(
            self.trainer,
            self.method,
            self.benchmark_args,
            output_dir=checkpoint_dir,
            report=False,
        )
        if self.method == "shadow":
            report_shadow_checkpoint_restore(checkpoint_dir, self.benchmark_args.shadow_model_name)
        return control


def load_trained_model(method, args, num_labels):
    adapter_path = method_dir(args, method)
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(
            f"Adapter checkpoint not found at {adapter_path}. Run with --mode train (or both) first."
        )
    base_model_name = resolve_checkpoint_base_model(method, args)
    base_model = load_base_model(args, num_labels, model_name=base_model_name)
    shadow_model = None
    shadow_model_name = None
    if method == "shadow":
        shadow_model_name = resolve_shadow_model_name(method, args)
        if shadow_model_name:
            shadow_model = load_shadow_model(args, shadow_model_name=shadow_model_name)
            for name, param in shadow_model.named_parameters():
                if param.requires_grad:
                    print(f">>> Trainable shadow model parameter: {name}")
        report_shadow_checkpoint_restore(adapter_path, shadow_model_name)
    return PeftModel.from_pretrained(base_model, adapter_path, shadow_model=shadow_model)


def build_trainer(
    method, args, model, tokenizer, train_ds, eval_ds, collator, compute_metrics, num_labels=None
):
    is_gsm8k = args.task == "gsm8k"
    fsdp_config = build_fsdp_config(args, method=method)
    training_args = TrainingArguments(
        output_dir=method_dir(args, method),
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        bf16_full_eval=args.bf16,
        fp16_full_eval=args.fp16 and not args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        fsdp=args.fsdp,
        fsdp_config=fsdp_config,
        save_strategy="steps" if args.save_steps and args.save_steps > 0 else "no",
        save_steps=args.save_steps if args.save_steps and args.save_steps > 0 else 500,
        save_total_limit=args.save_total_limit,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=not is_gsm8k,
    )

    shadow_lr_kwargs = (
        {"use_shadow_lr_scales": True, "shadow_lr_scale": args.shadow_lr_scale} if method == "shadow" else {}
    )
    if is_gsm8k:
        trainer = GSM8KTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            processing_class=tokenizer,
            gen_max_new_tokens=args.generation_max_length or DEFAULT_GSM8K_GENERATION_TOKENS,
            max_print_predictions=args.gsm8k_max_print_predictions,
            compute_eval_loss=args.gsm8k_eval_loss,
            distributed_eval=args.gsm8k_distributed_eval,
            **shadow_lr_kwargs,
        )
        if method == "shadow" and args.save_steps and args.save_steps > 0:
            callback = PeftAdapterCheckpointCallback(method, args)
            callback.set_trainer(trainer)
            trainer.add_callback(callback)
        return trainer
    trainer_cls = ShadowOptimizerTrainer if method == "shadow" else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        **shadow_lr_kwargs,
    )
    if method == "shadow" and args.save_steps and args.save_steps > 0:
        callback = PeftAdapterCheckpointCallback(method, args)
        callback.set_trainer(trainer)
        trainer.add_callback(callback)
    return trainer


def shadow_eval_modes(args, method):
    """Return ordered ShadowPEFT inference modes to run during eval."""
    if method != "shadow":
        return ["base_shadow"]
    if args.shadow_inference_mode == "both":
        return ["base_shadow", "shadow_only"]
    return [args.shadow_inference_mode]


def eval_metric_prefix(mode, method, modes):
    """Metric key prefix: dual eval uses shadow_only_* for the second pass; single mode uses primary keys."""
    if method != "shadow" or len(modes) == 1:
        return ""
    return "shadow_only" if mode == "shadow_only" else ""


def apply_shadow_inference_mode(model, mode):
    """Set ShadowPEFT inference mode on a PeftModel, if supported."""
    shadow_tuner = getattr(model, "base_model", None)
    if shadow_tuner is not None and hasattr(shadow_tuner, "set_inference_mode"):
        shadow_tuner.set_inference_mode(mode)
        return True
    return False


def print_eval_summary(args, label, result, prefix=""):
    """Print a one-line summary after each eval pass (GSM8K accuracy or cls/clm metric)."""
    if not is_main_process():
        return
    metric_name = metric_name_for(args.task)
    if prefix:
        metric_key = f"{prefix}_{metric_name}"
        samples_key = f"{prefix}_eval_samples"
    else:
        metric_key = metric_name
        samples_key = "eval_samples"
    metric = result.get(metric_key)
    if metric is None:
        return
    if args.task == "gsm8k":
        samples = result.get(samples_key)
        sample_str = f" ({int(samples)} samples)" if samples is not None else ""
        print(f"GSM8K {label} result: accuracy={metric:.4f}{sample_str}")
    else:
        print(f"{label} result: {metric_name}={metric:.4f}")


def evaluate_method(trainer, method, args):
    modes = shadow_eval_modes(args, method)
    result = {"shadow_inference_mode": args.shadow_inference_mode if method == "shadow" else None}

    for mode in modes:
        if method == "shadow":
            if not set_shadow_inference_mode(trainer, mode):
                main_print(f"Warning: could not set ShadowPEFT inference mode {mode!r}; skipping.")
                continue
            if mode == "shadow_only":
                main_print("\n--- Shadow-only eval (lightweight shadow path, no base forward pass) ---")

        if isinstance(trainer, GSM8KTrainer):
            trainer.gsm8k_inference_mode = mode

        prefix = eval_metric_prefix(mode, method, modes)
        metrics = trainer.evaluate()
        result.update(result_metrics_from_eval(metrics, args.task, prefix=prefix))

        label = mode if method == "shadow" else "eval"
        print_eval_summary(args, label, result, prefix=prefix)

    if method == "shadow" and modes != ["shadow_only"]:
        set_shadow_inference_mode(trainer, "base_shadow")
        if isinstance(trainer, GSM8KTrainer):
            trainer.gsm8k_inference_mode = "base_shadow"

    return result


def metric_name_for(task):
    return "perplexity" if task == "clm" else "accuracy"


def shadow_only_metric_name_for(task):
    return f"shadow_only_{metric_name_for(task)}"


def set_shadow_inference_mode(trainer, mode):
    """Switch ShadowPEFT inference mode on the unwrapped PEFT model, if supported."""
    peft_model = trainer.accelerator.unwrap_model(trainer.model)
    return apply_shadow_inference_mode(peft_model, mode)


def result_metrics_from_eval(metrics, task, prefix=""):
    """Extract eval_loss and the task metric from Trainer.evaluate() output."""
    key = f"{prefix}_" if prefix else ""
    eval_loss = metrics.get("eval_loss")
    result = {f"{key}eval_loss" if prefix else "eval_loss": eval_loss}
    shadow_loss = metrics.get("eval_shadow_loss")
    if shadow_loss is not None:
        result[f"{key}shadow_loss" if prefix else "shadow_loss"] = shadow_loss
    if task == "clm":
        result[f"{key}perplexity" if prefix else "perplexity"] = (
            math.exp(eval_loss) if eval_loss is not None else None
        )
    else:
        result[f"{key}accuracy" if prefix else "accuracy"] = metrics.get("eval_accuracy")
    if task == "gsm8k" and not prefix:
        result["eval_samples"] = metrics.get("eval_samples")
    elif task == "gsm8k" and prefix:
        result[f"{prefix}_eval_samples"] = metrics.get("eval_samples")
    return result


def configure_fsdp_for_peft(trainer):
    """Apply PEFT's FSDP auto-wrap policy before Trainer/accelerate wraps the model."""
    fsdp_plugin = getattr(trainer.accelerator.state, "fsdp_plugin", None)
    if fsdp_plugin is None:
        return
    from peft.utils.other import fsdp_auto_wrap_policy
    from peft.utils.peft_types import PeftType

    fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

    peft_config = getattr(trainer.model, "peft_config", None)
    if peft_config is not None:
        config = next(iter(peft_config.values()))
        if getattr(config, "peft_type", None) == PeftType.SHADOW and not fsdp_plugin.use_orig_params:
            main_print(
                "ShadowPEFT requires FSDP use_orig_params=True (mixed frozen/trainable params); enabling it."
            )
            fsdp_plugin.use_orig_params = True


def run_method(method, args, tokenizer, train_ds, eval_ds, collator, compute_metrics, num_labels):
    do_train = args.mode in ("train", "both")
    do_eval = args.mode in ("eval", "both")
    main_print(f"\n{'=' * 70}\nRunning method: {method} (mode={args.mode})\n{'=' * 70}")
    set_seed(args.seed)

    train_summary = load_train_summary(method, args) if do_eval and not do_train else {}

    if do_eval and not do_train:
        model = load_trained_model(method, args, num_labels)
        trainable = train_summary.get("trainable_params")
        total = train_summary.get("total_params")
        if trainable is None or total is None:
            trainable, total = model.get_nb_trainable_parameters()
    else:
        shadow_model = load_shadow_model(args) if method == "shadow" else None
        if shadow_model is not None:
            for name, param in shadow_model.named_parameters():
                if param.requires_grad:
                    print(f">>> Trainable shadow model parameter: {name}")
        model = apply_peft(load_base_model(args, num_labels), method, args, shadow_model=shadow_model)
        print(">>> model:", model)
        trainable, total = model.get_nb_trainable_parameters()

    if args.gradient_checkpointing:
        model.config.use_cache = False
    if is_main_process() and hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    trainer = build_trainer(
        method, args, model, tokenizer, train_ds, eval_ds, collator, compute_metrics, num_labels
    )

    train_time = train_summary.get("train_time_s")
    if do_train:
        configure_fsdp_for_peft(trainer)
        start = time.perf_counter()
        trainer.train()
        train_time = time.perf_counter() - start
        save_trained_adapter(trainer, method, args)
        save_train_summary(
            method,
            args,
            {
                "method": method,
                "model_name": args.model_name,
                "task": args.task,
                "shadow_model_name": args.shadow_model_name,
                "shadow_lr_scale": args.shadow_lr_scale,
                "trainable_params": trainable,
                "total_params": total,
                "trainable_pct": 100.0 * trainable / total,
                "train_time_s": train_time,
            },
        )
        main_print(f"Saved adapter to {method_dir(args, method)}")

    result = {
        "method": method,
        "mode": args.mode,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": 100.0 * trainable / total,
        "train_time_s": train_time,
    }

    if do_eval:
        result.update(evaluate_method(trainer, method, args))

    return result


def print_comparison(args, results, model_name=None):
    metric_name = metric_name_for(args.task)
    shadow_only_name = shadow_only_metric_name_for(args.task)
    has_shadow_only = any(shadow_only_name in r for r in results)
    display_model = model_name or args.model_name

    header = f"{'method':<10}{'trainable':>14}{'trainable%':>12}{'train_time(s)':>15}{metric_name:>14}"
    if has_shadow_only:
        header += f"{'shadow_only':>14}"
    main_print(f"\n{'=' * 70}\nComparison ({args.task}, {display_model})\n{'=' * 70}")
    main_print(header)
    main_print("-" * len(header))
    for r in results:
        metric = r.get(metric_name)
        metric_str = f"{metric:.4f}" if metric is not None else "n/a"
        row = (
            f"{r['method']:<10}{r['trainable_params']:>14,}{r['trainable_pct']:>11.3f}%"
            f"{r.get('train_time_s') or 0:>15.1f}{metric_str:>14}"
        )
        if has_shadow_only:
            shadow_only = r.get(shadow_only_name)
            shadow_only_str = f"{shadow_only:.4f}" if shadow_only is not None else "n/a"
            row += f"{shadow_only_str:>14}"
        main_print(row)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    eval_model_name = args.model_name
    if args.mode == "eval":
        eval_model_name = resolve_checkpoint_base_model(args.methods[0], args)
    tokenizer = build_tokenizer(eval_model_name)

    prepare = {"cls": prepare_cls_data, "clm": prepare_clm_data, "gsm8k": prepare_gsm8k_data}[args.task]
    train_ds, eval_ds, collator, compute_metrics, num_labels = prepare(args, tokenizer)
    if args.mode == "train":
        eval_ds = None
    elif args.mode == "eval":
        train_ds = None

    results = [
        run_method(method, args, tokenizer, train_ds, eval_ds, collator, compute_metrics, num_labels)
        for method in args.methods
    ]

    if args.mode in ("eval", "both"):
        print_comparison(args, results, model_name=eval_model_name)
    if is_main_process():
        out_path = os.path.join(args.output_dir, "results.json")
        with open(out_path, "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=2)
        print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
