# Copyright 2026-present the HuggingFace Inc. team.

"""Main entrypoint for RLMath experiments using TRL GRPOTrainer."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from collections.abc import Callable
from typing import Any

import torch
from data import load_rl_datasets
from reward import extract_boxed, math_reward_fn, safe_grade
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed
from trl import GRPOConfig, GRPOTrainer
from utils import (
    FILE_NAME_DEFAULT_TRAIN_PARAMS,
    RLTrainConfig,
    build_base_result,
    get_result_dir,
    load_train_config,
    now_iso,
    save_result,
)

from peft import PeftConfig, PeftModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_path", help="Path to experiment dir: experiments/<method>/<name>")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def validate_experiment_path(path: str) -> str:
    if not os.path.exists(FILE_NAME_DEFAULT_TRAIN_PARAMS):
        raise FileNotFoundError(f"Could not find {FILE_NAME_DEFAULT_TRAIN_PARAMS}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Experiment path does not exist: {path}")

    path = path.rstrip(os.path.sep)
    parts = path.split(os.path.sep)
    if len(parts) != 3 or parts[-3] != "experiments":
        raise ValueError("Experiment path must be experiments/<peft-method>/<experiment-name>")
    return os.path.join(parts[-2], parts[-1])


def load_peft_config_dict(exp_path: str) -> dict[str, Any] | None:
    adapter_path = os.path.join(exp_path, "adapter_config.json")
    if not os.path.exists(adapter_path):
        return None
    with open(adapter_path) as f:
        return json.load(f)


def load_peft_config(exp_path: str, total_step: int | None = None) -> PeftConfig | None:
    adapter_path = os.path.join(exp_path, "adapter_config.json")
    if not os.path.exists(adapter_path):
        return None
    with open(adapter_path) as f:
        cfg_dict = json.load(f)
    # AdaLoRA requires total_step to schedule rank pruning
    if cfg_dict.get("peft_type", "").upper() == "ADALORA" and total_step is not None:
        cfg_dict.setdefault("total_step", total_step)
    return PeftConfig.from_peft_type(**cfg_dict)


@torch.no_grad()
def evaluate_pass_at_1(
    *,
    model,
    tokenizer: AutoTokenizer,
    dataset,
    max_completion_length: int,
) -> float:
    correct = 0
    total = 0

    for row in dataset:
        inputs = tokenizer(row["prompt"], return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_completion_length,
            pad_token_id=tokenizer.pad_token_id,
        )
        completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)

        try:
            pred = extract_boxed(completion)
            is_correct = safe_grade(pred, row["ground_truth"])
        except ValueError:
            is_correct = False
        correct += int(is_correct)
        total += 1

    return correct / max(total, 1)


class RLResultCallback(TrainerCallback):
    """Persist training progress incrementally so long runs always leave artifacts."""

    def __init__(self, result: dict[str, Any], flush_fn: Callable[[], str], is_adalora: bool = False) -> None:
        self.result = result
        self.flush_fn = flush_fn
        self.is_adalora = is_adalora

    def _sync_metrics(self, state) -> None:
        self.result["train_info"]["metrics"] = list(state.log_history)
        self.result["train_info"]["last_update_at"] = now_iso()

    def on_train_begin(self, args, state, control, **kwargs):
        self.result["train_info"]["status"] = "running"
        self.result["train_info"]["phase"] = "train"
        self._sync_metrics(state)
        self.flush_fn()
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.result["train_info"]["status"] = "running"
        self.result["train_info"]["phase"] = "train"
        self._sync_metrics(state)
        self.flush_fn()
        return control

    def on_save(self, args, state, control, **kwargs):
        self.result["train_info"]["status"] = "running"
        self.result["train_info"]["phase"] = "train"
        self._sync_metrics(state)
        self.flush_fn()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.result["train_info"]["status"] = "train_completed"
        self.result["train_info"]["phase"] = "post_train"
        self.result["train_info"]["train_completed"] = True
        self._sync_metrics(state)
        self.flush_fn()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if not self.is_adalora:
            return control
        model = kwargs.get("model")
        if model is None:
            return control
        base_model = getattr(model, "base_model", None)
        if base_model is not None and hasattr(base_model, "update_and_allocate"):
            # AdaLoRA's update_ipt requires gradients on all lora params.
            # In GRPO some params may lack gradients; fill with zeros so
            # the importance calculation can proceed.
            for p in model.parameters():
                if p.requires_grad and p.grad is None:
                    p.grad = torch.zeros_like(p)
            base_model.update_and_allocate(state.global_step)
        return control


def _resolve_torch_dtype(dtype_name: str):
    if dtype_name in {"bfloat16", "float16", "float32"}:
        return getattr(torch, dtype_name)
    return None


def _remap_adapter_keys(checkpoint_dir: str) -> None:
    """Fix adapter key names for composite architectures (e.g. Qwen3.5).

    When a model has `model.language_model.layers` instead of `model.layers`,
    GRPOTrainer saves LoRA/AdaLoRA keys with the `language_model.` prefix, but
    PeftModel.from_pretrained expects them without it.  This rewrites the
    safetensors file and adapter_config.json in-place so loading works correctly.
    """
    import safetensors.torch as st

    # Remap safetensors weights
    adapter_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
    if not os.path.exists(adapter_path):
        return

    tensors = st.load_file(adapter_path)
    needs_remap = any("language_model." in k for k in tensors)
    if not needs_remap:
        return

    remapped = {}
    for k, v in tensors.items():
        new_key = k.replace(".language_model.", ".")
        remapped[new_key] = v
    st.save_file(remapped, adapter_path)

    # Fix adapter_config.json for AdaLoRA composite architecture keys
    config_path = os.path.join(checkpoint_dir, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        changed = False
        for field in ("rank_pattern", "lora_alpha"):
            mapping = cfg.get(field)
            if isinstance(mapping, dict) and any("language_model." in k for k in mapping):
                cfg[field] = {k.replace("language_model.", ""): v for k, v in mapping.items()}
                changed = True
        if changed:
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=2)


def _load_eval_model(
    *,
    model_id: str,
    checkpoint_dir: str,
    peft_cfg: PeftConfig | None,
    dtype_name: str,
):
    torch_dtype = _resolve_torch_dtype(dtype_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if peft_cfg is None:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir, torch_dtype=torch_dtype, device_map=device
        )
    else:
        _remap_adapter_keys(checkpoint_dir)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=device
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        model = model.merge_and_unload()
    model.eval()
    return model


def run_experiment(experiment_path: str, verbose: bool = False) -> str:
    experiment_name = validate_experiment_path(experiment_path)
    cfg: RLTrainConfig = load_train_config(experiment_path)
    set_seed(cfg.seed)

    peft_cfg_dict = load_peft_config_dict(experiment_path)
    peft_cfg = load_peft_config(experiment_path, total_step=cfg.max_steps)

    result = build_base_result(experiment_name, cfg, peft_cfg_dict)
    branch = result["run_info"]["peft_branch"]
    out_path: str | None = None

    def flush_result() -> str:
        out_dir = get_result_dir(result["train_info"]["status"], branch)
        return save_result(result, out_dir, experiment_name)

    t0 = time.perf_counter()
    status = "failed"
    error_msg = ""
    got_sigterm = {"value": False}

    def _sigterm_handler(signum, frame):
        got_sigterm["value"] = True
        raise KeyboardInterrupt("Received SIGTERM")

    prev_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        train_ds, test_ds = load_rl_datasets(
            dataset_name=cfg.dataset_name,
            dataset_config=cfg.dataset_config,
            train_split=cfg.dataset_train_split,
            test_split=cfg.dataset_test_split,
            train_subset_size=cfg.train_subset_size,
            eval_subset_size=cfg.eval_subset_size,
            seed=cfg.seed,
        )

        if verbose:
            print(f"Dataset loaded: {len(train_ds)} train, {len(test_ds)} test")

        output_dir = os.path.join("checkpoints", experiment_name.replace("/", "--"))

        # Build GRPOConfig with vLLM colocate support
        grpo_kwargs = dict(
            output_dir=output_dir,
            remove_unused_columns=False,
            max_steps=cfg.max_steps,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type=cfg.lr_scheduler_type,
            warmup_ratio=cfg.warmup_ratio,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            gradient_checkpointing=cfg.gradient_checkpointing,
            max_grad_norm=cfg.max_grad_norm,
            per_device_eval_batch_size=cfg.per_device_eval_batch_size,
            eval_strategy="no",
            logging_steps=cfg.logging_steps,
            save_steps=cfg.save_steps if cfg.save_checkpoint else 0,
            save_total_limit=cfg.save_total_limit,
            bf16=cfg.dtype == "bfloat16",
            fp16=cfg.dtype == "float16",
            max_completion_length=cfg.max_completion_length,
            num_generations=cfg.num_generations,
            num_generations_eval=cfg.num_generations_eval,
            generation_batch_size=cfg.generation_batch_size or cfg.num_generations,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            min_p=cfg.min_p,
            repetition_penalty=cfg.repetition_penalty,
            beta=cfg.beta,
            num_iterations=cfg.num_iterations,
            epsilon=cfg.epsilon,
            epsilon_high=cfg.epsilon_high,
            delta=cfg.delta,
            loss_type=cfg.loss_type,
            scale_rewards=cfg.scale_rewards,
            importance_sampling_level=cfg.importance_sampling_level,
            mask_truncated_completions=cfg.mask_truncated_completions,
            disable_dropout=cfg.disable_dropout,
            use_vllm=cfg.use_vllm,
            report_to=[],
        )

        # Add vLLM-specific params when vLLM is enabled
        if cfg.use_vllm:
            grpo_kwargs["vllm_mode"] = cfg.vllm_mode
            grpo_kwargs["vllm_gpu_memory_utilization"] = cfg.vllm_gpu_memory_utilization
            grpo_kwargs["vllm_enable_sleep_mode"] = cfg.vllm_enable_sleep_mode
            if cfg.vllm_max_model_length is not None:
                grpo_kwargs["vllm_max_model_length"] = cfg.vllm_max_model_length

        grpo_args = GRPOConfig(**grpo_kwargs)

        grpo_args.model_init_kwargs = {"dtype": cfg.dtype}

        is_adalora = bool((peft_cfg_dict or {}).get("peft_type", "").upper() == "ADALORA")
        trainer = GRPOTrainer(
            model=cfg.model_id,
            reward_funcs=math_reward_fn,
            args=grpo_args,
            train_dataset=train_ds,
            eval_dataset=None,
            peft_config=peft_cfg,
            callbacks=[RLResultCallback(result, flush_result, is_adalora=is_adalora)],
        )

        if verbose:
            print(f"Running GRPO for {experiment_name} on {cfg.dataset_name}...")
            trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in trainer.model.parameters())
            print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        train_output = trainer.train()
        log_history = trainer.state.log_history
        eval_ckpt_dir = os.path.join(output_dir, "eval_checkpoint")
        trainer.save_model(eval_ckpt_dir)
        result["train_info"].update(
            {
                "status": "train_completed",
                "phase": "post_train",
                "train_completed": True,
                "train_time": train_output.metrics.get("train_runtime"),
                "metrics": log_history,
                "checkpoint_path": eval_ckpt_dir,
                "last_update_at": now_iso(),
            }
        )
        out_path = flush_result()

        # Evaluate: single pass on test set (subset for speed).
        result["train_info"]["phase"] = "eval"
        result["train_info"]["last_update_at"] = now_iso()
        out_path = flush_result()

        if verbose:
            print("Running post-training eval on test set...")

        eval_size = min(cfg.eval_subset_size, len(test_ds))
        eval_ds = test_ds.select(range(eval_size))
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = _load_eval_model(
            model_id=cfg.model_id,
            checkpoint_dir=eval_ckpt_dir,
            peft_cfg=peft_cfg,
            dtype_name=cfg.dtype,
        )
        test_pass_at_1 = evaluate_pass_at_1(
            model=model,
            tokenizer=tokenizer,
            dataset=eval_ds,
            max_completion_length=cfg.max_completion_length,
        )

        last_log = next((x for x in reversed(log_history) if "reward" in x), {})

        result["train_info"].update(
            {
                "status": "success",
                "phase": "done",
                "train_completed": True,
                "train_time": train_output.metrics.get("train_runtime"),
                "metrics": log_history,
                "last_update_at": now_iso(),
            }
        )
        result["rl_eval_info"] = {
            "test_pass_at_1": test_pass_at_1,
            "reward_mean": last_log.get("reward"),
            "reward_std": last_log.get("reward_std"),
            "kl_mean": last_log.get("kl"),
            "num_eval_samples": eval_size,
        }
        if verbose:
            print(f"Test pass@1: {test_pass_at_1:.3f} ({eval_size} samples)")
        status = "success"

    except KeyboardInterrupt:
        status = "cancelled"
        error_msg = "terminated (SIGTERM)" if got_sigterm["value"] else "manually canceled"
    except torch.OutOfMemoryError as exc:
        status = "cancelled"
        error_msg = str(exc)
    except Exception as exc:
        status = "cancelled"
        error_msg = str(exc)
        import traceback
        traceback.print_exc()
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)
        total_time = time.perf_counter() - t0
        result["run_info"]["total_time"] = total_time
        result["train_info"]["status"] = status
        result["train_info"]["phase"] = "done"
        result["train_info"]["last_update_at"] = now_iso()
        if error_msg:
            result["train_info"]["error"] = error_msg
        out_path = flush_result()

    return out_path


def main() -> None:
    args = parse_args()
    out_path = run_experiment(args.experiment_path, verbose=args.verbose)
    print(f"Wrote result: {out_path}")


if __name__ == "__main__":
    main()
