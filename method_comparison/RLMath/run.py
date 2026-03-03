# Copyright 2026-present the HuggingFace Inc. team.

"""Main entrypoint for RLMath experiments using TRL GRPOTrainer."""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

import torch
from peft import PeftConfig
from transformers import AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer

from data import load_rl_datasets
from reward import cookbook_style_math_reward, compute_binary_reward, extract_gsm_answer
from utils import (
    FILE_NAME_DEFAULT_TRAIN_PARAMS,
    RESULT_PATH,
    RLTrainConfig,
    build_base_result,
    get_result_dir,
    load_train_config,
    save_result,
)


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


def load_peft_config(exp_path: str) -> PeftConfig | None:
    adapter_path = os.path.join(exp_path, "adapter_config.json")
    if not os.path.exists(adapter_path):
        return None
    return PeftConfig.from_pretrained(exp_path)


def make_reward_func():
    def reward_fn(completions, ground_truth, task, **kwargs):
        rewards = []
        for completion, gt, task_name in zip(completions, ground_truth, task):
            text = completion
            if isinstance(completion, list) and completion and isinstance(completion[0], dict):
                text = completion[0].get("content", "")
            if not isinstance(text, str):
                text = str(text)

            if task_name == "math":
                rewards.append(cookbook_style_math_reward(text, gt))
                continue
            else:
                pred = extract_gsm_answer(text)
            rewards.append(compute_binary_reward(pred, gt))
        return rewards

    return reward_fn


@torch.no_grad()
def evaluate_pass_at_1(
    *,
    model,
    tokenizer: AutoTokenizer,
    dataset,
    max_completion_length: int,
) -> tuple[float, float]:
    correct = 0
    rewards = 0.0

    for row in dataset:
        inputs = tokenizer(row["prompt"], return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_completion_length,
            pad_token_id=tokenizer.pad_token_id,
        )
        completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)

        if row["task"] == "math":
            reward = cookbook_style_math_reward(completion, row["ground_truth"])
            rewards += reward
            correct += int(reward == 1.0)
            continue
        else:
            pred = extract_gsm_answer(completion)
            reward = compute_binary_reward(pred, row["ground_truth"])
            rewards += reward
            correct += int(reward == 1.0)

    denom = max(len(dataset), 1)
    return correct / denom, rewards / denom


def run_experiment(experiment_path: str, verbose: bool = False) -> str:
    experiment_name = validate_experiment_path(experiment_path)
    cfg: RLTrainConfig = load_train_config(experiment_path)
    set_seed(cfg.seed)

    peft_cfg_dict = load_peft_config_dict(experiment_path)
    peft_cfg = load_peft_config(experiment_path)

    result = build_base_result(experiment_name, cfg, peft_cfg_dict)
    branch = result["run_info"]["peft_branch"]

    t0 = time.perf_counter()
    status = "success"

    try:
        train_ds, valid_ds, test_ds = load_rl_datasets(
            dataset_name=cfg.dataset_name,
            dataset_config=cfg.dataset_config,
            train_split=cfg.dataset_train_split,
            test_split=cfg.dataset_test_split,
            train_subset_size=cfg.train_subset_size,
            eval_subset_size=cfg.eval_subset_size,
            seed=cfg.seed,
        )

        output_dir = os.path.join("checkpoints", experiment_name.replace("/", "--"))
        grpo_args = GRPOConfig(
            output_dir=output_dir,
            remove_unused_columns=False,
            max_steps=cfg.max_steps,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type=cfg.lr_scheduler_type,
            warmup_ratio=cfg.warmup_ratio,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
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
            generation_batch_size=cfg.num_generations,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            beta=cfg.beta,
            epsilon=cfg.epsilon,
            loss_type=cfg.loss_type,
            scale_rewards=cfg.scale_rewards,
            importance_sampling_level=cfg.importance_sampling_level,
            use_vllm=cfg.use_vllm,
            report_to=[],
        )
        if cfg.dtype in {"bfloat16", "float16", "float32"}:
            dtype = getattr(torch, cfg.dtype)
        else:
            dtype = "auto"
        grpo_args.model_init_kwargs = {"dtype": dtype}

        trainer = GRPOTrainer(
            model=cfg.model_id,
            reward_funcs=make_reward_func(),
            args=grpo_args,
            train_dataset=train_ds,
            eval_dataset=None,
            peft_config=peft_cfg,
        )

        if verbose:
            print(f"Running GRPO for {experiment_name} on {cfg.dataset_name}...")

        train_output = trainer.train()

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = trainer.model
        test_pass_at_1, test_reward = evaluate_pass_at_1(
            model=model,
            tokenizer=tokenizer,
            dataset=test_ds,
            max_completion_length=cfg.max_completion_length,
        )
        valid_pass_at_1, valid_reward = evaluate_pass_at_1(
            model=model,
            tokenizer=tokenizer,
            dataset=valid_ds,
            max_completion_length=cfg.max_completion_length,
        )

        log_history = trainer.state.log_history
        last_log = next((x for x in reversed(log_history) if "reward" in x), {})

        result["train_info"].update(
            {
                "status": status,
                "train_time": train_output.metrics.get("train_runtime"),
                "file_size": None,
                "num_trainable_params": None,
                "metrics": log_history,
            }
        )
        result["rl_eval_info"] = {
            "valid_pass_at_1": valid_pass_at_1,
            "test_pass_at_1": test_pass_at_1,
            "valid_reward": valid_reward,
            "test_reward": test_reward,
            "reward_mean": last_log.get("reward"),
            "reward_std": last_log.get("reward_std"),
            "kl_mean": last_log.get("kl"),
            "frac_reward_zero_std": last_log.get("frac_reward_zero_std"),
            "num_eval_samples": len(test_ds),
        }

    except KeyboardInterrupt:
        status = "cancelled"
        result["train_info"]["status"] = status
    except Exception as exc:
        status = "failed"
        result["train_info"]["status"] = status
        result["train_info"]["error"] = str(exc)
        raise
    finally:
        total_time = time.perf_counter() - t0
        result["run_info"]["total_time"] = total_time
        if result["train_info"]["status"] == "running":
            result["train_info"]["status"] = status
        out_dir = get_result_dir(result["train_info"]["status"], branch)
        out_path = save_result(result, out_dir, experiment_name)

    return out_path


def main() -> None:
    args = parse_args()
    out_path = run_experiment(args.experiment_path, verbose=args.verbose)
    print(f"Wrote result: {out_path}")


if __name__ == "__main__":
    main()
