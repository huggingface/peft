# Copyright 2026-present the HuggingFace Inc. team.

"""Dataset loading for RL math tasks."""

from typing import Any

from datasets import Dataset, concatenate_datasets, get_dataset_config_names, load_dataset

from reward import extract_gsm_answer, extract_math_answer


def _to_prompt(question: str) -> str:
    return (
        "Solve the following problem. Show your reasoning.\n"
        "Provide the final answer in \\boxed{} format.\n\n"
        f"Problem: {question}\n\n"
        "Answer:"
    )


def _prepare_row(example: dict[str, Any], dataset_name: str) -> dict[str, str]:
    if dataset_name in {"openai/gsm8k", "gsm8k"}:
        question = example["question"]
        gt = extract_gsm_answer(example["answer"])
        task = "gsm"
    elif dataset_name in {"math", "EleutherAI/hendrycks_math", "hendrycks/competition_math"}:
        question = example.get("problem", "")
        gt = extract_math_answer(example.get("solution", ""))
        task = "math"
    elif dataset_name in {"deepmath", "zwhe99/DeepMath-103K"}:
        question = example.get("question", "")
        gt = extract_math_answer(str(example.get("final_answer", "")))
        task = "math"
    else:
        # Generic fallback for custom prompt-only datasets.
        question = example.get("prompt", "")
        gt = example.get("ground_truth", "")
        task = "custom"

    return {
        "prompt": _to_prompt(question),
        "ground_truth": gt,
        "task": task,
        "question": question,
    }


def load_rl_datasets(
    *,
    dataset_name: str,
    dataset_config: str | None,
    train_split: str,
    test_split: str,
    train_subset_size: int,
    eval_subset_size: int,
    seed: int,
) -> tuple[Dataset, Dataset, Dataset]:
    if dataset_name == "math":
        # Blog-adjacent setup used in cookbook: train on Hendrycks MATH without MATH-500 overlap, test on MATH-500.
        math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
        test_problems = {x["problem"] for x in math500}
        cfgs = get_dataset_config_names("EleutherAI/hendrycks_math")
        pieces = []
        for cfg in cfgs:
            for split in ("train", "test"):
                split_ds = load_dataset("EleutherAI/hendrycks_math", name=cfg, split=split)
                split_ds = split_ds.filter(lambda ex: ex["problem"] not in test_problems)
                pieces.append(split_ds)
        train = concatenate_datasets(pieces).shuffle(seed=seed)
        test = math500.shuffle(seed=seed)
    elif dataset_name == "deepmath":
        ds = load_dataset("zwhe99/DeepMath-103K", split="train").shuffle(seed=seed)
        train = ds
        test = ds
    else:
        hf_dataset = "openai/gsm8k" if dataset_name == "gsm8k" else dataset_name
        ds = load_dataset(hf_dataset, dataset_config)
        train = ds[train_split].shuffle(seed=seed)
        test = ds[test_split].shuffle(seed=seed)

    if train_subset_size > 0:
        train = train.select(range(min(train_subset_size, len(train))))

    valid_size = min(eval_subset_size, len(test) // 2 if len(test) > 1 else len(test))
    valid = test.select(range(valid_size)) if valid_size > 0 else test
    heldout = test.select(range(valid_size, min(valid_size + eval_subset_size, len(test)))) if len(test) > valid_size else test

    train = train.map(lambda x: _prepare_row(x, dataset_name), remove_columns=train.column_names)
    valid = valid.map(lambda x: _prepare_row(x, dataset_name), remove_columns=valid.column_names)
    heldout = heldout.map(lambda x: _prepare_row(x, dataset_name), remove_columns=heldout.column_names)

    return train, valid, heldout
