# Copyright 2026-present the HuggingFace Inc. team.

"""Dataset loading for RL math tasks, matching tinker-cookbook setup."""

from typing import Any, cast

from datasets import Dataset, concatenate_datasets, get_dataset_config_names, load_dataset
from reward import extract_boxed, extract_gsm_answer


# ---------------------------------------------------------------------------
# Prompt templates (matching tinker-cookbook)
# ---------------------------------------------------------------------------

MATH_SUFFIX = " Write your answer in \\boxed{} format."
GSM8K_SUFFIX = " Provide a numerical answer without units, written inside \\boxed{}."


def _math_prompt(question: str) -> str:
    return question + MATH_SUFFIX


def _gsm8k_prompt(question: str) -> str:
    return question + GSM8K_SUFFIX


# ---------------------------------------------------------------------------
# Dataset builders (matching tinker-cookbook structure)
# ---------------------------------------------------------------------------


def _get_hendrycks_math_test() -> Dataset:
    return cast(Dataset, load_dataset("HuggingFaceH4/MATH-500", name="default", split="test"))


def _get_hendrycks_math_train(seed: int) -> Dataset:
    """Hendrycks MATH train+test minus MATH-500 overlap (same as tinker-cookbook)."""
    test_problems: set[str] = {p["problem"] for p in _get_hendrycks_math_test()}
    configs = get_dataset_config_names("EleutherAI/hendrycks_math")
    pieces = []
    for cfg in configs:
        for split in ("train", "test"):
            ds = load_dataset("EleutherAI/hendrycks_math", name=cfg, split=split)
            ds = ds.filter(lambda ex: ex["problem"] not in test_problems)
            pieces.append(ds)
    return concatenate_datasets(pieces).shuffle(seed=seed)


def _prepare_math_row(example: dict[str, Any]) -> dict[str, str]:
    question = example["problem"]
    try:
        gt = extract_boxed(example["solution"])
    except ValueError:
        gt = ""
    return {"prompt": _math_prompt(question), "ground_truth": gt}


def _prepare_gsm8k_row(example: dict[str, Any]) -> dict[str, str]:
    question = example["question"]
    gt = extract_gsm_answer(example["answer"])
    return {"prompt": _gsm8k_prompt(question), "ground_truth": gt}


def _prepare_deepmath_row(example: dict[str, Any]) -> dict[str, str]:
    question = example.get("question", "")
    gt = str(example.get("final_answer", ""))
    return {"prompt": _math_prompt(question), "ground_truth": gt}


def load_rl_datasets(
    *,
    dataset_name: str,
    dataset_config: str | None,
    train_split: str,
    test_split: str,
    train_subset_size: int,
    eval_subset_size: int,
    seed: int,
) -> tuple[Dataset, Dataset]:
    if dataset_name == "math":
        train = _get_hendrycks_math_train(seed)
        test = _get_hendrycks_math_test()
        prep_fn = _prepare_math_row
    elif dataset_name in ("deepmath", "zwhe99/DeepMath-103K"):
        ds = load_dataset("zwhe99/DeepMath-103K", split="train").shuffle(seed=seed)
        n = len(ds)
        test_size = min(eval_subset_size * 2, n // 10)
        train = ds.select(range(test_size, n))
        test = ds.select(range(test_size))
        prep_fn = _prepare_deepmath_row
    elif dataset_name in ("gsm8k", "openai/gsm8k"):
        ds = load_dataset("openai/gsm8k", "main")
        train = ds[train_split].shuffle(seed=seed)
        test = ds[test_split].shuffle(seed=seed)
        prep_fn = _prepare_gsm8k_row
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if train_subset_size > 0:
        train = train.select(range(min(train_subset_size, len(train))))

    eval_size = min(eval_subset_size, len(test))
    test = test.select(range(eval_size))

    train = train.map(prep_fn, remove_columns=train.column_names)
    test = test.map(prep_fn, remove_columns=test.column_names)

    # Filter out rows with empty ground truth
    train = train.filter(lambda x: len(x["ground_truth"]) > 0)
    test = test.filter(lambda x: len(x["ground_truth"]) > 0)

    return train, test
