# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Reward helpers for RL math tasks (GSM8K/MATH-like)."""

import re
from fractions import Fraction


BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:/\d+)?")


def _to_canonical_number(value: str) -> str:
    value = value.strip().replace(",", "")
    if "/" in value:
        try:
            return str(float(Fraction(value)))
        except Exception:
            return value
    try:
        return str(float(value))
    except Exception:
        return value


def extract_gsm_answer(text: str) -> str:
    """Extract GSM-style final numeric answer from text."""
    if "####" in text:
        text = text.split("####")[-1]
    nums = NUM_RE.findall(text)
    return _to_canonical_number(nums[-1]) if nums else ""


def extract_boxed(text: str) -> str:
    """Extract last \\boxed{} content from a response."""
    boxed = BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()
    match = re.search(r"\\boxed\s+([a-zA-Z0-9./+-]+)", text)
    if match:
        return match.group(1).strip()
    raise ValueError("No boxed answer found")


def extract_math_answer(text: str) -> str:
    """Extract MATH-style final answer, preferring \\boxed{} values."""
    try:
        return _to_canonical_number(extract_boxed(text))
    except ValueError:
        pass
    nums = NUM_RE.findall(text)
    return _to_canonical_number(nums[-1]) if nums else ""


def compute_binary_reward(prediction: str, ground_truth: str) -> float:
    return 1.0 if prediction == ground_truth else 0.0


def cookbook_style_math_reward(response: str, answer: str) -> float:
    """Match the cookbook reward flow: extract boxed response and compare to final answer."""
    try:
        pred = _to_canonical_number(extract_boxed(response))
    except ValueError:
        return 0.0
    gt = _to_canonical_number(answer)
    return compute_binary_reward(pred, gt)
