# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Reward helpers for RL math tasks, ported from tinker-cookbook math_grading.py."""

import contextlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Callable, TypeVar

import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser

logger = logging.getLogger(__name__)
T = TypeVar("T")

# ======================================================================
# Boxed extraction (stack-based, handles nested braces)
# ======================================================================


def extract_boxed(text: str) -> str:
    """Extract the content of the last \\boxed{...} in text, handling nested braces."""
    boxed_strs = []
    stack = []
    for i, ch in enumerate(text):
        if ch == "{":
            stack.append(i)
        elif ch == "}":
            if not stack:
                raise ValueError("Unmatched }")
            last_open = stack.pop()
            if text[:last_open].endswith("\\boxed"):
                boxed_strs.append(text[last_open + 1 : i])
    if boxed_strs:
        return boxed_strs[-1]
    # Fallback: \boxed without braces, e.g. \boxed 2
    match = re.search(r"\\boxed\s+([a-zA-Z0-9]+)", text)
    if match:
        return match.group(1)
    raise ValueError("No boxed answer found")


# ======================================================================
# Normalization (from tinker-cookbook math_grading.py)
# ======================================================================


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a, b = substr[0], substr[1]
                if b != "{":
                    new_str += "{" + a + "}{" + b + "}" + substr[2:]
                else:
                    new_str += "{" + a + "}" + b + substr[2:]
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a_str, b_str = string.split("/")
    try:
        a, b = int(a_str), int(b_str)
        assert string == f"{a}/{b}"
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except (ValueError, AssertionError):
        return string


def _remove_right_units(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def _strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)
    return string


def normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    answer = answer.strip()
    m = re.search("^\\\\text\\{(?P<text>.+?)\\}$", answer)
    if m is not None:
        answer = m.group("text").strip()
    return _strip_string(str(answer))


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except (ValueError, OverflowError):
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except (ValueError, OverflowError):
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x_str: str) -> bool:
    try:
        x_str = _strip_properly_formatted_commas(x_str)
        x = float(x_str)
        return abs(x - int(round(x))) <= 1e-7
    except (ValueError, OverflowError):
        return False


def _str_to_int(x_str: str) -> int:
    return int(float(x_str.replace(",", "")))


def _inject_implicit_mixed_number(step: str) -> str:
    return re.compile("([0-9]) +([0-9])").sub("\\1+\\2", step)


def _strip_properly_formatted_commas(expr: str) -> str:
    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str | None:
    if expr is None:
        return None
    m = re.search("^\\\\text\\{(?P<text>.+?)\\}$", expr)
    if m is not None:
        expr = m.group("text")
    expr = expr.replace("\\%", "%").replace("\\$", "$").replace("$", "").replace("%", "")
    expr = expr.replace(" or ", " , ").replace(" and ", " , ")
    expr = expr.replace("million", "*10^6").replace("billion", "*10^9").replace("trillion", "*10^12")
    for unit in [
        "degree", "cm", "centimeter", "meter", "mile", "second", "minute",
        "hour", "day", "week", "month", "year", "foot", "feet", "inch", "yard",
    ]:
        expr = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub("\\^ *\\\\circ", "", expr)
    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]
    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        with contextlib.suppress(Exception):
            expr = _parse_latex(expr)
    expr = re.sub("- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "").replace("{", "").replace("}", "")
    expr = expr.lower()
    if _str_is_int(expr):
        expr = str(_str_to_int(expr))
    return expr


# ======================================================================
# Sympy grading (from tinker-cookbook)
# ======================================================================

BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = [r"\^[0-9]+\^", r"\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=sympy_parser.standard_transformations + (sympy_parser.implicit_multiplication_application,),
    )


def _parse_latex(expr: str) -> str:
    expr = expr.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)
    expr = expr.replace("\u221a", "sqrt").replace("\u03c0", "pi").replace("\u221e", "inf")
    expr = expr.replace("\u222a", "U").replace("\u00b7", "*").replace("\u00d7", "*")
    return expr.strip()


def should_allow_eval(expr: str) -> bool:
    letters = set(x for x in expr.replace("sqrt", "").replace("frac", "") if x.isalpha())
    if len(letters) > 2:
        return False
    for bad in BAD_SUBSTRINGS:
        if bad in expr:
            return False
    return all(re.search(bad, expr) is not None for bad in BAD_REGEXES)


def are_equal_under_sympy(gt_norm: str, given_norm: str) -> bool:
    try:
        expr = f"({gt_norm})-({given_norm})"
        if should_allow_eval(expr):
            return sympy.simplify(_sympy_parse(expr)) == 0
    except Exception:
        pass
    return False


def split_tuple(expr: str) -> list[str]:
    expr = _strip_properly_formatted_commas(expr)
    if not expr:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all(ch not in expr[1:-1] for ch in TUPLE_CHARS)
    ):
        return [e.strip() for e in expr[1:-1].split(",")]
    return [expr]


def grade_answer(given_answer: str, ground_truth: str) -> bool:
    """Robust grading: normalization + sympy simplification (from tinker-cookbook)."""
    if given_answer is None:
        return False
    gt_mathd = normalize_answer(ground_truth)
    given_mathd = normalize_answer(given_answer)
    if gt_mathd == given_mathd:
        return True
    gt_norm = _normalize(ground_truth)
    given_norm = _normalize(given_answer)
    if gt_norm is None:
        return False
    if gt_norm == given_norm:
        return True
    if not given_norm:
        return False
    gt_elems = split_tuple(gt_norm)
    given_elems = split_tuple(given_norm)
    if len(gt_elems) > 1 and (gt_norm[0] != given_norm[0] or gt_norm[-1] != given_norm[-1]):
        return False
    if len(gt_elems) != len(given_elems):
        return False
    for gt_e, given_e in zip(gt_elems, given_elems, strict=True):
        if _is_frac(gt_e) and _is_frac(given_e):
            if gt_e != given_e:
                return False
        elif _str_is_int(gt_e) != _str_is_int(given_e):
            return False
        elif not are_equal_under_sympy(gt_e, given_e):
            return False
    return True


def run_with_timeout(func: Callable[..., T], args: tuple = (), timeout_seconds: int = 5) -> T | None:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args)
        try:
            return future.result(timeout=timeout_seconds)
        except (FuturesTimeoutError, Exception) as e:
            logger.warning(f"Grading timed out or failed: {e}")
            return None


def safe_grade(given_answer: str, ground_truth: str, timeout: int = 5) -> bool:
    result = run_with_timeout(grade_answer, args=(given_answer, ground_truth), timeout_seconds=timeout)
    return bool(result)


# ======================================================================
# GSM8K-specific helpers
# ======================================================================


def extract_gsm_answer(text: str) -> str:
    """Extract GSM-style final answer from the #### line."""
    lines = text.splitlines()
    for line in reversed(lines):
        s = line.strip()
        if s.startswith("####"):
            content = s[4:].strip()
            if content.startswith(":"):
                content = content[1:].strip()
            return content.replace(",", "").strip()
    # Fallback: regex
    matches = re.findall(r"####\s*(.+)", text)
    if matches:
        return matches[-1].strip().replace(",", "")
    return ""


# ======================================================================
# Reward functions for TRL GRPOTrainer
# ======================================================================


def math_reward_fn(completions: list[str], ground_truth: list[str], **kwargs) -> list[float]:
    """Reward function for MATH-style tasks: extract \\boxed{} and grade with sympy."""
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        if isinstance(completion, list) and completion and isinstance(completion[0], dict):
            completion = completion[0].get("content", "")
        if not isinstance(completion, str):
            completion = str(completion)
        try:
            pred = extract_boxed(completion)
        except ValueError:
            rewards.append(0.0)
            continue
        rewards.append(1.0 if safe_grade(pred, gt) else 0.0)
    return rewards
