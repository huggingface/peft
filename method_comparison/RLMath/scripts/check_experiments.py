#!/usr/bin/env python3
"""Check status of RLMath experiments by scanning result JSONs.

Use after running experiments (e.g. via srun) to see status, errors, and metrics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TEMP_RESULTS = ROOT / "temporary_results"
CANCELLED = ROOT / "cancelled_results"
EXPERIMENTS_DIR = ROOT / "experiments"


def _result_dirs() -> list[Path]:
    return [RESULTS, TEMP_RESULTS, CANCELLED]


def _experiment_name_to_file(name: str) -> str:
    """e.g. lora/grpo-qwen3-4b-dapo-math-100 -> lora--grpo-qwen3-4b-dapo-math-100.json"""
    return name.replace("/", "--") + ".json"


def find_experiment_dirs() -> list[str]:
    """Return experiment names like 'lora/grpo-qwen3-4b-dapo-math-100'."""
    names = []
    if not EXPERIMENTS_DIR.exists():
        return names
    for method_dir in EXPERIMENTS_DIR.iterdir():
        if not method_dir.is_dir():
            continue
        for exp_dir in method_dir.iterdir():
            if exp_dir.is_dir() and (exp_dir / "adapter_config.json").exists():
                names.append(f"{method_dir.name}/{exp_dir.name}")
    return sorted(names)


def load_result(experiment_name: str) -> tuple[dict | None, str | None]:
    """Load result JSON for an experiment. Returns (result_dict, source_dir_name or None)."""
    fname = _experiment_name_to_file(experiment_name)
    for d in _result_dirs():
        p = d / fname
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f), d.name
            except Exception:
                return None, d.name
    return None, None


def check_experiments(verbose: bool = False) -> int:
    experiments = find_experiment_dirs()
    if not experiments:
        print("No experiment dirs found under experiments/<method>/<name>/", file=sys.stderr)
        return 1

    rows = []
    for name in experiments:
        result, source = load_result(name)
        if result is None:
            rows.append(
                {
                    "experiment": name,
                    "status": "no_result",
                    "source": source or "-",
                    "last_update": "-",
                    "error": "-",
                    "test_pass_at_1": None,
                    "reward_mean": None,
                }
            )
            continue
        train_info = result.get("train_info") or {}
        rl_eval = result.get("rl_eval_info") or {}
        status = train_info.get("status", "?")
        last = train_info.get("last_update_at", "-")
        err = train_info.get("error", "-") or "-"
        rows.append(
            {
                "experiment": name,
                "status": status,
                "source": source or "-",
                "last_update": last,
                "error": err[:60] + "..." if len(err) > 60 else err,
                "test_pass_at_1": rl_eval.get("test_pass_at_1"),
                "reward_mean": rl_eval.get("reward_mean") or (train_info.get("metrics") or [{}])[-1].get("reward"),
            }
        )

    # Print table
    print(f"{'Experiment':<45} {'Status':<12} {'Source':<18} {'test_pass@1':<12} {'reward_mean':<12}")
    print("-" * 100)
    for r in rows:
        pa1 = f"{r['test_pass_at_1']:.4f}" if r["test_pass_at_1"] is not None else "-"
        rm = f"{r['reward_mean']:.4f}" if r["reward_mean"] is not None else "-"
        print(f"{r['experiment']:<45} {r['status']:<12} {r['source']:<18} {pa1:<12} {rm:<12}")
    if verbose:
        print("\nLast update / error details:")
        for r in rows:
            if r["last_update"] != "-" or r["error"] != "-":
                print(f"  {r['experiment']}: last={r['last_update']} error={r['error']}")
    return 0


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Check RLMath experiment status from result JSONs.")
    p.add_argument("-v", "--verbose", action="store_true", help="Print last_update and error details.")
    args = p.parse_args()
    return check_experiments(verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
