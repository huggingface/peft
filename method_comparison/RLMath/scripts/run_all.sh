#!/bin/bash
# Run all RLMath experiments sequentially via srun (single GPU).
# Usage: bash scripts/run_all.sh [--dry-run] [--filter PATTERN]
#
# Options:
#   --dry-run      Print commands without executing
#   --filter PAT   Only run experiments matching PAT (e.g. "lora" or "adalora/grpo")
#   --skip-done    Skip experiments that already have a successful result

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="/mnt/data/kashif/miniconda3/envs/py312/bin/python"
PARTITION="hpc-mid"
MEM="128G"
TIME="02:00:00"
GPUS=1

DRY_RUN=false
FILTER=""
SKIP_DONE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --filter) FILTER="$2"; shift 2 ;;
        --skip-done) SKIP_DONE=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Collect all experiment directories
EXPERIMENTS=()
for exp_dir in experiments/*/*/; do
    exp_dir="${exp_dir%/}"
    # Must have training_params.json
    [[ -f "$exp_dir/training_params.json" ]] || continue
    # Apply filter
    [[ -n "$FILTER" ]] && [[ "$exp_dir" != *"$FILTER"* ]] && continue
    EXPERIMENTS+=("$exp_dir")
done

echo "Found ${#EXPERIMENTS[@]} experiments"
echo "========================================"

PASSED=0
FAILED=0
SKIPPED=0

for exp_dir in "${EXPERIMENTS[@]}"; do
    # Extract method/name for result file lookup
    method=$(basename "$(dirname "$exp_dir")")
    name=$(basename "$exp_dir")
    result_file="temporary_results/${method}--${name}.json"

    # Check if already done
    if $SKIP_DONE && [[ -f "$result_file" ]]; then
        status=$($PYTHON -c "import json; print(json.load(open('$result_file'))['train_info']['status'])" 2>/dev/null || echo "unknown")
        if [[ "$status" == "success" ]]; then
            echo "[SKIP] $exp_dir (already succeeded)"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
    fi

    echo ""
    echo "[RUN] $exp_dir"
    echo "--------------------------------------"

    CMD="srun --partition=$PARTITION --gpus=$GPUS --mem=$MEM --time=$TIME $PYTHON run.py $exp_dir -v"

    if $DRY_RUN; then
        echo "  DRY RUN: $CMD"
        continue
    fi

    if $CMD 2>&1 | tee "/tmp/rlmath-${method}--${name}.log"; then
        echo "[PASS] $exp_dir"
        PASSED=$((PASSED + 1))
    else
        echo "[FAIL] $exp_dir (exit code $?)"
        FAILED=$((FAILED + 1))
        # Continue to next experiment
    fi
done

echo ""
echo "========================================"
echo "Results: $PASSED passed, $FAILED failed, $SKIPPED skipped"
