#!/bin/bash
# Submit all RLMath experiments as individual sbatch jobs.
# Usage: bash scripts/submit_all.sh [--dry-run] [--filter PATTERN] [--skip-done]

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="/mnt/data/kashif/miniconda3/envs/py312/bin/python"
PARTITION="hpc-mid"
MEM="128G"
TIME="02:00:00"
GPUS=1
WORKDIR="$(pwd)"

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

SUBMITTED=0
SKIPPED=0

for exp_dir in experiments/*/*/; do
    exp_dir="${exp_dir%/}"
    [[ -f "$exp_dir/training_params.json" ]] || continue
    [[ -n "$FILTER" ]] && [[ "$exp_dir" != *"$FILTER"* ]] && continue

    method=$(basename "$(dirname "$exp_dir")")
    name=$(basename "$exp_dir")
    job_name="rl-${method}--${name}"
    log_dir="${WORKDIR}/logs"
    mkdir -p "$log_dir"
    log_file="${log_dir}/${method}--${name}-%j.out"

    # Check if already done
    if $SKIP_DONE; then
        for rdir in temporary_results results; do
            result_file="${rdir}/${method}--${name}.json"
            if [[ -f "$result_file" ]]; then
                status=$($PYTHON -c "import json; print(json.load(open('$result_file'))['train_info']['status'])" 2>/dev/null || echo "unknown")
                if [[ "$status" == "success" ]]; then
                    echo "[SKIP] $exp_dir (already succeeded)"
                    SKIPPED=$((SKIPPED + 1))
                    continue 2
                fi
            fi
        done
    fi

    if $DRY_RUN; then
        echo "[DRY] sbatch $exp_dir"
        SUBMITTED=$((SUBMITTED + 1))
        continue
    fi

    job_id=$(sbatch \
        --partition="$PARTITION" \
        --gpus="$GPUS" \
        --mem="$MEM" \
        --time="$TIME" \
        --job-name="$job_name" \
        --output="$log_file" \
        --chdir="$WORKDIR" \
        --wrap="$PYTHON run.py $exp_dir -v" \
        2>&1 | grep -oP '\d+')

    echo "[SUBMITTED] $exp_dir -> job $job_id"
    SUBMITTED=$((SUBMITTED + 1))
done

echo ""
echo "Submitted: $SUBMITTED, Skipped: $SKIPPED"
