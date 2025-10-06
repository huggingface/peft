#!/bin/bash

# ============================================================================
# Runnerscript_Group Experiment - Training & Evaluation Pipeline
# ============================================================================
# This script trains models with different training modes and ranks,
# then evaluates them using the eval_peft.py script.

set -e # Exit on any error

# ============================================================================
# CONFIGURATION - Edit these variables for different experiments
# ============================================================================
MODEL_NAME_OR_PATH="HuggingFaceTB/SmolLM2-1.7B"
# If not set externally, derive short name from model id
MODEL_SHORT_NAME="${MODEL_SHORT_NAME:-${MODEL_NAME_OR_PATH##*/}}" # Used for directory naming
# Resolve script directory and repo root dynamically
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# Allow overriding BASE_OUTPUT_DIR via environment; default near repo root
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$REPO_ROOT/train_results_group_exp}"

# --- Iteration Parameters ---
TRAINING_MODES=("qalora" "pissa_rank_analysis") 
LORA_RANKS=(1 2 4)

# --- Training Configuration ---
DATA_PATH="yahma/alpaca-cleaned"
DATASET_SPLIT="train[:10000]"
NUM_TRAIN_EPOCHS=2
PER_DEVICE_TRAIN_BATCH_SIZE=4
LEARNING_RATE=1e-4
MAX_LENGTH=2048
WARMUP_RATIO=0.03
LR_SCHEDULER_TYPE="cosine"
LOGGING_STEPS=10
SAVE_STEPS=5000 # Set high to save only at the end
BF16="True"

# --- Mode-Specific Parameters ---
QALORA_GROUP_SIZE=32
PISSA_NITER=4
# Sweep over multiple bit-widths and calibration datasets
# Example defaults; edit as needed
BITS_LIST=(2 3)
CALIBRATION_DATASETS=("c4" "alpaca-cleaned")
# BASE_OUTPUT_DIR will be kept as root; per-run directories will include bit and dataset tags

# --- System Configuration ---
# (no per-run CUDA_DEVICE here; rely on device_map in training script)

# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Function to fix adapter config by setting init_lora_weights to false
# This is often necessary before merging or evaluating PEFT adapters.
fix_adapter_config() {
    local adapter_path="$1"
    local config_file="$adapter_path/adapter_config.json"

    if [[ ! -f "$config_file" ]]; then
        log_error "Adapter config file not found: $config_file"
        return 1
    fi

    log_info "Fixing adapter config: $config_file"
    # Create a backup just in case
    cp "$config_file" "$config_file.backup"

    # Use jq if available (more robust), otherwise fallback to sed
    if command -v jq &> /dev/null; then
        jq '.init_lora_weights = false' "$config_file" > "$config_file.tmp" && mv "$config_file.tmp" "$config_file"
        log_success "Set 'init_lora_weights: false' in $config_file"
    else
        log_warning "jq not found. Using sed as a fallback to fix adapter config."
        sed -i 's/"init_lora_weights": true/"init_lora_weights": false/g' "$config_file"
        sed -i 's/"init_lora_weights": "daniel"/"init_lora_weights": false/g' "$config_file"
        log_success "Attempted to fix 'init_lora_weights' using sed."
    fi
}
export WANDB_PROJECT="qalora-finetuning"
# Main execution
main() {
    log_info "Starting Group Experiment Pipeline"
    mkdir -p "$BASE_OUTPUT_DIR"
    cd "$SCRIPT_DIR" || exit 1

    total_experiments=$((${#TRAINING_MODES[@]} * ${#LORA_RANKS[@]} * ${#BITS_LIST[@]} * ${#CALIBRATION_DATASETS[@]}))
    current_experiment=0

    for mode in "${TRAINING_MODES[@]}"; do
        for rank in "${LORA_RANKS[@]}"; do
            for bits in "${BITS_LIST[@]}"; do
                for dataset in "${CALIBRATION_DATASETS[@]}"; do
                    current_experiment=$((current_experiment + 1))
                    log_info "===== Running Experiment ${current_experiment}/${total_experiments}: MODE=${mode} RANK=${rank} BITS=${bits} DATASET=${dataset} ====="

                    # --- 1. Set up experiment name and directories ---
                    EXPERIMENT_NAME="${MODEL_SHORT_NAME}_${mode}_r${rank}_b${bits}_d${dataset}"
                    if [ "$mode" = "qalora" ]; then
                        EXPERIMENT_NAME="${EXPERIMENT_NAME}_group${QALORA_GROUP_SIZE}"
                    elif [ "$mode" = "pissa" ]; then
                        EXPERIMENT_NAME="${EXPERIMENT_NAME}_niter${PISSA_NITER}"
                    fi

                    TRAIN_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${EXPERIMENT_NAME}"
                    ADAPTER_DIR="${TRAIN_OUTPUT_DIR}/ft/adapter"
                    EVAL_OUTPUT_DIR="${BASE_OUTPUT_DIR}/eval_results/${EXPERIMENT_NAME}"

                    # --- 2. Training Phase ---
                    if [ -d "$ADAPTER_DIR" ]; then
                        log_warning "Adapter already exists at $ADAPTER_DIR. Skipping training."
                    else
                        log_info "Starting training for $EXPERIMENT_NAME"
                        python ultimate_train_collection.py \
                            --model_name_or_path="$MODEL_NAME_OR_PATH" \
                            --training_mode="$mode" \
                            --output_dir="$TRAIN_OUTPUT_DIR" \
                            --data_path="$DATA_PATH" \
                            --dataset_split="$DATASET_SPLIT" \
                            --dataset_field "instruction" "output" \
                            --lora_r="$rank" \
                            --qalora_group_size="$QALORA_GROUP_SIZE" \
                            --pissa_niter="$PISSA_NITER" \
                            --bits="$bits" \
                            --calibration_dataset="$dataset" \
                            --num_train_epochs="$NUM_TRAIN_EPOCHS" \
                            --per_device_train_batch_size="$PER_DEVICE_TRAIN_BATCH_SIZE" \
                            --learning_rate="$LEARNING_RATE" \
                            --lr_scheduler_type="$LR_SCHEDULER_TYPE" \
                            --warmup_ratio="$WARMUP_RATIO" \
                            --bf16="$BF16" \
                            --logging_steps="$LOGGING_STEPS" \
                            --save_steps="$SAVE_STEPS" \
                            --model_max_length="$MAX_LENGTH" \
                            --dataloader_pin_memory=False \
                            --remove_unused_columns=False \
                            --report_to="wandb" \
                            --gradient_accumulation_steps=1 \
                            --eval_steps=50 \

                        if [ $? -ne 0 ]; then
                            log_error "❌ Training failed for $EXPERIMENT_NAME. Skipping to next experiment."
                            continue
                        fi
                        log_success "✅ Training completed for $EXPERIMENT_NAME"
                    fi

                    # --- 3. Fix Adapter Config ---
                    if ! fix_adapter_config "$ADAPTER_DIR"; then
                        log_error "❌ Failed to fix adapter config for $EXPERIMENT_NAME. Skipping evaluation."
                        continue
                    fi
                    # Small delay between runs
                    sleep 5
                done
            done
        done
    done

    log_success "🎉🎉🎉 Group Experiment Pipeline Finished! 🎉🎉🎉"
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Check for required tools
    if ! command -v python &> /dev/null; then
        log_error "Python not found in PATH"
        exit 1
    fi
    main "$@"
fi