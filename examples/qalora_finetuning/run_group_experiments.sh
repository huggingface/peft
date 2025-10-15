#!/bin/bash

# ============================================================================
# Group Experiment - Training & Evaluation Pipeline
# ============================================================================
# This script trains models with different training modes and ranks,
# then evaluates them using the ultimate_train_collection.py script.

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_NAMES=(
    "HuggingFaceTB/SmolLM2-1.7B"
    "TinyLlama/TinyLlama_v1.1"
    # "meta-llama/Llama-3.2-1B"
    # "microsoft/phi-2"
)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$REPO_ROOT/train_results_group_exp}"

# --- Iteration Parameters ---
TRAINING_MODES=("qalora" "pissa_rank_analysis" "qalora_svd_error_two_adapter") 
LORA_RANKS=(8 16 32 64)
BITS_LIST=(2 3)
CALIBRATION_DATASETS=("c4" "alpaca-cleaned")
QALORA_GROUP_SIZES=(32 128)

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
SAVE_STEPS=5000
BF16="True"

# --- Mode-Specific Parameters ---
PISSA_NITER=4

# ============================================================================
# Helper Functions
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

export WANDB_PROJECT="qalora-finetuning-thesis"

# ============================================================================
# Main Execution
# ============================================================================
main() {
    log_info "Starting Group Experiment Pipeline"
    mkdir -p "$BASE_OUTPUT_DIR"
    cd "$SCRIPT_DIR" || exit 1

    for MODEL_NAME_OR_PATH in "${MODEL_NAMES[@]}"; do
        MODEL_SHORT_NAME="${MODEL_NAME_OR_PATH##*/}"
        log_info "========================================="
        log_info "Processing Model: ${MODEL_SHORT_NAME}"
        log_info "========================================="

        for mode in "${TRAINING_MODES[@]}"; do
            for rank in "${LORA_RANKS[@]}"; do
                for bits in "${BITS_LIST[@]}"; do
                    for dataset in "${CALIBRATION_DATASETS[@]}"; do
                        
                        if [ "$mode" = "qalora" ]; then
                            GROUP_SIZES_TO_RUN=("${QALORA_GROUP_SIZES[@]}")
                        else
                            GROUP_SIZES_TO_RUN=(128)
                        fi

                        for group_size in "${GROUP_SIZES_TO_RUN[@]}"; do
                            log_info "===== Running: MODEL=${MODEL_SHORT_NAME} MODE=${mode} RANK=${rank} BITS=${bits} DATASET=${dataset} GROUP=${group_size} ====="

                        EXPERIMENT_NAME="${MODEL_SHORT_NAME}_${mode}_r${rank}_b${bits}_d${dataset}"
                        if [ "$mode" = "qalora" ]; then
                            EXPERIMENT_NAME="${EXPERIMENT_NAME}_group${group_size}"
                        fi

                        TRAIN_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${EXPERIMENT_NAME}"
                        ADAPTER_DIR="${TRAIN_OUTPUT_DIR}/adapter"

                        if [ -d "$ADAPTER_DIR" ]; then
                            log_warning "Adapter exists at $ADAPTER_DIR. Skipping training."
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
                                --qalora_group_size="$group_size" \
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
                                --report_to="wandb"

                            if [ $? -ne 0 ]; then
                                log_error "Training failed for $EXPERIMENT_NAME. Skipping."
                                continue
                            fi
                            log_success "Training & Evaluation completed for $EXPERIMENT_NAME"
                        fi
                        sleep 5
                        done
                    done
                done
            done
        done
    done

    log_success "🎉 Group Experiment Pipeline Finished! 🎉"
}

main "$@"
