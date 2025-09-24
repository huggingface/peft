#!/bin/bash

# ============================================================================
# PiSSA Residual Quantization - Training & Evaluation Pipeline
# ============================================================================
# This script trains PiSSA models with different ranks and evaluates them
# across multiple quantization configurations.

set -e  # Exit on any error

# ============================================================================
# CONFIGURATION - Edit these variables for different models
# ============================================================================
MODEL_NAME_OR_PATH="Qwen/Qwen3-4B"
MODEL_SHORT_NAME="Qwen3-4B"  # Used for directory naming
SCRIPT_DIR="/home/nudel/Documents/peft/examples/qalora_finetuning"
BASE_OUTPUT_DIR="/home/nudel/Documents/peft/train_results_debugger"
LORA_RANKS=(32 64 128 256 512)
TRAINING_MODES=("qalora" "qlora") # Modes to iterate over

CUDA_DEVICE="0"

# Training Configuration
DATA_PATH="yahma/alpaca-cleaned"
DATASET_SPLIT="train[:10000]"
NUM_TRAIN_EPOCHS=2
PER_DEVICE_TRAIN_BATCH_SIZE=4
LEARNING_RATE=1e-4

# Evaluation Configuration
EVAL_TASKS="hellaswag,piqa,winogrande,arc_easy,arc_challenge,boolq,openbookqa,wikitext"
NUM_FEWSHOT=5
EVAL_LIMIT=100

# ============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Generate safe model name for file paths
get_model_name_safe() {
    echo "$MODEL_NAME_OR_PATH" | sed 's/\//_/g' | sed 's/-/_/g'
}

# Function to fix adapter config
fix_adapter_config() {
    local adapter_path="$1"
    local config_file="$adapter_path/adapter_config.json"
    
    if [[ -f "$config_file" ]]; then
        log_info "Fixing adapter config: $config_file"
        
        # Create backup
        cp "$config_file" "$config_file.backup"
        
        # Use jq to set init_lora_weights to false
        if command -v jq &> /dev/null; then
            jq '.init_lora_weights = false' "$config_file" > "$config_file.tmp" && mv "$config_file.tmp" "$config_file"
            log_success "Fixed init_lora_weights in $config_file"
        else
            # Fallback: use sed
            log_warning "jq not found, using sed fallback"
            sed -i 's/"init_lora_weights": "daniel"/"init_lora_weights": false/g' "$config_file"
            sed -i 's/"init_lora_weights": true/"init_lora_weights": false/g' "$config_file"
            log_success "Fixed init_lora_weights using sed"
        fi
    else
        log_error "Adapter config file not found: $config_file"
        return 1
    fi
}

# Robust model-specific directory finder - using global variable for return
FOUND_RESIDUAL_DIR=""

find_actual_residual_dir() {
    local base_dir="$1"
    local rank="$2"
    local training_mode="$3"
    local model_name_safe=$(get_model_name_safe)
    
    FOUND_RESIDUAL_DIR=""  # Reset global variable
    
    log_info "Searching for rank $rank, mode $training_mode models for $model_name_safe..."
    
    # Possible directory structures
    local possible_dirs=(
        "$base_dir/${training_mode}_quantized_residuals_r${rank}"
        "$base_dir/${training_mode}_quantized_residuals_r${rank}/${training_mode}_quantized_residuals_r${rank}"
    )
    
    for dir in "${possible_dirs[@]}"; do
        if [[ -d "$dir" ]]; then
            log_info "Checking directory: $dir"
            
            # Check for model-specific files
            local has_adapter=false
            local has_residual=false
            
            # Look for adapter with current model name
            if find "$dir" -name "daniel_adapter_r${rank}_*${model_name_safe}*" -type d 2>/dev/null | head -1 | grep -q .; then
                has_adapter=true
                log_info "Found adapter for $model_name_safe"
            fi
            
            # Look for residual weights with current model name
            if find "$dir" -name "w_res_*${model_name_safe}*_r${rank}_*" -type d 2>/dev/null | head -1 | grep -q .; then
                has_residual=true
                log_info "Found residual weights for $model_name_safe"
            fi
            
            # Must have both adapter and residual for this specific model
            if [[ "$has_adapter" == true && "$has_residual" == true ]]; then
                log_success "Valid model directory found: $dir"
                FOUND_RESIDUAL_DIR="$dir"  # Set global variable
                return 0
            else
                log_warning "Directory $dir exists but missing files for $model_name_safe"
                log_info "  Has adapter: $has_adapter"
                log_info "  Has residual: $has_residual"
            fi
        fi
    done
    
    log_error "Could not find valid residual models directory for rank $rank, mode $training_mode and model $model_name_safe"
    log_error "Searched in: ${possible_dirs[*]}"
    return 1
}

# Function to list all files in a directory for debugging
debug_list_directory() {
    local dir="$1"
    local title="$2"
    
    log_info "$title:"
    if [[ -d "$dir" ]]; then
        find "$dir" -type f -name "*" | head -20 | while read -r file; do
            echo "  $(basename "$file")"
        done
        local total_files=$(find "$dir" -type f | wc -l)
        if [[ $total_files -gt 20 ]]; then
            echo "  ... and $((total_files - 20)) more files"
        fi
    else
        echo "  Directory does not exist"
    fi
}

# Function to train model with specific rank
train_model() {
    local rank=$1
    local training_mode=$2
    local output_dir="$BASE_OUTPUT_DIR"
    
    log_info "Starting training for rank $rank with mode $training_mode..."
    log_info "Model: $MODEL_NAME_OR_PATH"
    log_info "Output directory: $output_dir"
    
    cd /home/nudel/Documents/peft
    
    export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
    export PYTHONPATH="/home/nudel/Documents/peft:${PYTHONPATH}"
    
    python "$SCRIPT_DIR/ultimate_train_collection.py" \
        --model_name_or_path="$MODEL_NAME_OR_PATH" \
        --output_dir="$output_dir" \
        --data_path="$DATA_PATH" \
        --dataset_split="$DATASET_SPLIT" \
        --dataset_field "instruction" "output" \
        --lora_r=$rank \
        --num_train_epochs=$NUM_TRAIN_EPOCHS \
        --per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE \
        --gradient_accumulation_steps=1 \
        --learning_rate=$LEARNING_RATE \
        --lr_scheduler_type="cosine" \
        --warmup_ratio=0.03 \
        --bf16=True \
        --logging_steps=10 \
        --save_steps=5000 \
        --eval_steps=500 \
        --training_mode="$training_mode" \
        --dataloader_pin_memory=False \
        --remove_unused_columns=False \
        --report_to="none"
    
    if [[ $? -eq 0 ]]; then
        log_success "Training completed for rank $rank with mode $training_mode"
        
        # Verify what was created
        local created_dir="$BASE_OUTPUT_DIR/${training_mode}_quantized_residuals_r${rank}"
        log_info "Expected training output in: $created_dir"
        
        if [[ -d "$created_dir" ]]; then
            debug_list_directory "$created_dir" "Contents of training output directory"
            
            # Find and fix adapter config
            local model_name_safe=$(get_model_name_safe)
            local found_adapter=$(find "$created_dir" -name "daniel_adapter_r${rank}_*${model_name_safe}*" -type d 2>/dev/null | head -1)
            
            if [[ -n "$found_adapter" ]]; then
                log_info "Found adapter at: $found_adapter"
                fix_adapter_config "$found_adapter"
            else
                log_warning "No adapter found for rank $rank, mode $training_mode and model $model_name_safe"
                log_info "Available adapters:"
                find "$created_dir" -name "daniel_adapter_*" -type d 2>/dev/null || echo "  None found"
            fi
        else
            log_error "Training output directory not found: $created_dir"
            return 1
        fi
    else
        log_error "Training failed for rank $rank with mode $training_mode"
        return 1
    fi
}

# Function to evaluate model with specific rank
evaluate_model() {
    local rank=$1
    local training_mode=$2
    local eval_output_dir="./residual_evaluation_results_${MODEL_SHORT_NAME}_${training_mode}_r${rank}"
    
    log_info "Starting evaluation for rank $rank from mode $training_mode..."
    
    # Find the actual residual models directory
    if ! find_actual_residual_dir "$BASE_OUTPUT_DIR" "$rank" "$training_mode"; then
        log_error "Could not find residual models directory for rank $rank and mode $training_mode"
        return 1
    fi
    
    local residual_dir="$FOUND_RESIDUAL_DIR"
    log_info "Found residual models dir: $residual_dir"
    log_info "Evaluation output dir: $eval_output_dir"
    
    # Debug: List directory contents
    debug_list_directory "$residual_dir" "Contents of residual directory"
    
    cd /home/nudel/Documents/peft
    
    export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
    export PYTHONPATH="/home/nudel/Documents/peft:${PYTHONPATH}"
    
    python "$SCRIPT_DIR/evaluate_residual_models.py" \
        --residual_models_dir="$residual_dir" \
        --output_dir="$eval_output_dir" \
        --tasks="$EVAL_TASKS" \
        --num_fewshot=$NUM_FEWSHOT \
        --limit=$EVAL_LIMIT \
        --save_results \
        --generate_latex
    
    if [[ $? -eq 0 ]]; then
        log_success "Evaluation completed for rank $rank from mode $training_mode"
        log_info "Results saved to: $eval_output_dir"
    else
        log_error "Evaluation failed for rank $rank from mode $training_mode"
        return 1
    fi
}

# Function to create combined results summary
create_combined_summary() {
    log_info "Creating combined results summary..."
    
    local summary_dir="./combined_residual_analysis_${MODEL_SHORT_NAME}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$summary_dir"
    
    # Copy all individual results
    for training_mode in "${TRAINING_MODES[@]}"; do
        for rank in "${LORA_RANKS[@]}"; do
            local eval_dir="./residual_evaluation_results_${MODEL_SHORT_NAME}_${training_mode}_r${rank}"
            if [[ -d "$eval_dir" ]]; then
                cp -r "$eval_dir" "$summary_dir/${training_mode}_rank_${rank}_results"
                log_info "Copied results for mode $training_mode, rank $rank"
            fi
        done
    done
    
    # Create a master summary file
    cat > "$summary_dir/README.md" << EOF
# PiSSA Residual Quantization Analysis Results

Generated on: $(date)

## Experiment Configuration
- Model: $MODEL_NAME_OR_PATH
- Model Short Name: $MODEL_SHORT_NAME
- Model Name Safe: $(get_model_name_safe)
- Training Modes Tested: ${TRAINING_MODES[*]}
- LoRA Ranks Tested: ${LORA_RANKS[*]}
- Quantization Bits: 2, 3, 4 (with various group sizes, evaluated by evaluate_residual_models.py)
- Evaluation Tasks: $EVAL_TASKS
- Few-shot Examples: $NUM_FEWSHOT
- Sample Limit: $EVAL_LIMIT

## Training Configuration
- Data Path: $DATA_PATH
- Dataset Split: $DATASET_SPLIT
- Epochs: $NUM_TRAIN_EPOCHS
- Batch Size: $PER_DEVICE_TRAIN_BATCH_SIZE
- Learning Rate: $LEARNING_RATE

## Directory Structure
EOF
    
    for training_mode in "${TRAINING_MODES[@]}"; do
        echo "### Training Mode: $training_mode" >> "$summary_dir/README.md"
        for rank in "${LORA_RANKS[@]}"; do
            echo "- \`${training_mode}_rank_${rank}_results/\`: Results for LoRA rank $rank" >> "$summary_dir/README.md"
        done
    done
    
    log_success "Combined summary created in: $summary_dir"
}

# Function to print current configuration
print_config() {
    log_info "Current Configuration:"
    echo "  Model: $MODEL_NAME_OR_PATH"
    echo "  Model Short Name: $MODEL_SHORT_NAME"
    echo "  Model Name Safe: $(get_model_name_safe)"
    echo "  Training Modes: ${TRAINING_MODES[*]}"
    echo "  LoRA Ranks: ${LORA_RANKS[*]}"
    echo "  Data: $DATA_PATH ($DATASET_SPLIT)"
    echo "  Training: $NUM_TRAIN_EPOCHS epochs, batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE, lr=$LEARNING_RATE"
    echo "  Evaluation: $EVAL_TASKS (fewshot=$NUM_FEWSHOT, limit=$EVAL_LIMIT)"
    echo "  CUDA Device: $CUDA_DEVICE"
    echo ""
}

# Function to cleanup old model files for different models
cleanup_old_models() {
    if [[ "$1" != "--force" ]]; then
        return 0
    fi
    
    local model_name_safe=$(get_model_name_safe)
    log_info "Cleaning up models that don't match current model ($model_name_safe)..."
    
    for training_mode in "${TRAINING_MODES[@]}"; do
        for rank in "${LORA_RANKS[@]}"; do
            local rank_dir="$BASE_OUTPUT_DIR/${training_mode}_quantized_residuals_r${rank}"
            if [[ -d "$rank_dir" ]]; then
                # Check if directory contains files for different model
                if ! (find "$rank_dir" -name "*${model_name_safe}*" | head -1 | grep -q .); then
                    log_warning "Found non-matching model files in $rank_dir, backing up..."
                    mv "$rank_dir" "${rank_dir}_backup_$(date +%Y%m%d_%H%M%S)"
                fi
            fi
        done
    done
}

# Main execution
main() {
    log_info "Starting PiSSA Residual Quantization Pipeline"
    print_config
    
    # Check if required directories exist
    if [[ ! -f "$SCRIPT_DIR/ultimate_train_collection.py" ]]; then
        log_error "Training script not found: $SCRIPT_DIR/ultimate_train_collection.py"
        exit 1
    fi
    
    if [[ ! -f "$SCRIPT_DIR/evaluate_residual_models.py" ]]; then
        log_error "Evaluation script not found: $SCRIPT_DIR/evaluate_residual_models.py"
        exit 1
    fi
    
    # Create base output directory
    mkdir -p "$BASE_OUTPUT_DIR"
    
    # Cleanup old models if requested
    cleanup_old_models "$1"
    
    local total_experiments=$((${#LORA_RANKS[@]} * ${#TRAINING_MODES[@]}))
    local completed_experiments=0
    
    for training_mode in "${TRAINING_MODES[@]}"; do
        log_info "===== PROCESSING TRAINING MODE: $training_mode ====="
        
        # Phase 1: Check for existing models OR train new ones
        log_info "===== PHASE 1: CHECKING FOR EXISTING MODELS ($training_mode) ====="
        successful_training=()
        need_training=()
        
        for rank in "${LORA_RANKS[@]}"; do
            if find_actual_residual_dir "$BASE_OUTPUT_DIR" "$rank" "$training_mode"; then
                log_success "Found existing models for rank $rank at: $FOUND_RESIDUAL_DIR"
                successful_training+=($rank)
                
                # Fix adapter config if needed
                local model_name_safe=$(get_model_name_safe)
                local found_adapter=$(find "$FOUND_RESIDUAL_DIR" -name "daniel_adapter_r${rank}_*${model_name_safe}*" -type d 2>/dev/null | head -1)
                
                if [[ -n "$found_adapter" ]]; then
                    log_info "Found adapter at: $found_adapter"
                    fix_adapter_config "$found_adapter"
                else
                    log_warning "Adapter not found for rank $rank"
                fi
            else
                log_warning "No existing models found for rank $rank"
                need_training+=($rank)
            fi
        done
        
        log_info "Found existing models for ranks: ${successful_training[*]}"
        log_info "Need training for ranks: ${need_training[*]}"
        
        # Phase 1b: Train missing models if any
        if [[ ${#need_training[@]} -gt 0 ]]; then
            log_info "===== PHASE 1b: TRAINING MISSING MODELS ($training_mode) ====="
            
            for rank in "${need_training[@]}"; do
                log_info "Training rank $rank ($training_mode) (${completed_experiments}/${total_experiments} total completed)"
                
                if train_model $rank $training_mode; then
                    successful_training+=($rank)
                    log_success "✅ Training successful for rank $rank ($training_mode)"
                else
                    log_error "❌ Training failed for rank $rank ($training_mode)"
                fi
                
                # Small delay between training runs
                sleep 5
            done
        fi
        
        # Phase 2: Evaluation
        log_info "===== PHASE 2: EVALUATING MODELS ($training_mode) ====="
        failed_evaluation=()
        successful_evaluation=()
        
        # Sort successful_training array to ensure consistent order
        IFS=$'\n' successful_training=($(sort -n <<<"${successful_training[*]}"))
        unset IFS

        for rank in "${successful_training[@]}"; do
            log_info "Evaluating rank $rank ($training_mode) (${completed_experiments}/${total_experiments} completed)"
            
            if evaluate_model $rank $training_mode; then
                successful_evaluation+=($rank)
                completed_experiments=$((completed_experiments + 1))
                log_success "✅ Evaluation successful for rank $rank ($training_mode)"
            else
                failed_evaluation+=($rank)
                log_error "❌ Evaluation failed for rank $rank ($training_mode)"
            fi
            
            # Small delay between evaluation runs
            sleep 5
        done
        
        # Evaluation summary for this mode
        log_info "===== EVALUATION SUMMARY FOR MODE: $training_mode ====="
        log_success "Successful evaluation: ${successful_evaluation[*]}"
        if [[ ${#failed_evaluation[@]} -gt 0 ]]; then
            log_error "Failed evaluation: ${failed_evaluation[*]}"
        fi
    done
    
    # Phase 3: Create combined summary
    log_info "===== PHASE 3: CREATING COMBINED SUMMARY ====="
    create_combined_summary
    
    # Final summary
    log_info "===== PIPELINE COMPLETED ====="
    log_info "Total experiments completed: ${completed_experiments}/${total_experiments}"
    
    if [[ $completed_experiments -eq $total_experiments ]]; then
        log_success "🎉 All models processed successfully!"
    else
        log_warning "⚠️  Some models failed. Check logs above for details."
    fi
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Check for required tools
    if ! command -v python &> /dev/null; then
        log_error "Python not found in PATH"
        exit 1
    fi
    
    # Run main function
    main "$@"
fi