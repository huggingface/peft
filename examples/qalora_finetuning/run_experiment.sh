#!/bin/bash

# Configuration from launch.json
BASE_MODEL="TinyLlama/TinyLlama_v1.1"
DATASET="yahma/alpaca-cleaned"
DATASET_SPLIT="train[:1]"

# Training parameters from launch.json
LEARNING_RATE="1e-4"
BATCH_SIZE=8
GRAD_ACCUM=1
NUM_EPOCHS=2
MAX_LENGTH=2048

# Training mode selection
TRAINING_MODE="qalora"  # Options: full, lora, qlora, qalora, pissa, corda
LORA_R=256
QALORA_GROUP_SIZE=32
PISSA_NITER=4

# Other training parameters from launch.json
WARMUP_RATIO=0.03
LR_SCHEDULER_TYPE="cosine"
LOGGING_STEPS=10
SAVE_STEPS=25000
EVAL_STEPS=500
BF16="True"
DATALOADER_PIN_MEMORY="False"
REMOVE_UNUSED_COLUMNS="False"
REPORT_TO="none"

# Evaluation parameters from launch.json
EVAL_TASKS="hellaswag,piqa,winogrande,arc_easy,arc_challenge,boolq,openbookqa,wikitext"
NUM_FEWSHOT=5
EVAL_BATCH_SIZE=1
EVAL_LIMIT=300

# Generate experiment names with parameters
DATASET_NAME=$(basename "$DATASET" | sed 's/-/_/g')
if [ "$TRAINING_MODE" = "qalora" ]; then
    EXPERIMENT_NAME="${DATASET_NAME}_${TRAINING_MODE}_r_${LORA_R}_group_${QALORA_GROUP_SIZE}"
elif [ "$TRAINING_MODE" = "pissa" ]; then
    EXPERIMENT_NAME="${DATASET_NAME}_${TRAINING_MODE}_r_${LORA_R}_niter_${PISSA_NITER}"
elif [ "$TRAINING_MODE" = "lora" ] || [ "$TRAINING_MODE" = "corda" ]; then
    EXPERIMENT_NAME="${DATASET_NAME}_${TRAINING_MODE}_r_${LORA_R}"
else
    EXPERIMENT_NAME="${DATASET_NAME}_${TRAINING_MODE}"
fi

OUTPUT_DIR="train_results_${EXPERIMENT_NAME}"
EVAL_DIR="eval_results_${EXPERIMENT_NAME}"

echo "🚀 Starting training experiment with configuration:"
echo "  Experiment name: $EXPERIMENT_NAME"
echo "  Base model: $BASE_MODEL"
echo "  Training mode: $TRAINING_MODE"
echo "  Output dir: $OUTPUT_DIR"
echo "  Eval dir: $EVAL_DIR"
echo "  Dataset: $DATASET"
echo "  LoRA rank: $LORA_R"
if [ "$TRAINING_MODE" = "qalora" ]; then
    echo "  QALoRA group size: $QALORA_GROUP_SIZE"
elif [ "$TRAINING_MODE" = "pissa" ]; then
    echo "  PiSSA iterations: $PISSA_NITER"
fi

# Phase 1: Training
echo "📚 Phase 1: Training with $TRAINING_MODE..."
python ultimate_train_collection.py \
    --model_name_or_path="$BASE_MODEL" \
    --training_mode="$TRAINING_MODE" \
    --lora_r="$LORA_R" \
    --qalora_group_size="$QALORA_GROUP_SIZE" \
    --pissa_niter="$PISSA_NITER" \
    --learning_rate="$LEARNING_RATE" \
    --per_device_train_batch_size="$BATCH_SIZE" \
    --data_path="$DATASET" \
    --dataset_split="$DATASET_SPLIT" \
    --dataset_field "instruction" "output" \
    --num_train_epochs="$NUM_EPOCHS" \
    --output_dir="$OUTPUT_DIR" \
    --model_max_length="$MAX_LENGTH" \
    --gradient_accumulation_steps="$GRAD_ACCUM" \
    --warmup_ratio="$WARMUP_RATIO" \
    --lr_scheduler_type="$LR_SCHEDULER_TYPE" \
    --logging_steps="$LOGGING_STEPS" \
    --save_steps="$SAVE_STEPS" \
    --eval_steps="$EVAL_STEPS" \
    --bf16="$BF16" \
    --dataloader_pin_memory="$DATALOADER_PIN_MEMORY" \
    --remove_unused_columns="$REMOVE_UNUSED_COLUMNS" \
    --report_to="$REPORT_TO" \
    --bits="3"

if [ $? -ne 0 ]; then
    echo "❌ Training failed!"
    exit 1
fi

echo "✅ Training completed!"

# Wait a moment for cleanup
sleep 2

# Phase 2: Evaluation
echo "📊 Phase 2: Evaluation on multiple benchmarks..."
python eval_peft.py \
    --model_name_or_path="/home/nudel/Documents/peft/examples/qalora_finetuning/train_results_alpaca_cleaned_qalora_r_4_group_32/ft/adapter" \
    --base_model="$BASE_MODEL" \
    --tasks="$EVAL_TASKS" \
    --num_fewshot="$NUM_FEWSHOT" \
    --per_device_eval_batch_size="$EVAL_BATCH_SIZE" \
    --test_generation \
    --output_dir="$EVAL_DIR" \
    --limit="$EVAL_LIMIT" \
    --bits="3"
    


if [ $? -ne 0 ]; then
    echo "❌ Evaluation failed!"
    exit 1
fi

echo "✅ Experiment completed successfully!"
echo "📁 Training results saved in: $OUTPUT_DIR"
echo "📁 Evaluation results saved in: $EVAL_DIR"