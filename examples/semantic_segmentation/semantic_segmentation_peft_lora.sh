

LOG_DIR="./logs"
OUTPUT_DIR="./outputs"
mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR

checkpoint=${1:-"nvidia/mit-b0"}
learning_rate=${2:-5e-4}
num_train_epochs=${3:-50}
per_device_train_batch_size=${4:-4}
per_device_eval_batch_size=${5:-2}

model_name=${model#*/}
log_file="${LOG_DIR}/${model_name}.log"
output_dir="${OUTPUT_DIR}/${model_name}"
# Run training script
echo "# ========================================================= #"
echo "training ${model_name}.."

python semantic_segmentation_peft_lora.py \
    --checkpoint $checkpoint \
    --learning_rate $learning_rate \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size $per_device_eval_batch_size \
    --output_dir $output_dir \
    2>&1 | tee $log_file
