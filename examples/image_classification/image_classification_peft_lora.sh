
LOG_DIR="./logs"
OUTPUT_DIR="./outputs"
mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR

model=${1:-"google/vit-base-patch16-224-in21k"}
lr=${3:-5e-3}
bs=${4:-128}
ep=${2:-5}

model_name=${model#*/}
log_file="${LOG_DIR}/${model_name}.log"
output_dir="${OUTPUT_DIR}/${model_name}"

/usr/bin/env python mlflow_image_classification_peft_lora.py \
    --model_checkpoint $model \
    --learning_rate $lr \
    --batch_size $bs \
    --num_train_epochs $ep \
    --output_dir $output_dir \
     2>&1 | tee $log_file

