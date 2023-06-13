#!/usr/bin/env bash

LOG_DIR="./logs"
OUTPUT_DIR="./outputs"
mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR

model=${1:-"roberta-large"}
tk=${2:-"mrpc"}
ep=${3:-20}
lr=${4:-1e-3}
bs=${5:-32}

model_name=${model#*/}
log_file="${LOG_DIR}/${model_name}.log"
output_dir="${OUTPUT_DIR}/${model_name}"

/usr/bin/env python p_tuning_refactor.py \
    --model $model \
    --task $tk \
    --num_epochs $ep \
    --lr $lr \
    --batch_size $bs \
    --output_dir $output_dir \
    2>&1 | tee $log_file
