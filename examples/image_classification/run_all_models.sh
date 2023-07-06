
bash ../setup.sh image_classification

input_file=$1
[[ ! -f $input_file ]] && echo "${input_file} not exist" && exit 1
LOG_DIR="./logs"
OUTPUT_DIR="./outputs"
mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR

lr=${2:-5e-3}
ep=${3:-5}

model_name=${model#*/}
log_file="${LOG_DIR}/${model_name}.log"
output_dir="${OUTPUT_DIR}/${model_name}"

while read model batch_size; do
    echo Running: $model $batch_size
    python image_classification_peft_lora.py \
    --model_checkpoint $model \
    --batch_size $batch_size \
    --learning_rate $lr \
    --num_train_epochs $ep \
    --output_dir $output_dir \
     2>&1 | tee $log_file

done < $input_file

