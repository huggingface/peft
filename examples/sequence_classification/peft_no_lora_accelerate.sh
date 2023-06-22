LOG_DIR="./logs"
OUTPUT_DIR="./outputs"
mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR

model=${1:-"roberta-large"}

model_name=${model#*/}
log_file="${LOG_DIR}/${model_name}.log"
output_dir="${OUTPUT_DIR}/${model_name}"
python -c "from accelerate.utils import write_basic_config; write_basic_config(mixed_precision='fp16')"
/usr/bin/env python peft_no_lora_accelerate.py --model_name_or_path $model \
	 --output_dir $output_dir \
	  2>&1 | tee $log_file


