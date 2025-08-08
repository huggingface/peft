# 设置可见GPU
export CUDA_VISIBLE_DEVICES=4,5

# 使用Bottleneck适配器进行微调
# 4bit量化配置已更新，可以尝试取消注释 --use_4bit 启用量化
# 8bit量化也已经修复，可以尝试 --use_8bit
# 注意: 同时使用deepspeed和量化可能会有兼容性问题，建议二选一使用

Tuning_method="Bottleneck"
target_modules="down_proj,up_proj"
model="Qwen2-7B-Instruct"
dataset="../dataset/alpaca/alpaca_data_copy.json"

model_name="/data1/lihanzuo/models/"$model

torchrun --nproc_per_node=2 --master_port=29500 sft_tuning.py \
  --seed 42 \
  --model_name $model_name \
  --data_path  $dataset \
  --tuning_method $Tuning_method --target_modules $target_modules \
  --output_dir "./"$model"-"$Tuning_method"-target_"$target_modules \
  --num_train_epochs 3 \
  --per_device_train_batch_size 12 \
  --per_device_eval_batch_size  10 \
  --gradient_accumulation_steps 2 \
  --eval_ratio 0.1 \
  --eval_and_save_steps 200 \
  --learning_rate 2e-5 \
  --max_seq_length 512 \
  --bf16 \
  --use_4bit   # 启用4bit量化 (推荐)
  # --use_8bit \  # 或者使用8bit量化
  # --deepspeed deepspeed_config_stage3.json \  # 或者使用Deepspeed
  # --wandb_project "qwen-bottleneck" \