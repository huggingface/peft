#!/bin/bash

# SBATCH  -J peft_test
# SBATCH  -N 1
# SBATCH  --gres=gpu:1
# SBATCH  --output=lora_%j.out
# SBATCH  --error=lora_%j.out
# SBATCH  --time=148:00:00
# SBATCH  --partition=LocalQ
# SBATCH  --ntasks=1


cd /mnt/Data/tqwu/peft

gpuid=0
rank=8
alpha=16


model_name_or_path="microsoft/Phi-3.5-mini-instruct"

for seed in {1..5}
do
  CUDA_VISIBLE_DEVICES=$gpuid python -u examples/moslora_llm_qa_finetune/finetune.py \
    --base_model $model_name_or_path \
    --data_path 'examples/moslora_llm_qa_finetune/commonsense_170k.json' \
    --output_dir examples/moslora_llm_qa_finetune/trained_models/LoRA_170_1e4_epoch1/$seed \
    --batch_size 16 \
    --micro_batch_size 16 \
    --num_epochs 1 \
    --learning_rate 1e-4 \
    --cutoff_len 256 \
    --val_set_size 160 \
    --adapter_name lora \
    --lora_r $rank \
    --lora_alpha $alpha \
    --target_modules "["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]" \
    --seed $seed

  for ds in ARC-Easy openbookqa social_i_qa ARC-Challenge winogrande piqa boolq 
  do
    CUDA_VISIBLE_DEVICES=$gpuid python -u examples/moslora_llm_qa_finetune/commonsense_evaluate.py \
      --ds_path "examples/moslora_llm_qa_finetune/dataset" \
      --dataset $ds \
      --batch_size 1 \
      --base_model $model_name_or_path \
      --lora_weights examples/moslora_llm_qa_finetune/trained_models/LoRA_170_1e4_epoch1/$seed \
      --save_dir examples/moslora_llm_qa_finetune/output_results/LoRA_170_1e4_epoch1/$seed
  done
done