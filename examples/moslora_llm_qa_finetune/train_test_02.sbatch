#!/bin/bash

# SBATCH  -J peft_test
# SBATCH  -N 1
# SBATCH  --gres=gpu:1
# SBATCH  --output=moslora.out
# SBATCH  --error=moslora.out
# SBATCH  --time=148:00:00
# SBATCH  --partition=LocalQ
# SBATCH  --ntasks=1

gpuid=2
rank=8
alpha=16


model_name_or_path="meta-llama/Meta-Llama-3.1-8B-Instruct"

for seed in {1..5}
do
  CUDA_VISIBLE_DEVICES=$gpuid python -u examples/moslora_llm_qa_finetune/finetune.py \
    --base_model $model_name_or_path \
    --data_path 'examples/moslora_llm_qa_finetune/commonsense_42k.json' \
    --output_dir examples/moslora_llm_qa_finetune/trained_models/MoSLoRA/$seed \
    --batch_size 16 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 256 \
    --val_set_size 120 \
    --adapter_name lora \
    --lora_r $rank \
    --lora_alpha $alpha \
    --target_modules "["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]" \
    --use_moslora --seed $seed

  for ds in ARC-Easy openbookqa social_i_qa ARC-Challenge winogrande piqa boolq hellaswag
  do
    CUDA_VISIBLE_DEVICES=$gpuid python -u examples/moslora_llm_qa_finetune/commonsense_evaluate.py \
      --ds_path "examples/moslora_llm_qa_finetune/dataset" \
      --dataset $ds \
      --batch_size 1 \
      --base_model $model_name_or_path \
      --lora_weights examples/moslora_llm_qa_finetune/trained_models/MoSLoRA/$seed \
      --save_dir examples/moslora_llm_qa_finetune/output_results/MoSLoRA/$seed
  done
done