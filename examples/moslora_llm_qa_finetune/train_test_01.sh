

gpuid=2
rank=4
alpha=8


model_name_or_path="microsoft/phi-1_5"

for seed in {1..3}
do
  CUDA_VISIBLE_DEVICES=$gpuid python -u examples/moslora_llm_qa_finetune/finetune.py \
    --base_model $model_name_or_path \
    --data_path 'examples/moslora_llm_qa_finetune/commonsense_42k.json' \
    --output_dir examples/moslora_llm_qa_finetune/trained_models/LoRA/$seed \
    --batch_size 16 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 3e-4 \
    --cutoff_len 256 \
    --val_set_size 120 \
    --adapter_name lora \
    --lora_r $rank \
    --use_gradient_checkpointing \
    --lora_alpha $alpha \
    --target_modules "["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]" \
    --seed $seed

  for ds in ARC-Easy openbookqa social_i_qa ARC-Challenge winogrande piqa boolq hellaswag
  do
    CUDA_VISIBLE_DEVICES=$gpuid python -u examples/moslora_llm_qa_finetune/commonsense_evaluate.py \
      --ds_path "examples/moslora_llm_qa_finetune/dataset" \
      --dataset $ds \
      --batch_size 1 \
      --base_model $model_name_or_path \
      --lora_weights examples/moslora_llm_qa_finetune/trained_models/LoRA/$seed \
      --save_dir examples/moslora_llm_qa_finetune/output_results/LoRA/$seed
  done
done