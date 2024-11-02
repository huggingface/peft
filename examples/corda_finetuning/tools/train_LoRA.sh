BASE_MODEL="meta-llama/Llama-2-7b-hf"
OUTPUT=$1

# math
# --data_path meta-math/MetaMathQA \
# --dataset_field query response \

# code:
#--data_path m-a-p/CodeFeedback-Filtered-Instruction 
#--dataset_field query answer

# Inst:
#--data_path fxmeng/WizardLM_evol_instruct_V2_143k 
#--dataset_field human assistant

python -u train_model.py \
    --model_name_or_path $BASE_MODEL \
    --output_dir $OUTPUT \
    --corda_mode False \
    --lora_r 128 \
    --data_path meta-math/MetaMathQA \
    --dataset_split "train[:100000]" \
    --dataset_field query response \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --report_to none
