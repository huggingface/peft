BASE_MODEL="/home/rwkv/JL/model/llama2-7b"
OUTPUT_PATH="/home/rwkv/JL/out_model/llama2-7b"
DATA_PATH="/home/rwkv/JL/model/MetaMathQA"

# batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus = 128
deepspeed --include=localhost:0 bone.py \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --model_name_or_path $BASE_MODEL \
    --use_lora True \
    --init_lora_weights "pissa_niter_4" \
    --target_modules "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj" \
    --lora_rank 40 \
    --lora_alpha 32 \
    --lora_dropout 0 \
    --data_path $DATA_PATH \
    --dataset_field query response \
    --dataset_split "train[:100000]"\
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 1 \
    --model_max_length 512 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0. \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --report_to "wandb" \
    --merge True \
    --run_name "pissa"
