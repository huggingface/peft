set -x

# Hugging Face mainland mirror and local caches
export HF_ENDPOINT="https://hf-mirror.com"
export HUGGINGFACE_HUB_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/mnt/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_ENABLE_HF_TRANSFER="1"

# Workaround for RTX 4000-series: disable NCCL P2P and IB communication paths
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export CUDA_VISIBLE_DEVICES="0"

export model_path="YOUR_MODEL_PATH" # e.g. "meta-llama/Llama-2-70b-hf"
export output_dir="./" # e.g. "./checkpoints"


function train(){
python -u train.py \
--seed 100 \
--model_name_or_path $model_path \
--dataset_name $2 \
--chat_template_format "chatml" \
--add_special_tokens False \
--append_concat_token False \
--splits "train,test" \
--max_seq_len 2048 \
--num_train_epochs 1 \
--logging_steps 5 \
--log_level "info" \
--logging_strategy "steps" \
--eval_strategy "epoch" \
--save_strategy "epoch" \
--bf16 True \
--packing True \
--learning_rate 1e-4 \
--lr_scheduler_type "cosine" \
--weight_decay 1e-4 \
--warmup_ratio 0.0 \
--max_grad_norm 1.0 \
--output_dir $output_dir/$1 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--gradient_accumulation_steps 8 \
--gradient_checkpointing True \
--use_reentrant True \
--dataset_text_field "content" \
--use_peft_lora True \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.1 \
--lora_target_modules "q_proj,v_proj" \
--use_8bit_quantization False \
--use_4bit_quantization False \
--use_nested_quant True \
--bnb_4bit_compute_dtype "bfloat16" \
--use_flash_attn True $3 > logs/$1_$(date "+%Y%m%d-%H%M%S").log 2>&1 &
}


train test_lorampo smangrul/ultrachat-10k-chatml --adapter_name=lora\ --lora_mpo


