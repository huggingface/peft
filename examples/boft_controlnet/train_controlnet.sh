PEFT_TYPE="boft"
BLOCK_NUM=8
BLOCK_SIZE=0
N_BUTTERFLY_FACTOR=1

export DATASET_NAME="oftverse/control-celeba-hq"
export PROJECT_NAME="controlnet_${PEFT_TYPE}"
export RUN_NAME="${PEFT_TYPE}_${BLOCK_NUM}${BLOCK_SIZE}${N_BUTTERFLY_FACTOR}"
export CONTROLNET_PATH=""

export MODEL_NAME="stabilityai/stable-diffusion-2-1"
# export MODEL_NAME="runwayml/stable-diffusion-v1-5"

export OUTPUT_DIR="./output/${DATASET_NAME}/${RUN_NAME}"

accelerate launch train_controlnet.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --resume_from_checkpoint=$RESUME_PATH \
  --controlnet_model_name_or_path=$CONTROLNET_PATH \
  --output_dir=$OUTPUT_DIR \
  --report_to="wandb" \
  --dataset_name=$DATASET_NAME \
  --resolution=512 \
  --learning_rate=1e-5 \
  --checkpointing_steps=500 \
  --max_train_steps=50000 \
  --validation_steps=5000 \
  --num_validation_images=12 \
  --train_batch_size=4 \
  --dataloader_num_workers=2 \
  --seed="0" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --wandb_project_name=$PROJECT_NAME \
  --wandb_run_name=$RUN_NAME \
  --enable_xformers_memory_efficient_attention \
  --use_boft \
  --boft_block_num=$BLOCK_NUM \
  --boft_block_size=$BLOCK_SIZE \
  --boft_n_butterfly_factor=$N_BUTTERFLY_FACTOR \
  --boft_dropout=0.1 \
  --boft_bias="boft_only" \