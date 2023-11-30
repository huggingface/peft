#!/bin/bash
usage() {
  echo "Usage: ${0} [-t|--type] [-d|--dataset] [-n|--block_num] [-s|--block_size] [-f|--n_butterfly_factor] [-v|--version]"  1>&2
  exit 1
}

while [[ $# -gt 0 ]];do
  key=${1}
  case ${key} in
    -t|--type)
      PEFT_TYPE=${2}
      shift 2
      ;;
    -d|--dataset)
      DATASET_NAME=${2}
      shift 2
      ;;
    -n|--block_num)
      BLOCK_NUM=${2}
      shift 2
      ;;
    -s|--block_size)
      BLOCK_SIZE=${2}
      shift 2
      ;;
    -f|--n_butterfly_factor)
      N_BUTTERFLY_FACTOR=${2}
      shift 2
      ;;
    -v|--version)
      VERSION=${2}
      shift 2
      ;;
    *)
      usage
      shift
      ;;
  esac
done


export HF_HOME="/is/cluster/yxiu/.cache"
export CUDA_HOME="/is/software/nvidia/cuda-11.7"

export PROJECT_NAME="controlnet_${DATASET_NAME}"
export RUN_NAME="${PEFT_TYPE}_${BLOCK_NUM}${BLOCK_SIZE}${N_BUTTERFLY_FACTOR}"
export CONTROLNET_PATH=""

if [ $VERSION = "stable" ]; then
  export MODEL_NAME="stabilityai/stable-diffusion-2-1"
elif [ $VERSION = "runway" ]; then
  export MODEL_NAME="runwayml/stable-diffusion-v1-5"
else
  echo "Invalid VERSION: ${VERSION}"
fi

export RUN_NAME="${RUN_NAME}_${VERSION}"
export OUTPUT_DIR="./data/output/${DATASET_NAME}/${RUN_NAME}"

if [ $DATASET_NAME = "ade20k" ]; then
  # export RESUME_PATH="latest"
  export RESUME_PATH=""
  export EPOCHS=20
else
  # export RESUME_PATH="checkpoint-33000"
  # export RESUME_PATH="latest"
  export RESUME_PATH=""
  export EPOCHS=40
fi

getenv=True
source /home/yxiu/miniconda3/bin/activate OPT

if [[ $PEFT_TYPE =~ ^(boft|oft)$ ]]; then

  accelerate launch --main_process_port=29577 train_controlnet.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --resume_from_checkpoint=$RESUME_PATH \
  --controlnet_model_name_or_path=$CONTROLNET_PATH \
  --output_dir=$OUTPUT_DIR \
  --report_to="wandb" \
  --dataset_name=$DATASET_NAME \
  --resolution=512 \
  --learning_rate=1e-5 \
  --checkpointing_steps=1000 \
  --validation_steps=100000000000 \
  --num_validation_images=12 \
  --num_train_epochs=$EPOCHS \
  --train_batch_size=4 \
  --dataloader_num_workers=8 \
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
  --boft_bias_fit \
  --boft_bias="boft_only" \

elif [ $PEFT_TYPE = "lora" ]; then

  accelerate launch --main_process_port=29577 train_controlnet.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --resume_from_checkpoint=$RESUME_PATH \
  --controlnet_model_name_or_path=$CONTROLNET_PATH \
  --output_dir=$OUTPUT_DIR \
  --report_to="wandb" \
  --dataset_name=$DATASET_NAME \
  --resolution=512 \
  --learning_rate=1e-5 \
  --checkpointing_steps=1000 \
  --validation_steps=100000000000 \
  --num_validation_images=12 \
  --num_train_epochs=$EPOCHS \
  --train_batch_size=4 \
  --dataloader_num_workers=8 \
  --seed="0" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --wandb_project_name=$PROJECT_NAME \
  --wandb_run_name=$RUN_NAME \
  --enable_xformers_memory_efficient_attention \
  --use_lora \
  --lora_r $BLOCK_NUM \
  --lora_alpha=27 \
  --lora_dropout=0.1 \
  --lora_bias="lora_only" \

else
  echo "Invalid PEFT_TYPE: ${PEFT_TYPE}"
fi
