#!/bin/bash
PEFT_TYPE="boft"
BLOCK_NUM=8
BLOCK_SIZE=0
N_BUTTERFLY_FACTOR=0
ITER_NUM=50000
# export HF_HOME="/is/cluster/yxiu/.cache"
export CUDA_HOME="/is/software/nvidia/cuda-11.7"
export RUN_NAME="${PEFT_TYPE}_${BLOCK_NUM}${BLOCK_SIZE}${N_BUTTERFLY_FACTOR}"

export MODEL_NAME="stabilityai/stable-diffusion-2-1"
# export MODEL_NAME="runwayml/stable-diffusion-v1-5"

export DATASET_NAME="oftverse/control-celeba-hq"
export CKPT_NAME="checkpoint-${ITER_NUM}"
export OUTPUT_DIR="./output/${DATASET_NAME}/${RUN_NAME}/${CKPT_NAME}"
export CONTROLNET_PATH="${OUTPUT_DIR}/controlnet/model.safetensors"
export UNET_PATH="${OUTPUT_DIR}/unet/${RUN_NAME}"

. /home/zqiu/miniconda3/etc/profile.d/conda.sh
conda activate peft

# CUDA_VISIBLE_DEVICES=0 python test_controlnet.py \
CUDA_VISIBLE_DEVICES=0 accelerate launch eval.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --controlnet_path=$CONTROLNET_PATH \
  --unet_path=$UNET_PATH \
  --adapter_name=$RUN_NAME \
  --output_dir=$OUTPUT_DIR \
  --dataset_name=$DATASET_NAME \
#!/bin/bash
PEFT_TYPE="boft"
BLOCK_NUM=8
BLOCK_SIZE=0
N_BUTTERFLY_FACTOR=0
ITER_NUM=50000
# export HF_HOME="/is/cluster/yxiu/.cache"
export CUDA_HOME="/is/software/nvidia/cuda-11.7"
export RUN_NAME="${PEFT_TYPE}_${BLOCK_NUM}${BLOCK_SIZE}${N_BUTTERFLY_FACTOR}"

export MODEL_NAME="stabilityai/stable-diffusion-2-1"
# export MODEL_NAME="runwayml/stable-diffusion-v1-5"

export DATASET_NAME="oftverse/control-celeba-hq"
export CKPT_NAME="checkpoint-${ITER_NUM}"
export OUTPUT_DIR="./output/${DATASET_NAME}/${RUN_NAME}/${CKPT_NAME}"
export CONTROLNET_PATH="${OUTPUT_DIR}/controlnet/model.safetensors"
export UNET_PATH="${OUTPUT_DIR}/unet/${RUN_NAME}"

. /home/zqiu/miniconda3/etc/profile.d/conda.sh
conda activate peft

# CUDA_VISIBLE_DEVICES=0 python test_controlnet.py \
CUDA_VISIBLE_DEVICES=0 accelerate launch eval.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --controlnet_path=$CONTROLNET_PATH \
  --unet_path=$UNET_PATH \
  --adapter_name=$RUN_NAME \
  --output_dir=$OUTPUT_DIR \
  --dataset_name=$DATASET_NAME \
  --vis_overlays \



