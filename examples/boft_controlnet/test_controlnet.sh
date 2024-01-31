PEFT_TYPE="boft"
BLOCK_NUM=8
BLOCK_SIZE=0
N_BUTTERFLY_FACTOR=1
ITER_NUM=50000

export RUN_NAME="${PEFT_TYPE}_${BLOCK_NUM}${BLOCK_SIZE}${N_BUTTERFLY_FACTOR}"

export MODEL_NAME="stabilityai/stable-diffusion-2-1"
# export MODEL_NAME="runwayml/stable-diffusion-v1-5"

export DATASET_NAME="oftverse/control-celeba-hq"
export CKPT_NAME="checkpoint-${ITER_NUM}"
export OUTPUT_DIR="./output/${DATASET_NAME}/${RUN_NAME}/${CKPT_NAME}"
export CONTROLNET_PATH="${OUTPUT_DIR}/controlnet/model.safetensors"
export UNET_PATH="${OUTPUT_DIR}/unet/${RUN_NAME}"
export RESULTS_PATH="${OUTPUT_DIR}/results"


accelerate launch test_controlnet.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --controlnet_path=$CONTROLNET_PATH \
  --unet_path=$UNET_PATH \
  --adapter_name=$RUN_NAME \
  --output_dir=$RESULTS_PATH \
  --dataset_name=$DATASET_NAME \


