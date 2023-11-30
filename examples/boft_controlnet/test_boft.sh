#!/bin/bash
usage() {
  echo "Usage: ${0} [-t|--type] [-d|--dataset] [-n|--block_num] [-s|--block_size] [-f|--n_butterfly_factor] [-i|--iter] [-v|--version]"  1>&2
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
    -i|--iter)
      ITER_NUM=${2}
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
export RUN_NAME="${PEFT_TYPE}_${BLOCK_NUM}${BLOCK_SIZE}${N_BUTTERFLY_FACTOR}"


if [ $VERSION = "stable" ]; then
  export MODEL_NAME="stabilityai/stable-diffusion-2-1"
elif [ $VERSION = "runway" ]; then
  export MODEL_NAME="runwayml/stable-diffusion-v1-5" 
else
  echo "Invalid VERSION: ${VERSION}"
fi

export RUN_NAME="${RUN_NAME}_${VERSION}"
export CKPT_NAME="checkpoint-${ITER_NUM}"
export OUTPUT_DIR="./data/output/${DATASET_NAME}/${RUN_NAME}/${CKPT_NAME}"
export CONTROLNET_PATH="${OUTPUT_DIR}/controlnet/model.safetensors"
export UNET_PATH="${OUTPUT_DIR}/unet/${RUN_NAME}"
export RESULTS_PATH="${OUTPUT_DIR}/results"

getenv=True
source /home/yxiu/miniconda3/bin/activate OPT

CUDA_VISIBLE_DEVICES=0 python test_controlnet.py \
--model_name=$MODEL_NAME \
--controlnet_path=$CONTROLNET_PATH \
--unet_path=$UNET_PATH \
--adapter_name=$RUN_NAME \
--output_dir=$RESULTS_PATH \
--dataset_name=$DATASET_NAME \


