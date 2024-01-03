# Fine-tuning for controllable generation with BOFT

## Set up your environment
Start by cloning the PEFT repository:

```python
git clone https://github.com/huggingface/peft
```

Navigate to the directory containing the training scripts for fine-tuning Dreambooth with LoRA:
```bash
cd peft/examples/boft_controlnet
```

Set up your environment: install PEFT, and all the required libraries. At the time of writing this guide we recommend installing PEFT from source.

```python
conda create --name peft python=3.10
conda activate peft
# conda install pytorch==2.0.2 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install xformers -c xformers
pip install -r requirements.txt
# pip install transformers accelerate evaluate datasets wandb diffusers==0.17.1
# pip install scikit-image opencv-python face-alignment==1.4.1
# pip install git+https://github.com/huggingface/peft
```

Download the example images for validation:
```python
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png
```

## Train controllable generation (ControlNet) with BOFT

Start with setting some hyperparamters for BOFT:
```bash
PEFT_TYPE="boft"
BLOCK_NUM=8
BLOCK_SIZE=0
N_BUTTERFLY_FACTOR=0
```

Here:


Navigate to the directory containing the training scripts for fine-tuning Stable Diffusion with BOFT for controllable generation:

```python
./train_controlnet.sh
```
or
```bash
PEFT_TYPE="boft"
BLOCK_NUM=8
BLOCK_SIZE=0
N_BUTTERFLY_FACTOR=0

export MODEL_NAME="stabilityai/stable-diffusion-2-1"
# export MODEL_NAME="runwayml/stable-diffusion-v1-5"

export DATASET_NAME="oftverse/control-celeba-hq"
export PROJECT_NAME="controlnet_${PEFT_TYPE}"
export RUN_NAME="${PEFT_TYPE}_${BLOCK_NUM}${BLOCK_SIZE}${N_BUTTERFLY_FACTOR}"
export CONTROLNET_PATH=""
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
  --checkpointing_steps=5000 \
  --max_train_steps=50000 \
  --validation_steps=2000 \
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
  --report_to="wandb" \
```

Run inference on the saved model to sample new images on the validation set:

```python
./test_controlnet.sh
```
or
```bash
#!/bin/bash
PEFT_TYPE="boft"
BLOCK_NUM=8
BLOCK_SIZE=0
N_BUTTERFLY_FACTOR=0
ITER_NUM=50000

export MODEL_NAME="stabilityai/stable-diffusion-2-1"
# export MODEL_NAME="runwayml/stable-diffusion-v1-5"

export RUN_NAME="${PEFT_TYPE}_${BLOCK_NUM}${BLOCK_SIZE}${N_BUTTERFLY_FACTOR}"
export DATASET_NAME="oftverse/control-celeba-hq"
export CKPT_NAME="checkpoint-${ITER_NUM}"
export OUTPUT_DIR="./output/${DATASET_NAME}/${RUN_NAME}/${CKPT_NAME}"
export CONTROLNET_PATH="${OUTPUT_DIR}/controlnet/model.safetensors"
export UNET_PATH="${OUTPUT_DIR}/unet/${RUN_NAME}"
export RESULTS_PATH="${OUTPUT_DIR}/results"

# CUDA_VISIBLE_DEVICES=0 python test_controlnet.py \
CUDA_VISIBLE_DEVICES=0 accelerate launch test_controlnet.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --controlnet_path=$CONTROLNET_PATH \
  --unet_path=$UNET_PATH \
  --adapter_name=$RUN_NAME \
  --output_dir=$RESULTS_PATH \
  --dataset_name=$DATASET_NAME \

```

Run evaluation on the sampled images to evaluate the landmark reprojection error:

```python
./eval.sh
```
or
```bash
PEFT_TYPE="boft"
BLOCK_NUM=8
BLOCK_SIZE=0
N_BUTTERFLY_FACTOR=0
ITER_NUM=50000

export MODEL_NAME="stabilityai/stable-diffusion-2-1"
# export MODEL_NAME="runwayml/stable-diffusion-v1-5"

export RUN_NAME="${PEFT_TYPE}_${BLOCK_NUM}${BLOCK_SIZE}${N_BUTTERFLY_FACTOR}"
export DATASET_NAME="oftverse/control-celeba-hq"
export CKPT_NAME="checkpoint-${ITER_NUM}"
export OUTPUT_DIR="./output/${DATASET_NAME}/${RUN_NAME}/${CKPT_NAME}"
export CONTROLNET_PATH="${OUTPUT_DIR}/controlnet/model.safetensors"
export UNET_PATH="${OUTPUT_DIR}/unet/${RUN_NAME}"

# CUDA_VISIBLE_DEVICES=0 python test_controlnet.py \
# CUDA_VISIBLE_DEVICES=0 
accelerate launch eval.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --controlnet_path=$CONTROLNET_PATH \
  --unet_path=$UNET_PATH \
  --adapter_name=$RUN_NAME \
  --output_dir=$OUTPUT_DIR \
  --dataset_name=$DATASET_NAME \
  --vis_overlays \

```