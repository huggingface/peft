<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->


# Fine-tuning for controllable generation with BOFT (ControlNet)

This guide demonstrates how to use BOFT, an orthogonal fine-tuning method, to fine-tune Stable Diffusion with either `stabilityai/stable-diffusion-2-1` or `runwayml/stable-diffusion-v1-5` model for controllable generation.

By using BOFT from 🤗 PEFT, we can significantly reduce the number of trainable parameters while still achieving impressive results in various fine-tuning tasks across different foundation models. BOFT enhances model efficiency by integrating full-rank orthogonal matrices with a butterfly structure into specific model blocks, such as attention blocks, mirroring the approach used in LoRA. During fine-tuning, only these inserted matrices are trained, leaving the original model parameters untouched. During inference, the trainable BOFT paramteres can be merged into the original model, eliminating any additional computational costs.

As a member of the **orthogonal finetuning** class, BOFT presents a systematic and principled method for fine-tuning. It possesses several unique properties and has demonstrated superior performance compared to LoRA in a variety of scenarios. For further details on BOFT, please consult the [PEFT's GitHub repo's concept guide OFT](https://https://huggingface.co/docs/peft/index), the [original BOFT paper](https://arxiv.org/abs/2311.06243) and the [original OFT paper](https://arxiv.org/abs/2306.07280).

In this guide we provide a controllable generation (ControlNet) fine-tuning script that is available in [PEFT's GitHub repo examples](https://github.com/huggingface/peft/tree/main/examples/boft_controlnet). This implementation is adapted from [diffusers's ControlNet](https://github.com/huggingface/diffusers/tree/main/examples/controlnet) and [Hecong Wu's ControlLoRA](https://github.com/HighCWu/ControlLoRA). You can try it out and finetune on your custom images.

## Set up your environment
Start by cloning the PEFT repository:

```bash
git clone https://github.com/huggingface/peft
```

Navigate to the directory containing the training scripts for fine-tuning Dreambooth with BOFT:
```bash
cd peft/examples/boft_controlnet
```

Set up your environment: install PEFT, and all the required libraries. At the time of writing this guide we recommend installing PEFT from source.

```bash
conda create --name peft python=3.10
conda activate peft
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install xformers -c xformers
pip install -r requirements.txt
pip install git+https://github.com/huggingface/peft
```

## Data

We use the [control-celeba-hq](https://huggingface.co/datasets/oftverse/control-celeba-hq) dataset for landmark-to-face controllable generation. We also provide evaluation scripts to evaluate the controllable generation performance. This task can be used to quantitatively compare different fine-tuning techniques.

```bash
export DATASET_NAME="oftverse/control-celeba-hq"
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

```bash
./train_controlnet.sh
```
or
```bash
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

Run inference on the saved model to sample new images from the validation set:

```bash
./test_controlnet.sh
```
or
```bash
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

accelerate launch test_controlnet.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --controlnet_path=$CONTROLNET_PATH \
  --unet_path=$UNET_PATH \
  --adapter_name=$RUN_NAME \
  --output_dir=$RESULTS_PATH \
  --dataset_name=$DATASET_NAME \

```

Run evaluation on the sampled images to evaluate the landmark reprojection error:

```bash
./eval.sh
```
or
```bash
ITER_NUM=50000

export MODEL_NAME="stabilityai/stable-diffusion-2-1"
# export MODEL_NAME="runwayml/stable-diffusion-v1-5"

export RUN_NAME="${PEFT_TYPE}_${BLOCK_NUM}${BLOCK_SIZE}${N_BUTTERFLY_FACTOR}"
export DATASET_NAME="oftverse/control-celeba-hq"
export CKPT_NAME="checkpoint-${ITER_NUM}"
export OUTPUT_DIR="./output/${DATASET_NAME}/${RUN_NAME}/${CKPT_NAME}"
export CONTROLNET_PATH="${OUTPUT_DIR}/controlnet/model.safetensors"
export UNET_PATH="${OUTPUT_DIR}/unet/${RUN_NAME}"

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