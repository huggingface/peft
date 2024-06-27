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

# DreamBooth fine-tuning with BOFT

This guide demonstrates how to use BOFT, an orthogonal fine-tuning method, to fine-tune Dreambooth with either `stabilityai/stable-diffusion-2-1` or `runwayml/stable-diffusion-v1-5` model.

By using BOFT from 🤗 PEFT, we can significantly reduce the number of trainable parameters while still achieving impressive results in various fine-tuning tasks across different foundation models. BOFT enhances model efficiency by integrating full-rank orthogonal matrices with a butterfly structure into specific model blocks, such as attention blocks, mirroring the approach used in LoRA. During fine-tuning, only these inserted matrices are trained, leaving the original model parameters untouched. During inference, the trainable BOFT paramteres can be merged into the original model, eliminating any additional computational costs.

As a member of the **orthogonal finetuning** class, BOFT presents a systematic and principled method for fine-tuning. It possesses several unique properties and has demonstrated superior performance compared to LoRA in a variety of scenarios. For further details on BOFT, please consult the [PEFT's GitHub repo's concept guide OFT](https://https://huggingface.co/docs/peft/index), the [original BOFT paper](https://arxiv.org/abs/2311.06243) and the [original OFT paper](https://arxiv.org/abs/2306.07280).

In this guide we provide a Dreambooth fine-tuning script that is available in [PEFT's GitHub repo examples](https://github.com/huggingface/peft/tree/main/examples/boft_dreambooth). This implementation is adapted from [peft's lora_dreambooth](https://github.com/huggingface/peft/tree/main/examples/lora_dreambooth). You can try it out and finetune on your custom images.

## Set up your environment

Start by cloning the PEFT repository:

```bash
git clone --recursive https://github.com/huggingface/peft
```

Navigate to the directory containing the training scripts for fine-tuning Dreambooth with BOFT:

```bash
cd peft/examples/boft_dreambooth
```

Set up your environment: install PEFT, and all the required libraries. At the time of writing this guide we recommend installing PEFT from source. The following environment setup should work on A100 and H100:

```bash
conda create --name peft python=3.10
conda activate peft
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install xformers -c xformers
pip install -r requirements.txt
pip install git+https://github.com/huggingface/peft
```

## Download the data

[dreambooth](https://github.com/google/dreambooth) dataset should have been automatically cloned in the following structure when running the training script.

```
boft_dreambooth
├── data
│   ├── data_dir
│   └── dreambooth
│       └── data
│           ├── backpack
│           └── backpack_dog
│           ...
```

You can also put your custom images into `boft_dreambooth/data/dreambooth`.

## Finetune Dreambooth with BOFT

```bash
./train_dreambooth.sh
```

or using the following script arguments:

```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="path-to-instance-images"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"
```

Here:

- `INSTANCE_DIR`: The directory containing the images that you intend to use for training your model.
- `CLASS_DIR`: The directory containing class-specific images. In this example, we use prior preservation to avoid overfitting and language-drift. For prior preservation, you need other images of the same class as part of the training process. However, these images can be generated and the training script will save them to a local path you specify here.
- `OUTPUT_DIR`: The destination folder for storing the trained model's weights.

To learn more about DreamBooth fine-tuning with prior-preserving loss, check out the [Diffusers documentation](https://huggingface.co/docs/diffusers/training/dreambooth#finetuning-with-priorpreserving-loss).

Launch the training script with `accelerate` and pass hyperparameters, as well as LoRa-specific arguments to it such as:

- `use_boft`: Enables BOFT in the training script.
- `boft_block_size`: the BOFT matrix block size across different layers, expressed in `int`. Smaller block size results in sparser update matrices with fewer trainable paramters. **Note**, please choose it to be dividable to most layer `in_features` dimension, e.g., 4, 8, 16. Also, you can only specify either `boft_block_size` or `boft_block_num`, but not both simultaneously, because `boft_block_size` x `boft_block_num` = layer dimension.
- `boft_block_num`: the number of BOFT matrix blocks across different layers, expressed in `int`. Fewer blocks result in sparser update matrices with fewer trainable paramters. **Note**, please choose it to be dividable to most layer `in_features` dimension, e.g., 4, 8, 16. Also, you can only specify either `boft_block_size` or `boft_block_num`, but not both simultaneously, because `boft_block_size` x `boft_block_num` = layer dimension.
- `boft_n_butterfly_factor`: the number of butterfly factors. **Note**, for `boft_n_butterfly_factor=1`, BOFT is the same as vanilla OFT, for `boft_n_butterfly_factor=2`, the effective block size of OFT becomes twice as big and the number of blocks become half.
- `bias`: specify if the `bias` paramteres should be traind. Can be `none`, `all` or `boft_only`.
- `boft_dropout`: specify the probability of multiplicative dropout.

Here's what the full set of script arguments may look like:

```bash
PEFT_TYPE="boft"
BLOCK_NUM=8
BLOCK_SIZE=0
N_BUTTERFLY_FACTOR=1

VALIDATION_PROMPT=${PROMPT_LIST[@]}
INSTANCE_PROMPT="a photo of ${UNIQUE_TOKEN} ${CLASS_TOKEN}"
CLASS_PROMPT="a photo of ${CLASS_TOKEN}"

export MODEL_NAME="stabilityai/stable-diffusion-2-1"
# export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export PROJECT_NAME="dreambooth_${PEFT_TYPE}"
export RUN_NAME="${SELECTED_SUBJECT}_${PEFT_TYPE}_${BLOCK_NUM}${BLOCK_SIZE}${N_BUTTERFLY_FACTOR}"
export INSTANCE_DIR="./data/dreambooth/dataset/${SELECTED_SUBJECT}"
export CLASS_DIR="./data/class_data/${CLASS_TOKEN}"
export OUTPUT_DIR="./data/output/${PEFT_TYPE}"


accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir="$CLASS_DIR" \
  --output_dir=$OUTPUT_DIR \
  --wandb_project_name=$PROJECT_NAME \
  --wandb_run_name=$RUN_NAME \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="$INSTANCE_PROMPT" \
  --validation_prompt="$VALIDATION_PROMPT" \
  --class_prompt="$CLASS_PROMPT" \
  --resolution=512 \
  --train_batch_size=1 \
  --num_dataloader_workers=2 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --use_boft \
  --boft_block_num=$BLOCK_NUM \
  --boft_block_size=$BLOCK_SIZE \
  --boft_n_butterfly_factor=$N_BUTTERFLY_FACTOR \
  --boft_dropout=0.1 \
  --boft_bias="boft_only" \
  --learning_rate=3e-5 \
  --max_train_steps=1010 \
  --checkpointing_steps=200 \
  --validation_steps=200 \
  --enable_xformers_memory_efficient_attention \
  --report_to="wandb" \
```

or use this training script:

```bash
./train_dreambooth.sh $idx
```

with the `$idx` corresponds to different subjects.

If you are running this script on Windows, you may need to set the `--num_dataloader_workers` to 0.

## Inference with a single adapter

To run inference with the fine-tuned model, simply run the jupyter notebook `dreambooth_inference.ipynb` for visualization with `jupyter notebook` under `./examples/boft_dreambooth`.
