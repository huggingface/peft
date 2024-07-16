<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# DreamBooth fine-tuning with HRA

This guide demonstrates how to use Householder reflection adaptation (HRA) method, to fine-tune Dreambooth with `stabilityai/stable-diffusion-2-1` model.

HRA provides a new perspective connecting LoRA to OFT and achieves encouraging performance in various downstream tasks.
HRA adapts a pre-trained model by multiplying each frozen weight matrix with a chain of r learnable Householder reflections (HRs).
HRA can be interpreted as either an OFT adapter or an adaptive LoRA. 
Consequently, it harnesses the advantages of both strategies, reducing parameters and computation costs while penalizing the loss of pre-training knowledge.
For further details on HRA, please consult the [original HRA paper](https://arxiv.org/abs/2405.17484).

In this guide we provide a Dreambooth fine-tuning script that is available in [PEFT's GitHub repo examples](https://github.com/huggingface/peft/tree/main/examples/hra_dreambooth). This implementation is adapted from [peft's boft_dreambooth](https://github.com/huggingface/peft/tree/main/examples/boft_dreambooth). 

You can try it out and fine-tune on your custom images.

## Set up your environment

Start by cloning the PEFT repository:

```bash
git clone --recursive https://github.com/huggingface/peft
```

Navigate to the directory containing the training scripts for fine-tuning Dreambooth with HRA:

```bash
cd peft/examples/hra_dreambooth
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
hra_dreambooth
├── data
│   └── dreambooth
│       └── dataset
│           ├── backpack
│           └── backpack_dog
│           ...
```

You can also put your custom images into `hra_dreambooth/data/dreambooth/dataset`.

## Fine-tune Dreambooth with HRA

```bash
class_idx=0
bash ./train_dreambooth.sh $class_idx
```

where the `$class_idx` corresponds to different subjects ranging from 0 to 29.

Launch the training script with `accelerate` and pass hyperparameters, as well as LoRa-specific arguments to it such as:

- `use_hra`: Enables HRA in the training script.
- `hra_r`: the number of HRs (i.e., r) across different layers, expressed in `int`. 
As r increases, the number of trainable parameters increases, which generally leads to improved performance.
However, this also results in higher memory consumption and longer computation times. 
Therefore, r is usually set to 8.
**Note**, please set r to an even number to avoid potential issues during initialization.
- `hra_apply_GS`: Applies Gram-Schmidt orthogonalization. Default is `false`.
- `hra_bias`: specify if the `bias` parameters should be trained. Can be `none`, `all` or `hra_only`.

If you are running this script on Windows, you may need to set the `--num_dataloader_workers` to 0.

To learn more about DreamBooth fine-tuning with prior-preserving loss, check out the [Diffusers documentation](https://huggingface.co/docs/diffusers/training/dreambooth#finetuning-with-priorpreserving-loss).

## Generate images with the fine-tuned model

To generate images with the fine-tuned model, simply run the jupyter notebook `dreambooth_inference.ipynb` for visualization with `jupyter notebook` under `./examples/hra_dreambooth`.
