import itertools
import os
import sys

import torch
import torch.utils.checkpoint
from diffusers import StableDiffusionPipeline
from diffusers.utils import check_min_version
from tqdm import tqdm

from peft import (
    PeftModel, LoraConfig, OFTConfig, BOFTConfig, 
    get_peft_model_state_dict, set_peft_model_state_dict
)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")


def get_lora_sd_pipeline(
    ckpt_dir, base_model_name_or_path, epoch, dtype=torch.float32, device="cuda", adapter_name="default"
):
    if base_model_name_or_path is None:
        raise ValueError("Please specify the base model name or path")

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_name_or_path, torch_dtype=dtype, requires_safety_checker=False
    ).to(device)

    load_adapter(pipe, ckpt_dir, epoch, adapter_name)

    if dtype in (torch.float16, torch.bfloat16):
        pipe.unet.half()

    pipe.to(device)

    return pipe


def load_adapter(pipe, ckpt_dir, epoch, adapter_name="default"):
    unet_sub_dir = os.path.join(ckpt_dir, f"unet/{epoch}", adapter_name)

    if isinstance(pipe.unet, PeftModel):
        pipe.unet.load_adapter(unet_sub_dir, adapter_name=adapter_name)
    else:
        pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_sub_dir, adapter_name=adapter_name)


def set_adapter(pipe, adapter_name):
    pipe.unet.set_adapter(adapter_name)
    if isinstance(pipe.text_encoder, PeftModel):
        pipe.text_encoder.set_adapter(adapter_name)


def merging_lora_with_base(pipe, ckpt_dir, adapter_name="default"):
    unet_sub_dir = os.path.join(ckpt_dir, "unet")
    text_encoder_sub_dir = os.path.join(ckpt_dir, "text_encoder")
    if isinstance(pipe.unet, PeftModel):
        pipe.unet.set_adapter(adapter_name)
    else:
        pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_sub_dir, adapter_name=adapter_name)
    pipe.unet = pipe.unet.merge_and_unload()

    if os.path.exists(text_encoder_sub_dir):
        if isinstance(pipe.text_encoder, PeftModel):
            pipe.text_encoder.set_adapter(adapter_name)
        else:
            pipe.text_encoder = PeftModel.from_pretrained(
                pipe.text_encoder, text_encoder_sub_dir, adapter_name=adapter_name
            )
        pipe.text_encoder = pipe.text_encoder.merge_and_unload()

    return pipe


def create_weighted_lora_adapter(pipe, adapters, weights, adapter_name="default"):
    pipe.unet.add_weighted_adapter(adapters, adapter_name)
    if isinstance(pipe.text_encoder, PeftModel):
        pipe.text_encoder.add_weighted_adapter(adapters, weights, adapter_name)

    return pipe



MODEL_NAME = "stabilityai/stable-diffusion-2-1"
# MODEL_NAME="runwayml/stable-diffusion-v1-5"

PEFT_TYPE="boft"
BLOCK_NUM=8
BLOCK_SIZE=0
N_BUTTERFLY_FACTOR=1
SELECTED_SUBJECT="backpack"
CLASS_TOKEN="backpack"
ITER = 200

PROJECT_NAME=f"dreambooth_{PEFT_TYPE}"
RUN_NAME=f"{SELECTED_SUBJECT}_{PEFT_TYPE}_{BLOCK_NUM}{BLOCK_SIZE}{N_BUTTERFLY_FACTOR}"
INSTANCE_DIR=f"./data/dreambooth/dataset/{SELECTED_SUBJECT}"
CLASS_DIR=f"./data/class_data/{CLASS_TOKEN}"
OUTPUT_DIR=f"./data/output/{PEFT_TYPE}"


pipe = get_lora_sd_pipeline(OUTPUT_DIR, MODEL_NAME, epoch=ITER, adapter_name=RUN_NAME)

load_adapter(pipe, OUTPUT_DIR, epoch=ITER, adapter_name=RUN_NAME)

# pipe = get_lora_sd_pipeline(os.path.join(base_path, f"data/output/{peft_type}"), MODEL_NAME, adapter_name="dog")

prompt = "qwe dog is on a wooden floor"
negative_prompt = "low quality, blurry, unfinished"

sys.exit()



pipe = get_lora_sd_pipeline(f"./data/output/{peft_type}", MODEL_NAME, epoch=picked_epoch, adapter_name="dog")
for dog_idx in [2, 3, 5, 6, 7, 8]:
    dog_lst.append(f"dog{dog_idx}")
    load_adapter(pipe, f"./data/output/{peft_type}", epoch=picked_epoch, adapter_name=f"dog{dog_idx}")

pbar = tqdm(list(itertools.product(dog_lst, repeat=2))[::-1])
for dog1, dog2 in pbar:
    new_dog_name = f"{dog1}_{dog2}"

    if dog1 == dog2:
        pipe.unet.set_adapter(dog1)
    else:
        pipe.unet.add_weighted_adapter([dog1, dog2], adapter_name=new_dog_name)
        pipe.unet.set_adapter(new_dog_name)

    for i in range(3):
        pbar.set_description(f"{new_dog_name}_{i}")
        image = pipe(prompt, num_inference_steps=50, guidance_scale=7, negative_prompt=negative_prompt).images[0]
        image.save(f"./data/output/mix/{new_dog_name}_{i}.png")
