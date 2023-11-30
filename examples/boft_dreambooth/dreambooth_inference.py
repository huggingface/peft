import itertools
import os
import sys

import torch
import torch.utils.checkpoint
from diffusers import StableDiffusionPipeline
from diffusers.utils import check_min_version
from tqdm import tqdm

sys.path.append("./peft/src")
from peft import PeftModel

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")


def get_lora_sd_pipeline(
    ckpt_dir,
    base_model_name_or_path,
    epoch,
    dtype=torch.float32,
    device="cuda",
    adapter_name="default"
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


MODEL_NAME = "stabilityai/stable-diffusion-2-1"
picked_epoch = 2000
peft_type = "boft"
dog_lst = ['dog']

prompt = "qwe dog is on a wooden table"
negative_prompt = "low quality, blurry, unfinished"

pipe = get_lora_sd_pipeline(
    f"./data/output/{peft_type}", MODEL_NAME, epoch=picked_epoch, adapter_name="dog"
)
for dog_idx in [2,3,5,6,7,8]:
    dog_lst.append(f"dog{dog_idx}")
    load_adapter(
        pipe, f"./data/output/{peft_type}", epoch=picked_epoch, adapter_name=f"dog{dog_idx}"
    )

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
        image = pipe(
            prompt, num_inference_steps=50, guidance_scale=7, negative_prompt=negative_prompt
        ).images[0]
        image.save(f"./data/output/mix/{new_dog_name}_{i}.png")
        

