import argparse
import sys
import os
import time
import numpy as np
from tqdm import tqdm
import torch
import contextlib
from safetensors.torch import load_file
import torch.utils.checkpoint
from diffusers import UniPCMultistepScheduler, DDIMScheduler
from diffusers.utils import check_min_version

from utils.light_controlnet import ControlNetModel
from utils.pipeline_controlnet import LightControlNetPipeline
from utils.unet_2d_condition import UNet2DConditionNewModel

sys.path.append("./peft/src")
from peft import PeftModel

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")
device = torch.device("cuda:0")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet testing script.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--controlnet_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained controlnet."
    )
    parser.add_argument(
        "--unet_path", type=str, default=None, required=True, help="Path to pretrained unet."
    )
    parser.add_argument(
        "--adapter_name", type=str, default=None, required=True, help="Name of the adapter to use."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, required=True, help="Path to output directory."
    )
    parser.add_argument(
        "--dataset_name", type=str, default=None, required=True, help="Name of the dataset."
    )

    return parser.parse_args(input_args)


def main(args):
    
    if args.dataset_name == "deepfashion":
        from utils.dataset import \
            DeepFashionDenseposeDataset as ControlNetDataset
    elif args.dataset_name == "ade20k":
        from utils.dataset import \
            ADE20kSegmDataset as ControlNetDataset
    elif args.dataset_name == "celebhq":
        from utils.dataset import \
            CelebHQDataset as ControlNetDataset

    val_dataset = ControlNetDataset(split="val", resolution=512, full=True)

    controlnet_path = args.controlnet_path
    unet_path = args.unet_path

    controlnet = ControlNetModel()
    controlnet.load_state_dict(load_file(controlnet_path))
    unet = UNet2DConditionNewModel.from_pretrained(args.model_name, subfolder="unet")
    unet = PeftModel.from_pretrained(unet, unet_path, adapter_name=args.adapter_name)

    pipe = LightControlNetPipeline.from_pretrained(
        args.model_name,
        controlnet=controlnet,
        unet=unet.model,
        torch_dtype=torch.float32,
        requires_safety_checker=False
    ).to(device)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)

    exist_lst = [int(img.split("_")[-1][:-4]) for img in os.listdir(args.output_dir)]
    all_lst = np.arange(len(val_dataset))
    idx_lst = [item for item in all_lst if item not in exist_lst]
    
    print("Number of images to be processed: ", len(idx_lst))
    
    np.random.seed(seed=int(time.time()))
    np.random.shuffle(idx_lst)
    
    for idx in tqdm(idx_lst):

        output_path = os.path.join(args.output_dir, f"pred_img_{idx:04d}.png")

        if not os.path.exists(output_path):
            
            data = val_dataset[idx]
            negative_prompt = "low quality, blurry, unfinished"

            with torch.no_grad():
                pred_img = pipe(
                    data["caption"], [data["conditioning_pixel_value"]],
                    num_inference_steps=50,
                    guidance_scale=7,
                    negative_prompt=negative_prompt
                ).images[0]

            pred_img.save(output_path)

    # control_img = Image.fromarray(
    #     (data["conditioning_pixel_value"] * 255).numpy().transpose(1, 2, 0).astype(np.uint8)
    # )
    # gt_img = Image.fromarray(
    #     ((data["pixel_value"] + 1.0) * 0.5 * 255).numpy().transpose(1, 2, 0).astype(np.uint8)
    # )


if __name__ == "__main__":
    args = parse_args()
    main(args)