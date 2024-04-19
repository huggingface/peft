# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The implementation is based on "Parameter-Efficient Orthogonal Finetuning
# via Butterfly Factorization" (https://arxiv.org/abs/2311.06243) in ICLR 2024.

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from diffusers import DDIMScheduler
from diffusers.utils import check_min_version
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.args_loader import parse_args
from utils.dataset import make_dataset
from utils.light_controlnet import ControlNetModel
from utils.pipeline_controlnet import LightControlNetPipeline
from utils.unet_2d_condition import UNet2DConditionNewModel


sys.path.append("../../src")
from peft import PeftModel


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")
device = torch.device("cuda:0")


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    val_dataset = make_dataset(args, tokenizer, accelerator, "test")

    controlnet_path = args.controlnet_path
    unet_path = args.unet_path

    controlnet = ControlNetModel()
    controlnet.load_state_dict(load_file(controlnet_path))
    unet = UNet2DConditionNewModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    unet = PeftModel.from_pretrained(unet, unet_path, adapter_name=args.adapter_name)

    pipe = LightControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=controlnet,
        unet=unet.model,
        torch_dtype=torch.float32,
        requires_safety_checker=False,
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
            data = val_dataset[idx.item()]
            negative_prompt = "low quality, blurry, unfinished"

            with torch.no_grad():
                pred_img = pipe(
                    data["text"],
                    [data["conditioning_pixel_values"]],
                    num_inference_steps=50,
                    guidance_scale=7,
                    negative_prompt=negative_prompt,
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
