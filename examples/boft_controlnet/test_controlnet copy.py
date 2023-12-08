import argparse
import sys
import os
import time
import numpy as np
from tqdm import tqdm
import torch
import contextlib
from pathlib import Path
from safetensors.torch import load_file
import torch.utils.checkpoint
from accelerate import Accelerator
from diffusers import UniPCMultistepScheduler, DDIMScheduler
from diffusers.utils import check_min_version

from utils.dataset import make_dataset
from utils.light_controlnet import ControlNetModel
from utils.pipeline_controlnet import LightControlNetPipeline
from utils.unet_2d_condition import UNet2DConditionNewModel
from utils.args_loader import parse_args

from transformers import AutoTokenizer

sys.path.append('../../src')
from peft import PeftModel

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")
device = torch.device("cuda:0")


detect_model = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D, device="cuda:0", flip_input=False
)

end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype=np.int32) - 1


def plot_kpts(image, kpts, color='g'):
    ''' Draw 68 key points
    Args: 
        image: the input image
        kpt: (68, 3).
    '''
    if color == 'r':
        c = (255, 0, 0)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (255, 0, 0)
    image = image.copy()
    kpts = kpts.copy()
    radius = max(int(min(image.shape[0], image.shape[1]) / 200), 1)
    for i in range(kpts.shape[0]):
        st = kpts[i, :2]
        if kpts.shape[1] == 4:
            if kpts[i, 3] > 0.5:
                c = (0, 255, 0)
            else:
                c = (0, 0, 255)
        image = cv2.circle(image, (int(st[0]), int(st[1])), radius, c, radius * 2)
        if i in end_list:
            continue
        ed = kpts[i + 1, :2]
        image = cv2.line(
            image, (int(st[0]), int(st[1])), (int(ed[0]), int(ed[1])), (255, 255, 255), radius
        )
    return image


def generate_landmark2d(inputpath, savepath, gt_lmk_folder, vis=False):

    # print(f'generate 2d landmarks')
    os.makedirs(savepath, exist_ok=True)

    imagepath_list = sorted(glob(f"{inputpath}/pred*.png"))

    for imagepath in tqdm(imagepath_list):

        name = Path(imagepath).stem
        idx = val_data[int(name.split('_')[-1])]["image"].split(".")[0]
        txt_path = os.path.join(savepath, f'{idx}.txt')
        overlap_path = os.path.join(savepath, f'{idx}_overlay.jpg')

        if (not os.path.exists(txt_path)) or (not os.path.exists(overlap_path)):

            image = imread(imagepath)[:, :, :3]
            out = detect_model.get_landmarks(image)
            if out is None:
                continue
            
            kpt = out[0].squeeze()
            np.savetxt(txt_path, kpt)

            if vis:
                image = cv2.imread(imagepath)
                image_point = plot_kpts(image, kpt)
                gt_image = cv2.resize(cv2.imread(os.path.join(gt_lmk_folder, f"{idx}_overlay.jpg")), (512, 512))
                cv2.imwrite(overlap_path, np.concatenate([image_point, gt_image], axis=1))


def landmark_comparison(lmk_folder, gt_lmk_folder):
    # print(f'calculate reprojection error')
    lmk_err = []

    pbar = tqdm(range(len(val_data)))
    for i in pbar:

        line = val_data[i]
        img_name = line["image"].split(".")[0]
        lmk1_path = os.path.join(gt_lmk_folder, f'{img_name}.txt')
        lmk1 = np.loadtxt(lmk1_path) / 2
        lmk2_path = os.path.join(lmk_folder, f'{img_name}.txt')
        if not os.path.exists(lmk2_path):
            print(f'{lmk2_path} not exist')
            continue
        lmk2 = np.loadtxt(lmk2_path)
        lmk_err.append(np.mean(np.linalg.norm(lmk1 - lmk2, axis=1)))
        pbar.set_description(f'lmk_err: {np.mean(lmk_err):.5f}')

    # print(np.mean(lmk_err))
    np.save(os.path.join(lmk_folder, 'lmk_err.npy'), lmk_err)


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
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, revision=args.revision, use_fast=False
        )
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
        requires_safety_checker=False
    ).to(device)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    gt_lmk_dir = os.path.join(args.output_dir, "gt_lmk")
    if not os.path.exists(gt_lmk_dir):
        os.makedirs(gt_lmk_dir, exist_ok=True)

    pred_lmk_dir = os.path.join(args.output_dir, "pred_lmk")
    if not os.path.exists(pred_lmk_dir):
        os.makedirs(pred_lmk_dir, exist_ok=True)        

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
                    data["text"], [data["conditioning_pixel_values"]],
                    num_inference_steps=50,
                    guidance_scale=7,
                    negative_prompt=negative_prompt
                ).images[0]

            pred_img.save(output_path)

            generate_landmark2d(pred_lmk_dir, gt_lmk_dir, vis=False)

    # control_img = Image.fromarray(
    #     (data["conditioning_pixel_value"] * 255).numpy().transpose(1, 2, 0).astype(np.uint8)
    # )
    # gt_img = Image.fromarray(
    #     ((data["pixel_value"] + 1.0) * 0.5 * 255).numpy().transpose(1, 2, 0).astype(np.uint8)
    # )

    if len(os.listdir(pred_lmk_dir)) == len(val_dataset) and len(os.listdir(gt_lmk_dir)) == len(val_dataset):
        landmark_comparison(pred_lmk_dir, gt_lmk_folder)
        print(args.method)


if __name__ == "__main__":
    args = parse_args()
    main(args)