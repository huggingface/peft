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

import glob
import os
from pathlib import Path

import cv2
import face_alignment
import numpy as np
import torch
from accelerate import Accelerator
from skimage.io import imread
from torchvision.utils import save_image
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.args_loader import parse_args
from utils.dataset import make_dataset


detect_model = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device="cuda:0", flip_input=False)

# with open('./data/celebhq-text/prompt_val_blip_full.json', 'rt') as f:    # fill50k, COCO
#     for line in f:
#         val_data = json.loads(line)

end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype=np.int32) - 1


def count_txt_files(directory):
    pattern = os.path.join(directory, "*.txt")
    txt_files = glob.glob(pattern)
    return len(txt_files)


def plot_kpts(image, kpts, color="g"):
    """Draw 68 key points
    Args:
        image: the input image
        kpt: (68, 3).
    """
    if color == "r":
        c = (255, 0, 0)
    elif color == "g":
        c = (0, 255, 0)
    elif color == "b":
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
        image = cv2.line(image, (int(st[0]), int(st[1])), (int(ed[0]), int(ed[1])), (255, 255, 255), radius)
    return image


def generate_landmark2d(dataset, input_dir, pred_lmk_dir, gt_lmk_dir, vis=False):
    print("Generate 2d landmarks ...")
    os.makedirs(pred_lmk_dir, exist_ok=True)

    imagepath_list = sorted(glob.glob(f"{input_dir}/pred*.png"))

    for imagepath in tqdm(imagepath_list):
        name = Path(imagepath).stem
        idx = int(name.split("_")[-1])
        pred_txt_path = os.path.join(pred_lmk_dir, f"{idx}.txt")
        gt_lmk_path = os.path.join(gt_lmk_dir, f"{idx}_gt_lmk.jpg")
        gt_txt_path = os.path.join(gt_lmk_dir, f"{idx}.txt")
        gt_img_path = os.path.join(gt_lmk_dir, f"{idx}_gt_img.jpg")

        if (not os.path.exists(pred_txt_path)) or (not os.path.exists(gt_txt_path)):
            image = imread(imagepath)  # [:, :, :3]
            out = detect_model.get_landmarks(image)
            if out is None:
                continue

            pred_kpt = out[0].squeeze()
            np.savetxt(pred_txt_path, pred_kpt)

            # Your existing code for obtaining the image tensor
            gt_lmk_img = dataset[idx]["conditioning_pixel_values"]
            save_image(gt_lmk_img, gt_lmk_path)

            gt_img = (dataset[idx]["pixel_values"]) * 0.5 + 0.5
            save_image(gt_img, gt_img_path)

            gt_img = (gt_img.permute(1, 2, 0) * 255).type(torch.uint8).cpu().numpy()
            out = detect_model.get_landmarks(gt_img)
            if out is None:
                continue

            gt_kpt = out[0].squeeze()
            np.savetxt(gt_txt_path, gt_kpt)
            # gt_image = cv2.resize(cv2.imread(gt_lmk_path), (512, 512))

            if vis:
                gt_lmk_image = cv2.imread(gt_lmk_path)

                # visualize predicted landmarks
                vis_path = os.path.join(pred_lmk_dir, f"{idx}_overlay.jpg")
                image = cv2.imread(imagepath)
                image_point = plot_kpts(image, pred_kpt)
                cv2.imwrite(vis_path, np.concatenate([image_point, gt_lmk_image], axis=1))

                # visualize gt landmarks
                vis_path = os.path.join(gt_lmk_dir, f"{idx}_overlay.jpg")
                image = cv2.imread(gt_img_path)
                image_point = plot_kpts(image, gt_kpt)
                cv2.imwrite(vis_path, np.concatenate([image_point, gt_lmk_image], axis=1))


def landmark_comparison(val_dataset, lmk_dir, gt_lmk_dir):
    print("Calculating reprojection error")
    lmk_err = []

    pbar = tqdm(range(len(val_dataset)))
    for i in pbar:
        # line = val_dataset[i]
        # img_name = line["image"].split(".")[0]
        lmk1_path = os.path.join(gt_lmk_dir, f"{i}.txt")
        lmk1 = np.loadtxt(lmk1_path)
        lmk2_path = os.path.join(lmk_dir, f"{i}.txt")

        if not os.path.exists(lmk2_path):
            print(f"{lmk2_path} not exist")
            continue

        lmk2 = np.loadtxt(lmk2_path)
        lmk_err.append(np.mean(np.linalg.norm(lmk1 - lmk2, axis=1)))
        pbar.set_description(f"lmk_err: {np.mean(lmk_err):.5f}")

    print("Reprojection error:", np.mean(lmk_err))
    np.save(os.path.join(lmk_dir, "lmk_err.npy"), lmk_err)


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

    gt_lmk_dir = os.path.join(args.output_dir, "gt_lmk")
    if not os.path.exists(gt_lmk_dir):
        os.makedirs(gt_lmk_dir, exist_ok=True)

    pred_lmk_dir = os.path.join(args.output_dir, "pred_lmk")
    if not os.path.exists(pred_lmk_dir):
        os.makedirs(pred_lmk_dir, exist_ok=True)

    input_dir = os.path.join(args.output_dir, "results")

    generate_landmark2d(val_dataset, input_dir, pred_lmk_dir, gt_lmk_dir, args.vis_overlays)

    if count_txt_files(pred_lmk_dir) == len(val_dataset) and count_txt_files(gt_lmk_dir) == len(val_dataset):
        landmark_comparison(val_dataset, pred_lmk_dir, gt_lmk_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
