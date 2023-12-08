import os
from glob import glob
from tqdm import tqdm
import numpy as np
from pathlib import Path
from skimage.io import imread
import cv2
import json
import argparse
import face_alignment
import torch
from torchvision.utils import save_image
from accelerate import Accelerator
from transformers import AutoTokenizer

from utils.dataset import make_dataset
from utils.args_loader import parse_args

detect_model = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D, device="cuda:0", flip_input=False
)

# with open('./data/celebhq-text/prompt_val_blip_full.json', 'rt') as f:    # fill50k, COCO
#     for line in f:
#         val_data = json.loads(line)

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


def generate_landmark2d(dataset, input_dir, save_dir, gt_lmk_folder, vis_dir=None):
    print(f'Generate 2d landmarks ...')
    os.makedirs(save_dir, exist_ok=True)

    imagepath_list = sorted(glob(f"{input_dir}/pred*.png"))

    for imagepath in tqdm(imagepath_list):
        name = Path(imagepath).stem
        idx = int(name.split('_')[-1])
        txt_path = os.path.join(save_dir, f'{idx}.txt')
        gt_lmk_path = os.path.join(gt_lmk_folder, f'{idx}_gt_lmk.jpg')

        if (not os.path.exists(txt_path)) or (not os.path.exists(gt_lmk_path)):
            image = imread(imagepath)[:, :, :3]
            out = detect_model.get_landmarks(image)
            if out is None:
                continue
            
            kpt = out[0].squeeze()
            np.savetxt(txt_path, kpt)

            # Your existing code for obtaining the image tensor
            gt_lmk_img = dataset[idx]['conditioning_pixel_values']
            save_image(gt_lmk_img, gt_lmk_path)
            # gt_image = cv2.resize(cv2.imread(gt_lmk_path), (512, 512))

            if vis_dir is not None:
                vis_path = os.path.join(vis_dir, f'{idx}_overlay.jpg')
                image = cv2.imread(imagepath)
                image_point = plot_kpts(image, kpt)
                gt_image = cv2.imread(gt_lmk_path)
                cv2.imwrite(vis_path, np.concatenate([image_point, gt_image], axis=1))
                sys.exit()


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


if __name__ == '__main__':
    args = parse_args()

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

    gt_lmk_dir = os.path.join(args.output_dir, "gt_lmk")
    if not os.path.exists(gt_lmk_dir):
        os.makedirs(gt_lmk_dir, exist_ok=True)

    pred_lmk_dir = os.path.join(args.output_dir, "pred_lmk")
    if not os.path.exists(pred_lmk_dir):
        os.makedirs(pred_lmk_dir, exist_ok=True)

    if args.vis_overlays:
        vis_dir = os.path.join(args.output_dir, "vis_overlays")
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir, exist_ok=True)
    else:
        vis_dir = None

    input_dir = os.path.join(args.output_dir, "results")

    generate_landmark2d(val_dataset, input_dir, pred_lmk_dir, gt_lmk_dir, vis_dir)
    
    if len(os.listdir(input_folder)) == len(val_dataset) and len(os.listdir(val_dataset)) == len(val_data):
        landmark_comparison(save_folder, gt_lmk_folder)
        print(args.method)