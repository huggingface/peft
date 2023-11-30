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

detect_model = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D, device="cuda:0", flip_input=False
)

with open('./data/celebhq-text/prompt_val_blip_full.json', 'rt') as f:    # fill50k, COCO
        for line in f:
            val_data = json.loads(line)

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

if __name__ == '__main__':
    
    # create parser
    parser = argparse.ArgumentParser(description='landmark generation')
    parser.add_argument('--method', type=str, default='boft_3205_25000_stable', help='method name')
    args = parser.parse_args()

    dataset = 'celebhq'
    method, config, iter, version = args.method.split('_')
    ckpt_dir = f"{method}_{config}_{version}/checkpoint-{iter}"

    gt_lmk_folder = './data/celebhq-text/celeba-hq-landmark2d'
    input_folder = f"./data/output/{dataset}/{ckpt_dir}/results"
    save_folder = f"./data/output/{dataset}/{ckpt_dir}/landmarks"

    generate_landmark2d(input_folder, save_folder, gt_lmk_folder, vis=True)
    
    if len(os.listdir(input_folder)) == len(val_data):
        landmark_comparison(save_folder, gt_lmk_folder)
        print(args.method)