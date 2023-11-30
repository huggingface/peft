import cv2 as cv
import numpy as np
import os
from glob import glob
from tqdm import tqdm

root = "./data/output"

# out_dir = os.path.join(root, "collect")
# os.makedirs(out_dir, exist_ok=True)

# for boft_path in tqdm(glob(f"{root}/boft/validation/2000/*/*.png")):
#     class_name = boft_path.split("/")[-2]
#     oft_path = boft_path.replace("boft", "oft")
#     lora_path = boft_path.replace("boft", "lora")
#     if not os.path.exists(oft_path) or not os.path.exists(lora_path):
#         continue
#     pad_margin = ((0, 0), (20, 20), (0, 0))
#     boft_img = np.pad(cv.imread(boft_path, cv.IMREAD_UNCHANGED), pad_margin, mode="constant", constant_values=255)
#     oft_img = np.pad(cv.imread(oft_path, cv.IMREAD_UNCHANGED), pad_margin, mode="constant", constant_values=255)
#     lora_img = np.pad(cv.imread(lora_path, cv.IMREAD_UNCHANGED), pad_margin, mode="constant", constant_values=255)

#     img = np.concatenate([lora_img[:,20:], oft_img, boft_img[:,:-20]], axis=1)
#     out_path = os.path.join(out_dir, class_name, os.path.basename(boft_path))
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     cv.imwrite(out_path, img)

raw_dir = "./data/dreambooth/dataset"
in_dir = os.path.join(root, "mix")
out_dir = os.path.join(root, "mix_large")
os.makedirs(out_dir, exist_ok=True)

dog_lst = ['dog', 'dog2', 'dog3', 'dog5', 'dog6', 'dog7', 'dog8']
per_size = 260
large_img = np.ones(shape=((len(dog_lst) + 1) * per_size, (len(dog_lst) + 1) * per_size, 3)) * 255

for idx, dog in enumerate(dog_lst):
    raw_img = np.random.choice(glob(f"{raw_dir}/{dog}/*"))

    large_img[0:per_size, (idx + 1) * per_size:(idx + 2) * per_size] = np.pad(
        cv.resize(cv.imread(raw_img, cv.IMREAD_UNCHANGED), (256, 256)), ((0, 4), (0, 4), (0, 0)),
        mode="constant",
        constant_values=255
    )

    large_img[(idx + 1) * per_size:(idx + 2) * per_size, 0:per_size] = np.pad(
        cv.resize(cv.imread(raw_img, cv.IMREAD_UNCHANGED), (256, 256)), ((0, 4), (0, 4), (0, 0)),
        mode="constant",
        constant_values=255
    )

for idx1, dog1 in enumerate(dog_lst):
    for idx2, dog2 in enumerate(dog_lst):
        all_files = glob(f"{in_dir}/{dog1}_{dog2}_*")
        if len(all_files) > 0:
            img_file = np.random.choice(all_files)
            large_img[(idx1 + 1) * per_size:(idx1 + 2) * per_size,
                      (idx2 + 1) * per_size:(idx2 + 2) * per_size] = np.pad(
                          cv.resize(cv.imread(img_file, cv.IMREAD_UNCHANGED), (256, 256)),
                          ((0, 4), (0, 4), (0, 0)),
                          mode="constant",
                          constant_values=255
                      )
cv.imwrite(f"{out_dir}/mosaic.png", large_img)
