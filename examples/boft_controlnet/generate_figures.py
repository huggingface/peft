import os
import json
from tqdm import tqdm
import numpy as np
from PIL import Image
from shutil import copyfile

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

if __name__ == '__main__':

    json_data = []
    lmk_err = {}
    data_dir = './data/celebhq-text'
    json_path = f"{data_dir}/prompt_val_blip_full.json"
    gt_lmk_folder = f"{data_dir}/celeba-hq-landmark2d"
    err_path = f'./data/output/celebhq/lmk_err.npy'

    with open(json_path, 'rt') as f:    # fill50k, COCO
        for line in f:
            json_data = json.loads(line)

    if not os.path.exists(err_path):
        for method in ['boft_3201', 'oft_1600', 'lora_1600']:

            for ckpt_name in [9000, 17000, 25000, 33000]:

                result_dir = f"./data/output/celebhq/{method}_stable/checkpoint-{ckpt_name}/results"
                lmks_dir = f"./data/output/celebhq/{method}_stable/checkpoint-{ckpt_name}/landmarks"

                lmk_err[method] = {ckpt_name: []}

                pbar = tqdm(range(len(json_data)))

                for idx in pbar:
                    img_name = json_data[int(idx)]["image"].split(".")[0]

                    lmk1_path = os.path.join(gt_lmk_folder, f'{img_name}.txt')
                    lmk2_path = os.path.join(lmks_dir, f'{img_name}.txt')

                    if not os.path.exists(lmk2_path):
                        lmk_err[method][ckpt_name].append(10**5)
                    else:
                        lmk1 = np.loadtxt(lmk1_path) / 2
                        lmk2 = np.loadtxt(lmk2_path)
                        lmk_err[method][ckpt_name].append(np.mean(np.linalg.norm(lmk1 - lmk2, axis=1)))

        np.save('./data/output/celebhq/lmk_err.npy', lmk_err)
    else:
        lmk_err = np.load(err_path, allow_pickle=True).item()
        
    subject_err = []
    
    for subject in range(len(json_data)):
        subject_per_err = 0.0
        boft_err = lmk_err['boft_3201'][33000][subject]
        oft_err = lmk_err['oft_1600'][33000][subject]
        lora_err = lmk_err['lora_1600'][33000][subject]
        if boft_err != 10**5 and oft_err != 10**5 and lora_err != 10**5 and boft_err < oft_err and boft_err < lora_err:
            oft_boft_diff = oft_err - boft_err
            lora_boft_diff = lora_err - boft_err
            if abs(oft_boft_diff - lora_boft_diff) < 5:
                subject_per_err = oft_boft_diff + lora_boft_diff
                subject_err.append(subject_per_err)
            else:
                subject_err.append(0)
        else:
            subject_err.append(0)
    
    top_n = 40
    top_idxs = np.array(subject_err).argsort()[::-1][:top_n]
    
    with open(f'./figures/top_{top_n}_subjects.txt', 'w') as f:
    
        for idx in top_idxs:
            all_imgs = []
            img_name = json_data[int(idx)]["image"].split(".")[0]
            # for ckpt_name in [9000, 17000, 25000, 33000]:
            for ckpt_name in [33000]:
                
                for method in ['boft_3201', 'oft_1600', 'lora_1600']:
                    
                    gen_img = f"./data/output/celebhq/{method}_stable/checkpoint-{ckpt_name}/results/pred_img_{idx:04d}.png"
                    lmk_img = f"./data/celebhq-text/celeba-hq-landmark2d_cond_512//{img_name}.png"
                    overlay_img = f"./data/output/celebhq/{method}_stable/checkpoint-{ckpt_name}/landmarks/{img_name}_overlay.jpg"
                    gt_img = f"./data/celebhq-text/celeba-hq-img/{img_name}.jpg"
                    
                    copyfile(src=gen_img, dst=f"./figures/raw/{method}_{ckpt_name}_{img_name}_gen.png")
                    copyfile(src=overlay_img, dst=f"./figures/raw/{method}_{ckpt_name}_{img_name}_overlay.png")
                    copyfile(src=gt_img, dst=f"./figures/raw/{method}_{ckpt_name}_{img_name}_gt.png")
                    copyfile(src=lmk_img, dst=f"./figures/raw/{method}_{ckpt_name}_{img_name}_lmk.png")
                    all_imgs.append(Image.open(overlay_img).resize((1024,512)))   
        
            f.write(f"{img_name}    {json_data[int(idx)]['prompt']}\n")
            # creates a new empty image, RGB mode, and size 444 by 95
            new_im = Image.new('RGB', (1024, 512*len(all_imgs)))

            for i, elem in enumerate(all_imgs):
                    new_im.paste(elem, (0,i*512))
            new_im.save(f'./figures/{img_name}.jpg')
        
    
    
    
