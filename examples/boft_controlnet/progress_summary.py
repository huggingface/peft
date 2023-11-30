import os
import argparse
import json
from termcolor import colored

if __name__ == '__main__':
    
    # create parser
    parser = argparse.ArgumentParser(description='file number comparison')
    parser.add_argument('--method', type=str, default='celebhq_boft_3205_stable', help='method name')
    args = parser.parse_args()
    
    dataset, method, config, version = args.method.split('_')
    
    json_data = []
    
    if dataset == 'celebhq':
        data_dir = './data/celebhq-text'
        json_path = f"{data_dir}/prompt_val_blip_full.json"
        ckpt_lst = [9000, 17000, 25000, 33000, 41000, 49000, 57000]
        with open(json_path, 'rt') as f:    # fill50k, COCO
            for line in f:
                json_data = json.loads(line)
    elif dataset == 'ade20k':
        data_dir = './data/ADE20K'
        json_path = f"{data_dir}/val/prompt_val_blip_full.json"
        ckpt_lst = [7000, 14000, 21000, 27000, 33000]
        with open(json_path, 'rt') as f:    # fill50k, COCO
            for line in f:
                json_data.append(json.loads(line))
    
    for iter in ckpt_lst:
        
        result_dir = f"./data/output/{dataset}/{method}_{config}_{version}/checkpoint-{iter}/results"
        
        if os.path.exists(result_dir):
            total_num = len(json_data)
            current_num = len(os.listdir(result_dir))
            tag = "green" if current_num == total_num else "red"
            print(colored(f"{args.method}-{iter}: {current_num}/{total_num}", tag))
        else:
            print(colored(f"First run inference on {args.method}-{iter} :)", "red"))
    
    