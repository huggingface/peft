import json
import os
import torch
import wandb
from diffusers import UniPCMultistepScheduler, DDIMScheduler
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.pipeline_controlnet import LightControlNetPipeline


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


class DeepFashionDenseposeDataset(Dataset):
    def __init__(self, split='train', resolution=512, full=False):

        self.data = []
        self.split = split
        self.resolution = resolution

        if full:
            json_path = './data/DeepFashion/{}/prompt_{}_blip_full.json'.format(split, split)
        else:
            json_path = './data/DeepFashion/{}/prompt_{}_blip.json'.format(split, split)

        with open(json_path, 'rt') as f:    # fill50k, COCO
            for line in f:
                self.data.append(json.loads(line))

        self.image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.conditioning_image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        prompt = item['prompt']

        source = Image.open(
            './data/DeepFashion/{}/densepose/'.format(self.split) + source_filename[7:]
        ).convert("RGB")
        target = Image.open(
            './data/DeepFashion/{}/color/'.format(self.split) + source_filename[7:]
        ).convert("RGB")

        image = self.image_transforms(target)
        conditioning_image = self.conditioning_image_transforms(source)

        return dict(
            pixel_value=image,
            conditioning_pixel_value=conditioning_image,
            caption=prompt,
        )


class ADE20kSegmDataset(Dataset):
    def __init__(self, split='train', resolution=512, full=False):

        self.data = []
        self.split = split
        self.resolution = resolution

        if full:
            json_path = './data/ADE20K/{}/prompt_{}_blip_full.json'.format(split, split)
        else:
            json_path = './data/ADE20K/{}/prompt_{}_blip.json'.format(split, split)

        with open(json_path, 'rt') as f:    # fill50k, COCO
            for line in f:
                self.data.append(json.loads(line))

        self.image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.conditioning_image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]
        source_filename = item['source']
        prompt = item['prompt']

        source = Image.open('./data/ADE20K/{}/segm/'.format(self.split) +
                            source_filename[7:]).convert("RGB")
        target = Image.open('./data/ADE20K/{}/color/'.format(self.split) +
                            source_filename[7:]).convert("RGB")

        image = self.image_transforms(target)
        conditioning_image = self.conditioning_image_transforms(source)

        return dict(
            pixel_value=image,
            conditioning_pixel_value=conditioning_image,
            caption=prompt,
        )


class CelebHQDataset(Dataset):
    def __init__(self, split='train', resolution=512, full=False):

        self.data = []
        self.split = split
        self.resolution = resolution

        if full:
            json_path = './data/celebhq-text/prompt_{}_blip_full.json'.format(split)
        else:
            json_path = './data/celebhq-text/prompt_{}_blip.json'.format(split)

        with open(json_path, 'rt') as f:    # fill50k, COCO
            for line in f:
                self.data.append(json.loads(line))

        self.image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.conditioning_image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ])

        self.data = self.data[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['image']
        prompt = item['prompt']

        source_path = './data/celebhq-text/celeba-hq-landmark2d_cond_512/' + \
            source_filename[:-4] + '.png'
        target_path = './data/celebhq-text/celeba-hq-img/' + source_filename

        source = Image.open(source_path).convert("RGB")
        target = Image.open(target_path).convert("RGB")

        image = self.image_transforms(target)
        conditioning_image = self.conditioning_image_transforms(source)

        return dict(
            pixel_value=image,
            conditioning_pixel_value=conditioning_image,
            caption=prompt,
        )


def collate_fn(examples, tokenizer):
    pixel_values = [example["pixel_value"] for example in examples]
    conditioning_pixel_values = [example["conditioning_pixel_value"] for example in examples]
    captions = [example["caption"] for example in examples]

    input_ids = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack(conditioning_pixel_values)
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format
                                                            ).float()

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
    }

    return batch