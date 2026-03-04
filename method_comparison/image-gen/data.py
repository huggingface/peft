# Copyright 2026-present the HuggingFace Inc. team.
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

"""Data handling for the image generation benchmark."""

import numpy as np
import torch
import torchvision.transforms as T
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoImageProcessor


class DreamBoothTrainDataset(Dataset):
    def __init__(
        self,
        images: list[Image.Image],
        prompts: list[str],
        repeats: int,
        transforms: AutoImageProcessor | None,
    ):
        self.images = images
        self.prompts = prompts
        self.repeats = repeats
        self.transforms = transforms

    def __len__(self):
        return len(self.images) * self.repeats

    def __getitem__(self, idx):
        index = idx % len(self.images)
        image = self.images[index]
        prompt = self.prompts[index]

        if self.transforms is None:
            # for valid/test, rely on Dino image processor
            return {
                "raw_image": image,
                "prompt": prompt,
            }

        output = self.transforms(image)
        return {
            "pixel_values": output,
            "prompt": prompt,
        }


def _to_rgb(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(image).convert("RGB")


def get_train_valid_test_datasets(*, train_config, print_fn=print):
    ds = load_dataset(train_config.dataset_id, split=train_config.dataset_split)
    image_column = train_config.image_column

    if image_column not in ds.column_names:
        raise ValueError(f"Column '{image_column}' not found in dataset {train_config.dataset_id}: {ds.column_names}")
    if len(ds) != len(train_config.instance_prompts):
        raise ValueError(
            f"Need 1 instance prompt per sample image, found {len(train_config.instance_prompts)} and "
            f"{len(ds)} instead."
        )

    train_size = len(ds) - train_config.valid_size - train_config.test_size
    if train_size < 1:
        raise ValueError(
            f"Dataset too small: need at least {1 + train_config.valid_size + train_config.test_size} rows, "
            f"found {len(ds)}"
        )

    np.random.seed(0)
    indices = np.arange(len(ds))
    np.random.shuffle(indices)

    idx_train = indices[:train_size]
    idx_valid = indices[train_size : train_size + train_config.valid_size]
    idx_test = indices[
        train_size + train_config.valid_size : train_size + train_config.valid_size + train_config.test_size
    ]

    ds_train = ds.select(idx_train)
    ds_valid = ds.select(idx_valid)
    ds_test = ds.select(idx_test)

    train_images = [_to_rgb(img) for img in ds_train[image_column]]
    valid_images = [_to_rgb(img) for img in ds_valid[image_column]]
    test_images = [_to_rgb(img) for img in ds_test[image_column]]

    train_prompts = [train_config.instance_prompts[i] for i in idx_train]
    valid_prompts = [train_config.instance_prompts[i] for i in idx_valid]
    test_prompts = [train_config.instance_prompts[i] for i in idx_test]

    # FIXME
    random_crop = False
    random_flip = False
    train_augmentations = T.Compose(
        [
            T.Resize(train_config.resolution, interpolation=T.InterpolationMode.BILINEAR),
            T.RandomCrop(train_config.resolution) if random_crop else T.CenterCrop(train_config.resolution),
            T.RandomHorizontalFlip(p=0.5) if random_flip else T.Lambda(lambda image: image),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]
    )

    train_dataset = DreamBoothTrainDataset(
        images=train_images,
        prompts=train_prompts,
        repeats=train_config.repeats,
        transforms=train_augmentations,
    )
    valid_dataset = DreamBoothTrainDataset(
        images=valid_images,
        prompts=valid_prompts,
        repeats=1,
        transforms=None,
    )
    test_dataset = DreamBoothTrainDataset(
        images=test_images,
        prompts=test_prompts,
        repeats=1,
        transforms=None,
    )

    print_fn(f"Dataset: {train_config.dataset_id}")
    print_fn(f"Raw rows: {len(ds)}")
    print_fn(f"Train rows: {len(train_dataset)}")
    print_fn(f"Valid rows: {len(valid_dataset)}")
    print_fn(f"Test rows: {len(test_dataset)}")

    return train_dataset, valid_dataset, test_dataset


def collate_fn(samples):
    pixel_values = torch.stack([sample["pixel_values"] for sample in samples])
    prompts = [sample["prompt"] for sample in samples]
    return {
        "pixel_values": pixel_values,
        "prompts": prompts,
    }
