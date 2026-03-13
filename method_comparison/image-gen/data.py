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
import torchvision.transforms as T
from datasets import load_dataset
from PIL import Image
from PIL.ImageOps import exif_transpose


def _to_rgb(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(image).convert("RGB")


def _build_train_pixel_values(images: list[Image.Image], resolution: int):
    size = resolution, resolution  # hard-code square
    train_augmentations = T.Compose(
        [
            T.Resize(size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]
    )
    return [train_augmentations(exif_transpose(image)) for image in images]


def get_train_valid_test_datasets(*, train_config, print_fn=print):
    ds = load_dataset(train_config.dataset_id, split=train_config.dataset_split)
    image_column = train_config.image_column

    train_size = len(ds) - train_config.valid_size - train_config.test_size

    prompts = train_config.instance_prompts
    if isinstance(prompts, str):
        prompts = [prompts] * len(ds)
    else:
        if len(ds) != len(prompts):
            raise ValueError(f"Need 1 instance prompt per sample image, found {len(prompts)} and {len(ds)} instead.")

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

    train_prompts = [prompts[i] for i in idx_train]
    valid_prompts = [prompts[i] for i in idx_valid]
    test_prompts = [prompts[i] for i in idx_test]

    train_dataset = {
        "pixel_values": _build_train_pixel_values(train_images, train_config.resolution),
        "prompts": train_prompts,
        "repeats": train_config.repeats,
    }
    valid_dataset = [
        {"raw_image": exif_transpose(image), "prompt": prompt} for image, prompt in zip(valid_images, valid_prompts)
    ]
    test_dataset = [
        {"raw_image": exif_transpose(image), "prompt": prompt} for image, prompt in zip(test_images, test_prompts)
    ]

    print_fn(f"Dataset: {train_config.dataset_id}")
    print_fn(f"Raw rows: {len(ds)}")
    print_fn(f"Train rows: {len(train_dataset['prompts']) * train_dataset['repeats']}")
    print_fn(f"Valid rows: {len(valid_dataset)}")
    print_fn(f"Test rows: {len(test_dataset)}")

    return train_dataset, valid_dataset, test_dataset
