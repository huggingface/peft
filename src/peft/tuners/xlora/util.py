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

import os
from typing import Optional, Union

import torch
from huggingface_hub import file_exists, hf_hub_download  # type: ignore
from huggingface_hub.utils import EntryNotFoundError  # type: ignore
from safetensors.torch import load_file as safe_load_file

from peft.utils.other import (
    infer_device,
)


def _load_classifier_weights(model_id: str, device: Optional[str] = None, **hf_hub_download_kwargs) -> dict:
    r"""
    A helper method to load the classifier weights from the HuggingFace Hub or locally.
    This is essentially `load_peft_weights`, but with the safetensors names changed.

    Args:
        model_id (`str`):
            The local path to the adapter weights or the name of the adapter to load from the HuggingFace Hub.
        device (`str`):
            The device to load the weights onto.
        hf_hub_download_kwargs (`dict`):
            Additional arguments to pass to the `hf_hub_download` method when loading from the HuggingFace Hub.
    """
    path = (
        os.path.join(model_id, hf_hub_download_kwargs["subfolder"])
        if hf_hub_download_kwargs.get("subfolder", None) is not None
        else model_id
    )

    SAFETENSORS_WEIGHTS_NAME = "xlora_classifier.safetensors"
    WEIGHTS_NAME = "xlora_classifier.pt"

    if device is None:
        device = infer_device()

    if os.path.exists(os.path.join(path, SAFETENSORS_WEIGHTS_NAME)):
        filename = os.path.join(path, SAFETENSORS_WEIGHTS_NAME)
        use_safetensors = True
    elif os.path.exists(os.path.join(path, WEIGHTS_NAME)):
        filename = os.path.join(path, WEIGHTS_NAME)
        use_safetensors = False
    else:
        token = hf_hub_download_kwargs.get("token", None)
        if token is None:
            token = hf_hub_download_kwargs.get("use_auth_token", None)

        hub_filename = (
            os.path.join(hf_hub_download_kwargs["subfolder"], SAFETENSORS_WEIGHTS_NAME)
            if hf_hub_download_kwargs.get("subfolder", None) is not None
            else SAFETENSORS_WEIGHTS_NAME
        )
        has_remote_safetensors_file = file_exists(
            repo_id=model_id,
            filename=hub_filename,
            revision=hf_hub_download_kwargs.get("revision", None),
            repo_type=hf_hub_download_kwargs.get("repo_type", None),
            token=token,
        )
        use_safetensors = has_remote_safetensors_file

        if has_remote_safetensors_file:
            # Priority 1: load safetensors weights
            filename = hf_hub_download(
                model_id,
                SAFETENSORS_WEIGHTS_NAME,
                **hf_hub_download_kwargs,
            )
        else:
            try:
                filename = hf_hub_download(model_id, WEIGHTS_NAME, **hf_hub_download_kwargs)
            except EntryNotFoundError:
                raise ValueError(
                    f"Can't find weights for {model_id} in {model_id} or in the Hugging Face Hub. "
                    f"Please check that the file {WEIGHTS_NAME} or {SAFETENSORS_WEIGHTS_NAME} is present at {model_id}."
                )

    if use_safetensors:
        if hasattr(torch.backends, "mps") and (device == torch.device("mps")):
            adapters_weights = safe_load_file(filename, device="cpu")
        else:
            adapters_weights = safe_load_file(filename, device=device)
    else:
        adapters_weights = torch.load(filename, map_location=torch.device(device))

    return adapters_weights


def _get_file_path_dir(load_directory: Union[str, os.PathLike], name: str, dir: str) -> str:
    if os.path.exists(os.path.join(load_directory, dir, name)):
        return os.path.join(load_directory, dir, name)
    return hf_hub_download(load_directory, filename=name, subfolder=dir)
