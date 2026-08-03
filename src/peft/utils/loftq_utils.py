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

# Reference code: https://github.com/yxli2123/LoftQ/blob/main/utils.py
# Reference paper: https://huggingface.co/papers/2310.08659

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Optional, Union

import torch
from accelerate.utils.memory import clear_device_cache
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, LocalEntryNotFoundError
from safetensors import SafetensorError, safe_open
from transformers.utils import cached_file
from transformers.utils.hub import get_checkpoint_shard_files

from peft.import_utils import is_bnb_4bit_available, is_bnb_available, is_xpu_available


def _low_rank_decomposition(weight, reduced_rank=32):
    """
    :param weight: The matrix to decompose, of shape (H, W) :param reduced_rank: the final rank :return:
    """
    matrix_dimension = len(weight.size())
    if matrix_dimension != 2:
        raise ValueError(f"Only support 2D matrix, but your input has {matrix_dimension} dimensions.")

    # Use SVD to decompose a matrix, default full_matrices is False to save parameters
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

    L = U @ (torch.sqrt(torch.diag(S)[:, 0:reduced_rank]))
    R = torch.sqrt(torch.diag(S)[0:reduced_rank, :]) @ Vh

    return {"L": L, "R": R, "U": U, "S": S, "Vh": Vh, "reduced_rank": reduced_rank}


@torch.no_grad()
def loftq_init(weight: Union[torch.Tensor, torch.nn.Parameter], num_bits: int, reduced_rank: int, num_iter=1):
    if is_bnb_available():
        import bitsandbytes as bnb
    else:
        raise ValueError("bitsandbytes is not available, please install it to use LoftQ.")

    if num_bits not in [4, 8]:
        raise ValueError("Only nf4 and int8 quantization is supported")
    if num_iter <= 0:
        raise ValueError("Number of iterations must be greater than 0")

    device = weight.device
    dtype = weight.dtype
    if not is_bnb_4bit_available() and num_bits == 4:
        raise RuntimeError(
            "Native support for nf4 is was removed in PEFT 0.20. Please install a recent version of bitsandbytes.",
        )
    compute_device = "xpu" if is_xpu_available() else "cuda"

    weight = weight.to(device=compute_device, dtype=torch.float32)
    res = weight.clone()
    for i in range(num_iter):
        clear_device_cache()
        # Quantization
        if num_bits == 4:
            qweight = bnb.nn.Params4bit(
                res.to("cpu"), requires_grad=False, compress_statistics=False, quant_type="nf4"
            ).to(compute_device)
            dequantized_weight = bnb.functional.dequantize_4bit(qweight.data, qweight.quant_state)
        elif num_bits == 8:
            qweight = bnb.nn.Int8Params(res.to("cpu"), requires_grad=False).to(device)
            dequantized_weight = bnb.functional.int8_vectorwise_dequant(qweight.data, qweight.SCB).to(compute_device)

        res = weight - dequantized_weight

        # Decompose the residual by SVD
        output = _low_rank_decomposition(res, reduced_rank=reduced_rank)
        L, R, reduced_rank = output["L"], output["R"], output["reduced_rank"]

        # don't prepare the residual if we're at the end
        if i + 1 == num_iter:
            break

        res = weight - torch.mm(L, R)

    lora_A, lora_B = R, L

    return dequantized_weight.to(device=device, dtype=dtype), lora_A, lora_B


@torch.no_grad()
def _loftq_init_new(qweight, weight, num_bits: int, reduced_rank: int):
    import bitsandbytes as bnb

    if num_bits != 4:
        raise ValueError("Only 4 bit quantization supported at the moment.")
    if not is_bnb_4bit_available():
        raise ValueError("bitsandbytes 4bit quantization is not available.")

    compute_device = "xpu" if is_xpu_available() else "cuda"
    dequantized_weight = bnb.functional.dequantize_4bit(qweight.data, qweight.quant_state)

    weight = weight.to(device=compute_device, dtype=torch.float32)
    residual = weight - dequantized_weight
    clear_device_cache()
    # Decompose the residualidual by SVD
    output = _low_rank_decomposition(residual, reduced_rank=reduced_rank)
    L, R, reduced_rank = output["L"], output["R"], output["reduced_rank"]
    return R, L


class _SafetensorLoader:
    """
    Simple utility class that loads tensors with safetensors from a single file or sharded files.

    Takes care of file name normalization etc.

    """

    def __init__(self, peft_model, model_path):
        if model_path is None:
            try:
                model_path = snapshot_download(peft_model.base_model.config._name_or_path, local_files_only=True)
            except (AttributeError, HFValidationError) as exc:
                raise ValueError(
                    "The provided model does not appear to be a transformers model or is a local model. In this case, "
                    "you must pass the model_path argument that points to the safetensors file."
                ) from exc
            except LocalEntryNotFoundError as exc:
                raise ValueError(
                    "The model.safetensors file must be present on disk, but it could not be found."
                ) from exc

        suffix = "model.safetensors"
        if not model_path.endswith(suffix):
            model_path = os.path.join(model_path, suffix)

        self.model_path = model_path
        self.base_model_prefix = getattr(peft_model.get_base_model(), "base_model_prefix", None)
        self.prefix = "base_model.model."
        self.is_sharded = False
        self.weight_map = None

        if not os.path.exists(model_path):
            # check if the file is sharded
            par_dir = model_path.rpartition(os.path.sep)[0]
            try:
                resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
                    par_dir, cached_file(par_dir, "model.safetensors.index.json")
                )
            except OSError as exc:
                raise FileNotFoundError(
                    f"Could not find file for {model_path}, ensure that there is a (sharded) safetensors file of the model."
                ) from exc

            self.is_sharded = True
            # maps from 'model-X-of-Y.safetensors' to full file path
            file_map = {k.rpartition(os.path.sep)[-1]: k for k in resolved_archive_file}
            self.weight_map = {k: file_map[v] for k, v in sharded_metadata["weight_map"].items()}

    def get_tensor(self, name):
        if not self.is_sharded:
            file_path = self.model_path
        else:
            file_path = self.weight_map[name]

        with safe_open(file_path, framework="pt", device="cpu") as f:
            try:
                tensor = f.get_tensor(name)
            except SafetensorError as exc:
                # no matching key found, we probably need to remove the base model prefix
                if self.base_model_prefix:
                    # remove 1 extra character for "."
                    name = name[len(self.base_model_prefix) + 1 :]
                    tensor = f.get_tensor(name)
                else:
                    raise
        return tensor


@torch.no_grad()
def replace_lora_weights_loftq(
    peft_model,
    model_path: Optional[str] = None,
    adapter_name: str = "default",
    callback: Optional[Callable[[torch.nn.Module, str], bool]] = None,
):
    """
    Replace the LoRA weights of a model quantized with bitsandbytes, using the LoftQ technique.

    The replacement is done on the fly by loading in the non-quantized weights from a locally stored safetensors model
    file and initializing the LoRA weights such that the quantization error between the original and quantized weights
    is minimized.

    As lazy loading is not possible with pickle, normal PyTorch checkpoint files cannot be supported.

    Depending on the model size, calling this function may take some time to finish.

    Args:
        peft_model (`PeftModel`):
            The model to replace the weights of. Must be a quantized PEFT model with LoRA layers.
        model_path (`Optional[str]`):
            The path to the model safetensors file. If the model is a Hugging Face model, this will be inferred from
            the model's config. Otherwise, it must be provided.
        adapter_name (`str`):
            The name of the adapter to replace the weights of. The default adapter name is "default".
        callback (`Optional[Callable[[PeftModel, str], bool]]`):
            A callback function that will be called after each module is replaced. The callback function should take
            the model and the name of the current module as input and return a boolean indicating whether the
            replacement should be kept. If the callback returns False, the replacement will be rolled back. This can be
            very useful to confirm that the LoftQ initialization actually decreases the quantization error of the
            model. As an example, this callback could generate logits for given input and compare it with the logits
            from the original, non-quanitzed model with the same input, and only return `True` if there is an
            improvement. As this is a greedy optimization, it's possible that calling this function multiple times
            yields incremental improvements.
    """
    if not is_bnb_4bit_available():
        raise ValueError("bitsandbytes must be installed and the model must be quantized in 4bits.")

    from peft.tuners.lora import Linear4bit

    # model_path = _check_model_path_loftq(model_path, peft_model)
    prefix = "base_model.model."
    any_match = False
    safetensor_loader = _SafetensorLoader(peft_model, model_path)

    # if too slow, consider adding tqdm as an option
    for name, module in peft_model.named_modules():
        if not isinstance(module, Linear4bit):
            continue

        if not name.startswith(prefix):
            raise TypeError("The passed model does not appear to be a valid PeftModel")

        any_match = True
        name = name[len(prefix) :]
        tensor = safetensor_loader.get_tensor(name + ".weight")

        reduced_rank = module.r[adapter_name]
        lora_A, lora_B = _loftq_init_new(module.weight, tensor, num_bits=4, reduced_rank=reduced_rank)
        if not callback:
            module.lora_A[adapter_name].weight.data = lora_A
            module.lora_B[adapter_name].weight.data = lora_B
            continue

        lora_A_before = module.lora_A[adapter_name].weight.data
        lora_B_before = module.lora_B[adapter_name].weight.data

        module.lora_A[adapter_name].weight.data = lora_A
        module.lora_B[adapter_name].weight.data = lora_B
        should_replace = callback(peft_model, name)
        if not should_replace:
            # roll back
            module.lora_A[adapter_name].weight.data = lora_A_before
            module.lora_B[adapter_name].weight.data = lora_B_before

        del lora_A_before, lora_B_before

    if not any_match:
        raise ValueError("No bnb LoRA module found on the model")
