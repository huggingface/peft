# Copyright 2024-present the HuggingFace Inc. team.
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

# Reference code: https://github.com/iboing/CorDA/blob/main/cordalib/decomposition.py
# Reference paper: https://arxiv.org/abs/2406.05223

import os
from typing import Any, Callable, Iterable, Optional

import torch
import torch.nn as nn
from attr import dataclass
from tqdm import tqdm

from peft.tuners.lora.config import CordaInitConfig, LoraConfig
from peft.tuners.lora.model import LoraModel
from peft.utils.other import get_pattern_key


@dataclass
class CordaEigens:
    S_WC: torch.Tensor
    U_WC: torch.Tensor
    V_WC: torch.Tensor


def target_modules(model: nn.Module, config: LoraConfig) -> Iterable[nn.Module]:
    """
    Iterate over CorDA target name and modules of a model. A module is a target if its name is in
    `config.target_modules` and is `nn.Linear`.
    """
    for name, module in model.named_modules():
        if LoraModel._check_target_module_exists(config, name) and isinstance(module, nn.Linear):
            yield name, module


def get_model_device(model: nn.Module) -> str:
    if hasattr(model, "module"):  # Handle DeepSpeed/DataParallel
        model = model.module
    return next(iter(model.parameters())).device.type


@torch.no_grad()
def preprocess_corda(
    model: nn.Module,
    lora_config: LoraConfig,
    init_config: CordaInitConfig = None,
):
    """
    Build necessary CorDA fields for a model.

    Args:
        model (`nn.Module`):
            Model to preprocess.
        lora_config (`LoraConfig`):
            Lora configuration of the model. `lora_config.corda_config` should be set.
        init_config (`CordaInitConfig`):
            CorDA initialization configuration. This is not merged with `lora_config` because it's not serializable.

    Upon completion, the following fields are set for each target module:
        corda_method (`Literal["ipm", "kpm"]`):
            CorDA method to apply. "ipm" for Instruction-Previewed Mode, "kpm" for Knowledge-Preserved Mode.
        rank (`int`):
            Rank of CorDA to apply.
        eigens.S_WC (`torch.Tensor`):
            Singular values of the weight matrix.
        eigens.U_WC (`torch.Tensor`):
            Left singular vectors of the weight matrix.
        eigens.V_WC (`torch.Tensor`):
            Right singular vectors of the weight matrix, multiplied by inverse of covariance matrix.
    """
    cache_file = lora_config.corda_config.get("cache_file")
    covariance_file = lora_config.corda_config.get("covariance_file")
    sample_count = lora_config.corda_config.get("sample_count")
    corda_method = lora_config.corda_config.get("corda_method")
    verbose = lora_config.corda_config.get("verbose")
    if init_config is None:
        init_config = CordaInitConfig()
    run_model = init_config.run_model
    hooked_model = init_config.hooked_model

    # If cache exists, skip building
    if cache_file is not None and os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
        cache = torch.load(cache_file, map_location=get_model_device(model))
        for name, module in target_modules(model, lora_config):
            module.corda_method = cache[f"{name}.corda_method"]
            module.rank = cache[f"{name}.rank"]
            module.eigens = CordaEigens(
                S_WC=cache[f"{name}.eigens.S_WC"],
                U_WC=cache[f"{name}.eigens.U_WC"],
                V_WC=cache[f"{name}.eigens.V_WC"],
            )
    else:
        # Specify CorDA method for each layer
        if corda_method is None:
            raise ValueError("corda_method is required when cache_file is not provided")
        for name, module in target_modules(model, lora_config):
            module.corda_method = corda_method

        # Specify CorDA rank for each layer
        for name, module in target_modules(model, lora_config):
            r_key = get_pattern_key(lora_config.rank_pattern.keys(), name)
            module.rank = lora_config.rank_pattern.get(r_key, lora_config.r)

        # Calculate covariance matrix
        calib_cov_distribution(model, lora_config, run_model, hooked_model, sample_count, covariance_file)

        # Calculate eigens
        collect_eigens(model, lora_config, verbose)

        # Crop CorDA eigens so that there's less to save
        crop_corda_eigens(model, lora_config)

        # Save cache to disk
        if cache_file is not None:
            cache: dict[str, Any] = {}
            for name, module in target_modules(model, lora_config):
                cache[f"{name}.corda_method"] = module.corda_method
                cache[f"{name}.rank"] = module.rank
                cache[f"{name}.eigens.S_WC"] = module.eigens.S_WC
                cache[f"{name}.eigens.U_WC"] = module.eigens.U_WC
                cache[f"{name}.eigens.V_WC"] = module.eigens.V_WC

            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            torch.save(cache, cache_file)


@torch.no_grad()
def calib_cov_distribution(
    model: nn.Module,
    config: LoraConfig,
    run_model: Optional[Callable[[], None]],
    hooked_model: Optional[nn.Module],
    sample_count: int,
    covariance_file: Optional[str],
):
    if covariance_file is not None and os.path.exists(covariance_file) and os.path.getsize(covariance_file) > 0:
        all_covariance_matrix = torch.load(covariance_file, map_location=get_model_device(model))
        for name, module in target_modules(model, config):
            module.covariance_matrix = all_covariance_matrix[name]
        return

    if run_model is None:
        raise ValueError("run_model must be specified when covariance file and cache file aren't built")
    if hooked_model is None:
        hooked_model = model
    hooked_model.eval()

    def hook(module, input, output):
        input = input[0].detach().squeeze(0).data  ## (context_length = 2048, dim)
        if config.corda_config.get("use_float32_for_covariance", True):
            input = input.float()
        input = input / torch.max(input).abs()

        # covariance = input.t() @ input ## (dim, dim)
        if torch.isnan(input).any():
            raise Exception("nan in input, break")
        if torch.isinf(input).any():
            raise Exception("inf in input, break")

        covariance = input.t().matmul(input)
        if torch.isnan(covariance).any():
            raise Exception("nan in covariance, break")
        if torch.isinf(covariance).any():
            raise Exception("inf in covariance, break")

        # calculate mean and std
        mean = input.mean(0)
        std = input.std(0)

        # add to module
        module.covariance_matrix += covariance / sample_count
        module.mean += mean / sample_count
        module.std += std / sample_count

        # module.covariance_matrix = (module.covariance_matrix + covariance) / 2
        del covariance, input

    for name, module in target_modules(hooked_model, config):
        module.covariance_matrix = 0
        module.mean = 0
        module.std = 0
        module.register_forward_hook(hook)

    run_model()

    if hooked_model is not model:
        targets = {}
        for name, module in target_modules(model, config):
            targets[name] = module
        for name, module in target_modules(hooked_model, config):
            # There can be modules used only in inference, but not training
            # Exclude modules not in target model to prevent KeyError in this case
            if name in targets:
                targets[name].covariance_matrix = module.covariance_matrix
                targets[name].mean = module.mean
                targets[name].std = module.std

    # Save covariance to disk
    if covariance_file is not None:
        all_covariance_matrix = {}
        for name, module in target_modules(model, config):
            all_covariance_matrix[name] = module.covariance_matrix
        os.makedirs(os.path.dirname(covariance_file), exist_ok=True)
        torch.save(all_covariance_matrix, covariance_file)


@torch.no_grad()
def collect_eigens(
    model: nn.Module,
    config: LoraConfig,
    verbose: bool,
):
    """Call collect_eigens_for_layer and store result in key `eigens` of each layer."""
    linear_modules = []
    for name, module in target_modules(model, config):
        linear_modules.append((name, module))
    if verbose:
        linear_modules = tqdm(linear_modules, desc="Collecting eigens")
    for name, module in linear_modules:
        module.eigens = collect_eigens_for_layer(module, config)


@torch.no_grad()
def collect_eigens_for_layer(
    linear: nn.Linear,
    config: LoraConfig,
) -> CordaEigens:
    w = linear.weight.data.float()
    out_dim = w.size(0)
    in_dim = w.size(1)
    min_dim = min(in_dim, out_dim)

    if not hasattr(linear, "covariance_matrix"):
        raise ValueError("build covariance matrix with calib_cov_distribution first")
    covariance_matrix = linear.covariance_matrix.float()

    damp = 0.01
    while True:
        compensate = torch.diag(
            torch.ones(covariance_matrix.size(0)).to(covariance_matrix.device)
            * torch.mean(torch.diag(covariance_matrix))
            * damp
        )
        fix_covariance_matrix = covariance_matrix + compensate
        cov_inv = torch.linalg.inv(fix_covariance_matrix)
        inv_error = torch.dist(
            fix_covariance_matrix @ cov_inv, torch.eye(covariance_matrix.size(0)).to(get_model_device(linear))
        ).item()
        if inv_error < 0.05:
            break
        else:
            damp = damp * 2
    w = w @ fix_covariance_matrix  ## w: out_dim, in_dim; covariance_matrix: in_dim, in_dim

    U, S, Vh = torch.linalg.svd(w, full_matrices=False)
    V = (Vh @ cov_inv).transpose(0, 1)

    # Sanity check, temporarily U and V are large, they will be crop after rank search
    if U.size(0) != out_dim or U.size(1) != min_dim:
        raise ValueError(f"U size mismatch: {U.size()} vs. ({out_dim}, {min_dim})")
    if S.size(0) != min_dim:
        raise ValueError(f"S size mismatch: {S.size()} vs. ({min_dim},)")
    if V.size(0) != in_dim or V.size(1) != min_dim:
        raise ValueError(f"V size mismatch: {V.size()} vs. ({in_dim}, {min_dim})")

    # Offload U and V to CPU, they consume too much memory
    U = U.cpu()
    V = V.cpu()
    return CordaEigens(
        S_WC=S,
        U_WC=U,
        V_WC=V,
    )


@torch.no_grad()
def crop_corda_eigens(model: nn.Module, config: LoraConfig):
    for name, module in target_modules(model, config):
        # We don't expect saving sliced tensor writes the whole tensor to disk,
        # so it's necessary to copy the tensors.
        # Reference: https://github.com/pytorch/pytorch/issues/40157
        if module.corda_method == "ipm":
            module.eigens.S_WC = module.eigens.S_WC[: module.rank].clone()
            module.eigens.U_WC = module.eigens.U_WC[:, : module.rank].clone().to(get_model_device(model))
            module.eigens.V_WC = module.eigens.V_WC[:, : module.rank].clone().to(get_model_device(model))
        elif module.corda_method == "kpm":
            module.eigens.S_WC = module.eigens.S_WC[-module.rank :].clone()
            module.eigens.U_WC = module.eigens.U_WC[:, -module.rank :].clone().to(get_model_device(model))
            module.eigens.V_WC = module.eigens.V_WC[:, -module.rank :].clone().to(get_model_device(model))
        else:
            raise ValueError("Invalid corda_method")

        # Sanity check
        if module.eigens.S_WC.size(0) != module.rank:
            raise ValueError(f"rank mismatch: {module.eigens.S_WC.size(0)} vs. {module.rank}")
        if module.eigens.U_WC.size(0) != module.weight.size(0):
            raise ValueError(f"U size mismatch: {module.eigens.U_WC.size(0)} vs. {module.weight.size(0)}")
        if module.eigens.U_WC.size(1) != module.rank:
            raise ValueError(f"U size mismatch: {module.eigens.U_WC.size(1)} vs. {module.rank}")
        if module.eigens.V_WC.size(0) != module.weight.size(1):
            raise ValueError(f"V size mismatch: {module.eigens.V_WC.size(0)} vs. {module.weight.size(1)}")
        if module.eigens.V_WC.size(1) != module.rank:
            raise ValueError(f"V size mismatch: {module.eigens.V_WC.size(1)} vs. {module.rank}")
