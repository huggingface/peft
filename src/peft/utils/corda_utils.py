import os
from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import PreTrainedModel

from peft.tuners.lora.config import LoraConfig
from peft.tuners.tuners_utils import check_target_module_exists


class CordaEigens:
    S_cov: torch.Tensor
    U_WC: torch.Tensor
    S_WC: torch.Tensor
    V_WC: torch.Tensor

    def __init__(
        self,
        S_cov: torch.Tensor,
        U_WC: torch.Tensor,
        S_WC: torch.Tensor,
        V_WC: torch.Tensor,
    ):
        self.S_cov = S_cov
        self.U_WC = U_WC
        self.S_WC = S_WC
        self.V_WC = V_WC


@torch.no_grad()
def target_modules(model: PreTrainedModel, config: LoraConfig) -> Iterable[nn.Module]:
    """
    Iterate over CorDA target name and modules of a model. A module is a target if its name is in
    `config.target_modules` and is `nn.Linear`.
    """
    for name, module in model.named_modules():
        if check_target_module_exists(config, name) and isinstance(module, nn.Linear):
            yield name, module


@torch.no_grad()
def preprocess_corda(
    model: PreTrainedModel,
    config: LoraConfig,
):
    """
    Build necessary CorDA fields for a model.

    The fields are:
        corda_method (`Literal["ipm", "kpm"]`):
            CorDA method to apply. "ipm" for Instruction-Previewed Mode, "kpm" for Knowledge-Preserved Mode.
        rank (`int`):
            Rank of CorDA to apply.
        eigens.S_cov (`torch.Tensor`):
            Singular values of the covariance matrix.
        eigens.U_WC (`torch.Tensor`):
            Left singular vectors of the weight matrix.
        eigens.S_WC (`torch.Tensor`):
            Singular values of the weight matrix.
        eigens.V_WC (`torch.Tensor`):
            Right singular vectors of the weight matrix.
    """
    run_model = config.corda_config.run_model
    cache_file = config.corda_config.cache_file
    covariance_file = config.corda_config.covariance_file
    sample_count = config.corda_config.sample_count
    corda_method = config.corda_config.corda_method
    corda_rank = config.r

    # If cache exists, skip building
    if os.path.exists(cache_file):
        print(f"CorDA cache file found: {cache_file}", flush=True)
        cache = torch.load(cache_file, map_location="cuda")
        for name, module in target_modules(model, config):
            if isinstance(module, nn.Linear) and f"{name}.corda_method" in cache:
                module.corda_method = cache[f"{name}.corda_method"]
                module.rank = cache[f"{name}.rank"]
                module.eigens = CordaEigens(
                    S_cov=cache[f"{name}.eigens.S_cov"],
                    U_WC=cache[f"{name}.eigens.U_WC"],
                    S_WC=cache[f"{name}.eigens.S_WC"],
                    V_WC=cache[f"{name}.eigens.V_WC"],
                )
        print(f"CorDA cache loaded from {cache_file}", flush=True)
        return

    # Otherwise, other args should not be None
    assert run_model is not None, "run_model is required when cache_file is not provided"
    assert sample_count is not None, "sample_count is required when cache_file is not provided"
    assert corda_method is not None, "corda_method is required when cache_file is not provided"

    # Specify CorDA method for each layer
    for name, module in target_modules(model, config):
        module.corda_method = corda_method

    # Calculate covariance matrix
    calib_cov_distribution(model, config, run_model, sample_count, covariance_file)

    # Calculate eigens
    collect_eigens(model, config)

    # Set rank for each layer
    for name, module in target_modules(model, config):
        module.rank = corda_rank

    # Crop CorDA eigens so that there's less to save
    crop_corda_eigens(model, config)

    # Save cache to disk
    if cache_file is not None:
        cache: dict[str, Any] = {}
        for name, module in target_modules(model, config):
            cache[f"{name}.corda_method"] = module.corda_method
            cache[f"{name}.rank"] = module.rank
            cache[f"{name}.eigens.S_cov"] = module.eigens.S_cov
            cache[f"{name}.eigens.U_WC"] = module.eigens.U_WC
            cache[f"{name}.eigens.S_WC"] = module.eigens.S_WC
            cache[f"{name}.eigens.V_WC"] = module.eigens.V_WC

        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        torch.save(cache, cache_file)
        print(f"CorDA cache saved at {cache_file}", flush=True)

    # Inject calculated rank and alpha patterns to model config
    config.rank_pattern = {}
    config.alpha_pattern = {}
    ratio = config.lora_alpha / config.r
    for name, module in target_modules(model, config):
        config.rank_pattern[name] = module.rank
        config.alpha_pattern[name] = module.rank * ratio

    # Clear run model callback as it's not serializable
    config.corda_config.run_model = None


@torch.no_grad()
def calib_cov_distribution(
    model: PreTrainedModel,
    config: LoraConfig,
    run_model: Callable[[], None],
    sample_count: int,
    covariance_file: Optional[str],
):
    if covariance_file is not None and os.path.exists(covariance_file):
        print(f"covariance file found: {covariance_file}", flush=True)
        all_covariance_matrix = torch.load(covariance_file, map_location="cuda")
        for name, module in target_modules(model, config):
            module.covariance_matrix = all_covariance_matrix[name]
        return

    model.eval()
    print(f"sample count: {sample_count}", flush=True)

    def hook(module, input, output):
        input = input[0].detach().squeeze(0).data  ## (context_length = 2048, dim)
        input = input.float()
        input = input / torch.max(input).abs()

        # covariance = input.t() @ input ## (dim, dim)
        if torch.isnan(input).any():
            print("nan detected", flush=True)
            raise Exception("nan in input, break")
        if torch.isinf(input).any():
            print("inf detected", flush=True)
            raise Exception("inf in input, break")

        covariance = input.t().matmul(input)
        if torch.isnan(covariance).any():
            print("nan detected", flush=True)
            raise Exception("nan in covariance, break")
        if torch.isinf(covariance).any():
            print("inf detected", flush=True)
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

    print("registering forward hook", flush=True)
    for name, module in target_modules(model, config):
        module.covariance_matrix = 0
        module.mean = 0
        module.std = 0
        module.register_forward_hook(hook)

    print("running model", flush=True)
    run_model()
    print("covariance matrices stored in model", flush=True)


@torch.no_grad()
def collect_eigens(
    model: PreTrainedModel,
    config: LoraConfig,
):
    """Call collect_eigens_for_layer and store result in key `eigens` of each layer."""
    linear_modules = []
    for name, module in target_modules(model, config):
        linear_modules.append((name, module))
    for name, module in tqdm(linear_modules, desc="Collecting eigens"):
        module.eigens = collect_eigens_for_layer(name, module)


@torch.no_grad()
def collect_eigens_for_layer(
    layername: str,
    linear: nn.Linear,
    svd_rank: Optional[int] = None,
    svd_niter: Optional[int] = None,
) -> CordaEigens:
    w = linear.weight.data.float()
    out_dim = w.size(0)
    in_dim = w.size(1)
    min_dim = min(in_dim, out_dim)

    assert hasattr(linear, "covariance_matrix")
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
        inv_error = torch.dist(fix_covariance_matrix @ cov_inv, torch.eye(covariance_matrix.size(0)).cuda()).item()
        print("damp:", damp, flush=True)
        print("inv_error:", inv_error, flush=True)
        if inv_error < 0.05:
            break
        else:
            damp = damp * 2
    w = w @ fix_covariance_matrix  ## w: out_dim, in_dim; covariance_matrix: in_dim, in_dim

    if svd_rank is not None and svd_niter is not None:
        _, S_cov, _ = torch.svd_lowrank(fix_covariance_matrix, q=svd_rank, niter=svd_niter)
        U, S, V = torch.svd_lowrank(w, q=svd_rank, niter=svd_niter)

        # expand S_cov, U, S and V to full size
        S_cov = torch.cat([S_cov, torch.zeros(in_dim - svd_rank).to(S_cov.device)])
        U = torch.cat([U, torch.zeros(out_dim, min_dim - svd_rank).to(U.device)], dim=1)
        S = torch.cat([S, torch.zeros(min_dim - svd_rank).to(S.device)])
        V = torch.cat([V, torch.zeros(in_dim, min_dim - svd_rank).to(V.device)], dim=1)
    else:
        S_cov = torch.linalg.svdvals(fix_covariance_matrix)
        U, S, Vh = torch.linalg.svd(w, full_matrices=False)
        V = Vh.transpose(0, 1)

    V = (V.t() @ cov_inv).transpose(0, 1)

    norm_of_VandCovinv = torch.sqrt((V**2).sum(dim=0))
    print("norm_of_VandConinv:", norm_of_VandCovinv, flush=True)

    S_adjusted = S * norm_of_VandCovinv
    print("S_adjusted:", S_adjusted, flush=True)

    # Reduce rank so that there's less to cache
    V = V / norm_of_VandCovinv  ## Normalized VandCovinv

    # Sanity check, temporarily U and V are large, they will be crop after rank search
    assert S_cov.size(0) == in_dim
    assert U.size(0) == out_dim
    assert U.size(1) == min_dim
    assert S.size(0) == min_dim
    assert V.size(0) == in_dim
    assert V.size(1) == min_dim

    # Offload U and V to CPU, they consume too much memory
    U = U.cpu()
    V = V.cpu()
    return CordaEigens(
        S_cov=S_cov,
        U_WC=U,
        S_WC=S_adjusted,
        V_WC=V,
    )


@torch.no_grad()
def crop_corda_eigens(model: PreTrainedModel, config: LoraConfig):
    for name, module in target_modules(model, config):
        # We don't expect saving sliced tensor writes the whole tensor to disk,
        # so it's necessary to copy the tensors.
        # Reference: https://github.com/pytorch/pytorch/issues/40157
        if module.corda_method == "ipm":
            module.eigens.U_WC = module.eigens.U_WC[:, : module.rank].clone()
            module.eigens.S_WC = module.eigens.S_WC[: module.rank].clone()
            module.eigens.V_WC = module.eigens.V_WC[:, : module.rank].clone()
        elif module.corda_method == "kpm":
            module.eigens.U_WC = module.eigens.U_WC[:, -module.rank :].clone()
            module.eigens.S_WC = module.eigens.S_WC[-module.rank :].clone()
            module.eigens.V_WC = module.eigens.V_WC[:, -module.rank :].clone()
        else:
            raise ValueError("Invalid corda_method")

        # Sanity check
        assert module.eigens.U_WC.size(0) == module.weight.size(0)
        assert module.eigens.U_WC.size(1) == module.rank
        assert module.eigens.S_WC.size(0) == module.rank
        assert module.eigens.V_WC.size(0) == module.weight.size(1)
        assert module.eigens.V_WC.size(1) == module.rank
    print("CorDA eigens cropped", flush=True)
