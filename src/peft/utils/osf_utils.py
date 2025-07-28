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
"""Utilities for Orthogonal Subspace Learning with Adaptive OSF."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


__all__ = [
    "auto_generate_target_osf_config",
    "create_osf_model_class",
    "decompose_weight_matrix",
    "optim_wrapper",
    "project_gradient_to_orthogonal_space",
    "reconstruct_weight_matrix",
    "wrap_model_with_osf",
]


def decompose_weight_matrix(weight: torch.Tensor, top_k: int) -> dict[str, Any]:
    """Perform an SVD of ``weight`` and split it into frozen and trainable parts."""
    device_local = weight.device
    orig_dtype = weight.dtype
    W = weight.to(torch.float32)
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    k = min(top_k, S.shape[0])

    svd = {
        "U_high": U[:, :k].contiguous().detach().to(device=device_local, dtype=orig_dtype),
        "S_high": S[:k].contiguous().detach().to(device=device_local, dtype=orig_dtype),
        "V_high": Vt[:k, :].contiguous().detach().to(device=device_local, dtype=orig_dtype),
        "U_low": nn.Parameter(U[:, k:].contiguous().detach().to(device=device_local, dtype=orig_dtype)),
        "S_low": nn.Parameter(S[k:].contiguous().detach().to(device=device_local, dtype=orig_dtype)),
        "V_low": nn.Parameter(Vt[k:, :].contiguous().detach().to(device=device_local, dtype=orig_dtype)),
        "rank_high": k,
    }
    return svd


def reconstruct_weight_matrix(svd_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """Reconstruct a weight matrix from its SVD components."""
    U_high = svd_dict["U_high"]
    S_high = svd_dict["S_high"]
    V_high = svd_dict["V_high"]
    U_low = svd_dict["U_low"]
    S_low = svd_dict["S_low"]
    V_low = svd_dict["V_low"]

    high_part = (
        torch.mm(U_high * S_high.unsqueeze(0), V_high)
        if U_high.numel() > 0 and S_high.numel() > 0
        else torch.zeros(U_low.size(0), V_low.size(1), device=U_high.device)
    )
    low_part = (
        torch.mm(U_low * S_low.unsqueeze(0), V_low)
        if U_low.numel() > 0 and S_low.numel() > 0
        else torch.zeros(U_high.size(0), V_high.size(1), device=U_low.device)
    )
    return high_part + low_part


def project_gradient_to_orthogonal_space(svd_dict: dict[str, Any]) -> None:
    """Project gradients of ``U_low`` and ``V_low`` to be orthogonal to the high rank space."""
    if svd_dict["U_low"].grad is None and svd_dict["S_low"].grad is None and svd_dict["V_low"].grad is None:
        return

    U_high = svd_dict["U_high"]
    V_high = svd_dict["V_high"]

    if svd_dict["U_low"].grad is not None:
        dU = svd_dict["U_low"].grad
        local_U_high = getattr(U_high, "to_local", lambda: U_high)()
        local_dU = getattr(dU, "to_local", lambda: dU)()
        if local_U_high.size(0) != local_dU.size(0):
            rank = torch.distributed.get_rank()
            start = rank * local_dU.size(0)
            end = start + local_dU.size(0)
            local_U_high = local_U_high[start:end]
        proj = local_U_high @ (local_U_high.transpose(0, 1) @ local_dU)
        local_dU.sub_(proj)
        if hasattr(dU, "_local_tensor"):
            dU._local_tensor.copy_(local_dU)
        else:
            dU.copy_(local_dU)

    if svd_dict["V_low"].grad is not None:
        dV = svd_dict["V_low"].grad
        local_V_high = getattr(V_high, "to_local", lambda: V_high)()
        local_dV = getattr(dV, "to_local", lambda: dV)()
        if local_V_high.size(1) != local_dV.size(1):
            rank = torch.distributed.get_rank()
            start = rank * local_dV.size(1)
            end = start + local_dV.size(1)
            local_V_high = local_V_high[:, start:end]
        proj = (local_dV @ local_V_high.transpose(0, 1)) @ local_V_high
        local_dV.sub_(proj)
        if hasattr(dV, "_local_tensor"):
            dV._local_tensor.copy_(local_dV)
        else:
            dV.copy_(local_dV)


def auto_generate_target_osf_config(model: nn.Module) -> dict[str, int]:
    """Create a mapping from parameter names to ``top_k`` based on layer size."""
    target_patterns = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
        "mlp.up_proj",
    ]
    config: dict[str, int] = {}
    for name, param in model.named_parameters():
        if any(pat in name for pat in target_patterns) and len(param.shape) == 2:
            top_k = int(math.floor(min(param.shape) * 0.5))
            full_rank = min(param.shape)
            if top_k >= full_rank:
                top_k = full_rank - 1
            config[name] = top_k
    return config


def create_osf_model_class(base_cls: type) -> type:
    """Create a subclass of ``base_cls`` where selected linear weights are replaced by SVD decompositions."""

    class ModelWithOSF(base_cls):
        def __init__(self, config, svd_config: dict[str, int] | None = None, initialize_svd: bool = True, **kwargs):
            super().__init__(config, **kwargs)
            self.svd_config = svd_config or {}
            self.name_mapping: dict[str, str] = {}
            self.svd_params = nn.ModuleDict()
            if initialize_svd:
                self._initialize_svd_parameters()

        @classmethod
        def from_pretrained(
            cls, pretrained_model_name_or_path, *model_args, svd_config: dict[str, int] | None = None, **kwargs
        ):
            model = super().from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                svd_config=svd_config or {},
                **kwargs,
            )
            if svd_config is None:
                svd_config = auto_generate_target_osf_config(model)
            model.svd_config = svd_config
            model.reinitialize_svd()
            return model

        def reinitialize_svd(self) -> None:
            self.name_mapping = {}
            self.svd_params = nn.ModuleDict()
            self._initialize_svd_parameters()

        def _get_module_by_name(self, name: str):
            parts = name.split(".")
            attr = parts[-1]
            mod = self
            for p in parts[:-1]:
                if hasattr(mod, p):
                    mod = getattr(mod, p)
                elif p.isdigit():
                    mod = mod[int(p)]
                else:
                    return None, None
            return mod, attr

        def _initialize_svd_parameters(self) -> None:
            for name, param in list(self.named_parameters()):
                if len(param.shape) == 2 and name in self.svd_config and self.svd_config[name] > 0:
                    top_k = self.svd_config[name]
                    svd_dict = decompose_weight_matrix(param.data, top_k=top_k)
                    safe_name = name.replace(".", "_")
                    self.name_mapping[name] = safe_name
                    self.register_buffer(f"{safe_name}_U_high", svd_dict["U_high"])
                    self.register_buffer(f"{safe_name}_S_high", svd_dict["S_high"])
                    self.register_buffer(f"{safe_name}_V_high", svd_dict["V_high"])

                    module_svd = nn.Module()
                    module_svd.U_low = svd_dict["U_low"]
                    module_svd.S_low = svd_dict["S_low"]
                    module_svd.V_low = svd_dict["V_low"]
                    module_svd.rank_high = svd_dict["rank_high"]
                    module_svd.safe_name = safe_name
                    self.svd_params[safe_name] = module_svd

                    mod, attr = self._get_module_by_name(name)
                    bias = mod.bias if hasattr(mod, "bias") else None

                    def make_forward(sn: str, bias: torch.Tensor | None):
                        def forward(x):
                            W = self._reconstruct_weight_by_safe_name(sn)
                            if W.dtype != x.dtype:
                                W = W.to(x.dtype)
                            return F.linear(x, W, bias)

                        return forward

                    mod.forward = make_forward(safe_name, bias)
                    param.requires_grad = False
                    mod._parameters.pop(attr, None)

        def _reconstruct_weight_by_safe_name(self, safe_name: str) -> torch.Tensor:
            U_high = getattr(self, f"{safe_name}_U_high")
            S_high = getattr(self, f"{safe_name}_S_high")
            V_high = getattr(self, f"{safe_name}_V_high")
            module_svd = self.svd_params[safe_name]
            svd_dict = {
                "U_high": U_high,
                "S_high": S_high,
                "V_high": V_high,
                "U_low": module_svd.U_low,
                "S_low": module_svd.S_low,
                "V_low": module_svd.V_low,
            }
            return reconstruct_weight_matrix(svd_dict)

        def project_gradients(self) -> None:
            for safe_name, module_svd in self.svd_params.items():
                svd_dict = {
                    "U_high": getattr(self, f"{safe_name}_U_high"),
                    "S_high": getattr(self, f"{safe_name}_S_high"),
                    "V_high": getattr(self, f"{safe_name}_V_high"),
                    "U_low": module_svd.U_low,
                    "S_low": module_svd.S_low,
                    "V_low": module_svd.V_low,
                }
                project_gradient_to_orthogonal_space(svd_dict)

        def prepare_state_dict_for_save(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            if not hasattr(self, "name_mapping"):
                return state_dict
            for orig, safe in self.name_mapping.items():
                U_high = state_dict.pop(f"{safe}_U_high")
                S_high = state_dict.pop(f"{safe}_S_high")
                V_high = state_dict.pop(f"{safe}_V_high")
                U_low = state_dict.pop(f"svd_params.{safe}.U_low")
                S_low = state_dict.pop(f"svd_params.{safe}.S_low")
                V_low = state_dict.pop(f"svd_params.{safe}.V_low")
                W = reconstruct_weight_matrix(
                    {
                        "U_high": U_high,
                        "S_high": S_high,
                        "V_high": V_high,
                        "U_low": U_low,
                        "S_low": S_low,
                        "V_low": V_low,
                    }
                )
                state_dict[orig] = W
            return state_dict

    ModelWithOSF.__name__ = f"{base_cls.__name__}WithOSF"
    return ModelWithOSF


def optim_wrapper(optimizer: torch.optim.Optimizer, model: nn.Module) -> torch.optim.Optimizer:
    """Wrap ``optimizer.step`` to project gradients before each update."""
    if not hasattr(model, "project_gradients"):
        return optimizer

    import types

    orig_step = optimizer.step

    def step(self, *args, **kwargs):
        model.project_gradients()
        return orig_step(*args, **kwargs)

    optimizer.step = types.MethodType(step, optimizer)
    return optimizer


def wrap_model_with_osf(model: nn.Module, svd_config: dict[str, int] | None = None) -> nn.Module:
    """Return a version of ``model`` where selected weights are decomposed using SVD.

    Parameters ---------- model:
        The model to wrap. It must expose a ``config`` attribute that will be passed to the wrapped class' constructor.
    svd_config:
        A mapping from parameter names to ``top_k`` ranks. If not provided, it is automatically generated based on the
        layer shapes using :func:`auto_generate_target_osf_config`.

    Returns ------- ``nn.Module``
        A new model instance with the same weights as ``model`` but with trainable low-rank parameters and frozen
        high-rank components.
    """

    svd_config = svd_config or auto_generate_target_osf_config(model)
    OSFCls = create_osf_model_class(model.__class__)
    wrapped = OSFCls(model.config, svd_config=svd_config, initialize_svd=False)
    wrapped.load_state_dict(model.state_dict())
    wrapped.reinitialize_svd()
    return wrapped
