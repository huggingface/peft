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

from __future__ import annotations

import re
import warnings
from typing import Any, Optional

import torch
import torch.nn as nn
from torch import svd_lowrank

from peft.tuners.tuners_utils import BaseTunerLayer, _get_in_out_features, check_adapters_to_merge
from peft.utils.integrations import gather_params_ctx
from peft.utils.other import transpose

from .config import PSOFTConfig


class OrthLayer(nn.Module):
    """
    r*r orthogonal transformation R used in PSOFT between A and B.
    Forward: output = input @ R.T
    """

    def __init__(
        self,
        size: int,
        orth: bool = True,
        mag_b: bool = True,
        mag_a: bool = True,
        use_cayley_neumann: bool = True,
        num_cayley_neumann_terms: int = 5,
        cayley_neumann_eps: Optional[float] = 0.9,
    ):
        super().__init__()
        self.size = size
        self.orth = orth
        self.mag_b = mag_b
        self.mag_a = mag_a
        self.use_cayley_neumann = use_cayley_neumann
        self.num_cayley_neumann_terms = num_cayley_neumann_terms
        self.cayley_neumann_eps = cayley_neumann_eps

        if orth:
            self.weight = nn.Parameter(torch.empty((size * (size - 1)) // 2))
            rows, cols = torch.triu_indices(size, size, 1)
            self.register_buffer("rows", rows, persistent=False)
            self.register_buffer("cols", cols, persistent=False)
        else:
            self.weight = nn.Parameter(torch.empty(size, size))

        self.vector_b = nn.Parameter(torch.empty(size)) if mag_b else None
        self.vector_a = nn.Parameter(torch.empty(size)) if mag_a else None

    def reset_parameters(self, init_weights: bool = True) -> None:
        params = [self.weight]
        if self.vector_b is not None:
            params.append(self.vector_b)
        if self.vector_a is not None:
            params.append(self.vector_a)

        if any(p.is_meta for p in params):
            return

        with torch.no_grad():
            if init_weights:
                if self.orth:
                    self.weight.zero_()
                else:
                    nn.init.eye_(self.weight)

                if self.vector_b is not None:
                    self.vector_b.fill_(1.0)
                if self.vector_a is not None:
                    self.vector_a.fill_(1.0)
            else:
                if self.orth:
                    nn.init.normal_(self.weight, mean=0.0, std=0.1)
                else:
                    nn.init.eye_(self.weight)
                    self.weight.add_(torch.randn_like(self.weight) * 0.1)

                if self.vector_b is not None:
                    self.vector_b.fill_(1.0)
                if self.vector_a is not None:
                    self.vector_a.fill_(1.0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        R = self.get_matrix()

        if input.device.type == "cpu" and input.dtype in (torch.float16, torch.bfloat16):
            compute_dtype = torch.float32
        else:
            compute_dtype = input.dtype

        out = (input.to(compute_dtype) @ R.to(compute_dtype).t()).to(input.dtype)
        return out

    # Adapted from the Cayley/Neumann-based orthogonal parametrization used in OFT v2
    # (PEFT implementation: https://github.com/huggingface/peft/blob/main/src/peft/tuners/oft/layer.py) #L104
    def _skew_symmetric(self) -> torch.Tensor:
        Q = torch.zeros((self.size, self.size), device=self.weight.device, dtype=self.weight.dtype)
        Q = Q.index_put((self.rows, self.cols), self.weight)
        Q = Q - Q.transpose(0, 1)
        return Q

    # Adapted from the Cayley/Neumann-based orthogonal parametrization used in OFT v2
    # (PEFT implementation: https://github.com/huggingface/peft/blob/main/src/peft/tuners/oft/layer.py) #L160
    def _project_Q(self, Q: torch.Tensor, eps: float = 0.9) -> torch.Tensor:
        norm = torch.linalg.norm(Q, ord="fro")
        if torch.isfinite(norm) and norm > eps:
            Q = Q * (eps / (norm + 1e-12))
        return Q

    def get_matrix(self) -> torch.Tensor:
        # non-orth case
        if not self.orth:
            R = self.weight
            if self.mag_b and self.vector_b is not None:
                R = self.vector_b[:, None] * R
            if self.mag_a and self.vector_a is not None:
                R = R * self.vector_a[None, :]
            return R

        Q = self._skew_symmetric()
        orig_dtype = Q.dtype
        cast_to_fp32 = (Q.device.type == "cpu") and (orig_dtype in (torch.float16, torch.bfloat16))
        if cast_to_fp32:
            Q = Q.float()

        id_mat = torch.eye(self.size, device=Q.device, dtype=Q.dtype)

        # Adapted from the Cayley/Neumann-based orthogonal parametrization used in OFT v2
        # (PEFT implementation: https://github.com/huggingface/peft/blob/main/src/peft/tuners/oft/layer.py) #L135
        if self.use_cayley_neumann:
            if self.cayley_neumann_eps is not None:
                Q = self._project_Q(Q, eps=self.cayley_neumann_eps)
            t = int(self.num_cayley_neumann_terms)

            R = id_mat.clone()
            if t > 1:
                R.add_(Q, alpha=2.0)
                if t > 2:
                    Q_squared = Q @ Q
                    R.add_(Q_squared, alpha=2.0)

                    Q_power = Q_squared
                    for _ in range(3, t - 1):
                        Q_power = Q_power @ Q
                        R.add_(Q_power, alpha=2.0)

                    Q_power = Q_power @ Q
                    R.add_(Q_power)
        else:
            R = torch.linalg.solve(id_mat + Q, id_mat - Q, left=False)

        if self.mag_b and self.vector_b is not None:
            R = self.vector_b[:, None] * R
        if self.mag_a and self.vector_a is not None:
            R = R * self.vector_a[None, :]

        if cast_to_fp32:
            R = R.to(orig_dtype)

        return R


class PSOFTLayer(BaseTunerLayer):
    adapter_layer_names: tuple[str, ...] = ("psoft_R",)
    other_param_names: tuple[str, ...] = (
        "r",
        "psoft_alpha",
        "scaling",
        "psoft_dropout",
        "psoft_svd",
        "psoft_svd_lowrank_niter",
        "init_psoft_weights",
    )

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        super().__init__()
        self.base_layer = base_layer

        # per-adapter hyperparams
        self.r: dict[str, int] = {}
        self.psoft_alpha: dict[str, float] = {}
        self.scaling: dict[str, float] = {}
        self.psoft_dropout = nn.ModuleDict({})
        self.psoft_svd: dict[str, str] = {}
        self.psoft_svd_lowrank_niter: dict[str, int] = {}
        self.init_psoft_weights: dict[str, Optional[str]] = {}

        # per-adapter trainable module
        self.psoft_R = nn.ModuleDict({})

        # per-adapter cache state
        self._psoft_cache_built: dict[str, bool] = {}
        self._psoft_cache_names: dict[str, tuple[str, str]] = {}  # adapter -> (A_buf_name, B_buf_name)

        self.merged_adapters: list[str] = []
        self._disable_adapters = False
        self.kwargs = kwargs

        self.fan_in_fan_out = False

        base_layer = self.get_base_layer()
        in_features, out_features = _get_in_out_features(base_layer)
        self.in_features = in_features
        self.out_features = out_features

    @staticmethod
    def _sanitize_adapter_name(adapter_name: str) -> str:
        return re.sub(r"[^0-9a-zA-Z_]", "_", adapter_name)

    def _get_cache_buffers(self, adapter_name: str) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if adapter_name not in self._psoft_cache_names:
            return None, None
        a_name, b_name = self._psoft_cache_names[adapter_name]
        return getattr(self, a_name, None), getattr(self, b_name, None)

    def _set_cache_buffers(self, adapter_name: str, A: torch.Tensor, B: torch.Tensor) -> None:
        safe = self._sanitize_adapter_name(adapter_name)
        a_name = f"_psoft_A_cache_{safe}"
        b_name = f"_psoft_B_cache_{safe}"

        if adapter_name not in self._psoft_cache_names:
            self.register_buffer(a_name, A, persistent=False)
            self.register_buffer(b_name, B, persistent=False)
            self._psoft_cache_names[adapter_name] = (a_name, b_name)
        else:
            setattr(self, a_name, A)
            setattr(self, b_name, B)

    def update_layer(self, adapter_name: str, config: PSOFTConfig, **kwargs: Any) -> None:
        r = int(config.r)
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.fan_in_fan_out = config.fan_in_fan_out

        self.r[adapter_name] = r
        self.psoft_alpha[adapter_name] = config.psoft_alpha
        self.scaling[adapter_name] = config.psoft_alpha / r

        self.psoft_dropout[adapter_name] = (
            nn.Dropout(p=config.psoft_dropout) if config.psoft_dropout > 0.0 else nn.Identity()
        )

        self.init_psoft_weights[adapter_name] = config.init_psoft_weights
        self.psoft_svd[adapter_name] = config.psoft_svd
        self.psoft_svd_lowrank_niter[adapter_name] = config.psoft_svd_lowrank_niter

        self.psoft_R[adapter_name] = OrthLayer(
            size=r,
            orth=config.psoft_orth,
            mag_b=config.psoft_mag_b,
            mag_a=config.psoft_mag_a,
            use_cayley_neumann=config.use_cayley_neumann,
            num_cayley_neumann_terms=config.num_cayley_neumann_terms,
            cayley_neumann_eps=config.cayley_neumann_eps,
        )

        do_init = getattr(config, "init_weights", True)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.psoft_R[adapter_name].reset_parameters(init_weights=do_init)
        self.psoft_R[adapter_name].requires_grad_(True)

        init_type = self.init_psoft_weights[adapter_name]
        if do_init and isinstance(init_type, str) and init_type.endswith("init"):
            base_w = self.get_base_layer().weight
            if base_w.is_meta:
                self._psoft_cache_built[adapter_name] = False
            else:
                with gather_params_ctx(base_w):
                    self._build_cache_once(adapter_name=adapter_name, init_type=init_type)

        self.set_adapter([adapter_name])

    # Adapted from the asymmetric SVD used in PiSSA
    # (PEFT implementation: https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py) #L316
    def _build_cache_once(self, adapter_name: str, init_type: str) -> None:
        if self._psoft_cache_built.get(adapter_name, False):
            return

        with torch.inference_mode(False), torch.no_grad():
            base = self.get_base_layer()
            weight = base.weight
            dtype = weight.dtype
            if dtype not in (torch.float32, torch.float16, torch.bfloat16):
                raise TypeError("PSOFT init requires float32/float16/bfloat16. Re-quantize after init if needed.")

            # W: (out, in) fp32
            W = transpose(weight.to(torch.float32), self.fan_in_fan_out)

            r = self.r[adapter_name]
            Vr, Sr, Uhr = self._compute_svd_factors(
                W,
                r,
                svd_mode=self.psoft_svd[adapter_name],
                niter=self.psoft_svd_lowrank_niter[adapter_name],
            )

            Sr_scaled = Sr / self.scaling[adapter_name]

            if init_type == "psoft_init":
                A = Uhr  # (r, in)
                B = Vr @ torch.diag(Sr_scaled)  # (out, r)
            elif init_type == "pissa_init":
                s_sqrt = torch.sqrt(Sr_scaled)
                A = torch.diag(s_sqrt) @ Uhr  # (r, in)
                B = Vr @ torch.diag(s_sqrt)  # (out, r)
            else:
                raise ValueError(f"Unknown init_psoft_weights: {init_type}")

            A = A.contiguous().detach()
            B = B.contiguous().detach()

            self._set_cache_buffers(adapter_name, A, B)
            self._psoft_cache_built[adapter_name] = True

    def _compute_svd_factors(self, weight: torch.Tensor, r: int, *, svd_mode: str, niter: int):
        # weight: (out, in) fp32
        if svd_mode == "full":
            U, S, Vh = torch.linalg.svd(weight.data, full_matrices=False)
            Vr = U[:, :r]  # (out, r)
            Sr = S[:r]  # (r,)
            Uhr = Vh[:r, :]  # (r, in)
        elif svd_mode == "lowrank":
            U, S, V = svd_lowrank(weight.data, q=r, niter=niter)  # V: (in, r)
            Vr = U[:, :r]
            Sr = S[:r]
            Uhr = V[:, :r].t()  # (r, in)
        else:
            raise ValueError(f"Unknown svd_mode: {svd_mode}")
        return Vr, Sr, Uhr


class Linear(nn.Module, PSOFTLayer):
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        config: PSOFTConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        PSOFTLayer.__init__(self, base_layer, **kwargs)

        self.fan_in_fan_out = config.fan_in_fan_out
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, config=config, **kwargs)

    def _get_R_matrix(self, adapter_name: str) -> torch.Tensor:
        return self.psoft_R[adapter_name].get_matrix()

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        """
        ΔW = scaling * B (R - id_mat) A
        Returns in base weight layout (respecting fan_in_fan_out).
        """
        if adapter_name not in self.psoft_R:
            raise KeyError(f"Adapter {adapter_name} not found in PSOFT layer.")

        if not self._psoft_cache_built.get(adapter_name, False):
            init_type = self.init_psoft_weights.get(adapter_name, None)
            if isinstance(init_type, str) and init_type.endswith("init"):
                base_w = self.get_base_layer().weight
                with gather_params_ctx(base_w):
                    self._build_cache_once(adapter_name, init_type)
            else:
                raise RuntimeError(f"PSOFT cache not built for adapter {adapter_name} and no init_type provided.")

        A, B = self._get_cache_buffers(adapter_name)
        base_w = self.get_base_layer().weight
        device = base_w.device
        out_dtype = base_w.dtype

        R = self._get_R_matrix(adapter_name)
        r = self.r[adapter_name]

        compute_dtype = (
            torch.float32 if (device.type == "cpu" and out_dtype in (torch.float16, torch.bfloat16)) else out_dtype
        )

        A_c = A.to(device=device, dtype=compute_dtype)
        B_c = B.to(device=device, dtype=compute_dtype)
        R_c = R.to(device=device, dtype=compute_dtype)

        id_mat = torch.eye(r, device=device, dtype=compute_dtype)
        delta = B_c @ (R_c - id_mat) @ A_c  # (out, in)
        delta = transpose(delta, self.fan_in_fan_out)
        delta = delta * self.scaling[adapter_name]

        return delta.to(dtype=out_dtype)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        base_layer = self.get_base_layer()

        for active_adapter in adapter_names:
            if active_adapter not in self.psoft_R:
                continue

            if safe_merge:
                orig_weight = base_layer.weight.data.clone()
                orig_dtype = orig_weight.dtype

                delta_weight = self.get_delta_weight(active_adapter)
                orig_weight += delta_weight.to(orig_dtype)

                if not torch.isfinite(orig_weight).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )

                base_layer.weight.data = orig_weight
            else:
                delta_weight = self.get_delta_weight(active_adapter)
                base_layer.weight.data += delta_weight.to(base_layer.weight.dtype)

            self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.", UserWarning)
            return

        weight = self.get_base_layer().weight

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()

            if active_adapter not in self.psoft_R:
                continue

            orig_dtype = weight.dtype
            delta_weight = self.get_delta_weight(active_adapter)
            weight.data -= delta_weight.to(orig_dtype)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)

        if self.merged:
            return self.base_layer(x, *args, **kwargs)

        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        psoft_keys = self.psoft_R.keys()
        for active_adapter in self.active_adapters:
            if active_adapter not in psoft_keys:
                continue

            if not self._psoft_cache_built.get(active_adapter, False):
                init_type = self.init_psoft_weights.get(active_adapter, None)
                if isinstance(init_type, str) and init_type.endswith("init"):
                    base_w = self.get_base_layer().weight
                    if getattr(base_w, "is_meta", False):
                        continue
                    with gather_params_ctx(base_w):
                        self._build_cache_once(active_adapter, init_type)
                else:
                    raise RuntimeError(
                        f"PSOFT cache not built for adapter {active_adapter} and no init_type provided."
                    )

            A, B = self._get_cache_buffers(active_adapter)

            dropout = self.psoft_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            R_layer = self.psoft_R[active_adapter]

            x_cast = self._cast_input_dtype(x, A.dtype)
            x_d = dropout(x_cast)

            A_c = A.to(device=x_d.device, dtype=x_d.dtype)
            B_c = B.to(device=x_d.device, dtype=x_d.dtype)

            xa = x_d @ A_c.t()
            xr = R_layer(xa)

            delta_y = (xr - xa) @ B_c.t()
            result = result + (delta_y * scaling).to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        return "psoft." + super().__repr__()


def dispatch_default(
    target: nn.Module,
    adapter_name: str,
    config: PSOFTConfig,
    parameter_name: Optional[str] = None,
    **kwargs,
) -> Optional[nn.Module]:
    new_module = None
    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if parameter_name is not None:
        return None

    if isinstance(target_base_layer, torch.nn.Linear):
        if config.fan_in_fan_out:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. Setting to False."
            )
            config.fan_in_fan_out = False
        new_module = Linear(target, adapter_name, config=config, **kwargs)

    return new_module
