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

import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from peft.tuners.tuners_utils import BaseTunerLayer, _get_in_out_features, check_adapters_to_merge
from peft.utils.other import transpose

from .._buffer_dict import BufferDict
from .config import FrodConfig


class FrodLayer(BaseTunerLayer):
    adapter_layer_names = ("frod_lambda_l", "frod_lambda_s_values")
    other_param_names = ("frod_V", "frod_U", "frod_s_indices", "frod_s_size")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.frod_dropout = nn.ModuleDict({})

        # Sparse S is parameterized by its COO values only.
        self.frod_lambda_l = nn.ParameterDict({})
        self.frod_lambda_s_values = nn.ParameterDict({})

        self.frod_s_indices: BufferDict = BufferDict({}, persistent=False)
        self.frod_s_size: BufferDict = BufferDict({}, persistent=False)
        self.frod_V: BufferDict = BufferDict({}, persistent=False)
        self.frod_U: BufferDict = BufferDict({}, persistent=False)

        self._disable_adapters = False
        self.merged_adapters = []
        self._frod_merged_delta = {}

        self.in_features, self.out_features = _get_in_out_features(self.get_base_layer())
        self.kwargs = kwargs

    def update_layer(
        self,
        adapter_name,
        frod_V: BufferDict,
        frod_s_indices: BufferDict,
        frod_s_size: BufferDict,
        config: FrodConfig,
    ):
        frod_dropout = config.frod_dropout
        init_weights = config.init_weights
        base_layer = self.get_base_layer()
        weight = transpose(base_layer.weight, self.fan_in_fan_out)
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        self.r[adapter_name] = self.out_features
        if frod_dropout > 0.0:
            frod_dropout_layer = nn.Dropout(p=frod_dropout)
        else:
            frod_dropout_layer = nn.Identity()

        self.frod_dropout[adapter_name] = frod_dropout_layer

        if frod_V is None or frod_s_indices is None or frod_s_size is None:
            raise ValueError("The FRoD projection buffers are missing. This should not happen.")
        if adapter_name not in frod_V:
            # FRoD projection buffers are shared across adapters for the same module category.
            reference_adapter = next(iter(frod_V))
            frod_V[adapter_name] = frod_V[reference_adapter]
            frod_s_indices[adapter_name] = frod_s_indices[reference_adapter]
            frod_s_size[adapter_name] = frod_s_size[reference_adapter]

        nnz = frod_s_indices[adapter_name].shape[1]
        self.frod_lambda_s_values[adapter_name] = nn.Parameter(torch.zeros(nnz, device=device, dtype=dtype))

        self.frod_V[adapter_name] = frod_V[adapter_name]
        self.frod_s_indices[adapter_name] = frod_s_indices[adapter_name]
        self.frod_s_size[adapter_name] = frod_s_size[adapter_name]

        # Keep cached projections on CPU and move them lazily in forward.
        self.frod_V[adapter_name] = self.frod_V[adapter_name].to(dtype=dtype, device="cpu")
        self.frod_s_indices[adapter_name] = self.frod_s_indices[adapter_name].to(device="cpu", dtype=torch.long)
        self.frod_s_size[adapter_name] = self.frod_s_size[adapter_name].to(device="cpu", dtype=torch.long)

        U, L = self._calculate_frod_u_and_lambda(self.frod_V[adapter_name], weight)
        U = U.to(dtype)
        L = L.to(device=device, dtype=dtype)
        self.frod_lambda_l[adapter_name] = nn.Parameter(L, requires_grad=True)
        if init_weights:
            self.reset_frod_parameters(adapter_name)
        else:
            # PEFT convention: init_weights=False should produce a non-identity adapter for merge tests.
            with torch.no_grad():
                nn.init.normal_(self.frod_lambda_s_values[adapter_name], std=0.05)
                self.frod_lambda_l[adapter_name].add_(torch.randn_like(self.frod_lambda_l[adapter_name]) * 0.05)

        self.frod_U[adapter_name] = U.cpu()
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def _calculate_frod_u_and_lambda(self, V, W):
        w = W.detach().to(torch.float64).cpu()
        v = V.detach().to(torch.float64).cpu()
        try:
            bi = torch.linalg.solve(v, w.T).T
        except RuntimeError:
            bi = w @ torch.linalg.pinv(v, rtol=1e-6).T
        lambda_l = torch.linalg.norm(bi, dim=0)
        u = torch.zeros_like(bi)
        nonzero = lambda_l > 1e-12
        u[:, nonzero] = bi[:, nonzero] / lambda_l[nonzero]
        return u.float(), lambda_l.float()

    def reset_frod_parameters(self, adapter_name):
        if adapter_name in self.frod_lambda_s_values:
            with torch.no_grad():
                nn.init.zeros_(self.frod_lambda_s_values[adapter_name])


class Linear(nn.Linear, FrodLayer):
    def __init__(
        self,
        base_layer,
        frod_V: BufferDict,
        frod_s_indices: BufferDict,
        frod_s_size: BufferDict,
        adapter_name: str,
        config: FrodConfig,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ) -> None:
        super(nn.Linear, self).__init__()
        FrodLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = config.fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, frod_V, frod_s_indices, frod_s_size, config=config)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        base_layer = self.get_base_layer()
        base_weight = base_layer.weight.data.clone()
        for active_adapter in adapter_names:
            if active_adapter in self.frod_lambda_l.keys():
                delta_weight = self._get_delta_weight(active_adapter, base_weight=base_weight)
                if safe_merge:
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += delta_weight
                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += delta_weight
                self._frod_merged_delta[active_adapter] = delta_weight
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.frod_lambda_l.keys():
                delta_weight = self._frod_merged_delta.pop(active_adapter, None)
                if delta_weight is None:
                    delta_weight = self.get_delta_weight(active_adapter)
                self.get_base_layer().weight.data -= delta_weight

    def get_delta_weight(self, adapter) -> torch.Tensor:
        return self._get_delta_weight(adapter)

    def _get_delta_weight(self, adapter, base_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        weight = self.get_base_layer().weight if base_weight is None else base_weight
        device = weight.device
        dtype = weight.dtype
        base_weight = transpose(weight, self.fan_in_fan_out)
        U = self.frod_U[adapter].to(device=device, dtype=dtype)
        V = self.frod_V[adapter].to(device=device, dtype=dtype)
        indices = self.frod_s_indices[adapter].to(device=U.device, dtype=torch.long)
        size = tuple(int(dim) for dim in self.frod_s_size[adapter].tolist())
        values = self.frod_lambda_s_values[adapter].to(U.device, U.dtype).clone()
        lambda_l = self.frod_lambda_l[adapter].to(device=U.device, dtype=U.dtype)

        S_sparse = torch.sparse_coo_tensor(indices, values, size).coalesce()
        S = S_sparse.to_dense()
        L = torch.diag_embed(lambda_l)
        frod_weight = U @ (S + L) @ V.T

        # FRoD parameterizes the adapted weight itself. Return only the difference so PEFT merge/unmerge and
        # disable-adapter behavior preserve the base model while the active adapter still replaces the base weight.
        return transpose(frod_weight - base_weight, self.fan_in_fan_out)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.frod_lambda_s_values:
                    continue

                target_dtype = x.dtype
                V = self.frod_V[active_adapter].to(device=x.device, dtype=target_dtype)
                U = self.frod_U[active_adapter].to(device=x.device, dtype=target_dtype)
                indices = self.frod_s_indices[active_adapter].to(device=x.device, dtype=torch.long)
                size = tuple(int(dim) for dim in self.frod_s_size[active_adapter].tolist())
                values = self.frod_lambda_s_values[active_adapter].to(device=x.device, dtype=target_dtype)
                lambda_l = self.frod_lambda_l[active_adapter].to(device=x.device, dtype=target_dtype)

                x = x.to(target_dtype)
                h = self.frod_dropout[active_adapter](x)

                batch_shape = h.shape[:-1]
                h_flat = h.reshape(-1, h.shape[-1])
                z_flat = torch.matmul(h_flat, V)

                # This block computes the sparse FRoD update z @ S.T with torch.sparse.mm, matching
                # F.linear(h, U @ (S + diag(lambda_l)) @ V.T).
                # CUDA sparse fp16/bf16 kernels are less reliable, so use fp32 here and cast the update back below.
                matmul_dtype = z_flat.dtype
                if z_flat.is_cuda and matmul_dtype in (torch.float16, torch.bfloat16):
                    matmul_dtype = torch.float32

                values = values.to(device=z_flat.device, dtype=matmul_dtype)
                z_flat_mm = z_flat.to(matmul_dtype)
                S_sparse = torch.sparse_coo_tensor(indices, values, size).coalesce()
                if S_sparse.dtype != matmul_dtype:
                    S_sparse = S_sparse.to(dtype=matmul_dtype)
                z_S_flat = torch.sparse.mm(S_sparse, z_flat_mm.t()).t()

                lambda_l = lambda_l.to(device=z_flat.device, dtype=matmul_dtype)
                z_L_flat = z_flat_mm * lambda_l

                U_mm = U.to(device=z_flat.device, dtype=matmul_dtype)
                out_add_flat = F.linear(z_S_flat + z_L_flat, U_mm)
                out_add_flat = out_add_flat.to(target_dtype)
                out_add = out_add_flat.reshape(*batch_shape, out_add_flat.shape[-1])
                base_weight = transpose(self.get_base_layer().weight, self.fan_in_fan_out).to(
                    device=x.device, dtype=target_dtype
                )
                base_out = F.linear(x, base_weight)

                result = result - base_out + out_add

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        # Match PEFT tuner convention so printed models show FRoD-wrapped layers as `frod.*`.
        rep = super().__repr__()
        return "frod." + rep

    def _move_adapter_to_device_of_base_layer(self, adapter_name: str, device: Optional[torch.device] = None) -> None:
        dtype = None
        weight = None
        if device is None:
            for weight_name in ("weight", "qweight"):
                weight = getattr(self.get_base_layer(), weight_name, None)
                if weight is not None:
                    device = weight.device
                    dtype = weight.dtype
                    break
            else:
                return

        for adapter_layer_name in self.adapter_layer_names:
            adapter_layer = getattr(self, adapter_layer_name, None)
            if not isinstance(adapter_layer, nn.ParameterDict):
                continue
            if adapter_name not in adapter_layer:
                continue
            param = adapter_layer[adapter_name]
            if param.is_meta:
                continue
            adapter_layer[adapter_name] = param.to(device, dtype=dtype)
