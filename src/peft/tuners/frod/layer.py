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

import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from numpy.linalg import inv
from torch import nn
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .._buffer_dict import BufferDict


class FRODLayer(BaseTunerLayer):
    adapter_layer_names = ("frod_lambda_l", "frod_lambda_s_values")
    other_param_names = ("frod_V", "frod_U", "frod_s_indices", "frod_s_size")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.frod_dropout = nn.ModuleDict({})

        # Sparse S is parameterized by its COO values only.
        self.frod_lambda_l = nn.ParameterDict({})
        self.frod_lambda_s_values = nn.ParameterDict({})

        self.frod_s_indices: Optional[BufferDict] = None
        self.frod_s_size: Optional[BufferDict] = None
        self.frod_V: Optional[BufferDict] = None
        self.frod_U: BufferDict = BufferDict({}, persistent=False)

        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name,
        frod_V: BufferDict,
        frod_s_indices: BufferDict,
        frod_s_size: BufferDict,
        frod_dropout,
        init_weights,
    ):
        base_layer = self.get_base_layer()
        weight = base_layer.weight.T if isinstance(base_layer, Conv1D) else base_layer.weight
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        param_dtype = dtype

        self.r[adapter_name] = self.out_features
        if frod_dropout > 0.0:
            frod_dropout_layer = nn.Dropout(p=frod_dropout)
        else:
            frod_dropout_layer = nn.Identity()

        self.frod_dropout.update(nn.ModuleDict({adapter_name: frod_dropout_layer}))

        if adapter_name not in frod_V:
            if not frod_V:
                raise ValueError("The FRoD projection buffers are empty. This should not happen.")
            frod_V[adapter_name] = next(iter(frod_V.values()))
            frod_s_indices[adapter_name] = next(iter(frod_s_indices.values()))
            frod_s_size[adapter_name] = next(iter(frod_s_size.values()))

        nnz = frod_s_indices[adapter_name].shape[1]
        self.frod_lambda_s_values[adapter_name] = nn.Parameter(torch.zeros(nnz, device=device, dtype=param_dtype))

        self.__dict__["frod_V"] = frod_V
        self.__dict__["frod_s_indices"] = frod_s_indices
        self.__dict__["frod_s_size"] = frod_s_size

        # Keep cached projections on CPU and move them lazily in forward.
        self.frod_V[adapter_name] = self.frod_V[adapter_name].to(dtype=param_dtype, device="cpu")
        self.frod_s_indices[adapter_name] = self.frod_s_indices[adapter_name].to(device="cpu", dtype=torch.long)
        self.frod_s_size[adapter_name] = self.frod_s_size[adapter_name].to(device="cpu", dtype=torch.long)

        U, L = self._calculate_frod_u_and_lambda(self.frod_V[adapter_name], weight)
        U = U.to(param_dtype)
        L = L.to(device=device, dtype=param_dtype)
        self.frod_lambda_l[adapter_name] = nn.Parameter(L, requires_grad=True)
        if init_weights:
            self.reset_frod_parameters(adapter_name)
        else:
            # PEFT convention: init_weights=False should produce a non-identity adapter for merge tests.
            with torch.no_grad():
                nn.init.normal_(self.frod_lambda_s_values[adapter_name], std=0.02)
                self.frod_lambda_l[adapter_name].add_(torch.randn_like(self.frod_lambda_l[adapter_name]) * 0.02)

        self.frod_U[adapter_name] = U.cpu()
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def _calculate_frod_u_and_lambda(self, V, W):
        w = W.detach().to(torch.float32).cpu().numpy()
        v = V.detach().to(torch.float32).cpu().numpy()
        try:
            v_inv_T = inv(v).T
        except np.linalg.LinAlgError:
            v_inv_T = np.linalg.pinv(v, rcond=1e-6).T
        Bi = w @ v_inv_T
        lambda_l = np.linalg.norm(Bi, axis=0)
        u = np.divide(Bi, lambda_l, out=np.zeros_like(Bi), where=lambda_l > 1e-8)
        U = torch.from_numpy(u).float()
        L = torch.from_numpy(lambda_l).float()
        return U, L

    def reset_frod_parameters(self, adapter_name):
        if adapter_name in self.frod_lambda_s_values:
            with torch.no_grad():
                nn.init.zeros_(self.frod_lambda_s_values[adapter_name])
        if adapter_name in self.frod_lambda_l:
            with torch.no_grad():
                nn.init.zeros_(self.frod_lambda_l[adapter_name])


class Linear(nn.Linear, FRODLayer):
    def __init__(
        self,
        base_layer,
        frod_V: BufferDict,
        frod_s_indices: BufferDict,
        frod_s_size: BufferDict,
        adapter_name: str,
        frod_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        init_weights: bool = True,
        **kwargs,
    ) -> None:
        super(nn.Linear, self).__init__()
        FRODLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, frod_V, frod_s_indices, frod_s_size, frod_dropout, init_weights)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.frod_lambda_l.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)
                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.frod_lambda_l.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        weight = self.get_base_layer().weight
        device = weight.device
        dtype = weight.dtype
        U = self.frod_U[adapter].to(device=device, dtype=dtype)
        V = self.frod_V[adapter].to(device=device, dtype=dtype)
        indices = self.frod_s_indices[adapter].to(device=U.device, dtype=torch.long)
        size_tensor = self.frod_s_size[adapter]
        if isinstance(size_tensor, torch.Tensor):
            size = tuple(int(dim) for dim in size_tensor.tolist())
        else:
            size = tuple(int(dim) for dim in size_tensor)
        values = self.frod_lambda_s_values[adapter].to(U.device, U.dtype).clone()
        lambda_l = self.frod_lambda_l[adapter].to(device=U.device, dtype=U.dtype)

        S_sparse = torch.sparse_coo_tensor(indices, values, size).coalesce()
        S = S_sparse.to_dense()
        L = torch.diag_embed(lambda_l)

        return transpose(U @ (S + L).T @ V.T, self.fan_in_fan_out)

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
                size_tensor = self.frod_s_size[active_adapter]
                if isinstance(size_tensor, torch.Tensor):
                    size = tuple(int(dim) for dim in size_tensor.tolist())
                else:
                    size = tuple(int(dim) for dim in size_tensor)
                values = self.frod_lambda_s_values[active_adapter].to(device=x.device, dtype=target_dtype)
                lambda_l = self.frod_lambda_l[active_adapter].to(device=x.device, dtype=target_dtype)

                x = x.to(target_dtype)
                h = self.frod_dropout[active_adapter](x)

                batch_shape = h.shape[:-1]
                h_flat = h.reshape(-1, h.shape[-1])
                z_flat = torch.matmul(h_flat, V)

                matmul_dtype = z_flat.dtype
                if z_flat.is_cuda and matmul_dtype in (torch.float16, torch.bfloat16):
                    matmul_dtype = torch.float32

                values = values.to(device=z_flat.device, dtype=matmul_dtype)
                z_flat_mm = z_flat.to(matmul_dtype)
                S_sparse = torch.sparse_coo_tensor(indices, values, size).coalesce()
                if S_sparse.dtype != matmul_dtype:
                    S_sparse = S_sparse.to(dtype=matmul_dtype)
                z_S_flat = torch.sparse.mm(S_sparse.t(), z_flat_mm.t()).t()

                lambda_l = lambda_l.to(device=z_flat.device, dtype=matmul_dtype)
                z_L_flat = z_flat_mm * lambda_l

                U_mm = U.to(device=z_flat.device, dtype=matmul_dtype)
                out_add_flat = F.linear(z_S_flat + z_L_flat, U_mm)
                out_add_flat = out_add_flat.to(target_dtype)
                out_add = out_add_flat.reshape(*batch_shape, out_add_flat.shape[-1])

                result = result + out_add

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
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
