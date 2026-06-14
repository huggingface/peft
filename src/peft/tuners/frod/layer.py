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
    other_param_names = ("frod_V", "frod_U", "frod_s_indices", "frod_s_size", "runtime_offload_base_weight")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.frod_dropout = nn.ModuleDict({})
        self.runtime_offload_base_weight = {}

        # Sparse S is parameterized by its COO values only.
        self.frod_lambda_l = nn.ParameterDict({})
        self.frod_lambda_s_values = nn.ParameterDict({})

        # These are references to tuner-level projection buffers. Registering per-layer copies would duplicate the
        # same shared V/S tensors on device when `model.to(...)` is called.
        self.frod_s_indices: Optional[BufferDict] = None
        self.frod_s_size: Optional[BufferDict] = None
        self.frod_V: Optional[BufferDict] = None
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
        weight = base_layer.weight.T if self.fan_in_fan_out else base_layer.weight
        device, dtype = self._get_adapter_target_device_dtype(base_layer.weight)

        self.r[adapter_name] = self.out_features
        self.runtime_offload_base_weight[adapter_name] = config.runtime_offload_base_weight
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

        frod_v = frod_V[adapter_name].to(dtype=dtype, device="cpu")
        frod_s_index = frod_s_indices[adapter_name].to(device="cpu", dtype=torch.long)

        nnz = frod_s_index.shape[1]
        self.frod_lambda_s_values[adapter_name] = nn.Parameter(torch.zeros(nnz, device=device, dtype=dtype))

        # Keep FrodModel as the only registered owner of these shared buffers. Assigning them as normal attributes
        # would register the same projection containers under every wrapped layer and duplicate traversal/state keys.
        self.__dict__["frod_V"] = frod_V
        self.__dict__["frod_s_indices"] = frod_s_indices
        self.__dict__["frod_s_size"] = frod_s_size

        U, L = self._calculate_frod_u_and_lambda(frod_v, weight)
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
        self._move_adapter_to_device_of_base_layer(adapter_name, device=device)
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
        adapter_deltas = []
        # FRoD deltas are computed against the current base weight, so compute all deltas before mutating it.
        for active_adapter in adapter_names:
            if active_adapter in self.frod_lambda_l.keys():
                adapter_deltas.append((active_adapter, self.get_delta_weight(active_adapter)))

        for active_adapter, delta_weight in adapter_deltas:
            delta_weight = delta_weight.to(device=base_layer.weight.device, dtype=base_layer.weight.dtype)
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

    def unload_and_optionally_merge_module(
        self, merge: bool, safe_merge: bool, adapter_names: Optional[list[str]]
    ) -> nn.Module:
        if merge:
            self.merge(safe_merge=safe_merge, adapter_names=adapter_names)
            self._move_base_weight_to_device_of_adapter(self.active_adapters[0] if self.active_adapters else None)
        else:
            self._move_base_weight_to_device_of_adapter(self.active_adapters[0] if self.active_adapters else None)
        return self.get_base_layer()

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
                base_weight = self.get_base_layer().weight
                delta_weight = delta_weight.to(device=base_weight.device, dtype=base_weight.dtype)
                base_weight.data -= delta_weight

    def get_delta_weight(self, adapter) -> torch.Tensor:
        self._move_base_weight_to_device_of_adapter(adapter)
        weight = self.get_base_layer().weight
        device = weight.device
        dtype = weight.dtype
        base_weight = transpose(weight, self.fan_in_fan_out)
        U, V, S_sparse, lambda_l = self._get_frod_tensors(adapter, device=device, dtype=dtype)
        S = S_sparse.to_dense()
        L = torch.diag_embed(lambda_l)
        frod_weight = U @ (S + L) @ V.T

        # FRoD parameterizes the adapted weight itself. Return only the difference so PEFT merge/unmerge and
        # disable-adapter behavior preserve the base model while the active adapter still replaces the base weight.
        return transpose(frod_weight - base_weight, self.fan_in_fan_out)

    def _get_frod_tensors(
        self, adapter: str, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.frod_V is None or self.frod_s_indices is None or self.frod_s_size is None:
            raise ValueError("FRoD projection buffers are missing. This should not happen.")

        U = self.frod_U[adapter].to(device=device, dtype=dtype)
        V = self.frod_V[adapter].to(device=device, dtype=dtype)
        indices = self.frod_s_indices[adapter].to(device=device, dtype=torch.long)
        size = tuple(int(dim) for dim in self.frod_s_size[adapter].tolist())
        values = self.frod_lambda_s_values[adapter].to(device=device, dtype=dtype)
        lambda_l = self.frod_lambda_l[adapter].to(device=device, dtype=dtype)
        S_sparse = torch.sparse_coo_tensor(indices, values, size).coalesce()
        return U, V, S_sparse, lambda_l

    def _sparse_activation_mm(self, z_flat: torch.Tensor, S_sparse: torch.Tensor) -> torch.Tensor:
        if S_sparse._nnz() == 0:
            return torch.zeros_like(z_flat)
        if z_flat.dtype in (torch.float16, torch.bfloat16):
            # Some backends do not implement sparse addmm for fp16/bf16. This computes z @ S.T directly from COO
            # entries and keeps the activation-side path in the requested dtype.
            rows, cols = S_sparse.indices()
            updates = z_flat.index_select(1, cols) * S_sparse.values()
            result = torch.zeros_like(z_flat)
            result.index_add_(1, rows, updates)
            return result
        return torch.sparse.mm(S_sparse, z_flat.t()).t()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            self._move_base_weight_to_device(x.device)
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            self._move_base_weight_to_device(x.device)
            result = self.base_layer(x, *args, **kwargs)
        else:
            target_dtype = x.dtype
            base_layer = self.get_base_layer()
            bias = base_layer.bias
            active_adapters = [
                active_adapter
                for active_adapter in self.active_adapters
                if active_adapter in self.frod_lambda_s_values
            ]
            skip_base_layer = len(active_adapters) == 1 and (
                isinstance(self.frod_dropout[active_adapters[0]], nn.Identity)
                or not self.frod_dropout[active_adapters[0]].training
            )
            # With one active adapter and no stochastic dropout, FRoD computes the reconstructed adapted weight
            # directly, so the target base linear result is not needed for the forward pass.
            if skip_base_layer:
                if self.runtime_offload_base_weight.get(active_adapters[0], False):
                    self._offload_base_weight_to_cpu()
                result = None
                base_out = None
            else:
                self._move_base_weight_to_device(x.device)
                result = self.base_layer(x, *args, **kwargs)
                base_out = result if bias is None else result - bias

            base_weight = None
            for active_adapter in active_adapters:
                U, V, S_sparse, lambda_l = self._get_frod_tensors(active_adapter, device=x.device, dtype=target_dtype)

                dropout = self.frod_dropout[active_adapter]
                h = dropout(x)

                batch_shape = h.shape[:-1]
                h_flat = h.reshape(-1, h.shape[-1])
                z_flat = torch.matmul(h_flat, V)

                z_S_flat = self._sparse_activation_mm(z_flat, S_sparse)
                z_L_flat = z_flat * lambda_l

                out_add_flat = F.linear(z_S_flat + z_L_flat, U)
                out_add = out_add_flat.reshape(*batch_shape, out_add_flat.shape[-1])
                # FRoD reconstructs the adapted weight directly, so subtract the base-weight contribution and only
                # accumulate the adapter delta.
                if skip_base_layer:
                    adapter_base_out = None
                elif isinstance(dropout, nn.Identity) or not dropout.training:
                    adapter_base_out = base_out
                else:
                    if base_weight is None:
                        base_weight = transpose(base_layer.weight, self.fan_in_fan_out).to(
                            device=x.device, dtype=target_dtype
                        )
                    adapter_base_out = F.linear(h, base_weight)

                if adapter_base_out is None:
                    result = out_add if bias is None else out_add + bias
                else:
                    result = result + out_add - adapter_base_out

            if result is None:
                self._move_base_weight_to_device(x.device)
                result = self.base_layer(x, *args, **kwargs)

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "frod." + rep

    def _move_adapter_to_device_of_base_layer(self, adapter_name: str, device: Optional[torch.device] = None) -> None:
        """Move trainable FRoD parameters while keeping shared projection buffers on CPU."""
        base_layer = self.get_base_layer()
        base_device, base_dtype = self._get_base_layer_device_and_dtype(base_layer)

        target_device = device if device is not None else base_device
        adapter_device, adapter_dtype = self._get_existing_adapter_device_dtype(adapter_name)
        if (target_device is None or target_device.type == "cpu") and adapter_device is not None:
            target_device = adapter_device
            if base_dtype is None:
                base_dtype = adapter_dtype
        if target_device is None:
            return

        target_dtype = None
        if base_dtype is not None and (base_dtype.is_floating_point or base_dtype.is_complex):
            target_dtype = base_dtype

        for adapter_layer_name in self.adapter_layer_names:
            adapter_layer = getattr(self, adapter_layer_name, None)
            if not isinstance(adapter_layer, nn.ParameterDict):
                continue
            if adapter_name not in adapter_layer:
                continue
            param = adapter_layer[adapter_name]
            if param.is_meta:
                continue
            if target_dtype is not None:
                adapter_layer[adapter_name] = param.to(target_device, dtype=target_dtype)
            else:
                adapter_layer[adapter_name] = param.to(target_device)

    def _get_existing_adapter_device_dtype(
        self, adapter_name: Optional[str] = None
    ) -> tuple[Optional[torch.device], Optional[torch.dtype]]:
        adapter_names = [adapter_name] if adapter_name is not None else []
        adapter_names.extend(name for name in self.frod_lambda_l.keys() if name != adapter_name)

        for name in adapter_names:
            for adapter_layer_name in self.adapter_layer_names:
                adapter_layer = getattr(self, adapter_layer_name, None)
                if not isinstance(adapter_layer, nn.ParameterDict) or name not in adapter_layer:
                    continue
                param = adapter_layer[name]
                if param.is_meta:
                    continue
                return param.device, param.dtype
        return None, None

    def _get_adapter_target_device_dtype(self, weight: torch.Tensor) -> tuple[torch.device, torch.dtype]:
        if weight.device.type != "cpu":
            return weight.device, weight.dtype

        adapter_device, adapter_dtype = self._get_existing_adapter_device_dtype()
        if adapter_device is not None:
            return adapter_device, adapter_dtype
        return weight.device, weight.dtype

    def _move_base_weight_to_device(self, device: torch.device) -> None:
        weight = self.get_base_layer().weight
        if weight.is_meta or weight.device == device:
            return
        weight.data = weight.data.to(device=device)
        if weight.grad is not None:
            weight.grad = weight.grad.to(device=device)

    def _move_base_weight_to_device_of_adapter(self, adapter_name: Optional[str]) -> None:
        adapter_device, _ = self._get_existing_adapter_device_dtype(adapter_name)
        bias = getattr(self.get_base_layer(), "bias", None)
        bias_device = getattr(bias, "device", None)
        if bias_device is not None and (adapter_device is None or adapter_device.type == "cpu"):
            adapter_device = bias_device
        if adapter_device is not None:
            self._move_base_weight_to_device(adapter_device)

    def _offload_base_weight_to_cpu(self) -> None:
        weight = self.get_base_layer().weight
        if weight.is_meta or weight.device.type == "cpu":
            return
        weight.data = weight.data.cpu()
        if weight.grad is not None:
            weight.grad = weight.grad.cpu()
