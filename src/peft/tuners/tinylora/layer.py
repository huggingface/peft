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

from __future__ import annotations

import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .._buffer_dict import BufferDict


class TinyLoraLayer(BaseTunerLayer):
    """
    TinyLoRA layer implementation.

    TinyLoRA is based on LoRA-XS and uses SVD decomposition of frozen weights. The key innovation is replacing the
    trainable r×r matrix R with:
        R = sum_i(v[i] * P[i])
    where v is a tiny trainable vector and P_i are fixed random projection matrices.

    The forward pass computes:
        result += lora_B(R(lora_A(x)))
    where lora_A and lora_B are frozen SVD components.
    """

    # List all names of layers that may contain adapter weights
    # Note: tinylora_v is stored at model level and shared across layers
    # The layer stores a reference to it (like VeRA does with vera_A/vera_B)
    adapter_layer_names = ("tinylora_v",)
    other_param_names = ("tinylora_A", "tinylora_B", "tinylora_P")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.u = {}
        self.tinylora_dropout = nn.ModuleDict({})

        # Stores reference to shared trainable vectors (at model level)
        # The key will be like "{adapter_name}_group_{group_idx}"
        self.tinylora_v: Optional[nn.ParameterDict] = None
        self.tinylora_v_key: dict[str, str] = {}  # adapter_name -> key in tinylora_v

        # Frozen SVD components as buffers (following LoRA-XS convention)
        # tinylora_A corresponds to V from SVD (shape: r x in_features)
        # tinylora_B corresponds to U @ diag(S) from SVD (shape: out_features x r)
        self.tinylora_A = BufferDict({}, persistent=True)
        self.tinylora_B = BufferDict({}, persistent=True)

        # Fixed random projection tensors P ∈ R^{u×r×r}
        self.tinylora_P = BufferDict({}, persistent=True)

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        # Track layer index for seeding
        self._layer_idx: Optional[int] = None

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        else:
            # Fallback for other layer types
            in_features, out_features = None, None

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def _all_available_adapter_names(self) -> list[str]:
        """Return a sorted list of all available adapter names.

        Override the base class method because TinyLoRA's tinylora_v uses keys like '{adapter_name}_group_{idx}'
        instead of just the adapter name. We use tinylora_v_key which maps adapter_name -> key in tinylora_v.
        """
        adapter_names = set()
        # Use tinylora_v_key which has adapter names as keys
        adapter_names.update(self.tinylora_v_key.keys())
        # Also check other_param_names which use adapter names directly
        for name in self.other_param_names:
            attr = getattr(self, name, None)
            if attr is not None and hasattr(attr, "keys"):
                adapter_names.update(attr.keys())
        return sorted(adapter_names)

    def set_layer_idx(self, idx: int):
        """Set the layer index, used for deterministic seeding of projection matrices."""
        self._layer_idx = idx

    def _get_layer_seed(self, adapter_name: str, base_seed: int) -> int:
        """Get a deterministic seed for this layer's projection matrices."""
        if self._layer_idx is not None:
            return base_seed + self._layer_idx
        # Fallback: use hash of the adapter name
        return base_seed + hash(adapter_name) % 10000

    def update_layer(
        self,
        adapter_name: str,
        tinylora_v: nn.ParameterDict,
        tinylora_v_key: str,
        r: int,
        u: int,
        tinylora_dropout: float,
        init_weights: bool,
        projection_seed: int,
        fan_in_fan_out: bool = False,
        inference_mode: bool = False,
        **kwargs,
    ):
        """Initialize layer with SVD decomposition and projection tensors."""
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        if u <= 0:
            raise ValueError(f"`u` should be a positive integer value but the value passed is {u}")

        self.u[adapter_name] = u

        if tinylora_dropout > 0.0:
            tinylora_dropout_layer = nn.Dropout(p=tinylora_dropout)
        else:
            tinylora_dropout_layer = nn.Identity()

        self.tinylora_dropout.update(nn.ModuleDict({adapter_name: tinylora_dropout_layer}))

        # Store reference to shared v
        self.tinylora_v = tinylora_v
        self.tinylora_v_key[adapter_name] = tinylora_v_key

        # Compute truncated SVD of base weights (following LoRA-XS convention)
        # actual_r may be less than r if matrix dimensions are smaller
        actual_r = self._init_svd(adapter_name, r, fan_in_fan_out)
        self.r[adapter_name] = actual_r

        # Initialize random projection tensors P using the actual rank
        self._init_projection(adapter_name, u, actual_r, projection_seed)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=inference_mode)

    def _init_svd(self, adapter_name: str, r: int, fan_in_fan_out: bool) -> int:
        """
        Compute truncated SVD of base weights and store as frozen buffers.

        Following LoRA-XS convention:
        - W = U @ S @ V^T (full SVD)
        - We store: tinylora_A = V[:r, :] (shape: r x in_features)
        - We store: tinylora_B = U[:, :r] @ diag(S[:r]) (shape: out_features x r)

        This allows: delta_W = tinylora_B @ R @ tinylora_A

        Returns:
            int: The actual rank used (may be less than r if matrix dimensions are smaller)
        """
        base_layer = self.get_base_layer()
        weight = base_layer.weight.data

        # Handle Conv1D which stores weights as (in, out)
        weight = transpose(weight, fan_in_fan_out)

        dtype = weight.dtype
        device = weight.device

        # Compute SVD in float32 for numerical stability
        # W has shape (out_features, in_features)
        weight_fp32 = weight.float()
        U, S, Vh = torch.linalg.svd(weight_fp32, full_matrices=False)

        # The actual rank is limited by the matrix dimensions
        max_rank = min(weight.shape[0], weight.shape[1])
        actual_r = min(r, max_rank)

        # Truncate to actual rank
        # U: (out_features, min(out, in)) -> U_r: (out_features, actual_r)
        # S: (min(out, in),) -> S_r: (actual_r,)
        # Vh: (min(out, in), in_features) -> V_r: (actual_r, in_features)
        U_r = U[:, :actual_r].to(dtype)
        S_r = S[:actual_r].to(dtype)
        V_r = Vh[:actual_r, :].to(dtype)

        # Store in LoRA-XS convention:
        # tinylora_A = V_r (actual_r x in_features) - acts like lora_A
        # tinylora_B = U_r @ diag(S_r) (out_features x actual_r) - acts like lora_B
        # Use .contiguous() to ensure tensors can be saved with safetensors
        self.tinylora_A[adapter_name] = V_r.contiguous()
        self.tinylora_B[adapter_name] = (
            U_r * S_r.unsqueeze(0)
        ).contiguous()  # Broadcasting: (out, actual_r) * (1, actual_r)

        return actual_r

    def _init_projection(self, adapter_name: str, u: int, r: int, base_seed: int):
        """Initialize fixed random projection tensors P ∈ R^{u×r×r}."""
        seed = self._get_layer_seed(adapter_name, base_seed)
        gen = torch.Generator().manual_seed(seed)

        # P has shape (u, r, r)
        # Note: The paper describes P as "fixed random matrices" but does not specify the distribution.
        # We use Gaussian (torch.randn) with 1/√r normalization, which is standard for random projections
        # (see Johnson-Lindenstrauss lemma: https://en.wikipedia.org/wiki/Johnson-Lindenstrauss_lemma).
        P = torch.randn(u, r, r, generator=gen)
        P = P / (r**0.5)

        self.tinylora_P[adapter_name] = P

    def _compute_R(self, adapter_name: str) -> torch.Tensor:
        """Reconstruct R matrix from v and P: R = sum_i(v[i] * P[i])."""
        v_key = self.tinylora_v_key[adapter_name]
        v = self.tinylora_v[v_key]  # Shape: (u,)
        P = self.tinylora_P[adapter_name]  # Shape: (u, r, r)

        # Move P to same device/dtype as v
        P = P.to(device=v.device, dtype=v.dtype)

        # R = sum over i of v[i] * P[i]
        R = torch.einsum("i,ijk->jk", v, P)  # Shape: (r, r)
        return R

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        """
        Compute delta_W = tinylora_B @ R @ tinylora_A for merging.

        Returns weight update in shape (out_features, in_features).
        """
        A = self.tinylora_A[adapter_name]  # (r, in_features)
        B = self.tinylora_B[adapter_name]  # (out_features, r)
        R = self._compute_R(adapter_name)  # (r, r)

        device = A.device
        dtype = A.dtype

        # Move components to same device/dtype
        B = B.to(device=device, dtype=dtype)
        R = R.to(device=device, dtype=dtype)

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        if cast_to_fp32:
            A = A.float()
            B = B.float()
            R = R.float()

        # delta_W = B @ R @ A
        # B: (out, r), R: (r, r), A: (r, in)
        # Result: (out, in)
        delta = B @ R @ A

        if cast_to_fp32:
            delta = delta.to(dtype=dtype)

        return delta


class Linear(nn.Linear, TinyLoraLayer):
    """TinyLoRA implemented in a dense layer."""

    def __init__(
        self,
        base_layer: nn.Module,
        tinylora_v: nn.ParameterDict,
        tinylora_v_key: str,
        adapter_name: str,
        r: int = 2,
        u: int = 64,
        tinylora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        init_weights: bool = True,
        projection_seed: int = 42,
        **kwargs,
    ) -> None:
        # this gets the init from nn.Linear's super perspective, i.e. nn.Module.__init__
        super(nn.Linear, self).__init__()
        TinyLoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            tinylora_v,
            tinylora_v_key,
            r,
            u,
            tinylora_dropout,
            init_weights,
            projection_seed,
            fan_in_fan_out=fan_in_fan_out,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights.

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.tinylora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    delta_weight = transpose(delta_weight, self.fan_in_fan_out)
                    orig_weights += delta_weight

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    delta_weight = transpose(delta_weight, self.fan_in_fan_out)
                    base_layer.weight.data += delta_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """Unmerge all merged adapters from the base weights."""
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.tinylora_A.keys():
                delta_weight = self.get_delta_weight(active_adapter)
                delta_weight = transpose(delta_weight, self.fan_in_fan_out)
                self.get_base_layer().weight.data -= delta_weight

    def delete_adapter(self, adapter_name: str) -> None:
        """Delete an adapter from the layer.

        Override to handle tinylora_v_key which maps adapter names to shared v keys.
        """
        # Delete from tinylora_v_key (adapter name -> v key mapping)
        if adapter_name in self.tinylora_v_key:
            del self.tinylora_v_key[adapter_name]

        # Delete from other params that use adapter name directly
        for attr in self.other_param_names:
            param_dict = getattr(self, attr, None)
            if param_dict is not None and adapter_name in param_dict:
                del param_dict[adapter_name]

        # Delete r and u tracking
        if adapter_name in self.r:
            del self.r[adapter_name]
        if adapter_name in self.u:
            del self.u[adapter_name]

        # Delete dropout layer
        if adapter_name in self.tinylora_dropout:
            del self.tinylora_dropout[adapter_name]

        # Handle active adapters
        if adapter_name in self.active_adapters:
            active_adapters = self.active_adapters[:]
            active_adapters.remove(adapter_name)
            if active_adapters:
                self.set_adapter(active_adapters)
            else:
                remaining_adapters = self._all_available_adapter_names()
                if not remaining_adapters:
                    self.set_adapter([])
                else:
                    new_active_adapter = remaining_adapters[0]
                    warnings.warn(
                        f"Adapter {adapter_name} was active which is now deleted. Setting active adapter to "
                        f"{new_active_adapter}."
                    )
                    self.set_adapter(new_active_adapter)

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
                if active_adapter not in self.tinylora_A.keys():
                    continue

                A = self.tinylora_A[active_adapter]  # (r, in_features)
                B = self.tinylora_B[active_adapter]  # (out_features, r)
                R = self._compute_R(active_adapter)  # (r, r)

                dropout = self.tinylora_dropout[active_adapter]
                x_dropped = dropout(x)
                x_dropped = x_dropped.to(A.dtype)

                # Move components to input device
                device = x_dropped.device
                A = A.to(device)
                B = B.to(device)
                R = R.to(device)

                # Forward computation following LoRA-XS pattern:
                # delta = x @ A^T @ R^T @ B^T
                # Using F.linear(x, W) = x @ W^T:
                # h = F.linear(x, A) -> x @ A^T -> (batch, seq, r)
                # h = F.linear(h, R) -> h @ R^T -> (batch, seq, r)
                # delta = F.linear(h, B) -> h @ B^T -> (batch, seq, out)
                h = F.linear(x_dropped, A)  # (batch, seq, r)
                h = F.linear(h, R)  # (batch, seq, r)
                delta = F.linear(h, B)  # (batch, seq, out)

                result = result + delta

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "tinylora." + rep


class Embedding(nn.Module, TinyLoraLayer):
    """TinyLoRA implemented in an Embedding layer."""

    def __init__(
        self,
        base_layer: nn.Module,
        tinylora_v: nn.ParameterDict,
        tinylora_v_key: str,
        adapter_name: str,
        r: int = 2,
        u: int = 64,
        tinylora_dropout: float = 0.0,
        init_weights: bool = True,
        projection_seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__()
        TinyLoraLayer.__init__(self, base_layer, **kwargs)

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            tinylora_v,
            tinylora_v_key,
            r,
            u,
            tinylora_dropout,
            init_weights,
            projection_seed,
        )

    def update_layer(
        self,
        adapter_name: str,
        tinylora_v: nn.ParameterDict,
        tinylora_v_key: str,
        r: int,
        u: int,
        tinylora_dropout: float,
        init_weights: bool,
        projection_seed: int,
        inference_mode: bool = False,
        **kwargs,
    ):
        """Initialize layer with SVD decomposition and projection tensors."""
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        if u <= 0:
            raise ValueError(f"`u` should be a positive integer value but the value passed is {u}")

        self.u[adapter_name] = u

        if tinylora_dropout > 0.0:
            tinylora_dropout_layer = nn.Dropout(p=tinylora_dropout)
        else:
            tinylora_dropout_layer = nn.Identity()

        self.tinylora_dropout.update(nn.ModuleDict({adapter_name: tinylora_dropout_layer}))

        # Store reference to shared v
        self.tinylora_v = tinylora_v
        self.tinylora_v_key[adapter_name] = tinylora_v_key

        # Compute truncated SVD of embedding weights
        actual_r = self._init_svd_embedding(adapter_name, r)
        self.r[adapter_name] = actual_r

        # Initialize random projection tensors P using the actual rank
        self._init_projection(adapter_name, u, actual_r, projection_seed)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=inference_mode)

    def _init_svd_embedding(self, adapter_name: str, r: int) -> int:
        """
        Compute truncated SVD of embedding weights and store as frozen buffers.

        Embedding weight shape: (num_embeddings, embedding_dim) We treat this as W where:
        - W = U @ S @ V^T (full SVD)
        - tinylora_A = V[:r, :] (shape: r x embedding_dim)
        - tinylora_B = U[:, :r] @ diag(S[:r]) (shape: num_embeddings x r)

        Returns:
            int: The actual rank used (may be less than r if dimensions are smaller)
        """
        base_layer = self.get_base_layer()
        weight = base_layer.weight.data  # (num_embeddings, embedding_dim)

        dtype = weight.dtype
        device = weight.device

        # Compute SVD in float32 for numerical stability
        weight_fp32 = weight.float()
        U, S, Vh = torch.linalg.svd(weight_fp32, full_matrices=False)

        # The actual rank is limited by the matrix dimensions
        max_rank = min(weight.shape[0], weight.shape[1])
        actual_r = min(r, max_rank)

        # Truncate to actual rank
        U_r = U[:, :actual_r].to(dtype)
        S_r = S[:actual_r].to(dtype)
        V_r = Vh[:actual_r, :].to(dtype)

        # Store in LoRA-XS convention:
        # tinylora_A = V_r (actual_r x embedding_dim)
        # tinylora_B = U_r @ diag(S_r) (num_embeddings x actual_r)
        self.tinylora_A[adapter_name] = V_r.contiguous()
        self.tinylora_B[adapter_name] = (U_r * S_r.unsqueeze(0)).contiguous()

        return actual_r

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """Merge the active adapter weights into the base weights."""
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter in self.tinylora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    orig_weights = base_layer.weight.data.clone()
                    delta_weight = self.get_delta_weight(active_adapter)
                    orig_weights += delta_weight

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    base_layer.weight.data += delta_weight

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """Unmerge all merged adapters from the base weights."""
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.tinylora_A.keys():
                delta_weight = self.get_delta_weight(active_adapter)
                self.get_base_layer().weight.data -= delta_weight

    def delete_adapter(self, adapter_name: str) -> None:
        """Delete an adapter from the layer."""
        if adapter_name in self.tinylora_v_key:
            del self.tinylora_v_key[adapter_name]

        for attr in self.other_param_names:
            param_dict = getattr(self, attr, None)
            if param_dict is not None and adapter_name in param_dict:
                del param_dict[adapter_name]

        if adapter_name in self.r:
            del self.r[adapter_name]
        if adapter_name in self.u:
            del self.u[adapter_name]

        if adapter_name in self.tinylora_dropout:
            del self.tinylora_dropout[adapter_name]

        if adapter_name in self.active_adapters:
            active_adapters = self.active_adapters[:]
            active_adapters.remove(adapter_name)
            if active_adapters:
                self.set_adapter(active_adapters)
            else:
                remaining_adapters = self._all_available_adapter_names()
                if not remaining_adapters:
                    self.set_adapter([])
                else:
                    new_active_adapter = remaining_adapters[0]
                    warnings.warn(
                        f"Adapter {adapter_name} was active which is now deleted. Setting active adapter to "
                        f"{new_active_adapter}."
                    )
                    self.set_adapter(new_active_adapter)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)

        if self.merged:
            return self.base_layer(x, *args, **kwargs)

        result = self.base_layer(x, *args, **kwargs)

        for active_adapter in self.active_adapters:
            if active_adapter not in self.tinylora_A.keys():
                continue

            A = self.tinylora_A[active_adapter]  # (r, embedding_dim)
            B = self.tinylora_B[active_adapter]  # (num_embeddings, r)
            R = self._compute_R(active_adapter)  # (r, r)

            dropout = self.tinylora_dropout[active_adapter]

            # Move components to input device
            device = result.device
            dtype = result.dtype
            A = A.to(device=device, dtype=dtype)
            B = B.to(device=device, dtype=dtype)
            R = R.to(device=device, dtype=dtype)

            # For embedding, we need to:
            # 1. Look up B[x] to get the low-rank representation (batch, seq, r)
            # 2. Multiply by R to get (batch, seq, r)
            # 3. Multiply by A to get the delta (batch, seq, embedding_dim)
            # delta = B[x] @ R @ A

            # B[x]: embedding lookup in the low-rank space
            after_B = F.embedding(x, B)  # (batch, seq, r)
            after_B = dropout(after_B)

            # Multiply by R and A
            after_R = after_B @ R  # (batch, seq, r)
            delta = after_R @ A  # (batch, seq, embedding_dim)

            result = result + delta

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "tinylora." + rep
