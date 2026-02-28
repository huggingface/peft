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

import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from peft.tuners._buffer_dict import BufferDict
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge

from .utils import clustering_Z, get_trainable_subspaces, seg_locations, slice_pca


class AdamssLayer(BaseTunerLayer):
    """
    Base Adamss layer that stores adapter-specific information.
    """

    # All names of layers that may contain adapter weights (trainable or frozen)
    # Use ModuleDict for proper state_dict key format that PEFT's save/load can handle
    # Only include trainable parameter containers here (not buffers)
    # adamss_resW and adamss_newB are buffers (frozen), not trainable parameters
    adapter_layer_names = ("adamss_A", "adamss_B")
    # other_param_names specifies additional non-tensor metadata
    other_param_names = (
        "num_subspaces",
        "train_subspace_index",
        "seg_result",
        "scatter_index",
        "exp_avg_ipt_A",
        "exp_avg_ipt_B",
        "exp_avg_unc_A",
        "exp_avg_unc_B",
    )

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        # Adapter-specific attributes
        # Use ModuleDict containing ParameterList for proper state_dict keys (e.g., adamss_A.default.0)
        self.adamss_A = nn.ModuleDict({})  # Will contain ParameterList per adapter
        self.adamss_B = nn.ModuleDict({})  # Will contain ParameterList per adapter
        # Use BufferDict for frozen weights (keys like adamss_resW.default)
        self.adamss_resW = BufferDict(persistent=True)
        self.adamss_newB = BufferDict(persistent=True)
        self.num_subspaces = {}
        self.train_subspace_index = {}
        self.seg_result = {}
        self.scatter_index = {}
        # ASA importance tracking (per-adapter lists indexed by subspace)
        self.exp_avg_ipt_A = {}
        self.exp_avg_ipt_B = {}
        self.exp_avg_unc_A = {}
        self.exp_avg_unc_B = {}
        self._disable_adapters = False
        self.merged_adapters = []

        # Mark base layer parameters as not trainable
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def reset_importance(self, adapter_name: str) -> None:
        """
        Clear stored importance stats for an adapter.

        Called after each masking interval to restart EMA accumulation for the
        next importance scoring window. Without the reset the scores from early
        training steps would dominate later masking decisions.
        """
        if adapter_name in self.exp_avg_ipt_A:
            n = len(self.exp_avg_ipt_A[adapter_name])
            self.exp_avg_ipt_A[adapter_name] = [None] * n
            self.exp_avg_ipt_B[adapter_name] = [None] * n
            self.exp_avg_unc_A[adapter_name] = [None] * n
            self.exp_avg_unc_B[adapter_name] = [None] * n

    def update_importance(
        self, adapter_name: str, importance_beta: float = 0.85, uncertainty_beta: float = 0.85
    ) -> None:
        """
        Update importance scores using current gradients (called explicitly by AdamssAsaCallback).
        Matches adamss_pkg behavior of using accumulated gradients.

        Args:
            adapter_name: Name of the adapter to update importance for.
            importance_beta: EMA coefficient for importance averaging (0.8-0.95 typical).
            uncertainty_beta: EMA coefficient for uncertainty averaging (0.8-0.95 typical).
        """
        if adapter_name not in self.exp_avg_ipt_A:
            return
        if adapter_name not in self.adamss_A:
            return

        param_list_A = self.adamss_A[adapter_name]
        param_list_B = self.adamss_B[adapter_name]

        ipt_A = self.exp_avg_ipt_A[adapter_name]
        ipt_B = self.exp_avg_ipt_B[adapter_name]
        unc_A = self.exp_avg_unc_A[adapter_name]
        unc_B = self.exp_avg_unc_B[adapter_name]

        for i in range(self.num_subspaces[adapter_name]):
            for param_list, ipt_list, unc_list in [
                (param_list_A, ipt_A, unc_A),
                (param_list_B, ipt_B, unc_B),
            ]:
                param = param_list[i]
                if param.grad is not None:
                    if ipt_list[i] is None:
                        ipt_list[i] = torch.zeros_like(param)
                        unc_list[i] = torch.zeros_like(param)

                    # Calculate importance: |w * g|
                    ipt = (param * param.grad).abs().detach()

                    # Update uncertainty BEFORE updating importance (matches adamss_pkg)
                    diff = (ipt - ipt_list[i]).abs()
                    unc_list[i].mul_(uncertainty_beta).add_(diff, alpha=1 - uncertainty_beta)

                    # Then update importance
                    ipt_list[i].mul_(importance_beta).add_(ipt, alpha=1 - importance_beta)


    def update_layer(
        self,
        adapter_name: str,
        r: int,
        num_subspaces: int,
        subspace_rank: int,
        init_weights: str,
        use_asa: bool = False,
        inference_mode: bool = False,
        use_dynamic_rank: bool = False,
        svd_threshold: float = 0.1,
        **kwargs,
    ) -> None:
        """
        Update layer with Adamss adapter.

        This method initializes the Adamss decomposition for the weight matrix
        using SVD, clustering, and QR initialization.
        """

        # Get the base weight info
        weight = self.get_base_layer().weight
        bias = self.get_base_layer().bias
        device = weight.device
        dtype = weight.dtype
        _, in_features = weight.shape

        # Adjust in_features for bias column
        if bias is not None:
            in_features += 1

        # Prepare weight tensor (add bias as extra column if present)
        if bias is not None:
            weight_with_bias = torch.cat((weight, bias.unsqueeze(1)), dim=1)
        else:
            weight_with_bias = weight

        # Reshape to 4D tensor for slice_pca: (1, 1, out_features, in_features)
        weight_tensor = weight_with_bias.unsqueeze(0).unsqueeze(0).to(torch.float32).to(device)

        # Perform SVD decomposition with diagnostics in case of failure
        try:
            res = slice_pca(weight_tensor, r, device, torch.float32)
        except Exception as e:
            raise RuntimeError(
                f"slice_pca raised an exception for layer {adapter_name} (shape={tuple(weight_tensor.shape)}, dtype={weight_tensor.dtype}, device={device}): {e}"
            ) from e

        VVT, UU = res

        # Cluster the column space
        cluster_idx, effective_num_subspaces = clustering_Z(UU[0, 0, :, :], num_subspaces, 10)

        # Get segment locations
        seg_result = seg_locations(cluster_idx)

        # Get trainable subspace indices
        train_subspace_index = get_trainable_subspaces(effective_num_subspaces)

        # Store metadata
        self.num_subspaces[adapter_name] = effective_num_subspaces
        self.train_subspace_index[adapter_name] = train_subspace_index
        self.seg_result[adapter_name] = seg_result

        # Store residual weight and projection matrix in BufferDict (frozen, device-aware)
        # BufferDict handles registration, keys like 'adamss_resW.{adapter_name}' match expected pattern
        self.adamss_resW[adapter_name] = weight_with_bias.detach().to(dtype)
        self.adamss_newB[adapter_name] = VVT[0, 0, :, :].detach().to(dtype)

        # Use user-specified subspace_rank
        rank_per_subspace = subspace_rank

        # Initialize trainable subspace parameters
        # Collect parameters in lists, then create ParameterList for proper state_dict keys
        A_params = []
        B_params = []

        for i in range(self.num_subspaces[adapter_name]):
            subspace_idx = train_subspace_index[i]
            seg_indices = seg_result[subspace_idx]

            # Extract subspace data: use ALL r columns from SVD (not just r/K)
            # This matches adamss_pkg where A matrices have shape (rank_i, r)
            subspace_data = UU[0, 0, seg_indices, :r]
            # subspace_data shape: (len(seg_indices), r)

            # Determine actual rank based on configuration
            if use_dynamic_rank:
                # Compute Gram matrix Z = subspace_data @ subspace_data.T
                Z_row = subspace_data @ subspace_data.T  # (len(seg_indices), len(seg_indices))

                # Compute SVD to get row space decomposition
                _U_row, S_row, _V_row = torch.svd_lowrank(Z_row, q=min(Z_row.shape), niter=2)

                # Dynamic rank selection using SVD threshold
                # Dynamically determine actual rank using configurable threshold
                # Following adamss_pkg: num_ii_jj = min((S > threshold * S[0]).sum().item(), args.adamss_ri)
                threshold_mask = S_row > svd_threshold * S_row[0]
                actual_rank = min(threshold_mask.sum().item(), subspace_rank, len(S_row))

                # Handle edge case: ensure at least rank 1
                if actual_rank == 0:
                    actual_rank = 1

            else:
                # Fixed rank: match adamss_pkg behavior exactly
                # adamss_pkg uses: num_ii_jj = min(len(seg_result[indx]), args.adamss_ri)
                actual_rank = min(len(seg_indices), subspace_rank)

                # Ensure at least rank 1
                if actual_rank == 0:
                    actual_rank = 1

            # Initialize A using QR decomposition to match adamss_pkg exactly
            # adamss_pkg: Q, R = torch.linalg.qr((UU[seg_result[indx],:]).T @ A[i], mode='reduced')
            # where A[i] is U from SVD of subspace Gram matrix

            # First, compute Gram matrix for this subspace
            Z_subspace = subspace_data @ subspace_data.T  # (len(seg_indices), len(seg_indices))

            # SVD of Gram matrix to get A_intermediate
            U_gram, _S_gram, _ = torch.svd_lowrank(Z_subspace, q=min(Z_subspace.shape[0], actual_rank), niter=2)
            A_intermediate = U_gram[:, :actual_rank]  # (len(seg_indices), actual_rank)

            # Now apply QR decomposition: (UU[seg_indices, :r]).T @ A_intermediate
            # UU[0, 0, seg_indices, :r] is subspace_data with shape (len(seg_indices), r)
            matrix_for_qr = subspace_data.T @ A_intermediate  # (r, actual_rank)
            Q, _R = torch.linalg.qr(matrix_for_qr, mode="reduced")

            # A_init is Q.T to match adamss_pkg
            A_init = Q.T.contiguous()  # (actual_rank, r)

            # Initialize B matrix
            # When init_weights='orthogonal', use zeros for identity operation (standard training init)
            # When init_weights=None/False, use random values so adapter produces non-zero effect
            if init_weights == "orthogonal":
                # Zero initialization - adapter produces identity at start
                # Note: A parameters may not update in step 1 since ∂Loss/∂A ∝ B and B=0
                # But after step 1, B becomes non-zero and A will get gradients in step 2+
                B_init = torch.zeros(len(seg_indices), actual_rank, dtype=dtype, device=device)
            else:
                # Random initialization for testing - produces non-zero adapter output
                B_init = torch.randn(len(seg_indices), actual_rank, dtype=dtype, device=device) * 0.01

            # Collect A and B parameters
            A_params.append(nn.Parameter(A_init.to(dtype)))
            B_params.append(nn.Parameter(B_init.to(dtype)))

        # Create ParameterLists and store in ModuleDict
        # This creates state_dict keys like 'adamss_A.default.0', 'adamss_A.default.1', etc.
        self.adamss_A[adapter_name] = nn.ParameterList(A_params)
        self.adamss_B[adapter_name] = nn.ParameterList(B_params)

        # Initialize ASA tracking if enabled
        if use_asa:
            n = effective_num_subspaces
            self.exp_avg_ipt_A[adapter_name] = [None] * n
            self.exp_avg_ipt_B[adapter_name] = [None] * n
            self.exp_avg_unc_A[adapter_name] = [None] * n
            self.exp_avg_unc_B[adapter_name] = [None] * n

            # Hooks are not used; importance is updated explicitly via update_importance
            # to match adamss_pkg behavior (using accumulated gradients).

        # Compute scatter_index for forward pass
        self.scatter_index[adapter_name] = torch.cat(
            [
                self.seg_result[adapter_name][self.train_subspace_index[adapter_name][i]]
                for i in range(self.num_subspaces[adapter_name])
            ],
            dim=0,
        ).long()

        # Move adapter to device of base layer (important for low_cpu_mem_usage support)
        self._move_adapter_to_device_of_base_layer(adapter_name)

        # Set requires_grad via parent class set_adapter (works with ParameterList)
        self.set_adapter(self.active_adapters, inference_mode=inference_mode)


class Linear(nn.Module, AdamssLayer):
    """
    Adamss-adapted Linear layer.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        r: int = 500,
        num_subspaces: int = 5,
        subspace_rank: int = 1,
        init_weights: str = "orthogonal",
        use_asa: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        AdamssLayer.__init__(self, base_layer, **kwargs)

        # Set in_features and out_features from base layer (required for PEFT compatibility)
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        # Initialize the adapter
        inference_mode = kwargs.pop("inference_mode", False)
        self.update_layer(
            adapter_name,
            r,
            num_subspaces,
            subspace_rank,
            init_weights,
            use_asa,
            inference_mode=inference_mode,
            **kwargs,
        )
        self._active_adapter = adapter_name
        self.dtype = base_layer.weight.dtype

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """Handle shape mismatches that arise when loading checkpoints.

        AdaMSS B parameter shapes depend on KMeans clustering of the weight
        matrix.  Because KMeans is non-deterministic, loading a checkpoint into
        a freshly-initialised model can produce different segment sizes,
        causing shape mismatches.  This override detects mismatches and
        replaces placeholder parameters with correctly-shaped tensors before
        the default ``load_state_dict`` logic runs.
        """
        for key, value in state_dict.items():
            if not key.startswith(prefix):
                continue
            local_key = key[len(prefix):]
            # Walk dot-separated path to reach the attribute
            parts = local_key.split(".")
            try:
                target = self
                for part in parts[:-1]:
                    target = getattr(target, part) if not part.isdigit() else target[int(part)]
                last = parts[-1]
                if last.isdigit():
                    current = target[int(last)]
                else:
                    current = getattr(target, last)
            except (AttributeError, IndexError, KeyError):
                continue

            if isinstance(current, (torch.Tensor, nn.Parameter)) and current.shape != value.shape:
                new_param = nn.Parameter(torch.empty_like(value), requires_grad=current.requires_grad)
                if last.isdigit():
                    target[int(last)] = new_param
                else:
                    setattr(target, last, new_param)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass with Adamss adaptation.
        """
        previous_dtype = x.dtype

        if self._disable_adapters or not self._active_adapter:
            # When adapters are disabled, we need to handle the case where
            # the adapter was merged - in that case, base_layer.weight contains
            # the merged weights and we need to use original weights instead.
            if self.merged:
                # Save merged adapters list before unmerge (unmerge clears it)
                adapters_to_remerge = list(self.merged_adapters)
                # Temporarily unmerge to get original behavior
                self.unmerge()
                result = self.base_layer(x, *args, **kwargs)
                # Re-merge after the forward pass
                self.merge(adapter_names=adapters_to_remerge)
            else:
                result = self.base_layer(x, *args, **kwargs)
        else:
            # Cast input to layer dtype
            x = x.to(self.dtype)

            # Add bias column if needed
            # Handle different input shapes: (batch, features) or (batch, seq, features)
            if x.dim() == 2:
                ones = torch.ones(x.shape[0], 1, device=x.device, dtype=self.dtype)
            else:  # x.dim() == 3 or more
                ones = torch.ones(*x.shape[:-1], 1, device=x.device, dtype=self.dtype)
            newx = torch.cat((x, ones), dim=-1)

            # resW is the frozen original weight — identical for all adapters,
            # just need any valid adapter key to retrieve it from the BufferDict.
            first_active_adapter = None
            for adapter in self.active_adapters:
                if adapter in self.adamss_A:
                    first_active_adapter = adapter
                    break

            if first_active_adapter is None:
                result = self.base_layer(x, *args, **kwargs)
            else:
                # Compute base output from residual weight (frozen original weight)
                result = F.linear(newx, self.adamss_resW[first_active_adapter].to(self.dtype))

                # Iterate over all active adapters and add their deltas
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.adamss_A:
                        continue

                    # Project input into low-rank space
                    projected = F.linear(newx, self.adamss_newB[active_adapter].to(self.dtype))  # (..., r)

                    # Apply A and B transformations per subspace
                    # TODO: if all subspaces have equal output dimensions, this loop could
                    # be replaced with a batched matmul/einsum for better performance.
                    # Currently B_i shapes vary per subspace (clustering-dependent), so we loop.
                    subspace_chunks = []
                    for i in range(self.num_subspaces[active_adapter]):
                        A_i = self.adamss_A[active_adapter][i].to(self.dtype)  # (ri, r)
                        B_i = self.adamss_B[active_adapter][i].to(self.dtype)  # (seg_len_i, ri)
                        subspace_out = F.linear(projected, A_i)  # (..., ri)
                        subspace_out = F.linear(subspace_out, B_i)  # (..., seg_len_i)
                        subspace_chunks.append(subspace_out)

                    # Concatenate subspace results and scatter to output dimension order
                    adapter_delta = torch.cat(subspace_chunks, dim=-1)
                    scatter_index_tensor = self.scatter_index[active_adapter].to(adapter_delta.device)

                    if adapter_delta.dim() == 2:
                        index = scatter_index_tensor.unsqueeze(0).expand(adapter_delta.shape[0], -1)
                        adapter_delta = torch.zeros(
                            adapter_delta.shape[0], result.shape[-1], device=adapter_delta.device, dtype=adapter_delta.dtype
                        ).scatter(1, index, adapter_delta)
                    else:
                        index = scatter_index_tensor.unsqueeze(0).unsqueeze(0).expand(*adapter_delta.shape[:-1], -1)
                        adapter_delta = torch.zeros(
                            *adapter_delta.shape[:-1], result.shape[-1], device=adapter_delta.device, dtype=adapter_delta.dtype
                        ).scatter(-1, index, adapter_delta)

                    result = result + adapter_delta

        # Cast back to original dtype
        return result.to(previous_dtype)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights.

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.adamss_A:
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weight = base_layer.weight.data.clone()
                    orig_dtype = orig_weight.dtype
                    delta_weight = self.get_delta_weight(active_adapter)
                    orig_weight += delta_weight.to(orig_dtype)

                    if not torch.isfinite(orig_weight).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weight

                    # Also update bias if present
                    if base_layer.bias is not None:
                        orig_bias = base_layer.bias.data.clone()
                        delta_bias = self.get_delta_bias(active_adapter)
                        orig_bias += delta_bias.to(orig_dtype)
                        if not torch.isfinite(orig_bias).all():
                            raise ValueError(
                                f"NaNs detected in the merged bias. The adapter {active_adapter} seems to be broken"
                            )
                        base_layer.bias.data = orig_bias
                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    base_layer.weight.data += delta_weight

                    # Also update bias if present
                    if base_layer.bias is not None:
                        delta_bias = self.get_delta_bias(active_adapter)
                        base_layer.bias.data += delta_bias

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.adamss_A:
                base_layer = self.get_base_layer()
                weight = base_layer.weight
                orig_dtype = weight.dtype
                delta_weight = self.get_delta_weight(active_adapter)
                weight.data -= delta_weight.to(orig_dtype)

                # Also update bias if present
                if base_layer.bias is not None:
                    delta_bias = self.get_delta_bias(active_adapter)
                    base_layer.bias.data -= delta_bias.to(orig_dtype)

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        For AdaMSS, the forward computes:
            result = newx @ resW.T + scatter(newx @ newB.T @ A.T @ B.T)

        Since resW = original_weight_with_bias, the delta is just the adapter path:
            delta = scatter(B @ A @ newB)

        We extract the weight portion (excluding bias column) as the delta to add
        to the base layer's weight.

        Args:
            adapter_name (str): The name of the adapter for which the delta weight should be computed.
        """
        device = self.get_base_layer().weight.device
        dtype = self.get_base_layer().weight.dtype
        base_weight = self.get_base_layer().weight

        # Get buffers
        newB = self.adamss_newB[adapter_name]  # Shape: (r, in_features + 1)

        # Get parameter lists
        param_list_A = self.adamss_A[adapter_name]
        param_list_B = self.adamss_B[adapter_name]

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32
        cast_to_fp32 = device.type == "cpu" and dtype in (torch.float16, torch.bfloat16)
        compute_dtype = torch.float32 if cast_to_fp32 else dtype

        newB = newB.to(device).to(compute_dtype)

        # Compute the adapter contribution (scattered B @ A @ newB)
        # delta_weight has shape (out_features, in_features + 1)
        out_features = base_weight.shape[0]
        in_features_plus_1 = newB.shape[1]

        # Initialize delta weight for the adapter path only
        delta_weight = torch.zeros(out_features, in_features_plus_1, device=device, dtype=compute_dtype)

        # Compute the transformation for each subspace
        # For subspace i: contribution = B_i @ A_i @ newB
        # where A_i has shape (rank_i, r), B_i has shape (seg_len_i, rank_i)
        chunks = []
        for i in range(self.num_subspaces[adapter_name]):
            A_i = param_list_A[i].to(device).to(compute_dtype)  # Shape: (rank_i, r)
            B_i = param_list_B[i].to(device).to(compute_dtype)  # Shape: (seg_len_i, rank_i)

            # B_i @ A_i @ newB gives shape (seg_len_i, in_features+1)
            chunk = B_i @ A_i @ newB
            chunks.append(chunk)

        # Concatenate chunks and scatter to correct positions
        if chunks:
            combined = torch.cat(chunks, dim=0)  # Shape: (total_seg_len, in_features+1)
            # Use scatter_index for assignment
            scatter_idx = self.scatter_index[adapter_name].to(delta_weight.device)
            delta_weight[scatter_idx] = combined

        # Extract just the weight portion (excluding the bias column)
        # delta_weight[:, :-1] is the weight delta
        # delta_weight[:, -1] is the bias delta
        output_tensor = delta_weight[:, :-1]  # Shape: (out_features, in_features)

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

        return output_tensor

    def get_delta_bias(self, adapter_name: str) -> torch.Tensor:
        """
        Compute the bias delta for the given adapter.

        This is the last column of the adapter path contribution (B @ A @ newB).

        Args:
            adapter_name (str): The name of the adapter for which the bias delta should be computed.
        """
        device = self.get_base_layer().weight.device
        dtype = self.get_base_layer().weight.dtype
        base_weight = self.get_base_layer().weight

        # Get buffers
        newB = self.adamss_newB[adapter_name]  # Shape: (r, in_features + 1)

        # Get parameter lists
        param_list_A = self.adamss_A[adapter_name]
        param_list_B = self.adamss_B[adapter_name]

        cast_to_fp32 = device.type == "cpu" and dtype in (torch.float16, torch.bfloat16)
        compute_dtype = torch.float32 if cast_to_fp32 else dtype

        newB = newB.to(device).to(compute_dtype)

        out_features = base_weight.shape[0]
        in_features_plus_1 = newB.shape[1]

        # Initialize delta weight for the adapter path only
        delta_weight = torch.zeros(out_features, in_features_plus_1, device=device, dtype=compute_dtype)

        # Compute the transformation for each subspace
        chunks = []
        for i in range(self.num_subspaces[adapter_name]):
            A_i = param_list_A[i].to(device).to(compute_dtype)
            B_i = param_list_B[i].to(device).to(compute_dtype)
            chunk = B_i @ A_i @ newB
            chunks.append(chunk)

        if chunks:
            combined = torch.cat(chunks, dim=0)
            scatter_idx = self.scatter_index[adapter_name].to(delta_weight.device)
            delta_weight[scatter_idx] = combined

        # Extract the bias portion (last column)
        output_tensor = delta_weight[:, -1]  # Shape: (out_features,)

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

        return output_tensor

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "adamss." + rep
