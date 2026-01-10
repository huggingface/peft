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

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer


class AdaMSSLayer(BaseTunerLayer):
    """
    Base AdaMSS layer that stores adapter-specific information.
    """

    # All names of layers that may contain adapter weights (trainable only)
    adapter_layer_names = ("adamss_A", "adamss_B")
    # other_param_names specifies additional parameter names that are stored
    other_param_names = ("adamss_resW", "adamss_newB", "KK", "TrainSubsp_indx", "seg_result", "newindex", "exp_avg_ipt", "exp_avg_unc")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        # Adapter-specific attributes
        self.adamss_A = nn.ParameterDict({})
        self.adamss_B = nn.ParameterDict({})
        self.adamss_resW = {}  # Regular dict for frozen weights
        self.adamss_newB = {}  # Regular dict for frozen weights
        self.KK = {}
        self.TrainSubsp_indx = {}
        self.seg_result = {}
        self.newindex = {}
        # ASA importance tracking
        self.exp_avg_ipt = {}  # Exponential moving average of importance
        self.exp_avg_unc = {}  # Exponential moving average of uncertainty
        self._disable_adapters = False
        self.merged_adapters = []
        # self._asa_update_enabled = False # Flag removed as hooks are removed
        
        # Mark base layer parameters as not trainable
        for param in self.base_layer.parameters():
            param.requires_grad = False
    
    def set_asa_update_enabled(self, enabled: bool) -> None:
        """Deprecated: No longer needed as hooks are removed."""
        pass

    def reset_importance(self, adapter_name: str) -> None:
        """Clear stored importance stats for an adapter (aligns with adamss_pkg)."""
        if adapter_name in self.exp_avg_ipt:
            self.exp_avg_ipt[adapter_name] = {}
        if adapter_name in self.exp_avg_unc:
            self.exp_avg_unc[adapter_name] = {}

    def _create_importance_hook(self, adapter_name, key, beta1=0.85, beta2=0.85):
        """
        Deprecated: Hooks are no longer used.
        """
        return None

    def update_importance(self, adapter_name: str, beta1: float = 0.85, beta2: float = 0.85) -> None:
        """
        Update importance scores using current gradients (called explicitly by ASACallback).
        Matches adamss_pkg behavior of using accumulated gradients.
        """
        if adapter_name not in self.exp_avg_ipt:
            return

        exp_avg_ipt = self.exp_avg_ipt[adapter_name]
        exp_avg_unc = self.exp_avg_unc[adapter_name]

        # Iterate over all parameters for this adapter
        for i in range(self.KK[adapter_name]):
            key_A = f"{adapter_name}_A_{i}"
            key_B = f"{adapter_name}_B_{i}"
            
            for key, param_dict in [(key_A, self.adamss_A), (key_B, self.adamss_B)]:
                if key in param_dict:
                    param = param_dict[key]
                    if param.grad is not None:
                        if key not in exp_avg_ipt:
                            exp_avg_ipt[key] = torch.zeros_like(param)
                            exp_avg_unc[key] = torch.zeros_like(param)
                        
                        # Calculate importance: |w * g|
                        ipt = (param * param.grad).abs()
                        
                        # Update EMA
                        exp_avg_ipt[key].mul_(beta1).add_(ipt, alpha=1 - beta1)
                        
                        # Update Uncertainty
                        diff = (ipt - exp_avg_ipt[key]).abs()
                        exp_avg_unc[key].mul_(beta2).add_(diff, alpha=1 - beta2)
    
    def mask_to_target(self, adapter_name: str, target_kk: int, verbose: bool = False) -> None:
        """
        Mask (freeze) less important subspaces to reach target_kk active subspaces.
        """
        if verbose:
            print(f"[DEBUG][mask_to_target] Starting processing for adapter: {adapter_name}, target_kk={target_kk}")
        if adapter_name not in self.exp_avg_ipt:
            if verbose:
                print(f"[DEBUG][mask_to_target] Adapter {adapter_name} not in exp_avg_ipt, skipping")
            return  # ASA not enabled
        
        num_subspaces = self.KK[adapter_name]
        if target_kk >= num_subspaces:
            return

        exp_avg_ipt = self.exp_avg_ipt[adapter_name]
        exp_avg_unc = self.exp_avg_unc[adapter_name]

        if verbose:
            print(f"[DEBUG][mask_to_target] exp_avg_ipt keys: {list(exp_avg_ipt.keys())}")
            print(f"[DEBUG][mask_to_target] exp_avg_unc keys: {list(exp_avg_unc.keys())}")

        subspace_scores = []
        for i in range(num_subspaces):
            key_A = f"{adapter_name}_A_{i}"
            key_B = f"{adapter_name}_B_{i}"

            if key_A not in exp_avg_ipt or key_B not in exp_avg_ipt:
                continue

            score_A = (exp_avg_ipt[key_A] * exp_avg_unc[key_A]).mean()
            score_B = (exp_avg_ipt[key_B] * exp_avg_unc[key_B]).mean()
            subspace_scores.append((i, score_A + score_B))

        # Debug: Print subspace scores for verification
        if verbose:
            print(f"[DEBUG][mask_to_target] Subspace scores for {adapter_name}:")
            for idx, score in subspace_scores:
                print(f"  Subspace {idx}: Score={score}")

        if len(subspace_scores) <= target_kk and target_kk > 0:
            return

        if target_kk <= 0:
            active_indices = set()
        else:
            scores_tensor = torch.stack([s for _, s in subspace_scores])
            kth = torch.kthvalue(-scores_tensor, target_kk).values.item()
            threshold = -kth
            active_indices = {idx for idx, score in subspace_scores if score > threshold}

            if len(active_indices) < target_kk:
                subspace_scores.sort(key=lambda x: x[1], reverse=True)
                active_indices = {idx for idx, _ in subspace_scores[:target_kk]}

        # Debug: Print active indices after thresholding
        if verbose:
            print(f"[DEBUG][mask_to_target] Active indices for {adapter_name}: {sorted(active_indices)}")

        for i in range(num_subspaces):
            key_A = f"{adapter_name}_A_{i}"
            key_B = f"{adapter_name}_B_{i}"
            
            if key_A in self.adamss_A and key_B in self.adamss_B:
                should_train = i in active_indices
                self.adamss_A[key_A].requires_grad = should_train
                self.adamss_B[key_B].requires_grad = should_train
                if not should_train:
                    self.adamss_A[key_A].grad = None
                    self.adamss_B[key_B].grad = None

        # Debug print: Output requires_grad status and statistics for all adamss parameters
        if verbose:
            print(f"[DEBUG][mask_to_target] {adapter_name} parameter requires_grad status:")
            trainable_count = 0
            total_count = 0
            for key, param in self.adamss_A.items():
                print(f"  {key}: requires_grad={param.requires_grad}")
                total_count += param.numel()
                if param.requires_grad:
                    trainable_count += param.numel()
            for key, param in self.adamss_B.items():
                print(f"  {key}: requires_grad={param.requires_grad}")
                total_count += param.numel()
                if param.requires_grad:
                    trainable_count += param.numel()
            print(f"[DEBUG][mask_to_target] Current trainable parameters: {trainable_count} / {total_count} ({100*trainable_count/total_count:.2f}%)")

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        num_subspaces: int,
        subspace_rank: int,
        init_weights: str,
        use_asa: bool = False,
        **kwargs,
    ) -> None:
        """
        Update layer with AdaMSS adapter.
        
        This method initializes the AdaMSS decomposition for the weight matrix.
        """
        if adapter_name in self.adamss_A:
            # Adapter already exists, skip
            return
        
        # Extract dynamic rank configuration from kwargs
        use_dynamic_rank = kwargs.get('use_dynamic_rank', False)
        svd_threshold = kwargs.get('svd_threshold', 0.1)

        # Import here to avoid circular dependency
        from .utils import slicePCA, clustering_Z, seg_locations, get_trainable_subspaces_all

        # Get the base weight
        weight = self.get_base_layer().weight
        bias = self.get_base_layer().bias
        device = weight.device
        dtype = weight.dtype

        # Prepare weight tensor (add bias as extra column if present)
        if bias is not None:
            weight_with_bias = torch.cat((weight, bias.unsqueeze(1)), dim=1)
        else:
            weight_with_bias = weight

        # Reshape to 4D tensor for slicePCA: (1, 1, out_features, in_features)
        weight_tensor = weight_with_bias.unsqueeze(0).unsqueeze(0).to(torch.float32).to(device)

        # Perform SVD decomposition with diagnostics in case of failure
        try:
            res = slicePCA(weight_tensor, r, device, torch.float32)
        except Exception as e:
            raise RuntimeError(f"slicePCA raised an exception for layer {adapter_name} (shape={tuple(weight_tensor.shape)}, dtype={weight_tensor.dtype}, device={device}): {e}") from e
        if res is None:
            # Try fallback to native adamss implementation
            try:
                from adamss_pkg.lrr import slicePCA as ad_slicePCA
                res2 = ad_slicePCA(weight_tensor, r, device, torch.float32)
                if res2 is not None:
                    VVT, UU = res2
                    # write a small debug note
                    try:
                        with open('/tmp/peft_adamss_fallback.log', 'a') as f:
                            f.write(f"Fallback to adamss_pkg.lrr.slicePCA for layer {adapter_name}\n")
                    except Exception:
                        pass
                else:
                    raise RuntimeError("adamss_pkg.lrr.slicePCA also returned None")
            except Exception as e:
                # collect lightweight diagnostics to help debugging
                try:
                    has_nan = bool(torch.isnan(weight_tensor).any().item())
                    mn = float(weight_tensor.min().item())
                    mx = float(weight_tensor.max().item())
                except Exception:
                    has_nan = 'unknown'
                    mn = 'unknown'
                    mx = 'unknown'
                raise RuntimeError(f"slicePCA returned None for layer {adapter_name} and fallback failed: shape={tuple(weight_tensor.shape)}, dtype={weight_tensor.dtype}, device={device}, has_nan={has_nan}, min={mn}, max={mx}; fallback_error={e}") from e
        else:
            VVT, UU = res

        # Store projection matrices
        newA = UU  # (1, 1, out_features, r)
        newB = VVT  # (1, 1, r, in_features)

        # Cluster the column space
        indx, KK_list = clustering_Z(UU[0, 0, :, :], num_subspaces, 10)

        # Get segment locations
        seg_results = seg_locations(1, indx)

        # Get trainable subspace indices
        TrainSubsp_indx = get_trainable_subspaces_all(1, KK_list)

        # Store metadata
        self.KK[adapter_name] = KK_list[0]
        self.TrainSubsp_indx[adapter_name] = TrainSubsp_indx[0]
        self.seg_result[adapter_name] = seg_results[0]

        # Store residual weight and projection matrix as buffers (frozen, device-aware)
        # Using register_buffer ensures they move with the model to GPU/CPU
        self.register_buffer(
            f"adamss_resW_{adapter_name}", 
            weight_with_bias.detach().to(dtype),
            persistent=False
        )
        self.register_buffer(
            f"adamss_newB_{adapter_name}",
            newB[0, 0, :, :].detach().to(dtype),
            persistent=False
        )
        # Store metadata for accessing buffers dynamically
        self.adamss_resW[adapter_name] = f"adamss_resW_{adapter_name}"
        self.adamss_newB[adapter_name] = f"adamss_newB_{adapter_name}"

        # Calculate effective rank per subspace (r/K)
        # Following paper: each subspace uses R_k = R/K columns from SVD decomposition
        # rank_per_subspace = r // num_subspaces
        
        # Override with user specified subspace_rank if provided (default 1)
        rank_per_subspace = subspace_rank
        # print(f"      [INFO] Using rank_per_subspace = {rank_per_subspace} (r={r} / K={num_subspaces})")
        print(f"      [INFO] Using rank_per_subspace = {rank_per_subspace} (user specified)")

        # Initialize trainable subspace parameters
        for i in range(self.KK[adapter_name]):
            indx_i = TrainSubsp_indx[0][i]
            seg_indices = seg_results[0][indx_i]

            # Extract subspace data: use ALL r columns from SVD (not just r/K)
            # This matches adamss_pkg where A matrices have shape (rank_i, r)
            subspace_data = newA[0, 0, seg_indices, :r]
            # subspace_data shape: (len(seg_indices), r)
            
            # Determine actual rank based on configuration
            if use_dynamic_rank:
                # Compute Gram matrix Z = subspace_data @ subspace_data.T
                Z_row = subspace_data @ subspace_data.T  # (len(seg_indices), len(seg_indices))
                
                # Compute SVD to get row space decomposition
                U_row, S_row, V_row = torch.svd_lowrank(Z_row, q=min(Z_row.shape), niter=2)
                
                # Dynamic rank selection using SVD threshold
                # Dynamically determine actual rank using configurable threshold
                # Following adamss_pkg: num_ii_jj = min((S > threshold * S[0]).sum().item(), args.adamss_ri)
                threshold_mask = S_row > svd_threshold * S_row[0]
                actual_rank = min(threshold_mask.sum().item(), subspace_rank, len(S_row))
                
                # Handle edge case: ensure at least rank 1
                if actual_rank == 0:
                    actual_rank = 1
                
                print(f"      [INFO] Subspace {i}: dynamic rank = {actual_rank} (threshold {svd_threshold} from {len(S_row)} row singular values)")
            else:
                # Fixed rank: match adamss_pkg behavior exactly
                # adamss_pkg uses: num_ii_jj = min(len(seg_result[indx]), args.adamss_ri)
                actual_rank = min(len(seg_indices), subspace_rank)
                
                # Ensure at least rank 1
                if actual_rank == 0:
                    actual_rank = 1
                
                # print(f"      [INFO] Subspace {i}: fixed-rank = {actual_rank} (seg_indices={len(seg_indices)})")
            
            # Initialize A using QR decomposition to match adamss_pkg exactly
            # adamss_pkg: Q, R = torch.linalg.qr((newA[seg_result[indx],:]).T @ A[i], mode='reduced')
            # where A[i] is U from SVD of subspace Gram matrix
            
            # First, compute Gram matrix for this subspace
            Z_subspace = subspace_data @ subspace_data.T  # (len(seg_indices), len(seg_indices))
            
            # SVD of Gram matrix to get A_intermediate
            U_gram, S_gram, _ = torch.svd_lowrank(Z_subspace, q=min(Z_subspace.shape[0], actual_rank), niter=2)
            A_intermediate = U_gram[:, :actual_rank]  # (len(seg_indices), actual_rank)
            
            # Now apply QR decomposition: (newA[seg_indices, :r]).T @ A_intermediate
            # newA[0, 0, seg_indices, :r] is subspace_data with shape (len(seg_indices), r)
            matrix_for_qr = subspace_data.T @ A_intermediate  # (r, actual_rank)
            Q, R = torch.linalg.qr(matrix_for_qr, mode='reduced')
            
            # A_init is Q.T to match adamss_pkg
            A_init = Q.T.contiguous()  # (actual_rank, r)
            
            # Initialize B matrix to zeros (matches adamss_pkg)
            # B maps from actual_rank dimensions back to len(seg_indices) dimensions
            B_init = torch.zeros(len(seg_indices), actual_rank, dtype=dtype, device=device)
            
            # Register A and B parameters
            # A maps from r (full SVD rank) dimensions to actual_rank dimensions
            # Shape: (actual_rank, r) - matches adamss_pkg structure
            self.adamss_A[f"{adapter_name}_A_{i}"] = nn.Parameter(A_init.to(dtype))

            # B maps from actual_rank dimensions back to len(seg_indices) dimensions
            # PyTorch Linear expects weight shape: (out_features, in_features)
            # So B should be: (len(seg_indices), actual_rank)
            self.adamss_B[f"{adapter_name}_B_{i}"] = nn.Parameter(B_init.to(dtype))

        # Explicitly enable gradients for A and B parameters
        for i in range(self.KK[adapter_name]):
            self.adamss_A[f"{adapter_name}_A_{i}"].requires_grad = True
            self.adamss_B[f"{adapter_name}_B_{i}"].requires_grad = True

        # Initialize ASA tracking if enabled
        if use_asa:
            self.exp_avg_ipt[adapter_name] = {}
            self.exp_avg_unc[adapter_name] = {}
            
            # Hooks are not used; importance is updated explicitly via update_importance
            # to match adamss_pkg behavior (using accumulated gradients).

        # Compute newindex for forward pass
        self.newindex[adapter_name] = np.concatenate(
            [self.seg_result[adapter_name][self.TrainSubsp_indx[adapter_name][i]] 
             for i in range(self.KK[adapter_name])],
            axis=0
        )

    def set_adapter(self, adapter_names: str | list[str], inference_mode: bool = False) -> None:
        """
        Set the active adapter(s) for AdaMSS layer.
        
        Custom implementation because our parameter keys use format "{adapter_name}_A_{i}"
        instead of just "{adapter_name}", so we need custom matching logic.
        """
        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        # Set active adapters
        for layer_name in self.adapter_layer_names:
            module_dict = getattr(self, layer_name)
            for key, param in module_dict.items():
                # Check if key starts with any active adapter name
                is_active = any(key.startswith(f"{adapter_name}_") for adapter_name in adapter_names)
                
                if is_active and not inference_mode:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        self._active_adapter = adapter_names[0] if len(adapter_names) == 1 else adapter_names


class Linear(nn.Module, AdaMSSLayer):
    """
    AdaMSS-adapted Linear layer.
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
        AdaMSSLayer.__init__(self, base_layer, **kwargs)

        # Initialize the adapter
        self.update_layer(adapter_name, r, num_subspaces, subspace_rank, init_weights, use_asa, **kwargs)
        self._active_adapter = adapter_name
        self.dtype = base_layer.weight.dtype

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass with AdaMSS adaptation.
        """
        previous_dtype = x.dtype

        if self._disable_adapters or not self._active_adapter:
            # Use original layer
            return self.base_layer(x, *args, **kwargs)

        # Get active adapter
        adapter_name = self._active_adapter
# Check if adapter exists (check for first A matrix)
        if f"{adapter_name}_A_0" not in self.adamss_A:
            return self.base_layer(x, *args, **kwargs)

        # Cast input to layer dtype
        x = x.to(self.dtype)

        # Add bias column if needed
        # Handle different input shapes: (batch, features) or (batch, seq, features)
        if x.dim() == 2:
            ones = torch.ones(x.shape[0], 1, device=x.device, dtype=self.dtype)
        else:  # x.dim() == 3 or more
            ones = torch.ones(*x.shape[:-1], 1, device=x.device, dtype=self.dtype)
        newx = torch.cat((x, ones), dim=-1)

        # Get buffers dynamically (ensures correct device)
        resW = getattr(self, self.adamss_resW[adapter_name])
        newB = getattr(self, self.adamss_newB[adapter_name])

        # Compute residual path: x @ resW^T
        x1 = F.linear(newx, resW)

        # Compute AdaMSS path
        x2 = F.linear(newx, newB)  # Shape: (..., r)

        # Get r from newB shape
        # newB has shape (r, in_features+1), so r is the first dimension
        r = newB.shape[0]
        
        # No splitting needed! Each A matrix takes the full x2 of dimension r
        # This matches adamss_pkg where A matrices have shape (rank_i, r)
        
        # Apply A and B transformations per subspace
        x6_chunks = []
        for i in range(self.KK[adapter_name]):
            # Get A and B for this subspace
            A_i = self.adamss_A[f"{adapter_name}_A_{i}"]  # Shape: (ri, r) - takes FULL r
            B_i = self.adamss_B[f"{adapter_name}_B_{i}"]  # Shape: (len(seg_indices_i), ri)
            
            # Apply transformations: x2 @ A^T @ B^T
            x5_i = F.linear(x2, A_i)  # (..., r) @ (ri, r)^T -> (..., ri)
            x6_i = F.linear(x5_i, B_i)  # (..., ri) @ (len(seg_indices_i), ri)^T -> (..., len(seg_indices_i))
            x6_chunks.append(x6_i)
        
        # Concatenate results from all subspaces
        x6 = torch.cat(x6_chunks, dim=-1)

        # Scatter to correct positions
        # Handle both 2D (batch, features) and 3D (batch, seq, features) inputs
        # CRITICAL FIX: x7 should match x1's shape (full output_features), not x6's shape
        x7 = torch.zeros_like(x1)
        if x6.dim() == 2:
            # 2D input: (batch, features)
            x7[:, self.newindex[adapter_name]] = x6
        else:
            # 3D input: (batch, seq, features)
            x7[:, :, self.newindex[adapter_name]] = x6

        # Combine residual and AdaMSS paths
        result = x1 + x7

        # Cast back to original dtype
        result = result.to(previous_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "adamss." + rep
