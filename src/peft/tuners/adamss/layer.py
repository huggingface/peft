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
        
        # Mark base layer parameters as not trainable
        for param in self.base_layer.parameters():
            param.requires_grad = False
    
    def update_importance(self, adapter_name: str, beta1: float = 0.85, beta2: float = 0.85) -> None:
        """
        Update importance scores for ASA.
        Called during training to track parameter importance.
        """
        if adapter_name not in self.exp_avg_ipt:
            return  # ASA not enabled for this adapter
        
        num_subspaces = self.KK[adapter_name]
        
        # Collect gradients and parameters (including frozen ones with zero grad)
        all_grads = []
        all_params = []
        
        for i in range(num_subspaces):
            key_A = f"{adapter_name}_A_{i}"
            key_B = f"{adapter_name}_B_{i}"
            
            if key_A in self.adamss_A and key_B in self.adamss_B:
                param_A = self.adamss_A[key_A]
                param_B = self.adamss_B[key_B]
                
                # Always collect params, use zero grad if frozen
                all_params.append(param_A.view(-1))
                if param_A.grad is not None:
                    all_grads.append(param_A.grad.view(-1))
                else:
                    all_grads.append(torch.zeros_like(param_A.view(-1)))
                
                all_params.append(param_B.view(-1))
                if param_B.grad is not None:
                    all_grads.append(param_B.grad.view(-1))
                else:
                    all_grads.append(torch.zeros_like(param_B.view(-1)))
        
        if not all_grads:
            return
        
        # Concatenate all gradients and parameters
        grads = torch.cat(all_grads)
        params = torch.cat(all_params)
        
        # Calculate sensitivity: |grad * param|
        ipt = (grads * params).abs()
        
        # Update exponential moving averages
        exp_avg_ipt = self.exp_avg_ipt[adapter_name]
        exp_avg_unc = self.exp_avg_unc[adapter_name]
        
        # Ensure tensors are on the same device
        if exp_avg_ipt.device != ipt.device:
            exp_avg_ipt = exp_avg_ipt.to(ipt.device)
            exp_avg_unc = exp_avg_unc.to(ipt.device)
        
        # Update importance with EMA
        exp_avg_ipt.mul_(beta1).add_(ipt, alpha=1 - beta1)
        
        # Update uncertainty (variance) with EMA
        diff = (ipt - exp_avg_ipt).abs()
        exp_avg_unc.mul_(beta2).add_(diff, alpha=1 - beta2)
        
        # Store back (now on correct device)
        self.exp_avg_ipt[adapter_name] = exp_avg_ipt
        self.exp_avg_unc[adapter_name] = exp_avg_unc
    
    def mask_to_target(self, adapter_name: str, target_kk: int) -> None:
        """
        Mask (freeze) less important subspaces to reach target_kk active subspaces.
        """
        if adapter_name not in self.exp_avg_ipt:
            return  # ASA not enabled
        
        num_subspaces = self.KK[adapter_name]
        if target_kk >= num_subspaces:
            # No masking needed
            return
        
        # Calculate combined importance score: ipt * unc
        exp_avg_ipt = self.exp_avg_ipt[adapter_name]
        exp_avg_unc = self.exp_avg_unc[adapter_name]
        ipt_score = exp_avg_ipt * exp_avg_unc
        
        # Calculate per-subspace importance
        subspace_scores = []
        offset = 0
        
        for i in range(num_subspaces):
            key_A = f"{adapter_name}_A_{i}"
            key_B = f"{adapter_name}_B_{i}"
            
            if key_A in self.adamss_A and key_B in self.adamss_B:
                param_A = self.adamss_A[key_A]
                param_B = self.adamss_B[key_B]
                
                size_A = param_A.numel()
                size_B = param_B.numel()
                total_size = size_A + size_B
                
                # Average importance for this subspace
                subspace_score = ipt_score[offset:offset + total_size].mean()
                subspace_scores.append((i, subspace_score.item()))
                offset += total_size
        
        # Sort by importance (descending)
        subspace_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Keep top target_kk subspaces, freeze the rest
        active_indices = set(idx for idx, _ in subspace_scores[:target_kk])
        
        for i in range(num_subspaces):
            key_A = f"{adapter_name}_A_{i}"
            key_B = f"{adapter_name}_B_{i}"
            
            if key_A in self.adamss_A and key_B in self.adamss_B:
                should_train = i in active_indices
                self.adamss_A[key_A].requires_grad = should_train
                self.adamss_B[key_B].requires_grad = should_train

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

        # Perform SVD decomposition
        VVT, UU = slicePCA(weight_tensor, r, device, torch.float32)

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

        # Initialize trainable subspace parameters
        for i in range(self.KK[adapter_name]):
            indx_i = TrainSubsp_indx[0][i]
            seg_indices = seg_results[0][indx_i]

            # Calculate actual rank for this subspace
            actual_rank = min(len(seg_indices), subspace_rank)

            # Initialize A matrix with orthogonal initialization
            Q, _, _ = torch.svd_lowrank(
                (newA[0, 0, seg_indices, :]).T,
                q=actual_rank,
                niter=2,
                M=None
            )
            # Use flat key format: adapter_name_A_i (no dots allowed in parameter names)
            # Call .contiguous() to ensure memory is contiguous for safetensors saving
            self.adamss_A[f"{adapter_name}_A_{i}"] = nn.Parameter(Q.T.contiguous().to(dtype))

            # Initialize B matrix with zeros
            B_shape = (len(seg_indices), actual_rank)
            self.adamss_B[f"{adapter_name}_B_{i}"] = nn.Parameter(torch.zeros(B_shape, dtype=dtype).contiguous())

        # Explicitly enable gradients for A and B parameters
        for i in range(self.KK[adapter_name]):
            self.adamss_A[f"{adapter_name}_A_{i}"].requires_grad = True
            self.adamss_B[f"{adapter_name}_B_{i}"].requires_grad = True

        # Initialize ASA tracking if enabled
        if use_asa:
            total_params = sum(
                self.adamss_A[f"{adapter_name}_A_{i}"].numel() + self.adamss_B[f"{adapter_name}_B_{i}"].numel()
                for i in range(self.KK[adapter_name])
            )
            self.exp_avg_ipt[adapter_name] = torch.zeros(total_params, device=device, dtype=dtype)
            self.exp_avg_unc[adapter_name] = torch.zeros(total_params, device=device, dtype=dtype)

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
        x2 = F.linear(newx, newB)

        # Stack A and B matrices using flat keys
        stacked_A = torch.cat(
            [self.adamss_A[f"{adapter_name}_A_{i}"] for i in range(self.KK[adapter_name])],
            dim=0
        )
        stacked_B = torch.block_diag(
            *[self.adamss_B[f"{adapter_name}_B_{i}"] for i in range(self.KK[adapter_name])]
        )

        # Apply A and B transformations
        x5 = F.linear(x2, stacked_A)
        x6 = F.linear(x5, stacked_B)

        # Scatter to correct positions
        # Handle both 2D (batch, features) and 3D (batch, seq, features) inputs
        x7 = torch.zeros_like(x6)
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
