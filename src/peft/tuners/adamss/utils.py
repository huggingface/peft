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

import numpy as np
import torch
from sklearn.cluster import KMeans


def slicePCA(tensor, r, device, dtype=torch.float32):
    """
    Perform slice-wise PCA (SVD) on 4D tensor.
    
    Args:
        tensor: 4D tensor of shape (B, C, H, W)
        r: rank for low-rank approximation
        device: computation device
        dtype: data type
        
    Returns:
        VVT: Right singular vectors (B, C, r, W)
        UU: Left singular vectors (B, C, H, r)
    """
    tensor = tensor.to(device)
    B, C, H, W = tensor.shape
    UU = torch.zeros(B, C, H, r, dtype=dtype, device=device)
    VVT = torch.zeros(B, C, r, W, dtype=dtype, device=device)
    
    for i in range(B):
        for j in range(C):
            U, _, V = torch.svd_lowrank(tensor[i, j, :, :], q=r, niter=2, M=None)
            UU[i, j, :, :] = U[:, 0:r]
            VVT[i, j, :, :] = V[:, 0:r].T
            
    return VVT, UU


def clustering_Z(VT, K, iternum):
    """
    Cluster the columns of VT into K subspaces using K-Means.
    
    Args:
        VT: Matrix to cluster (columns are clustered)
        K: Number of clusters
        iternum: Maximum iterations for K-Means
        
    Returns:
        indxx: List of cluster assignments
        KK: List of K values
    """
    KK = []
    indxx = []
    for i in range(1):
        kmeans = KMeans(n_clusters=K, init='random', n_init=1, max_iter=iternum, random_state=123456789)
        idx = kmeans.fit_predict(VT.cpu().numpy())
        indxx.append(idx)
        KK.append(K)
    return indxx, KK


def seg_locations(WW_shape0, index):
    """
    Segment indices into locations based on cluster assignments.
    
    Args:
        WW_shape0: Number of layers
        index: Cluster assignments
        
    Returns:
        locations: List of locations for each cluster
    """
    locations = []
    for ii in range(WW_shape0):
        K = index[ii].max().item() + 1
        location = []
        for i in range(K):
            location.append(np.where(index[ii] == i)[0])
        locations.append(location)
    return locations


def get_trainable_subspaces_all(num_layers, KK):
    """
    Get all trainable subspace indices.
    
    Args:
        num_layers: Number of layers
        KK: List of number of subspaces per layer
        
    Returns:
        top_indicess: List of trainable subspace indices
    """
    top_indicess = []
    for ii in range(num_layers):
        top_indices = []
        for i in range(KK[ii]):
            top_indices.append(i)
        top_indicess.append(top_indices)
    return top_indicess
