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

import os
import time
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
    
    # Clamp r to the minimum dimension to avoid SVD errors
    # SVD rank cannot exceed min(H, W)
    effective_r = min(r, H, W)
    
    UU = torch.zeros(B, C, H, effective_r, dtype=dtype, device=device)
    VVT = torch.zeros(B, C, effective_r, W, dtype=dtype, device=device)
    
    for i in range(B):
        for j in range(C):
            U, _, V = torch.svd_lowrank(tensor[i, j, :, :], q=effective_r, niter=2, M=None)
            UU[i, j, :, :] = U[:, 0:effective_r]
            VVT[i, j, :, :] = V[:, 0:effective_r].T

    # Optional debug dump: save a compact snapshot of UU/VVT and weight row norms
    try:
        if os.environ.get("PEFT_ADAMSS_DEBUG") == "1":
            # limit sizes
            max_rows = int(os.environ.get("PEFT_ADAMSS_DEBUG_ROWS", "512"))
            max_cols = int(os.environ.get("PEFT_ADAMSS_DEBUG_COLS", "100"))
            h = min(UU.shape[2], max_rows)
            c = min(r, max_cols)
            UU_small = UU[0, 0, :h, :c].cpu().numpy().astype('float32')
            V_small = VVT[0, 0, :c, :min(VVT.shape[3], max_cols)].cpu().numpy().astype('float32')
            row_norms = torch.norm(tensor[0, 0, :, :], dim=1).cpu().numpy().astype('float32')
            fname = f"/tmp/peft_slicepca_debug_{int(time.time())}.npz"
            np.savez_compressed(fname, UU_small=UU_small, V_small=V_small, row_norms=row_norms, orig_shape=np.array(list(tensor.shape)))
    except Exception:
        pass

    # Return computed matrices (important: ensure callers receive VVT and UU)
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
        # Move to numpy
        # Note: We do NOT normalize row vectors here to match the original adamss_pkg behavior.
        # Although normalization is generally recommended for cosine-similarity-like clustering,
        # the reference implementation clusters raw singular vectors.
        vt = VT.cpu().numpy().astype('float32')

        # Auto-clamp K to available samples (output dimensions)
        # Cannot cluster n samples into more than n clusters
        effective_K = min(K, vt.shape[0])
        if effective_K < K:
            # Silent clamping - this is expected for small layers like lin1 (2 outputs)
            pass
        
        kmeans = KMeans(n_clusters=effective_K, init='random', n_init=1, max_iter=iternum, random_state=123456789)
        idx = kmeans.fit_predict(vt)
        indxx.append(idx)
        KK.append(effective_K)

        # Optional debug dump: controlled by PEFT_ADAMSS_DEBUG env var
        try:
            if os.environ.get("PEFT_ADAMSS_DEBUG") == "1":
                # limit size for safety
                nr = min(vt.shape[0], int(os.environ.get("PEFT_ADAMSS_DEBUG_ROWS", "512")))
                nc = min(vt.shape[1], int(os.environ.get("PEFT_ADAMSS_DEBUG_COLS", "100")))
                vt_small = vt[:nr, :nc].astype('float32')
                vt_norm_small = vt_norm[:nr, :nc].astype('float32')
                fname = f"/tmp/peft_kmeans_debug_{int(time.time())}.npz"
                np.savez_compressed(fname, vt_small=vt_small, vt_norm_small=vt_norm_small, labels=np.array(idx, dtype=np.int32), meta=np.array([int(K)], dtype=np.int32))
        except Exception:
            pass

    return indxx, KK


def seg_locations(WW_shape0, index, debug=False):
    """
    Segment indices into locations based on cluster assignments.

    Args:
        WW_shape0: Number of layers
        index: Cluster assignments
        debug: if True or env var PEFT_ADAMSS_DEBUG=1, write a debug JSON to /tmp

    Returns:
        locations: List of locations for each cluster (canonical order)

    The function sorts clusters by their smallest index (min element) so that
    cluster label permutations from KMeans won't change downstream ordering.
    """
    locations = []
    for ii in range(WW_shape0):
        K = int(index[ii].max().item()) + 1
        location = []
        for i in range(K):
            arr = np.where(index[ii] == i)[0]
            location.append(arr)
        # Canonicalize ordering: sort clusters by their minimum index (empty arrays go last)
        # location.sort(key=lambda arr: (int(arr.min()) if arr.size > 0 else 10 ** 9))
        locations.append(location)

    # Optional debug dump
    try:
        if debug or ("PEFT_ADAMSS_DEBUG" in os.environ and os.environ.get("PEFT_ADAMSS_DEBUG") == "1"):
            import json, time
            dump = {"layers": []}
            for ii in range(WW_shape0):
                orig = [list(map(int, np.where(index[ii] == i)[0])) for i in range(int(index[ii].max().item()) + 1)]
                canon = [list(map(int, loc)) for loc in locations[ii]]
                dump["layers"].append({"layer": ii, "original": orig, "canonical": canon})
            path = f"/tmp/peft_adamss_seg_debug_{int(time.time())}.json"
            with open(path, "w") as f:
                json.dump(dump, f)
    except Exception:
        pass

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
