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

import torch


def slice_pca(tensor, r, device, dtype=torch.float32, random_seed=0):
    """
    Perform slice-wise PCA (SVD) on 4D tensor.

    Args:
        tensor: 4D tensor of shape (B, C, H, W)
        r: rank for low-rank approximation
        device: computation device
        dtype: data type
        random_seed: seed for the random projection used by `torch.svd_lowrank`, so the decomposition is
            deterministic and reproducible across save/load

    Returns:
        VVT: Right singular vectors (B, C, r, W) UU: Left singular vectors (B, C, H, r)
    """
    tensor = tensor.to(device)
    B, C, H, W = tensor.shape

    # Clamp r to the minimum dimension to avoid SVD errors
    # SVD rank cannot exceed min(H, W)
    effective_r = min(r, H, W)

    UU = torch.zeros(B, C, H, effective_r, dtype=dtype, device=device)
    VVT = torch.zeros(B, C, effective_r, W, dtype=dtype, device=device)

    # torch.svd_lowrank draws a random projection internally, so its result (and hence the downstream
    # clustering and scatter_index) depends on the RNG state. Seed a forked RNG with the configurable
    # random_seed so the result is deterministic (torch.svd_lowrank does not accept a generator argument);
    # fork_rng leaves the global RNG stream untouched.
    fork_devices = [device] if torch.device(device).type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(random_seed)
        for i in range(B):
            for j in range(C):
                U, _, V = torch.svd_lowrank(tensor[i, j, :, :], q=effective_r, niter=2, M=None)
                UU[i, j, :, :] = U[:, 0:effective_r]
                VVT[i, j, :, :] = V[:, 0:effective_r].T
    # Return computed matrices (important: ensure callers receive VVT and UU)
    return VVT, UU


def clustering_Z(VT, num_subspaces, iternum):
    """
    Cluster the rows of VT into K subspaces using K-Means.

    Args:
        VT: Matrix to cluster (rows are clustered)
        num_subspaces: Number of clusters (subspaces)
        iternum: Maximum iterations for K-Means

    Returns:
        cluster_idx: Cluster assignments as a ``torch.LongTensor`` effective_num_subspaces: Actual number of subspaces
        used (``int``)

    Note:
        This function requires scikit-learn to be installed. Install it with: pip install scikit-learn
    """
    # Local import with helpful error message
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        raise ImportError(
            "scikit-learn is required for AdaMSS initialization. Please install it with: pip install scikit-learn"
        )

    # Move to numpy for sklearn compatibility
    # Note: We do NOT normalize row vectors here to match the original adamss_pkg behavior.
    # Although normalization is generally recommended for cosine-similarity-like clustering,
    # the reference implementation clusters raw singular vectors.
    vt = VT.detach().cpu().numpy().astype("float32")

    # Auto-clamp num_subspaces to available samples (output dimensions)
    # Cannot cluster n samples into more than n clusters
    effective_num_subspaces = min(num_subspaces, vt.shape[0])

    kmeans = KMeans(
        n_clusters=effective_num_subspaces, init="random", n_init=1, max_iter=iternum, random_state=123456789
    )
    idx = kmeans.fit_predict(vt)

    return torch.from_numpy(idx).long(), effective_num_subspaces


def seg_locations(index):
    """
    Segment indices into locations based on cluster assignments.

    Args:
        index: Cluster assignments as a ``torch.LongTensor``

    Returns:
        location: Dict mapping cluster id to ``torch.LongTensor`` of row indices.
            Clusters are ordered by their smallest index so that KMeans label permutations do not affect downstream
            ordering.
    """
    K = int(index.max().item()) + 1
    location = {}
    for i in range(K):
        location[i] = torch.where(index == i)[0]
    return location


def get_trainable_subspaces(num_subspaces):
    """
    Get all trainable subspace indices.

    Args:
        num_subspaces: Number of subspaces

    Returns:
        List of trainable subspace indices (``list[int]``)
    """
    return list(range(num_subspaces))
