"""
Visualization utilities for residual quantization and SVD analysis.

All functions accept PyTorch tensors and save figures to disk.
They return the output file path(s) for convenience.
"""
from __future__ import annotations

import os
from typing import Optional, List, Dict, Tuple

import torch

# Matplotlib is imported inside functions to avoid backend initialization


def _parse_pool_factor(pool_factor: Optional[Tuple[int, int] | int | None]) -> Optional[Tuple[int, int]]:
    if pool_factor is None:
        return None
    if isinstance(pool_factor, int):
        return (max(1, pool_factor), max(1, pool_factor))
    if isinstance(pool_factor, tuple) and len(pool_factor) == 2:
        r, c = int(pool_factor[0]), int(pool_factor[1])
        return (max(1, r), max(1, c))
    raise ValueError(f"Invalid pool_factor: {pool_factor}")


def _pool_2d_absmax_with_sign(M: torch.Tensor, r: int, c: int) -> torch.Tensor:
    """Pool 2D by selecting the element with maximum absolute value (keeping its sign)."""
    import torch.nn.functional as F
    if r == 1 and c == 1:
        return M
    H, W = M.shape
    Ht = (H // r) * r
    Wt = (W // c) * c
    if Ht == 0 or Wt == 0:
        return M
    X = M[:Ht, :Wt]
    # Use unfold to extract non-overlapping patches of size r x c
    X4 = X.unsqueeze(0).unsqueeze(0)  # [1,1,Ht,Wt]
    patches = F.unfold(X4, kernel_size=(r, c), stride=(r, c))  # [1, r*c, L]
    L = patches.shape[-1]
    # Find indices of max absolute value within each patch
    abs_p = patches.abs()
    idx = abs_p.argmax(dim=1, keepdim=True)  # [1,1,L]
    # Gather the signed value at idx
    pooled = torch.gather(patches, 1, idx).squeeze(0).squeeze(0)  # [L]
    out = pooled.view(Ht // r, Wt // c)
    return out


def _pool_2d_avg(M: torch.Tensor, r: int, c: int) -> torch.Tensor:
    import torch.nn.functional as F
    if r == 1 and c == 1:
        return M
    H, W = M.shape
    Ht = (H // r) * r
    Wt = (W // c) * c
    if Ht == 0 or Wt == 0:
        return M
    X = M[:Ht, :Wt]
    X4 = X.unsqueeze(0).unsqueeze(0)
    # Average pooling
    pooled = F.avg_pool2d(X4, kernel_size=(r, c), stride=(r, c))  # [1,1,H',W']
    return pooled.squeeze(0).squeeze(0)


def _pool_2d_max(M: torch.Tensor, r: int, c: int) -> torch.Tensor:
    import torch.nn.functional as F
    if r == 1 and c == 1:
        return M
    H, W = M.shape
    Ht = (H // r) * r
    Wt = (W // c) * c
    if Ht == 0 or Wt == 0:
        return M
    X = M[:Ht, :Wt]
    X4 = X.unsqueeze(0).unsqueeze(0)
    pooled = F.max_pool2d(X4, kernel_size=(r, c), stride=(r, c))
    return pooled.squeeze(0).squeeze(0)


def _maybe_pool(M: torch.Tensor, pool_factor: Optional[Tuple[int, int]], pool_mode: str) -> torch.Tensor:
    if pool_factor is None:
        return M
    r, c = pool_factor
    mode = (pool_mode or "absmax").lower()
    if mode in ("absmax", "absmax_sign", "abs-max"):
        return _pool_2d_absmax_with_sign(M, r, c)
    if mode in ("avg", "mean"):
        return _pool_2d_avg(M, r, c)
    if mode in ("max",):
        return _pool_2d_max(M, r, c)
    raise ValueError(f"Unsupported pool_mode: {pool_mode}")


def save_layer_heatmaps(
    layer_name: str,
    out_dir: str,
    W_original: torch.Tensor,
    W_quantized_model: torch.Tensor,
    R_true: torch.Tensor,
    R_quant: torch.Tensor,
    *,
    rank: Optional[int] = None,
    W_svd: Optional[torch.Tensor] = None,
    log_scale_threshold: float = 1e-3,
    pool_factor: Optional[Tuple[int, int] | int | None] = None,
    pool_mode: str = "absmax",
) -> str:
    """
    Save a 2x3 heatmap panel comparing:
      1) W_original
      2) W_svd (if provided or computed via `rank`)
      3) W_res (true residual)
      4) Q(W_res) (quantized residual)
      5) Q(W_res) + W_svd (reconstruction)
      6) W_quantized_model (for reference)

    All inputs are expected in the same layout (out_features x in_features).

    Returns the path to the saved PNG.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    plt.switch_backend("Agg")
    os.makedirs(out_dir, exist_ok=True)

    # Optional SVD computation (CPU)
    if W_svd is None and rank is not None and rank > 0:
        with torch.no_grad():
            W_cpu = W_original.detach().cpu()
            U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
            r = min(int(rank), S.shape[0])
            U_r = U[:, :r]
            Vh_r = Vh[:r, :]
            S_r = S[:r].unsqueeze(1)
            W_svd = U_r @ (S_r * Vh_r)

    # Apply optional pooling for visualization (not for metrics)
    pf = _parse_pool_factor(pool_factor)
    W_vis = _maybe_pool(W_original.detach().cpu(), pf, pool_mode) if pf else W_original.detach().cpu()
    Wq_vis = _maybe_pool(W_quantized_model.detach().cpu(), pf, pool_mode) if pf else W_quantized_model.detach().cpu()
    R_vis = _maybe_pool(R_true.detach().cpu(), pf, pool_mode) if pf else R_true.detach().cpu()
    Rq_vis = _maybe_pool(R_quant.detach().cpu(), pf, pool_mode) if pf else R_quant.detach().cpu()
    Wsvd_vis = (_maybe_pool(W_svd.detach().cpu(), pf, pool_mode) if (pf and W_svd is not None) else (W_svd.detach().cpu() if W_svd is not None else None))

    W_np = W_vis.numpy()
    Wq_np = Wq_vis.numpy()
    R_np = R_vis.numpy()
    Rq_np = Rq_vis.numpy()
    Wsvd_np = Wsvd_vis.numpy() if Wsvd_vis is not None else None
    Wrec_np = (Rq_np + Wsvd_np) if Wsvd_np is not None else None

    vmax = float(
        max(
            [
                abs(W_np).max(),
                abs(Wq_np).max(),
                abs(R_np).max(),
                abs(Rq_np).max(),
                *( [abs(Wsvd_np).max()] if Wsvd_np is not None else [] ),
                *( [abs(Wrec_np).max()] if Wrec_np is not None else [] ),
            ]
        )
        or 1.0
    )
    vmin = -vmax
    norm = SymLogNorm(linthresh=log_scale_threshold, vmin=vmin, vmax=vmax, base=10)

    fig, axes = plt.subplots(2, 3, figsize=(24, 10))
    fig.suptitle(f"Layer: {layer_name} (SymLog)", fontsize=18)
    cmap = "coolwarm"

    im0 = axes[0, 0].imshow(W_np, cmap=cmap, norm=norm)
    axes[0, 0].set_title("1. W_original")
    axes[0, 0].grid(False)

    if Wsvd_np is not None:
        axes[0, 1].imshow(Wsvd_np, cmap=cmap, norm=norm)
        suffix = f" (Top-{rank})" if rank is not None else ""
        axes[0, 1].set_title(f"2. W_svd{suffix}")
    else:
        axes[0, 1].imshow(torch.zeros_like(W_vis).cpu().numpy(), cmap=cmap, norm=norm)
        axes[0, 1].set_title("2. W_svd (n/a)")
    axes[0, 1].grid(False)

    axes[1, 0].imshow(R_np, cmap=cmap, norm=norm)
    axes[1, 0].set_title("3. W_res (true)")
    axes[1, 0].grid(False)

    axes[1, 1].imshow(Rq_np, cmap=cmap, norm=norm)
    axes[1, 1].set_title("4. Q(W_res)")
    axes[1, 1].grid(False)

    if Wrec_np is not None:
        axes[0, 2].imshow(Wrec_np, cmap=cmap, norm=norm)
        axes[0, 2].set_title("5. Reconstruction Q(W_res)+W_svd")
    else:
        axes[0, 2].set_title("5. Reconstruction (n/a)")
        axes[0, 2].axis("off")
    axes[0, 2].grid(False)

    if Wrec_np is not None:
        axes[1, 2].imshow(Wq_np, cmap=cmap, norm=norm)
        axes[1, 2].set_title("6. Quantized model W")
    else:
        axes[1, 2].set_title("6. (n/a)")
        axes[1, 2].axis("off")
    axes[1, 2].grid(False)

    # Place a single colorbar to the far right (outside the subplots)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axes[0, 2])
    cax = divider.append_axes("right", size="2.5%", pad=0.15)
    cb = plt.colorbar(im0, cax=cax)
    cb.set_label("Weight value")

    plt.tight_layout(rect=[0, 0.03, 0.98, 0.95])

    fname = layer_name.replace(".", "_") + "_full_analysis.png"
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_svd_coverage(
    layer_name: str,
    out_dir: str,
    W_original: torch.Tensor,
    ranks: List[int],
) -> Dict[str, str]:
    """
    Save SVD energy coverage plot and CSV. Returns dict with file paths.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import csv

    plt.switch_backend("Agg")
    os.makedirs(out_dir, exist_ok=True)

    s = torch.linalg.svdvals(W_original.cpu()).numpy()
    s2 = s ** 2
    tot = float(np.sum(s2)) or 1e-12
    cum = np.cumsum(s2) / tot

    max_rank = s.shape[0]
    ranks = [int(r) for r in ranks if 1 <= int(r) <= max_rank]
    if not ranks:
        return {}

    coverage_pct = [float(cum[r - 1] * 100.0) for r in ranks]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(1, max_rank + 1), cum * 100.0, color="gray", alpha=0.5, label="Cumulative energy")
    ax.scatter(ranks, coverage_pct, color="tab:blue", label="Requested ranks")
    for r, p in zip(ranks, coverage_pct):
        ax.annotate(f"{p:.1f}%", (r, p), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_title(f"SVD Coverage vs. Rank: {layer_name}")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Coverage (%)")
    ax.set_xscale("log")
    ax.set_ylim(0, 100)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend()

    safe_prefix = layer_name.replace(".", "_")
    png_path = os.path.join(out_dir, f"{safe_prefix}_svd_coverage.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=120)
    plt.close(fig)

    csv_path = os.path.join(out_dir, f"{safe_prefix}_svd_coverage.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "coverage_percent"])
        for r, p in zip(ranks, coverage_pct):
            w.writerow([r, f"{p:.6f}"])

    return {"png": png_path, "csv": csv_path}


def save_svd_components(
    layer_name: str,
    out_dir: str,
    W_original: torch.Tensor,
    *,
    rank: Optional[int] = None,
    coverage_ranks: Optional[List[int]] = None,
) -> str:
    """
    Save a multi-panel SVD visualization with singular values (log/linear) and
    heatmaps of U_r and Vh_r when rank is provided.
    Returns the saved PNG path.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    plt.switch_backend("Agg")
    os.makedirs(out_dir, exist_ok=True)

    if coverage_ranks is None:
        coverage_ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    W_cpu = W_original.cpu()
    U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
    s_np = S.numpy()
    n_sv = s_np.shape[0]
    idx = np.arange(n_sv)

    s2 = s_np ** 2
    tot = float(np.sum(s2)) or 1e-12
    cum = np.cumsum(s2) / tot

    cov_ranks = [int(r) for r in coverage_ranks if 1 <= int(r) <= n_sv]
    cov_pcts = [float(cum[r - 1] * 100.0) for r in cov_ranks]

    cmap = plt.get_cmap("tab20")

    def plot_singular(ax, yscale: Optional[str], title: str):
        ax.bar(idx, s_np, color="lightgray", label=f"All SVs (n={n_sv})")
        if rank is not None and rank >= 1:
            ax.bar(idx[:rank], s_np[:rank], color="cornflowerblue", alpha=0.7, label=f"Kept SVs (≤ {rank})")
        for i, (r, p) in enumerate(zip(cov_ranks, cov_pcts)):
            c = cmap(i % 20)
            ax.axvline(x=r - 0.5, color=c, linestyle="--", linewidth=1.8)
            ax.plot(r - 1, s_np[r - 1], marker="o", color=c, markersize=5, label=f"r={r}: {p:.1f}%")
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("SV index")
        ax.set_ylabel("Singular values" + (" (log)" if yscale == "log" else ""))
        if yscale:
            ax.set_yscale(yscale)
        ax.grid(True, which="both", ls="--", alpha=0.4)
        ax.legend(ncol=4, fontsize=8)

    fig = plt.figure(figsize=(20, 18))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.2, 1.2, 1.0])

    ax_log = fig.add_subplot(gs[0, :])
    plot_singular(ax_log, "log", f"SVD (log) – {layer_name}")

    ax_lin = fig.add_subplot(gs[1, :])
    plot_singular(ax_lin, None, "SVD (linear)")

    if rank is not None and rank >= 1:
        use_r = min(rank, n_sv)
        U_r = U[:, :use_r].numpy()
        Vh_r = Vh[:use_r, :].numpy()

        ax_u = fig.add_subplot(gs[2, 0])
        im_u = ax_u.imshow(U_r, cmap="viridis", aspect="auto")
        ax_u.set_title(f"U_r, shape={U_r.shape}")
        fig.colorbar(im_u, ax=ax_u, fraction=0.046, pad=0.02)

        ax_vh = fig.add_subplot(gs[2, 1])
        im_v = ax_vh.imshow(Vh_r, cmap="viridis", aspect="auto")
        ax_vh.set_title(f"Vh_r, shape={Vh_r.shape}")
        fig.colorbar(im_v, ax=ax_vh, fraction=0.046, pad=0.02)

        kept_pct = float(s2[:use_r].sum() / (tot or 1e-12) * 100.0)
        supt = f"SVD components – {layer_name}\nTop-{use_r} cover {kept_pct:.2f}%"
    else:
        supt = f"SVD – {layer_name}"

    fig.suptitle(supt, fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = os.path.join(out_dir, layer_name.replace(".", "_") + "_svd_4charts_with_coverage.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def compute_reconstruction_errors(
    W_original: torch.Tensor,
    W_svd: torch.Tensor,
    R_true: torch.Tensor,
    R_quant: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute relative errors (svd, quant, total) normalized by ||W_original||.
    """
    with torch.no_grad():
        svd_err = W_original - W_svd
        quant_err = R_true - R_quant
        total_err = svd_err + quant_err  # == W_original - (R_quant + W_svd)

        denom = torch.linalg.norm(W_original).clamp_min(1e-12)
        return {
            "svd_error": (torch.linalg.norm(svd_err) / denom).item(),
            "quant_error": (torch.linalg.norm(quant_err) / denom).item(),
            "total_error": (torch.linalg.norm(total_err) / denom).item(),
        }


def _load_tensor(path: str) -> torch.Tensor:
    import numpy as np
    if path.endswith((".pt", ".pth")):
        t = torch.load(path, map_location="cpu")
        if isinstance(t, torch.Tensor):
            return t
        raise ValueError(f"Tensor file {path} did not contain a torch.Tensor")
    elif path.endswith(".npy"):
        arr = np.load(path)
        return torch.from_numpy(arr)
    else:
        raise ValueError(f"Unsupported tensor file format: {path}")


def save_adapter_comparison(
    layer_name: str,
    out_dir: str,
    adapters: Dict[str, torch.Tensor],
    log_scale_threshold: float = 1e-3,
    share_color_scale: bool = True,
    vpercent: Optional[float] = None,
    zero_eps: float = 0.0,
    pool_factor: Optional[Tuple[int, int] | int | None] = None,
    pool_mode: str = "absmax",
) -> Dict[str, str]:
    """
    Plot a heatmap panel for multiple adapter deltas (ΔW = B@A*scaling).
    Also saves CSVs with norms and pairwise Frobenius distances.
    Returns dict with file paths.
    """
    import numpy as np
    import csv
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    plt.switch_backend("Agg")
    os.makedirs(out_dir, exist_ok=True)

    if not adapters:
        raise ValueError("No adapters provided to save_adapter_comparison")

    # Tensors for metrics (full res)
    tags = list(adapters.keys())
    mats_t = [adapters[t].detach().cpu().to(torch.float32) for t in tags]

    # Pooled copies for visualization only
    pf = _parse_pool_factor(pool_factor)
    mats_vis = [(_maybe_pool(T, pf, pool_mode) if pf else T) for T in mats_t]
    mats = [M.numpy() for M in mats_vis]

    # Helper for vmax
    def vmax_for(arr: np.ndarray) -> float:
        absv = np.abs(arr).ravel()
        if absv.size == 0:
            return 1.0
        if vpercent is not None:
            import numpy as _np
            return float(_np.percentile(absv, vpercent))
        return float(absv.max())

    if share_color_scale:
        vmax = max(max(1e-12, vmax_for(M)) for M in mats) or 1.0
        vmins = [-(vmax)] * len(mats)
        vmaxs = [vmax] * len(mats)
    else:
        vmaxs = [max(1e-12, vmax_for(M)) for M in mats]
        vmins = [-(vm) for vm in vmaxs]

    n = len(tags)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols + 1.5, 6 * rows))
    if n == 1:
        import numpy as _np
        axes = _np.array([[axes]])
    import numpy as _np
    axes = _np.array(axes).reshape(rows, cols)

    cmap = "coolwarm"
    last_ax = None
    last_im = None
    for idx, tag in enumerate(tags):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        M = mats[idx]
        norm = SymLogNorm(linthresh=log_scale_threshold, vmin=vmins[idx], vmax=vmaxs[idx], base=10)
        im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
        last_ax = ax
        last_im = im
        # stats from full-res tensor
        T = mats_t[idx]
        fro = float(T.norm().item())
        mabs = float(T.abs().max().item()) if T.numel() else 0.0
        zeros = float(((T.abs() <= max(zero_eps, 0.0)).float().mean().item())) if T.numel() else 1.0
        ax.set_title(f"{tag}  shape={list(T.shape)}\n||.||_F={fro:.2e}  max|.|={mabs:.2e}  zero<=eps%={zeros*100:.2f}")
        ax.grid(False)

    # Hide any unused axes
    for k in range(n, rows * cols):
        r = k // cols
        c = k % cols
        axes[r, c].axis("off")

    # Colorbar to the far right of the last axis
    if last_ax is not None and last_im is not None:
        divider = make_axes_locatable(last_ax)
        cax = divider.append_axes("right", size="2.5%", pad=0.15)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=last_im.norm)
        cb = plt.colorbar(sm, cax=cax)
        cb.set_label("ΔW value")

    fig.suptitle(f"Adapters ΔW comparison – {layer_name} (SymLog)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 0.98, 0.95])

    safe = layer_name.replace(".", "_")
    png_path = os.path.join(out_dir, f"{safe}_adapters_comparison.png")
    plt.savefig(png_path, dpi=150)
    plt.close(fig)

    # Norm CSV (Frobenius norm and max abs) + zero fraction
    norms_csv = os.path.join(out_dir, f"{safe}_adapters_norms.csv")
    with open(norms_csv, "w", newline="") as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow(["tag", "fro_norm", "max_abs", f"zero_frac(|x|<= {zero_eps:g})"]) 
        for tag, T in zip(tags, mats_t):
            fro = float(T.norm().item())
            mabs = float(T.abs().max().item()) if T.numel() else 0.0
            zf = float(((T.abs() <= max(zero_eps, 0.0)).float().mean().item())) if T.numel() else 1.0
            w.writerow([tag, f"{fro:.6e}", f"{mabs:.6e}", f"{zf:.6f}"])

    # Pairwise Frobenius distances (only when shapes match)
    pairs_csv = os.path.join(out_dir, f"{safe}_adapters_pairwise_fro.csv")
    with open(pairs_csv, "w", newline="") as f:
        import csv as _csv
        w = _csv.writer(f)
        w.writerow(["tag_i", "tag_j", "fro_norm_diff"]) 
        for i in range(n):
            for j in range(i + 1, n):
                if mats_t[i].shape == mats_t[j].shape:
                    diff = (mats_t[i] - mats_t[j]).norm().item()
                    w.writerow([tags[i], tags[j], f"{float(diff):.6e}"])
                else:
                    w.writerow([tags[i], tags[j], "NA(shape_mismatch)"])

    return {"png": png_path, "csv_norms": norms_csv, "csv_pairwise": pairs_csv}


def main():
    """CLI to generate visualizations from saved tensors."""
    import argparse

    parser = argparse.ArgumentParser(description="Visualize residual quantization and SVD components")
    parser.add_argument("--layer-name", type=str, required=True, help="Layer name for titles and filenames")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for figures")

    # Inputs
    parser.add_argument("--w-original", type=str, required=True, help="Path to W_original tensor (.pt/.npy)")
    parser.add_argument("--w-quantized", type=str, required=True, help="Path to W_quantized_model tensor (.pt/.npy)")
    parser.add_argument("--r-true", type=str, required=True, help="Path to R_true tensor (.pt/.npy)")
    parser.add_argument("--r-quant", type=str, required=True, help="Path to R_quant tensor (.pt/.npy)")
    parser.add_argument("--w-svd", type=str, default=None, help="Optional path to precomputed W_svd (.pt/.npy)")
    parser.add_argument("--rank", type=int, default=None, help="Optional rank to compute W_svd if not provided")

    # Options
    parser.add_argument("--log-scale-threshold", type=float, default=1e-3, help="SymLog linthresh")
    parser.add_argument(
        "--coverage-ranks",
        type=str,
        default="1,2,4,8,16,32,64,128,256,512",
        help="Comma-separated ranks for coverage plot",
    )
    parser.add_argument(
        "--plots",
        type=str,
        default="heatmaps,coverage,components",
        help="Comma-separated selection: heatmaps,coverage,components",
    )
    parser.add_argument("--print-errors", action="store_true", help="Compute and print reconstruction errors")

    # Pooling options for heatmaps
    parser.add_argument("--pool", type=str, default=None, help="Optional pooling factor. Examples: '32' or '8,16' (rows,cols). Uses absmax by default.")
    parser.add_argument("--pool-mode", type=str, default="absmax", choices=["absmax", "max", "avg"], help="Pooling reduction: absmax (signed), max, or avg")

    args = parser.parse_args()

    # Load tensors
    W_original = _load_tensor(args.w_original)
    W_quantized = _load_tensor(args.w_quantized)
    R_true = _load_tensor(args.r_true)
    R_quant = _load_tensor(args.r_quant)
    W_svd = _load_tensor(args.w_svd) if args.w_svd else None

    plots = {p.strip().lower() for p in args.plots.split(",")}
    cov_ranks = [int(x.strip()) for x in args.coverage_ranks.split(",") if x.strip()]

    # Parse pool factor
    pf = None
    if args.pool:
        try:
            if "," in args.pool:
                pr, pc = args.pool.split(",", 1)
                pf = (int(pr), int(pc))
            else:
                pf = int(args.pool)
        except Exception as e:
            raise ValueError(f"Invalid --pool value '{args.pool}': {e}")

    # Heatmaps
    if "heatmaps" in plots:
        save_layer_heatmaps(
            layer_name=args.layer_name,
            out_dir=args.out_dir,
            W_original=W_original,
            W_quantized_model=W_quantized,
            R_true=R_true,
            R_quant=R_quant,
            rank=args.rank,
            W_svd=W_svd,
            log_scale_threshold=args.log_scale_threshold,
            pool_factor=pf,
            pool_mode=args.pool_mode,
        )

    # Coverage
    if "coverage" in plots:
        save_svd_coverage(
            layer_name=args.layer_name,
            out_dir=args.out_dir,
            W_original=W_original,
            ranks=cov_ranks,
        )

    # Components
    if "components" in plots:
        save_svd_components(
            layer_name=args.layer_name,
            out_dir=args.out_dir,
            W_original=W_original,
            rank=args.rank,
            coverage_ranks=cov_ranks,
        )

    # Errors
    if args.print_errors:
        if W_svd is None and args.rank is not None and args.rank > 0:
            with torch.no_grad():
                W_cpu = W_original.detach().cpu()
                U, S, Vh = torch.linalg.svd(W_cpu, full_matrices=False)
                r = min(int(args.rank), S.shape[0])
                U_r = U[:, :r]
                Vh_r = Vh[:r, :]
                S_r = S[:r].unsqueeze(1)
                W_svd = U_r @ (S_r * Vh_r)
        if W_svd is None:
            print("[warn] W_svd not provided and no rank specified. Skipping error metrics.")
        else:
            metrics = compute_reconstruction_errors(W_original, W_svd, R_true, R_quant)
            print("Reconstruction errors (relative to ||W_original||):", metrics)


if __name__ == "__main__":
    main()
