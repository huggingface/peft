#!/usr/bin/env python3
"""
Unified CLI to dump tensors from your residual quantization artifacts and plot visualizations.

Subcommands:
  - dump:     Extract W_original, W_svd, R_true, R_quant, W_quantized_model for one layer
  - plot:     Generate plots from saved .pt/.npy tensors (uses vis_tools)
  - pipeline: Run dump then plot in one go
  - compare_adapters: Compare multiple LoRA adapters on a single layer

This script follows your naming pattern for quantized residuals:
  <experiment_dir>/quantized_residuals_r{rank}/
    w_res_{model_clean}_r{rank}_daniel_{bits}bit_gs{group_size}_{dataset}

Where model_clean == model_name_or_path with '/' and '\\' replaced by '_'.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional, List, Tuple

import torch
from transformers import AutoModelForCausalLM

# Local imports
from vis_tools import (
    save_layer_heatmaps,
    save_svd_coverage,
    save_svd_components,
    compute_reconstruction_errors,
    save_adapter_comparison,
)

import json
from typing import Dict
from safetensors.torch import load_file as safetensors_load


def _get_module_by_name(model, dotted_name: str):
    cur = model
    for part in dotted_name.split("."):
        if not hasattr(cur, part):
            raise KeyError(f"Module path not found: {dotted_name} (missing '{part}')")
        cur = getattr(cur, part)
    return cur


def _pick_module_device_and_dtype(mod: torch.nn.Module) -> Tuple[torch.device, torch.dtype]:
    """Pick the best available device and dtype for running a forward-only recovery.
    Preference: keep existing device; if CPU-only and CUDA available, try CUDA. Use fp16 on CUDA, fp32 on CPU.
    """
    dev: Optional[torch.device] = None
    for t in list(mod.parameters()) + list(mod.buffers()):
        if hasattr(t, "device"):
            dev = t.device
            break
    if dev is None:
        dev = torch.device("cpu")
    if dev.type == "cpu" and torch.cuda.is_available():
        # We'll try CUDA later only if CPU forward fails.
        pass
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    return dev, dtype


def _recover_linear_weight_via_forward(
    qmod: torch.nn.Module,
    in_features: int,
    out_features: int,
    *,
    max_chunk: int = 1024,
) -> torch.Tensor:
    """Recover a module's effective float32 weight using its forward pass.

    For a Linear-like module with forward y = x @ W^T + b, we feed identity rows in chunks and
    reconstruct W from the outputs. Works for quantized wrappers (TorchQuantLinear, GPTQLinear, etc.).
    """
    # Decide device/dtype
    dev, act_dtype = _pick_module_device_and_dtype(qmod)

    def _try_forward_on_device(device: torch.device) -> torch.Tensor:
        # Move the submodule if needed (we created the full model only for extraction, so moving is OK)
        try:
            qmod_on_dev = qmod
        except Exception:
            qmod_on_dev = qmod
        try:
            if hasattr(qmod_on_dev, "to") and (next(iter(qmod_on_dev.buffers()), None) is None or next(iter(qmod_on_dev.buffers()), None).device != device):
                try:
                    qmod_on_dev = qmod_on_dev.to(device)
                except Exception:
                    # Some quant modules don't support .to() fully; hope they're already on a usable device
                    pass
        except StopIteration:
            pass

        # Allocate output accumulator on device
        Y = torch.empty((in_features, out_features), dtype=act_dtype if device.type == "cuda" else torch.float32, device=device)

        bias = getattr(qmod_on_dev, "bias", None)
        if isinstance(bias, torch.Tensor):
            bias_dev = bias.to(device=device, dtype=Y.dtype)
        else:
            bias_dev = None

        # We'll create a full eye once if feasible, then slice. If OOM, fall back to chunk construction.
        eye_tensor: Optional[torch.Tensor] = None
        try:
            eye_tensor = torch.eye(in_features, dtype=act_dtype if device.type == "cuda" else torch.float32, device=device)
        except RuntimeError:
            eye_tensor = None

        chunk = max(1, min(max_chunk, in_features))
        with torch.no_grad():
            start = 0
            while start < in_features:
                end = min(in_features, start + chunk)
                if eye_tensor is not None:
                    x = eye_tensor[start:end]
                else:
                    # Build a sparse-ish identity chunk without holding full eye in memory
                    x = torch.zeros((end - start, in_features), dtype=act_dtype if device.type == "cuda" else torch.float32, device=device)
                    idx = torch.arange(start, end, device=device)
                    x[torch.arange(0, end - start, device=device), idx - start] = 1  # place ones on the diagonal slice
                    # The above indexing builds 1s in positions (row=i-start, col=i)
                    # Alternative using scatter would also work but this is efficient enough for moderate chunks
                out = qmod_on_dev(x)
                out = out.to(Y.dtype)
                if bias_dev is not None:
                    out = out - bias_dev  # remove bias broadcast
                Y[start:end] = out
                start = end
        # Y = I @ W^T = W^T rows stacked. So W = Y^T
        W = Y.transpose(0, 1).to(dtype=torch.float32).detach()
        return W

    # First try current device
    try:
        return _try_forward_on_device(dev)
    except Exception as e1:
        # If failed on CPU and CUDA exists, try CUDA once
        if dev.type == "cpu" and torch.cuda.is_available():
            try:
                return _try_forward_on_device(torch.device("cuda"))
            except Exception as e2:
                raise RuntimeError(f"Recover via forward failed on CPU ({e1}) and CUDA ({e2}).")
        raise RuntimeError(f"Recover via forward failed on {dev}: {e1}")


def dump_tensors(
    model_name_or_path: str,
    experiment_dir: str,
    rank: int,
    group_size: int,
    bits: int,
    dataset: str,
    layer_name: str,  # e.g. "model.layers.5.mlp.down_proj"
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # 1) W_original from base FP model
    print(f"[dump] Loading base model: {model_name_or_path}")
    base = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.float32, trust_remote_code=True
    )
    mod = _get_module_by_name(base, layer_name)
    if not hasattr(mod, "weight"):
        raise RuntimeError(f"{layer_name} has no .weight in the base model")
    W_original = mod.weight.detach().cpu()
    torch.save(W_original, os.path.join(out_dir, "W_original.pt"))
    # Keep W_original in memory for shapes
    del mod
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2) R_quant from quantized residual model
    model_clean = model_name_or_path.replace("/", "_").replace("\\", "_")
    qres_dir = os.path.join(experiment_dir, f"quantized_residuals_r{rank}")
    qres_name = f"w_res_{model_clean}_r{rank}_daniel_{bits}bit_gs{group_size}_{dataset}"
    qres_path = os.path.join(qres_dir, qres_name)

    print(f"[dump] Loading quantized residual model: {qres_path}")
    qres = AutoModelForCausalLM.from_pretrained(qres_path, device_map="auto", trust_remote_code=True)

    # Try common ways to recover a float weight for the residual layer
    try:
        qmod = _get_module_by_name(qres, layer_name)
        if hasattr(qmod, "weight") and isinstance(getattr(qmod, "weight"), torch.Tensor):
            R_quant = qmod.weight.detach().float().cpu()
        elif hasattr(qmod, "dequantize_weight") and callable(getattr(qmod, "dequantize_weight")):
            # Some GPTQ/Torch-Quant modules expose a direct dequantize_weight() API
            try:
                w = qmod.dequantize_weight()
                # Some APIs may return a tuple (W, *rest)
                if isinstance(w, tuple) and len(w) > 0 and isinstance(w[0], torch.Tensor):
                    w = w[0]
                if not isinstance(w, torch.Tensor):
                    raise TypeError("dequantize_weight() did not return a Tensor")
                R_quant = w.detach().float().cpu()
            except Exception:
                # Fall through to other strategies
                raise
        elif hasattr(qmod, "to_float") and callable(getattr(qmod, "to_float")):
            try:
                fmod = qmod.to_float()
                if hasattr(fmod, "weight"):
                    R_quant = fmod.weight.detach().float().cpu()
                else:
                    raise AttributeError("to_float() returned module without .weight")
            except Exception:
                # Fall through to other strategies
                raise
        else:
            raise AttributeError("No direct .weight, .dequantize_weight or .to_float available")
    except Exception:
        # Fallback 1: state_dict direct key
        sd = qres.state_dict()
        key = f"{layer_name}.weight"
        if key in sd:
            R_quant = sd[key].detach().float().cpu()
        else:
            # Fallback 2: reconstruct via forward pass on identity chunks
            print("[dump] Direct weight not found; reconstructing via forward pass on identity chunks …")
            in_features = W_original.shape[1]
            out_features = W_original.shape[0]
            R_quant = _recover_linear_weight_via_forward(qmod, in_features, out_features)
            R_quant = R_quant.detach().cpu()
    finally:
        del qres
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    R_quant = R_quant.T
    torch.save(R_quant, os.path.join(out_dir, "R_quant.pt"))

    # 3) Compute W_svd and R_true via SVD of W_original
    print(f"[dump] Computing SVD rank={rank} for {layer_name}")
    with torch.no_grad():
        U, S, Vh = torch.linalg.svd(W_original, full_matrices=False)
        r = min(int(rank), S.shape[0])
        U_r = U[:, :r]
        Vh_r = Vh[:r, :]
        S_r = S[:r].unsqueeze(1)
        W_svd = U_r @ (S_r * Vh_r)
        R_true = W_original - W_svd

    torch.save(W_svd, os.path.join(out_dir, "W_svd.pt"))
    torch.save(R_true, os.path.join(out_dir, "R_true.pt"))

    # 4) Convenience: W_quantized_model ≈ R_quant + W_svd
    W_quantized_model = R_quant + W_svd
    torch.save(W_quantized_model, os.path.join(out_dir, "W_quantized_model.pt"))

    print(f"[dump] Saved tensors to: {out_dir}")


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


def run_plot(
    layer_name: str,
    out_dir: str,
    w_original: str,
    w_quantized: str,
    r_true: str,
    r_quant: str,
    *,
    w_svd: Optional[str] = None,
    rank: Optional[int] = None,
    log_scale_threshold: float = 1e-3,
    coverage_ranks: Optional[List[int]] = None,
    do_heatmaps: bool = True,
    do_coverage: bool = True,
    do_components: bool = True,
    print_errors: bool = True,
) -> None:
    W_original = _load_tensor(w_original)
    W_quant = _load_tensor(w_quantized)
    R_true = _load_tensor(r_true)
    R_quant = _load_tensor(r_quant)
    W_svd_tensor = _load_tensor(w_svd) if w_svd else None

    if do_heatmaps:
        save_layer_heatmaps(
            layer_name=layer_name,
            out_dir=out_dir,
            W_original=W_original,
            W_quantized_model=W_quant,
            R_true=R_true,
            R_quant=R_quant,
            rank=rank,
            W_svd=W_svd_tensor,
            log_scale_threshold=log_scale_threshold,
        )

    cov_ranks = coverage_ranks or [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    if do_coverage:
        save_svd_coverage(
            layer_name=layer_name,
            out_dir=out_dir,
            W_original=W_original,
            ranks=cov_ranks,
        )

    if do_components:
        save_svd_components(
            layer_name=layer_name,
            out_dir=out_dir,
            W_original=W_original,
            rank=rank,
            coverage_ranks=cov_ranks,
        )

    if print_errors:
        # compute errors; if W_svd not provided and rank set, recompute
        if W_svd_tensor is None and rank is not None and rank > 0:
            with torch.no_grad():
                U, S, Vh = torch.linalg.svd(W_original, full_matrices=False)
                r = min(int(rank), S.shape[0])
                U_r = U[:, :r]
                Vh_r = Vh[:r, :]
                S_r = S[:r].unsqueeze(1)
                W_svd_tensor = U_r @ (S_r * Vh_r)
        if W_svd_tensor is not None:
            metrics = compute_reconstruction_errors(W_original, W_svd_tensor, R_true, R_quant)
            print("Errors (relative):", metrics)
        else:
            print("[warn] No W_svd and no rank; skip error metrics")


def _find_adapter_files(root: str):
    """Find adapter_model.safetensors and adapter_config.json under root or common subfolders."""
    candidates = []
    for d in [root, os.path.join(root, "adapter")]:
        model_p = os.path.join(d, "adapter_model.safetensors")
        cfg_p = os.path.join(d, "adapter_config.json")
        if os.path.isfile(model_p) and os.path.isfile(cfg_p):
            candidates.append((model_p, cfg_p))
    if not candidates:
        # search one level deeper
        try:
            for name in os.listdir(root):
                d = os.path.join(root, name)
                if os.path.isdir(d):
                    model_p = os.path.join(d, "adapter_model.safetensors")
                    cfg_p = os.path.join(d, "adapter_config.json")
                    if os.path.isfile(model_p) and os.path.isfile(cfg_p):
                        candidates.append((model_p, cfg_p))
                        break
        except FileNotFoundError:
            pass
    if not candidates:
        raise FileNotFoundError(f"No adapter_model.safetensors + adapter_config.json found under: {root}")
    return candidates[0]


def _read_adapter_scaling(adapter_config_json: str) -> float:
    try:
        with open(adapter_config_json, "r") as f:
            cfg = json.load(f)
        lora_alpha = cfg.get("lora_alpha", cfg.get("lora_alpha_r"))
        r = cfg.get("r", cfg.get("lora_r"))
        if lora_alpha is not None and r not in (None, 0):
            return float(lora_alpha) / float(r)
    except Exception:
        pass
    return 1.0


def _find_lora_keys(state_dict: Dict[str, torch.Tensor], layer_name: str):
    token = layer_name.strip()
    keys = list(state_dict.keys())

    def match_keys(suffix: str):
        return [k for k in keys if token in k and k.endswith(suffix)]

    a_keys = match_keys(".lora_A.default.weight") or match_keys(".lora_A.weight")
    b_keys = match_keys(".lora_B.default.weight") or match_keys(".lora_B.weight")

    if not a_keys or not b_keys:
        def loose_match(suffix: str):
            return [k for k in keys if k.endswith(suffix) and f".{token}." in k]
        a_keys = a_keys or loose_match(".lora_A.default.weight") or loose_match(".lora_A.weight")
        b_keys = b_keys or loose_match(".lora_B.default.weight") or loose_match(".lora_B.weight")

    if not a_keys or not b_keys:
        raise KeyError(f"Could not locate LoRA A/B keys for layer '{layer_name}' in adapter state_dict.")

    a_key = max(a_keys, key=len)
    b_key = max(b_keys, key=len)
    return a_key, b_key


def load_adapter_delta(adapter_root: str, layer_name: str) -> torch.Tensor:
    model_path, cfg_path = _find_adapter_files(adapter_root)
    scaling = _read_adapter_scaling(cfg_path)
    sd = safetensors_load(model_path, device="cpu")

    a_key, b_key = _find_lora_keys(sd, layer_name)
    A = sd[a_key].to(torch.float32)
    B = sd[b_key].to(torch.float32)

    if A.dim() != 2 or B.dim() != 2 or B.shape[1] != A.shape[0]:
        raise ValueError(f"Unexpected shapes for {layer_name}: A={tuple(A.shape)} B={tuple(B.shape)}")

    delta = (B @ A) * float(scaling)
    return delta.contiguous()


def _recover_module_weight_any(qmod: torch.nn.Module) -> torch.Tensor:
    """Best-effort recovery of a Linear-like weight matrix (out_features x in_features).
    Tries: direct .weight -> dequantize_weight() -> to_float().weight -> forward recovery.
    """
    # Direct weight
    w = getattr(qmod, "weight", None)
    if isinstance(w, torch.Tensor):
        return w.detach().float().cpu().contiguous()

    # Dequantize API
    if hasattr(qmod, "dequantize_weight") and callable(getattr(qmod, "dequantize_weight")):
        try:
            w = qmod.dequantize_weight()
            if isinstance(w, tuple) and len(w) > 0 and isinstance(w[0], torch.Tensor):
                w = w[0]
            if isinstance(w, torch.Tensor):
                return w.detach().float().cpu().contiguous()
        except Exception:
            pass

    # to_float() fallback
    if hasattr(qmod, "to_float") and callable(getattr(qmod, "to_float")):
        try:
            fmod = qmod.to_float()
            w = getattr(fmod, "weight", None)
            if isinstance(w, torch.Tensor):
                return w.detach().float().cpu().contiguous()
        except Exception:
            pass

    # Forward reconstruction
    in_features = getattr(qmod, "in_features", None) or getattr(qmod, "infeatures", None)
    out_features = getattr(qmod, "out_features", None) or getattr(qmod, "outfeatures", None)
    if isinstance(in_features, int) and isinstance(out_features, int):
        W = _recover_linear_weight_via_forward(qmod, in_features, out_features)
        return W.detach().float().cpu().contiguous()

    raise RuntimeError("Unable to recover weight: no .weight/.dequantize_weight/.to_float and unknown in/out features")


def _load_model_weight(model_path: str, layer_name: str) -> torch.Tensor:
    """Load a model and extract a layer's (out x in) weight on CPU (handles quantized layers)."""
    mdl = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, trust_remote_code=True)
    qmod = _get_module_by_name(mdl, layer_name)
    W = _recover_module_weight_any(qmod)
    del mdl
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return W


def _cmd_compare_adapters(args):
    # Build adapter map from --adapters-json (file path or inline JSON) and repeated --adapter TAG=PATH
    adapter_map = {}

    def _merge_mapping(new_map: Dict[str, str], source: str):
        for k, v in new_map.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise ValueError(f"Invalid entry in adapters mapping from {source}: {k} -> {v}")
            if k in adapter_map:
                raise ValueError(f"Duplicate adapter tag '{k}' from {source} (already defined)")
            adapter_map[k.strip()] = v.strip()

    # 1) From JSON mapping/list
    if getattr(args, "adapters_json", None):
        raw = args.adapters_json
        loaded = None
        # If it's a file path, read it; else treat as inline JSON
        if os.path.isfile(raw):
            with open(raw, "r") as f:
                loaded = json.load(f)
            src = f"file:{raw}"
        else:
            try:
                loaded = json.loads(raw)
                src = "inline-json"
            except json.JSONDecodeError as e:
                raise ValueError(f"--adapters-json is neither a file nor valid JSON: {raw} ({e})")

        # Accept shapes:
        #  - dict {tag: path}
        #  - list of {"tag":..., "path":...}
        #  - list of [tag, path]
        if isinstance(loaded, dict):
            _merge_mapping(loaded, src)
        elif isinstance(loaded, list):
            tmp = {}
            for i, item in enumerate(loaded):
                if isinstance(item, dict) and "tag" in item and "path" in item:
                    tmp[str(item["tag"])] = str(item["path"])
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    tmp[str(item[0])] = str(item[1])
                else:
                    raise ValueError(f"Unsupported entry at index {i} in {src}: {item}")
            _merge_mapping(tmp, src)
        else:
            raise ValueError(f"Unsupported JSON structure in {src}. Use an object or a list of entries.")

    # 2) From repeated --adapter TAG=PATH
    if getattr(args, "adapter", None):
        for spec in args.adapter:
            if "=" not in spec:
                raise ValueError(f"--adapter expects TAG=PATH, got: {spec}")
            tag, path = spec.split("=", 1)
            tag = tag.strip()
            path = path.strip()
            if not tag or not path:
                raise ValueError(f"Invalid adapter spec: {spec}")
            if tag in adapter_map:
                raise ValueError(f"Duplicate adapter tag '{tag}' from --adapter (already defined)")
            adapter_map[tag] = path

    if not adapter_map and not getattr(args, "weight", None) and not getattr(args, "model", None):
        raise ValueError("No inputs specified. Use --adapters-json/--adapter and/or --weight and/or --model TAG=PATH.")

    # Load adapter deltas
    deltas = {}
    for tag, root in adapter_map.items():
        deltas[tag] = load_adapter_delta(root, args.layer_name)

    # Load raw weights from tensor files
    raw_weights: Dict[str, torch.Tensor] = {}
    if getattr(args, "weight", None):
        for spec in args.weight:
            if "=" not in spec:
                raise ValueError(f"--weight expects TAG=PATH, got: {spec}")
            tag, path = spec.split("=", 1)
            tag = tag.strip(); path = path.strip()
            if tag in deltas or tag in raw_weights:
                raise ValueError(f"Duplicate tag '{tag}' across inputs")
            W = _load_tensor(path).to(torch.float32).contiguous()
            raw_weights[tag] = W

    # Load raw weights directly from model folders
    if getattr(args, "model", None):
        for spec in args.model:
            if "=" not in spec:
                raise ValueError(f"--model expects TAG=PATH, got: {spec}")
            tag, path = spec.split("=", 1)
            tag = tag.strip(); path = path.strip()
            if tag in deltas or tag in raw_weights:
                raise ValueError(f"Duplicate tag '{tag}' across inputs")
            W = _load_model_weight(path, args.layer_name)
            raw_weights[tag] = W

    # Merge all panels
    panels = {**deltas, **raw_weights}
    if not panels:
        raise ValueError("Nothing to visualize.")
    
    # ✅ NEW: Compute delta panels from --delta TAG1,TAG2 or TAG1:TAG2
    if getattr(args, "delta", None):
        for spec in args.delta:
            # Parse TAG1,TAG2 or TAG1:TAG2
            sep = "," if "," in spec else ":"
            parts = spec.split(sep, 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid --delta spec '{spec}'. Expected TAG1,TAG2 or TAG1:TAG2")
            tag1, tag2 = parts[0].strip(), parts[1].strip()
            
            # Check both tags exist
            if tag1 not in panels:
                raise ValueError(f"Delta source tag '{tag1}' not found in loaded panels: {list(panels.keys())}")
            if tag2 not in panels:
                raise ValueError(f"Delta target tag '{tag2}' not found in loaded panels: {list(panels.keys())}")
            
            # Compute delta
            W1 = panels[tag1]
            W2 = panels[tag2]
            if W1.shape != W2.shape:
                raise ValueError(f"Shape mismatch for delta {tag1},{tag2}: {W1.shape} vs {W2.shape}")
            
            delta_tag = f"{tag2}-{tag1}"  # Name convention: "ft-pre" means ft minus pre
            if delta_tag in panels:
                raise ValueError(f"Delta tag '{delta_tag}' already exists (duplicate or naming conflict)")
            
            panels[delta_tag] = (W2 - W1).contiguous()
            print(f"[delta] Computed {delta_tag} = {tag2} - {tag1}")

    # Optional shape check (must match for a shared pooling factor)
    shapes = {tuple(t.shape) for t in panels.values()}
    if len(shapes) > 1:
        print(f"[warn] Input tensors have differing shapes: {shapes}. Plotting may fail unless pooling brings them to a common size.")

    # Parse optional pooling factor
    pf = None
    if getattr(args, "pool", None):
        try:
            if "," in args.pool:
                pr, pc = args.pool.split(",", 1)
                pf = (int(pr), int(pc))
            else:
                pf = int(args.pool)
        except Exception as e:
            raise ValueError(f"Invalid --pool value '{args.pool}': {e}")

    os.makedirs(args.out_dir, exist_ok=True)
    out = save_adapter_comparison(
        layer_name=args.layer_name,
        out_dir=args.out_dir,
        adapters=panels,
        log_scale_threshold=args.log_scale_threshold,
        share_color_scale=not getattr(args, "per_panel_scale", False),
        vpercent=getattr(args, "vpercent", None),
        zero_eps=getattr(args, "zero_eps", 0.0),
        pool_factor=pf,
        pool_mode=getattr(args, "pool_mode", "absmax"),
    )
    print("Saved:", out)


def main():
    ap = argparse.ArgumentParser(description="Analyze a single layer of residual quantization pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # dump
    ap_dump = sub.add_parser("dump", help="Dump tensors for one layer from experiment artifacts")
    ap_dump.add_argument("--model-name-or-path", required=True)
    ap_dump.add_argument("--experiment-dir", required=True)
    ap_dump.add_argument("--rank", type=int, required=True)
    ap_dump.add_argument("--group-size", type=int, required=True)
    ap_dump.add_argument("--bits", type=int, required=True)
    ap_dump.add_argument("--dataset", required=True)
    ap_dump.add_argument("--layer-name", required=True)
    ap_dump.add_argument("--out-dir", required=True)

    # plot
    ap_plot = sub.add_parser("plot", help="Generate plots from dumped tensors")
    ap_plot.add_argument("--layer-name", required=True)
    ap_plot.add_argument("--out-dir", required=True)
    ap_plot.add_argument("--w-original", required=True)
    ap_plot.add_argument("--w-quantized", required=True)
    ap_plot.add_argument("--r-true", required=True)
    ap_plot.add_argument("--r-quant", required=True)
    ap_plot.add_argument("--w-svd")
    ap_plot.add_argument("--rank", type=int)
    ap_plot.add_argument("--log-scale-threshold", type=float, default=1e-3)
    ap_plot.add_argument("--coverage-ranks", type=str, default="1,2,4,8,16,32,64,128,256,512")

    # pipeline
    ap_pipe = sub.add_parser("pipeline", help="Run dump then plot")
    ap_pipe.add_argument("--model-name-or-path", required=True)
    ap_pipe.add_argument("--experiment-dir", required=True)
    ap_pipe.add_argument("--rank", type=int, required=True)
    ap_pipe.add_argument("--group-size", type=int, required=True)
    ap_pipe.add_argument("--bits", type=int, required=True)
    ap_pipe.add_argument("--dataset", required=True)
    ap_pipe.add_argument("--layer-name", required=True)
    ap_pipe.add_argument("--out-base", required=True, help="Base dir to create tensors/ and plots/ subfolders")
    ap_pipe.add_argument("--log-scale-threshold", type=float, default=1e-3)
    ap_pipe.add_argument("--rank-for-svd", type=int, help="Rank for SVD (defaults to --rank)")
    ap_pipe.add_argument("--coverage-ranks", type=str, default="1,2,4,8,16,32,64,128,256,512")

    # compare_adapters
    ap_cmp = sub.add_parser("compare_adapters", help="Compare adapters and/or raw weights on a single layer")
    ap_cmp.add_argument("--layer-name", required=True, type=str)
    ap_cmp.add_argument("--out-dir", required=True, type=str)
    ap_cmp.add_argument(
        "--adapter",
        action="append",
        required=False,
        help=(
            "Repeatable: TAG=PATH to adapter folder (contains adapter_model.safetensors & adapter_config.json). "
            "Example: pre=/.../quantized_residuals_r4/daniel_adapter_r4_...  "
            "ft=/.../SmolLM2-.../ft/adapter  qalora=/.../_qalora_.../ft/adapter"
        ),
    )
    ap_cmp.add_argument(
        "--adapters-json",
        type=str,
        required=False,
        help=(
            "JSON mapping (either a path to a .json file or inline JSON). "
            "Accepted forms: {\"pre\": \"/path\", ...} or "
            "[{\"tag\": \"pre\", \"path\": \"/path\"}, ...] or [[\"pre\", \"/path\"], ...]"
        ),
    )
    # New: include arbitrary weights
    ap_cmp.add_argument(
        "--weight",
        action="append",
        required=False,
        help=(
            "Repeatable: TAG=PATH to a tensor file (.pt/.npy) to visualize as a panel. "
            "Useful for W_original.pt, R_quant.pt, W_svd.pt, etc."
        ),
    )
    ap_cmp.add_argument(
        "--model",
        action="append",
        required=False,
        help=(
            "Repeatable: TAG=PATH to a model directory; the weight of --layer-name will be loaded from the model. "
            "Works with FP and quantized models (best-effort recovery)."
        ),
    )
    ap_cmp.add_argument("--log-scale-threshold", type=float, default=1e-3)
    ap_cmp.add_argument(
        "--delta",
        action="append",
        required=False,
        help=(
            "Repeatable: specify two known tags to compute a delta panel TAG2 - TAG1. "
            "Format 'TAG1,TAG2' or 'TAG1:TAG2'. Example: --delta pre,ft"
        ),
    )
    # Visualization controls
    ap_cmp.add_argument("--per-panel-scale", action="store_true", help="Use independent color scales per panel instead of sharing one.")
    ap_cmp.add_argument("--vpercent", type=float, default=None, help="If set (e.g., 99.9), color scale vmax uses this percentile of |ΔW| (shared or per-panel).")
    ap_cmp.add_argument("--zero-eps", "--zero_eps", dest="zero_eps", type=float, default=0.0, help="Report zero fraction as |ΔW|<=zero_eps.")
    ap_cmp.add_argument("--pool", type=str, default=None, help="Optional pooling factor. Examples: '32' or '8,16' (rows,cols). Uses absmax by default.")
    ap_cmp.add_argument("--pool-mode", type=str, default="absmax", choices=["absmax", "max", "avg"], help="Pooling reduction: absmax (signed), max, or avg")

    args = ap.parse_args()

    if args.cmd == "dump":
        dump_tensors(
            model_name_or_path=args.model_name_or_path,
            experiment_dir=args.experiment_dir,
            rank=args.rank,
            group_size=args.group_size,
            bits=args.bits,
            dataset=args.dataset,
            layer_name=args.layer_name,
            out_dir=args.out_dir,
        )
        return

    if args.cmd == "plot":
        cov = [int(x.strip()) for x in args.coverage_ranks.split(",") if x.strip()]
        run_plot(
            layer_name=args.layer_name,
            out_dir=args.out_dir,
            w_original=args.w_original,
            w_quantized=args.w_quantized,
            r_true=args.r_true,
            r_quant=args.r_quant,
            w_svd=args.w_svd,
            rank=args.rank,
            log_scale_threshold=args.log_scale_threshold,
            coverage_ranks=cov,
        )
        return

    if args.cmd == "pipeline":
        tensors_dir = os.path.join(args.out_base, "tensors")
        plots_dir = os.path.join(args.out_base, "plots")
        os.makedirs(tensors_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        dump_tensors(
            model_name_or_path=args.model_name_or_path,
            experiment_dir=args.experiment_dir,
            rank=args.rank,
            group_size=args.group_size,
            bits=args.bits,
            dataset=args.dataset,
            layer_name=args.layer_name,
            out_dir=tensors_dir,
        )
        cov = [int(x.strip()) for x in args.coverage_ranks.split(",") if x.strip()]
        run_plot(
            layer_name=args.layer_name,
            out_dir=plots_dir,
            w_original=os.path.join(tensors_dir, "W_original.pt"),
            w_quantized=os.path.join(tensors_dir, "W_quantized_model.pt"),
            r_true=os.path.join(tensors_dir, "R_true.pt"),
            r_quant=os.path.join(tensors_dir, "R_quant.pt"),
            w_svd=os.path.join(tensors_dir, "W_svd.pt"),
            rank=(args.rank_for_svd or args.rank),
            log_scale_threshold=args.log_scale_threshold,
            coverage_ranks=cov,
        )
        return

    if args.cmd == "compare_adapters":
        return _cmd_compare_adapters(args)


if __name__ == "__main__":
    main()
