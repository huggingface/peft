#!/usr/bin/env python
"""Benchmark: DoRA standard vs kernel implementation.

Compares memory usage and runtime performance of use_dora=True vs
use_dora="kernel" on synthetic data with realistic dimensions.

Usage::

    python scripts/benchmark_dora_kernel.py --in-features 4096 --out-features 4096 --rank 8
    python scripts/benchmark_dora_kernel.py --in-features 11008 --out-features 4096 --rank 64
"""

import argparse
import time

import torch
import torch.nn as nn

from peft import LoraConfig, get_peft_model


def measure_forward(model, x, n_warmup=3, n_runs=10):
    """Measure average forward time and peak memory."""
    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(n_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg_ms = 1000.0 * sum(times) / len(times)
    peak_mem_mb = 0.0
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
    return avg_ms, peak_mem_mb


def measure_backward(model, x, n_warmup=3, n_runs=10):
    """Measure average backward time and peak memory."""
    # Warmup
    for _ in range(n_warmup):
        out = model(x)
        out.sum().backward()
        model.zero_grad()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(n_runs):
        model.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(x)
        out.sum().backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg_ms = 1000.0 * sum(times) / len(times)
    peak_mem_mb = 0.0
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
    return avg_ms, peak_mem_mb


class SingleLinearModel(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.lin(x)


def main():
    parser = argparse.ArgumentParser(description="Benchmark DoRA standard vs kernel")
    parser.add_argument("--in-features", type=int, default=4096, help="Input dimension")
    parser.add_argument("--out-features", type=int, default=4096, help="Output dimension")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--bias", action="store_true", help="Use bias in linear layer")
    parser.add_argument("--eval-only", action="store_true", help="Only benchmark eval mode (forward)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Dimensions: in={args.in_features}, out={args.out_features}, rank={args.rank}")
    print(f"Batch size: {args.batch_size}, bias={args.bias}")
    print()

    torch.manual_seed(42)
    x = torch.randn(args.batch_size, args.in_features, device=device)

    configs = {
        "standard": LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            target_modules=["lin"],
            use_dora=True,
            init_lora_weights=False,
        ),
        "kernel": LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            target_modules=["lin"],
            use_dora="kernel",
            init_lora_weights=False,
        ),
    }

    # --- Eval mode (forward only) ---
    print("=" * 70)
    print("EVAL MODE (forward only)")
    print("=" * 70)
    results = {}
    for name, config in configs.items():
        torch.manual_seed(0)
        model = SingleLinearModel(args.in_features, args.out_features, bias=args.bias).to(device)
        torch.manual_seed(42)
        peft = get_peft_model(model, config)
        peft.eval()

        # Sync weights between models
        results[name] = peft

    # Sync LoRA weights between standard and kernel
    std_layer = results["standard"].base_model.model.lin
    kern_layer = results["kernel"].base_model.model.lin
    kern_layer.lora_A["default"].weight.data = std_layer.lora_A["default"].weight.data.clone()
    kern_layer.lora_B["default"].weight.data = std_layer.lora_B["default"].weight.data.clone()
    kern_layer.lora_magnitude_vector["default"].weight.data = (
        std_layer.lora_magnitude_vector["default"].weight.data.clone()
    )

    # Verify outputs match
    with torch.no_grad():
        out_std = results["standard"](x)
        out_kern = results["kernel"](x)
    max_diff = (out_std - out_kern).abs().max().item()
    print(f"Output max abs diff: {max_diff:.2e}")
    print(f"Outputs match: {torch.allclose(out_std, out_kern, atol=1e-4)}")
    print()

    for name in ["standard", "kernel"]:
        avg_ms, peak_mem = measure_forward(results[name], x)
        print(f"  {name:12s}: {avg_ms:.3f} ms/forward, peak_mem={peak_mem:.1f} MB")

    # --- Training mode (forward + backward) ---
    if not args.eval_only:
        print()
        print("=" * 70)
        print("TRAINING MODE (forward + backward)")
        print("=" * 70)
        for name in ["standard", "kernel"]:
            results[name].train()
            avg_ms, peak_mem = measure_backward(results[name], x)
            print(f"  {name:12s}: {avg_ms:.3f} ms/step, peak_mem={peak_mem:.1f} MB")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    fwd_std = measure_forward(results["standard"], x)
    fwd_kern = measure_forward(results["kernel"], x)
    speedup = fwd_std[0] / fwd_kern[0] if fwd_kern[0] > 0 else float("inf")
    mem_ratio = fwd_kern[1] / fwd_std[1] if fwd_std[1] > 0 else float("inf")
    print(f"Forward speedup: {speedup:.2f}x")
    if torch.cuda.is_available():
        print(f"Forward memory ratio (kernel/std): {mem_ratio:.2f}x")


if __name__ == "__main__":
    main()