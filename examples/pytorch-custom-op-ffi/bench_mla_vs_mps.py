#!/usr/bin/env python3
"""MLA decompression throughput: MFA (via FFI) vs torch.matmul on MPS.

The MLA forward is two GEMMs:  K = kv_latent @ Wk,  V = kv_latent @ Wv
with M = batch*seq, K = kv_latent_dim, N = num_heads*head_dim. Both the FFI
call and the MPS reference do the same two GEMMs, so TFLOPS are comparable.

The FFI forward is synchronous (commit + waitUntilCompleted), so wall-clock
per call approximates GPU time. The MPS path is bracketed with
torch.mps.synchronize(). Each reports the min over timed trials after warmup.
"""

import argparse
import statistics
import time

import metal_sdpa_extension as ext
import torch

SHAPES = [
    # (batch, seq, heads, head_dim, kv_latent_dim)
    (1, 512, 8, 128, 512),
    (1, 1024, 8, 128, 512),
    (1, 2048, 8, 128, 512),
]


def flops_two_gemms(M: int, N: int, K: int) -> float:
    # 2 GEMMs (K and V), each a MAC = 2 flops.
    return 2.0 * 2.0 * float(M) * float(N) * float(K)


def time_mla(ctx, kv, batch, heads, seq, head_dim, kv_latent_dim, iters: int) -> float:
    for _ in range(5):
        ctx.forward(kv, batch, heads, seq, head_dim, kv_latent_dim)
    times = []
    for _ in range(iters):
        torch.mps.synchronize()
        start = time.perf_counter()
        ctx.forward(kv, batch, heads, seq, head_dim, kv_latent_dim)
        times.append(time.perf_counter() - start)
    return min(times)


def time_mps(kv, wk, wv, iters: int) -> float:
    for _ in range(5):
        kv @ wk
        kv @ wv
    torch.mps.synchronize()
    times = []
    for _ in range(iters):
        torch.mps.synchronize()
        start = time.perf_counter()
        k = kv @ wk
        v = kv @ wv
        torch.mps.synchronize()
        times.append(time.perf_counter() - start)
    return min(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS device not available")
    device = torch.device("mps")
    torch.manual_seed(0)

    print(f"\nMLA decompression: MFA (FFI) vs torch.matmul (MPS), FP16")
    import subprocess

    try:
        device_name = (
            subprocess.check_output(
                ["system_profiler", "SPDisplaysDataType"], text=True
            )
            .split("Chipset Model:")[1]
            .split("\n")[0]
            .strip()
        )
    except Exception:
        device_name = "Apple Silicon"
    print(f"device: {device_name}")
    print("-" * 78)
    print(
        f"{'shape (M,K,N)':<22}{'MFA ms':>9}{'MFA TFLOPS':>13}{'MPS ms':>9}{'MPS TFLOPS':>13}{'speedup':>9}"
    )
    print("-" * 78)

    for batch, seq, heads, head_dim, kv_latent_dim in SHAPES:
        total_dim = heads * head_dim
        M, K, N = batch * seq, kv_latent_dim, total_dim
        fl = flops_two_gemms(M, N, K)

        kv = torch.randn(M, K, device=device, dtype=torch.float16) * 0.1
        wk = torch.randn(K, N, device=device, dtype=torch.float16) * 0.1
        wv = torch.randn(K, N, device=device, dtype=torch.float16) * 0.1

        ctx = ext.MlaContext()
        ctx.load_weights(wk, wv)

        mla_s = time_mla(
            ctx, kv, batch, heads, seq, head_dim, kv_latent_dim, args.iters
        )
        mps_s = time_mps(kv, wk, wv, args.iters)

        mfa_tflops = fl / mla_s / 1e12
        mps_tflops = fl / mps_s / 1e12
        speedup = mps_s / mla_s

        shape_str = f"({M},{K},{N})"
        print(
            f"{shape_str:<22}{mla_s*1e3:>9.3f}{mfa_tflops:>13.2f}"
            f"{mps_s*1e3:>9.3f}{mps_tflops:>13.2f}{speedup:>9.2f}x"
        )

        # Correctness spot-check at this shape.
        k, v = ctx.forward(kv, batch, heads, seq, head_dim, kv_latent_dim)
        k_ref = (kv.float() @ wk.float()).half()
        max_diff = (k.float() - k_ref.float()).abs().max().item()
        assert max_diff < 0.5, f"correctness drift at {shape_str}: maxDiff={max_diff}"

    print("-" * 78)


if __name__ == "__main__":
    main()
