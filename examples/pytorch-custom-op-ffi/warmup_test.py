#!/usr/bin/env python3
"""
Test the effect of warmup on Metal SDPA performance.
Author: bghira
"""

import time

import metal_sdpa_extension
import torch
import torch.nn.functional as F


def test_warmup_effect():
    """Test how warmup affects performance across different sizes"""
    print("Metal SDPA Warmup Effect Analysis")
    print("Author: bghira")
    print("Based on bghira's insights about pipeline caching")
    print("=" * 60)

    if not metal_sdpa_extension.is_metal_available():
        print("❌ Metal not available")
        return

    test_cases = [
        ("Small", 64, 16),
        ("Medium", 128, 32),
        ("Large", 256, 64),
        ("Very Large", 512, 64),
    ]

    print(
        f"{'Size':12} | {'Cold Metal':>12} | {'Warm Metal':>12} | {'PyTorch':>10} | {'Cold vs PT':>10} | {'Warm vs PT':>10}"
    )
    print("-" * 80)

    for name, seq_len, head_dim in test_cases:
        q = torch.randn(seq_len, head_dim, dtype=torch.float32)
        k = torch.randn(seq_len, head_dim, dtype=torch.float32)
        v = torch.randn(seq_len, head_dim, dtype=torch.float32)

        # Cold start (first call)
        start = time.perf_counter()
        _ = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
        cold_time = (time.perf_counter() - start) * 1000

        # Warm up with multiple calls
        for _ in range(10):
            _ = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

        # Warm performance
        start = time.perf_counter()
        _ = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
        warm_time = (time.perf_counter() - start) * 1000

        # PyTorch baseline
        start = time.perf_counter()
        _ = F.scaled_dot_product_attention(q, k, v)
        pytorch_time = (time.perf_counter() - start) * 1000

        cold_ratio = pytorch_time / cold_time
        warm_ratio = pytorch_time / warm_time

        print(
            f"{name:12} | {cold_time:9.2f} ms | {warm_time:9.2f} ms | {pytorch_time:7.2f} ms | {cold_ratio:7.2f}x | {warm_ratio:7.2f}x"
        )

    print("-" * 80)
    print("\nKey Insights:")
    print("- Cold start includes pipeline compilation/caching overhead")
    print("- Warm performance shows true computational capability")
    print("- bghira's observation: warmup reveals MFA's true performance")


def test_context_reuse():
    """Test if keeping context alive improves performance"""
    print(f"\n{'='*60}")
    print("Context Reuse Test")

    q = torch.randn(256, 64, dtype=torch.float32)
    k = torch.randn(256, 64, dtype=torch.float32)
    v = torch.randn(256, 64, dtype=torch.float32)

    # Multiple calls in sequence (context should stay alive)
    times = []
    for i in range(20):
        start = time.perf_counter()
        _ = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    print(f"Call 1 (cold):     {times[0]:.2f} ms")
    print(f"Call 2-5 (warm):   {sum(times[1:5])/4:.2f} ms avg")
    print(f"Call 6-10:         {sum(times[5:10])/5:.2f} ms avg")
    print(f"Call 11-20:        {sum(times[10:20])/10:.2f} ms avg")

    improvement = times[0] / (sum(times[10:20]) / 10)
    print(f"Improvement factor: {improvement:.2f}x")


def main():
    test_warmup_effect()
    test_context_reuse()


if __name__ == "__main__":
    main()
