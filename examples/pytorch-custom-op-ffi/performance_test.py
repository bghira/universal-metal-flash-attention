#!/usr/bin/env python3
"""
Performance benchmark for Metal SDPA backend.
Author: bghira
"""

import time

import metal_sdpa_extension
import numpy as np
import torch
import torch.nn.functional as F


def benchmark_case(name, q, k, v, num_warmup=5, num_trials=20):
    """Benchmark a specific case"""
    print(f"\n=== {name} ===")
    print(f"Shape: Q={list(q.shape)}, K={list(k.shape)}, V={list(v.shape)}")
    print(f"Dtype: {q.dtype}")

    # Warmup
    for _ in range(num_warmup):
        _ = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
        _ = F.scaled_dot_product_attention(q, k, v)

    # Benchmark Metal SDPA
    metal_times = []
    for _ in range(num_trials):
        start = time.perf_counter()
        metal_output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
        end = time.perf_counter()
        metal_times.append(end - start)

    # Benchmark PyTorch SDPA
    torch_times = []
    for _ in range(num_trials):
        start = time.perf_counter()
        torch_output = F.scaled_dot_product_attention(q, k, v)
        end = time.perf_counter()
        torch_times.append(end - start)

    # Calculate statistics
    metal_mean = np.mean(metal_times) * 1000  # Convert to ms
    metal_std = np.std(metal_times) * 1000
    torch_mean = np.mean(torch_times) * 1000
    torch_std = np.std(torch_times) * 1000

    speedup = torch_mean / metal_mean if metal_mean > 0 else 0

    # Check accuracy
    diff = torch.abs(metal_output - torch_output).max().item()
    rel_diff = (
        (diff / torch.abs(torch_output).max().item())
        if torch_output.abs().max() > 0
        else 0
    )

    print(f"Metal SDPA:   {metal_mean:.2f} ± {metal_std:.2f} ms")
    print(f"PyTorch SDPA: {torch_mean:.2f} ± {torch_std:.2f} ms")
    print(f"Speedup:      {speedup:.2f}x")
    print(f"Max diff:     {diff:.2e}")
    print(f"Rel diff:     {rel_diff:.2e}")

    return {
        "name": name,
        "metal_mean": metal_mean,
        "torch_mean": torch_mean,
        "speedup": speedup,
        "accuracy": diff,
    }


def main():
    print("Metal Flash Attention Performance Benchmark")
    print("Author: bghira")
    print("=" * 60)

    if not metal_sdpa_extension.is_metal_available():
        print("❌ Metal not available")
        return

    print("✅ Metal available")
    version = metal_sdpa_extension.get_version()
    print(f"MFA Version: {version}")

    results = []

    # Test different sizes
    test_cases = [
        ("Small (32x16)", torch.randn(32, 16, dtype=torch.float32)),
        ("Medium (128x32)", torch.randn(128, 32, dtype=torch.float32)),
        ("Large (512x64)", torch.randn(512, 64, dtype=torch.float32)),
        ("Very Large (1024x64)", torch.randn(1024, 64, dtype=torch.float32)),
    ]

    for name, q in test_cases:
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        try:
            result = benchmark_case(name, q, k, v)
            results.append(result)
        except Exception as e:
            print(f"❌ {name}: {e}")

    # Test different data types
    print(f"\n{'='*60}")
    print("Data Type Comparison (512x32)")

    base_shape = (512, 32)
    for dtype_name, dtype in [("float32", torch.float32), ("float16", torch.float16)]:
        q = torch.randn(*base_shape, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        try:
            result = benchmark_case(f"{dtype_name}", q, k, v, num_trials=10)
            results.append(result)
        except Exception as e:
            print(f"❌ {dtype_name}: {e}")

    # Test 4D tensors (single head)
    print(f"\n{'='*60}")
    print("4D Tensor Test (single head)")

    q_4d = torch.randn(
        4, 128, 1, 32, dtype=torch.float32
    )  # batch=4, seq=128, heads=1, dim=32
    k_4d = torch.randn_like(q_4d)
    v_4d = torch.randn_like(q_4d)

    try:
        result = benchmark_case("4D Single Head", q_4d, k_4d, v_4d)
        results.append(result)
    except Exception as e:
        print(f"❌ 4D Single Head: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")

    if results:
        valid_results = [r for r in results if r["speedup"] > 0]
        if valid_results:
            avg_speedup = np.mean([r["speedup"] for r in valid_results])
            max_speedup = max([r["speedup"] for r in valid_results])
            best_case = max(valid_results, key=lambda x: x["speedup"])

            print(f"Average Speedup: {avg_speedup:.2f}x")
            print(f"Best Speedup:    {max_speedup:.2f}x ({best_case['name']})")
            print(f"Worst Speedup:   {min([r['speedup'] for r in valid_results]):.2f}x")

            # Performance classification
            if avg_speedup > 2.0:
                performance = "🚀 Excellent"
            elif avg_speedup > 1.5:
                performance = "✅ Good"
            elif avg_speedup > 1.0:
                performance = "🆗 Moderate"
            else:
                performance = "⚠️ Poor"

            print(f"\nOverall Performance: {performance}")

            # Accuracy summary
            max_error = max([r["accuracy"] for r in results if "accuracy" in r])
            print(f"Max Accuracy Error: {max_error:.2e}")

            if max_error < 1e-4:
                print("✅ Accuracy: Excellent")
            elif max_error < 1e-3:
                print("✅ Accuracy: Good")
            else:
                print("⚠️ Accuracy: Check required")
        else:
            print("❌ No valid performance results")
    else:
        print("❌ No benchmark results available")


if __name__ == "__main__":
    main()
