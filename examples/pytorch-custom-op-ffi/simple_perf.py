#!/usr/bin/env python3
"""
Simple performance test for Metal SDPA backend.
Author: bghira
"""

import torch
import torch.nn.functional as F
import metal_sdpa_extension
import time


def simple_benchmark():
    """Simple performance test"""
    print("Simple Metal SDPA Performance Test")
    print("Author: bghira")
    print("=" * 40)

    if not metal_sdpa_extension.is_metal_available():
        print("❌ Metal not available")
        return

    # Test case: medium size
    seq_len, head_dim = 256, 32
    q = torch.randn(seq_len, head_dim, dtype=torch.float32)
    k = torch.randn(seq_len, head_dim, dtype=torch.float32)
    v = torch.randn(seq_len, head_dim, dtype=torch.float32)

    print(f"Testing shape: {q.shape}")

    # Single warmup
    print("Warming up...")
    _ = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
    _ = F.scaled_dot_product_attention(q, k, v)

    # Time Metal SDPA (single run)
    print("Timing Metal SDPA...")
    start = time.perf_counter()
    metal_output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
    metal_time = time.perf_counter() - start

    # Time PyTorch SDPA (single run)
    print("Timing PyTorch SDPA...")
    start = time.perf_counter()
    torch_output = F.scaled_dot_product_attention(q, k, v)
    torch_time = time.perf_counter() - start

    # Results
    metal_ms = metal_time * 1000
    torch_ms = torch_time * 1000
    speedup = torch_time / metal_time if metal_time > 0 else 0

    # Accuracy
    diff = torch.abs(metal_output - torch_output).max().item()

    print(f"\nResults:")
    print(f"Metal SDPA:   {metal_ms:.2f} ms")
    print(f"PyTorch SDPA: {torch_ms:.2f} ms")
    print(f"Speedup:      {speedup:.2f}x")
    print(f"Max diff:     {diff:.2e}")

    # Performance assessment
    if speedup > 1.5:
        assessment = "🚀 Good performance"
    elif speedup > 1.0:
        assessment = "🆗 Moderate performance"
    elif speedup > 0.5:
        assessment = "⚠️ Slower than PyTorch"
    else:
        assessment = "❌ Much slower than PyTorch"

    print(f"Assessment:   {assessment}")

    if diff < 1e-4:
        print("Accuracy:     ✅ Excellent")
    else:
        print(f"Accuracy:     ⚠️ Check required (diff={diff:.2e})")


if __name__ == "__main__":
    simple_benchmark()
