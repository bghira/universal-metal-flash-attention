#!/usr/bin/env python3
"""
Test GPU latency measurement for Metal SDPA backend.
Author: bghira
"""

import torch
import metal_sdpa_extension
import ctypes


def test_gpu_latency():
    """Test the GPU latency measurement function"""
    print("GPU Latency Measurement Test")
    print("Author: bghira")
    print("=" * 40)

    if not metal_sdpa_extension.is_metal_available():
        print("❌ Metal not available")
        return

    # Test case
    q = torch.randn(512, 64, dtype=torch.float32)
    k = torch.randn(512, 64, dtype=torch.float32)
    v = torch.randn(512, 64, dtype=torch.float32)

    print(f"Testing with shape: {q.shape}")

    # Run operation
    output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

    # Try to get GPU latency (this requires accessing the Swift context)
    # For now, just verify the operation completed
    print(f"✅ Operation completed successfully")
    print(f"Output shape: {output.shape}")
    print(f"Has NaN: {torch.isnan(output).any()}")

    # The GPU latency function would need to be exposed through the Python bindings
    # For now, we can measure end-to-end latency as a proxy
    import time

    num_runs = 10
    times = []

    for _ in range(num_runs):
        start = time.perf_counter()
        _ = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times) * 1000  # Convert to ms
    print(f"Average end-to-end latency: {avg_time:.2f} ms")


if __name__ == "__main__":
    test_gpu_latency()
