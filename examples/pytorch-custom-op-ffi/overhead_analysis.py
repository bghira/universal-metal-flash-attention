#!/usr/bin/env python3
"""
Analyze overhead sources in Metal SDPA backend on Apple Silicon.
Author: bghira
"""

import torch
import torch.nn.functional as F
import metal_sdpa_extension
import time


def analyze_overhead():
    """Analyze where the overhead is coming from"""
    print("Metal SDPA Overhead Analysis on Apple Silicon")
    print("Author: bghira")
    print("=" * 50)

    if not metal_sdpa_extension.is_metal_available():
        print("❌ Metal not available")
        return

    # Small tensor case where overhead is most visible
    q = torch.randn(64, 16, dtype=torch.float32)
    k = torch.randn(64, 16, dtype=torch.float32)
    v = torch.randn(64, 16, dtype=torch.float32)

    print(f"Analyzing tensor shape: {q.shape}")
    print(f"Total elements: {q.numel()}")
    print(f"Memory per tensor: {q.numel() * 4} bytes ({q.numel() * 4 / 1024:.1f} KB)")

    # Break down the overhead sources
    print("\n=== Overhead Analysis ===")

    # 1. Function call overhead
    num_trials = 1000

    # Measure pure Python function call overhead
    def dummy_function(a, b, c):
        return a

    start = time.perf_counter()
    for _ in range(num_trials):
        _ = dummy_function(q, k, v)
    python_overhead = (time.perf_counter() - start) / num_trials * 1000

    print(f"1. Python function call overhead: {python_overhead:.4f} ms")

    # 2. PyTorch operation baseline
    start = time.perf_counter()
    for _ in range(num_trials):
        _ = F.scaled_dot_product_attention(q, k, v)
    pytorch_time = (time.perf_counter() - start) / num_trials * 1000

    print(f"2. PyTorch SDPA time: {pytorch_time:.4f} ms")

    # 3. Metal extension call
    start = time.perf_counter()
    for _ in range(num_trials):
        _ = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
    metal_time = (time.perf_counter() - start) / num_trials * 1000

    print(f"3. Metal SDPA time: {metal_time:.4f} ms")

    # 4. Tensor operations (similar to what happens inside)
    start = time.perf_counter()
    for _ in range(num_trials):
        q_cpu = (
            q.to(torch.kCPU).contiguous()
            if not (q.device.type == "cpu" and q.is_contiguous())
            else q
        )
        k_cpu = (
            k.to(torch.kCPU).contiguous()
            if not (k.device.type == "cpu" and k.is_contiguous())
            else k
        )
        v_cpu = (
            v.to(torch.kCPU).contiguous()
            if not (v.device.type == "cpu" and v.is_contiguous())
            else v
        )
        output = torch.empty_like(q_cpu)
    tensor_ops_time = (time.perf_counter() - start) / num_trials * 1000

    print(f"4. Tensor operations overhead: {tensor_ops_time:.4f} ms")

    # Analysis
    print(f"\n=== Overhead Breakdown ===")
    metal_overhead = metal_time - pytorch_time
    print(f"Total Metal overhead: {metal_overhead:.4f} ms")

    # Likely sources on Apple Silicon:
    print(f"\nLikely overhead sources (not data transfer on unified memory):")
    print(f"- C++/Python FFI calls: ~{tensor_ops_time:.4f} ms")
    print(f"- Metal kernel dispatch: ~{metal_overhead - tensor_ops_time:.4f} ms")
    print(f"- Swift FFI layer: (included in Metal kernel dispatch)")
    print(f"- Metal command encoding: (included in Metal kernel dispatch)")

    # Efficiency analysis
    efficiency = pytorch_time / metal_time * 100
    print(f"\nEfficiency for small tensors: {efficiency:.1f}%")

    if efficiency < 50:
        print("❌ Very inefficient for small tensors")
    elif efficiency < 80:
        print("⚠️ Inefficient for small tensors")
    else:
        print("✅ Reasonable efficiency")

    print(f"\nConclusion: The overhead is NOT from memory transfer")
    print(f"(Apple Silicon unified memory), but from:")
    print(f"1. Metal kernel launch overhead")
    print(f"2. C++/Swift FFI overhead")
    print(f"3. Metal command buffer setup")
    print(f"4. Small tensor size not amortizing fixed costs")


def test_memory_behavior():
    """Test memory allocation behavior"""
    print(f"\n{'='*50}")
    print("Memory Allocation Analysis")

    # Test if tensors are actually copied
    q = torch.randn(64, 16, dtype=torch.float32)
    q_orig_data_ptr = q.data_ptr()

    # This is what happens inside the Metal backend
    q_cpu = q.to(torch.kCPU).contiguous()
    q_cpu_data_ptr = q_cpu.data_ptr()

    print(f"Original tensor data_ptr: {hex(q_orig_data_ptr)}")
    print(f"CPU/contiguous data_ptr:  {hex(q_cpu_data_ptr)}")

    if q_orig_data_ptr == q_cpu_data_ptr:
        print("✅ No memory copy (tensor already CPU contiguous)")
    else:
        print("⚠️ Memory copy occurred (tensor was not CPU contiguous)")

    # Check contiguity
    print(f"Original is_contiguous: {q.is_contiguous()}")
    print(f"Original device: {q.device}")


def main():
    analyze_overhead()
    test_memory_behavior()


if __name__ == "__main__":
    main()
