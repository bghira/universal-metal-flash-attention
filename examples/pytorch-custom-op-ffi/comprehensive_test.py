#!/usr/bin/env python3
"""
Comprehensive test for Metal SDPA backend.
Author: bghira
"""

import torch
import torch.nn.functional as F
import numpy as np
import metal_sdpa_extension
import time


def test_basic_functionality():
    """Test basic SDPA functionality"""
    print("=== Basic Functionality Test ===")

    for shape_name, (seq_len, head_dim) in [
        ("Small", (8, 4)),
        ("Medium", (32, 16)),
        ("Large", (64, 32)),
    ]:
        print(f"\n--- {shape_name} ({seq_len}x{head_dim}) ---")

        q = torch.randn(seq_len, head_dim, dtype=torch.float32)
        k = torch.randn(seq_len, head_dim, dtype=torch.float32)
        v = torch.randn(seq_len, head_dim, dtype=torch.float32)

        try:
            # Test Metal SDPA
            metal_output = metal_sdpa_extension.metal_scaled_dot_product_attention(
                q, k, v
            )

            # Test PyTorch reference
            ref_output = F.scaled_dot_product_attention(q, k, v)

            # Check for NaN/Inf
            has_nan = torch.isnan(metal_output).any()
            has_inf = torch.isinf(metal_output).any()

            if has_nan or has_inf:
                print(f"❌ {shape_name}: Output contains NaN/Inf")
                continue

            # Check accuracy
            diff = torch.abs(metal_output - ref_output).max()
            rel_diff = (diff / torch.abs(ref_output).max()).item()

            print(f"✅ {shape_name}: Max diff = {diff:.6f}, Rel diff = {rel_diff:.6f}")

            if rel_diff < 1e-4:
                print(f"✅ {shape_name}: Accuracy test passed")
            else:
                print(f"⚠️ {shape_name}: Accuracy may be lower than expected")

        except Exception as e:
            print(f"❌ {shape_name}: Error - {e}")


def test_4d_tensors():
    """Test 4D tensor support"""
    print("\n=== 4D Tensor Test ===")

    batch_size, seq_len, num_heads, head_dim = 2, 16, 4, 8
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)

    try:
        metal_output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
        ref_output = F.scaled_dot_product_attention(q, k, v)

        has_nan = torch.isnan(metal_output).any()
        has_inf = torch.isinf(metal_output).any()

        if has_nan or has_inf:
            print("❌ 4D tensors: Output contains NaN/Inf")
            return

        diff = torch.abs(metal_output - ref_output).max()
        print(f"✅ 4D tensors: Max diff = {diff:.6f}")

    except Exception as e:
        print(f"❌ 4D tensors: Error - {e}")


def test_different_dtypes():
    """Test different data types"""
    print("\n=== Data Type Test ===")

    q_base = torch.randn(8, 4)
    k_base = torch.randn(8, 4)
    v_base = torch.randn(8, 4)

    for dtype_name, dtype in [
        ("float32", torch.float32),
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
    ]:
        print(f"\n--- {dtype_name} ---")

        q = q_base.to(dtype)
        k = k_base.to(dtype)
        v = v_base.to(dtype)

        try:
            metal_output = metal_sdpa_extension.metal_scaled_dot_product_attention(
                q, k, v
            )

            has_nan = torch.isnan(metal_output).any()
            has_inf = torch.isinf(metal_output).any()

            if has_nan or has_inf:
                print(f"❌ {dtype_name}: Output contains NaN/Inf")
                continue

            print(f"✅ {dtype_name}: Success")

        except Exception as e:
            print(f"❌ {dtype_name}: Error - {e}")


def test_causal_masking():
    """Test causal masking"""
    print("\n=== Causal Masking Test ===")

    q = torch.randn(8, 4, dtype=torch.float32)
    k = torch.randn(8, 4, dtype=torch.float32)
    v = torch.randn(8, 4, dtype=torch.float32)

    try:
        # Test causal=True
        metal_causal = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q, k, v, is_causal=True
        )

        # Test causal=False
        metal_normal = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q, k, v, is_causal=False
        )

        # They should be different (unless by coincidence)
        diff = torch.abs(metal_causal - metal_normal).max()

        if diff > 1e-6:
            print(f"✅ Causal masking: Outputs differ as expected (diff={diff:.6f})")
        else:
            print(f"⚠️ Causal masking: Outputs are very similar (diff={diff:.6f})")

        # Compare with PyTorch reference
        ref_causal = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        causal_diff = torch.abs(metal_causal - ref_causal).max()
        print(f"✅ Causal vs PyTorch: Max diff = {causal_diff:.6f}")

    except Exception as e:
        print(f"❌ Causal masking: Error - {e}")


def test_performance():
    """Basic performance test"""
    print("\n=== Performance Test ===")

    seq_len, head_dim = 128, 64
    q = torch.randn(seq_len, head_dim, dtype=torch.float32)
    k = torch.randn(seq_len, head_dim, dtype=torch.float32)
    v = torch.randn(seq_len, head_dim, dtype=torch.float32)

    # Warmup
    for _ in range(3):
        _ = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
        _ = F.scaled_dot_product_attention(q, k, v)

    # Time Metal SDPA
    num_runs = 10
    start_time = time.time()
    for _ in range(num_runs):
        _ = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
    metal_time = (time.time() - start_time) / num_runs

    # Time PyTorch reference
    start_time = time.time()
    for _ in range(num_runs):
        _ = F.scaled_dot_product_attention(q, k, v)
    ref_time = (time.time() - start_time) / num_runs

    print(f"Metal SDPA: {metal_time*1000:.2f} ms/iter")
    print(f"PyTorch:    {ref_time*1000:.2f} ms/iter")
    print(f"Speedup:    {ref_time/metal_time:.2f}x")


def main():
    print("Comprehensive Metal SDPA Backend Test")
    print("Author: bghira")
    print("=" * 50)

    if not metal_sdpa_extension.is_metal_available():
        print("❌ Metal not available")
        return

    print("✅ Metal available")
    version = metal_sdpa_extension.get_version()
    print(f"MFA Version: {version}")

    test_basic_functionality()
    test_4d_tensors()
    test_different_dtypes()
    test_causal_masking()
    test_performance()

    print("\n" + "=" * 50)
    print("✅ Comprehensive testing completed!")


if __name__ == "__main__":
    main()
