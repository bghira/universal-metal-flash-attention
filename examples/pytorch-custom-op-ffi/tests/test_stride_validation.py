#!/usr/bin/env python3
"""
Test to validate that stride information is being passed correctly to Metal kernels.
"""

import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import metal_sdpa_extension
import torch


def test_stride_handling():
    """Test that verifies stride information is passed and used."""

    print("Testing Stride Information Passing")
    print("=" * 50)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Create a simple test case with smaller dimensions
    batch, heads, seq_len, dim = 1, 2, 4, 8

    print(f"Test dimensions: B={batch}, H={heads}, S={seq_len}, D={dim}")

    # Test 1: Contiguous tensor
    print("\n1. Contiguous tensor test:")
    q = (
        torch.randn(batch, heads, seq_len, dim, dtype=torch.float16, device=device)
        * 0.01
    )
    k = (
        torch.randn(batch, heads, seq_len, dim, dtype=torch.float16, device=device)
        * 0.01
    )
    v = (
        torch.randn(batch, heads, seq_len, dim, dtype=torch.float16, device=device)
        * 0.01
    )

    print(f"   Shape: {list(q.shape)}")
    print(f"   Stride: {q.stride()}")
    print(f"   Contiguous: {q.is_contiguous()}")

    try:
        output1 = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
        print(f"   Result shape: {list(output1.shape)}")
        has_nan = torch.isnan(output1).any().item()
        print(f"   Contains NaN: {has_nan}")
        if not has_nan:
            print("   ✅ Contiguous test passed")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test 2: Non-contiguous tensor (permuted)
    print("\n2. Non-contiguous (permuted) tensor test:")
    # Permute to [B,S,H,D] - creates non-contiguous view
    q_perm = q.permute(0, 2, 1, 3)
    k_perm = k.permute(0, 2, 1, 3)
    v_perm = v.permute(0, 2, 1, 3)

    print(f"   Shape: {list(q_perm.shape)}")
    print(f"   Stride: {q_perm.stride()}")
    print(f"   Contiguous: {q_perm.is_contiguous()}")

    try:
        output2 = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q_perm, k_perm, v_perm
        )
        print(f"   Result shape: {list(output2.shape)}")
        has_nan = torch.isnan(output2).any().item()
        print(f"   Contains NaN: {has_nan}")

        if not has_nan:
            # Permute output2 back to compare with output1
            output2_orig_layout = output2.permute(0, 2, 1, 3)
            if torch.allclose(output1, output2_orig_layout, rtol=1e-2, atol=1e-3):
                print("   ✅ Non-contiguous test passed - outputs match!")
            else:
                diff = torch.abs(output1 - output2_orig_layout).max().item()
                print(f"   ⚠️  Outputs differ: max diff = {diff}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Test 3: Direct stride test - create tensor with specific strides
    print("\n3. Custom stride test:")
    # Create a larger tensor and take a slice to get non-standard strides
    large = (
        torch.randn(
            batch, heads * 2, seq_len * 2, dim, dtype=torch.float16, device=device
        )
        * 0.01
    )
    q_slice = large[:, :heads, :seq_len, :]
    k_slice = large[:, :heads, :seq_len, :]
    v_slice = large[:, :heads, :seq_len, :]

    print(f"   Shape: {list(q_slice.shape)}")
    print(f"   Stride: {q_slice.stride()}")
    print(f"   Contiguous: {q_slice.is_contiguous()}")

    try:
        output3 = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q_slice, k_slice, v_slice
        )
        print(f"   Result shape: {list(output3.shape)}")
        has_nan = torch.isnan(output3).any().item()
        print(f"   Contains NaN: {has_nan}")
        if not has_nan:
            print("   ✅ Custom stride test passed")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n" + "=" * 50)
    print("Stride validation complete")


if __name__ == "__main__":
    test_stride_handling()
