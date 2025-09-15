#!/usr/bin/env python3
"""
Debug script testing the original case that produced NaNs.
Author: bghira
"""

import metal_sdpa_extension
import torch


def test_original_problematic_case():
    """Test with the exact parameters that were producing NaNs before"""
    print("=== Testing Original Case (32x16) ===")

    # This was the exact case from the original test
    seq_len, head_dim = 32, 16
    q = torch.randn(seq_len, head_dim, dtype=torch.float32)
    k = torch.randn(seq_len, head_dim, dtype=torch.float32)
    v = torch.randn(seq_len, head_dim, dtype=torch.float32)

    print(f"Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")
    print(f"Input dtypes: Q={q.dtype}, K={k.dtype}, V={v.dtype}")

    try:
        output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")

        has_nan = torch.isnan(output).any()
        has_inf = torch.isinf(output).any()

        print(f"Has NaN: {has_nan}")
        print(f"Has Inf: {has_inf}")

        if not has_nan and not has_inf:
            print("✅ Output is valid!")
            print(f"Output range: [{output.min():.6f}, {output.max():.6f}]")
            print(f"Output mean: {output.mean():.6f}")
            print(f"Output std: {output.std():.6f}")

            # Compare with PyTorch reference
            ref_output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            diff = torch.abs(output - ref_output).max()
            print(f"Max difference from PyTorch reference: {diff:.6f}")

            if diff < 1e-4:
                print("✅ Results match PyTorch reference within tolerance")
            else:
                print("⚠️ Results differ from PyTorch reference")
        else:
            print("❌ Output contains NaN or Inf")

    except Exception as e:
        print(f"❌ Error during computation: {e}")


def test_4d_tensors():
    """Test with 4D tensors (batch dimension)"""
    print("\n=== Testing 4D Tensors ===")

    batch_size, seq_len, num_heads, head_dim = 2, 16, 4, 8
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)

    print(f"Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")

    try:
        output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

        has_nan = torch.isnan(output).any()
        has_inf = torch.isinf(output).any()

        print(f"Has NaN: {has_nan}")
        print(f"Has Inf: {has_inf}")

        if not has_nan and not has_inf:
            print("✅ 4D tensors work correctly!")

            # Compare with PyTorch reference
            ref_output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            diff = torch.abs(output - ref_output).max()
            print(f"Max difference from PyTorch reference: {diff:.6f}")
        else:
            print("❌ 4D tensors produce NaN/Inf")

    except Exception as e:
        print(f"❌ Error with 4D tensors: {e}")


def test_different_precisions():
    """Test with different floating point precisions"""
    print("\n=== Testing Different Precisions ===")

    for dtype_name, dtype in [
        ("float16", torch.float16),
        ("float32", torch.float32),
        ("bfloat16", torch.bfloat16),
    ]:
        print(f"\n--- Testing {dtype_name} ---")

        q = torch.randn(8, 4, dtype=dtype)
        k = torch.randn(8, 4, dtype=dtype)
        v = torch.randn(8, 4, dtype=dtype)

        try:
            output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

            has_nan = torch.isnan(output).any()
            has_inf = torch.isinf(output).any()

            print(f"  Has NaN: {has_nan}")
            print(f"  Has Inf: {has_inf}")

            if not has_nan and not has_inf:
                print(f"  ✅ {dtype_name} works correctly!")
            else:
                print(f"  ❌ {dtype_name} produces NaN/Inf")

        except Exception as e:
            print(f"  ❌ Error with {dtype_name}: {e}")


def main():
    print("Original Case NaN Debug")
    print("Author: bghira")
    print("=" * 50)

    if not metal_sdpa_extension.is_metal_available():
        print("❌ Metal not available")
        return

    print("✅ Metal available")

    test_original_problematic_case()
    test_4d_tensors()
    test_different_precisions()


if __name__ == "__main__":
    main()
