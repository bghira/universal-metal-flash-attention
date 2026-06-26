#!/usr/bin/env python3
"""
Simple test to demonstrate the stride/non-contiguous tensor issue.

This is a minimal test that shows the current implementation produces
NaN values when given non-contiguous tensors from permute operations.
"""

import os
import sys

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import metal_sdpa_extension
import torch


def test_non_contiguous_issue():
    """Test that demonstrates NaN output with non-contiguous tensors."""

    print("Testing Non-Contiguous Tensor Handling")
    print("=" * 50)

    # Create FLUX layout tensors [B,H,S,D]
    batch, heads, seq_len, dim = 1, 12, 77, 64
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Configuration: B={batch}, H={heads}, S={seq_len}, D={dim}")
    print(f"Device: {device}\n")

    # Create tensors in FLUX layout
    torch.manual_seed(42)
    q_flux = (
        torch.randn(batch, heads, seq_len, dim, dtype=torch.float16, device=device)
        * 0.1
    )
    k_flux = (
        torch.randn(batch, heads, seq_len, dim, dtype=torch.float16, device=device)
        * 0.1
    )
    v_flux = (
        torch.randn(batch, heads, seq_len, dim, dtype=torch.float16, device=device)
        * 0.1
    )

    print("Test 1: Contiguous FLUX layout tensors")
    print("-" * 40)
    print(f"Shape: {list(q_flux.shape)}")
    print(f"Contiguous: {q_flux.is_contiguous()}")
    print(f"Stride: {q_flux.stride()}")

    try:
        output_flux = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q_flux, k_flux, v_flux
        )
        has_nan = torch.isnan(output_flux).any().item()
        has_inf = torch.isinf(output_flux).any().item()
        print(f"Result: NaN={has_nan}, Inf={has_inf}")
        if not has_nan and not has_inf:
            print("✅ PASS: Contiguous tensors work correctly")
        else:
            print("❌ FAIL: Contiguous tensors produce invalid output")
    except Exception as e:
        print(f"❌ FAIL: Exception - {e}")

    print("\nTest 2: Non-contiguous permuted tensors")
    print("-" * 40)

    # Permute to Metal layout [B,S,H,D] - creates non-contiguous views
    q_metal = q_flux.permute(0, 2, 1, 3)
    k_metal = k_flux.permute(0, 2, 1, 3)
    v_metal = v_flux.permute(0, 2, 1, 3)

    print(f"Shape: {list(q_metal.shape)}")
    print(f"Contiguous: {q_metal.is_contiguous()}")
    print(f"Stride: {q_metal.stride()}")
    print(f"Expected stride (if contiguous): {(seq_len*heads*dim, heads*dim, dim, 1)}")

    try:
        output_metal = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q_metal, k_metal, v_metal
        )
        has_nan = torch.isnan(output_metal).any().item()
        has_inf = torch.isinf(output_metal).any().item()
        print(f"Result: NaN={has_nan}, Inf={has_inf}")

        if has_nan:
            print("❌ FAIL: Non-contiguous tensors produce NaN values")
            print("   This confirms the stride handling issue!")
        elif has_inf:
            print("❌ FAIL: Non-contiguous tensors produce Inf values")
        else:
            print("✅ PASS: Non-contiguous tensors handled correctly")

    except Exception as e:
        print(f"❌ FAIL: Exception - {e}")
        print("   This might be a GPU memory access fault")

    print("\nTest 3: Contiguous copies of permuted tensors")
    print("-" * 40)

    # Make contiguous copies
    q_contig = q_metal.contiguous()
    k_contig = k_metal.contiguous()
    v_contig = v_metal.contiguous()

    print(f"Shape: {list(q_contig.shape)}")
    print(f"Contiguous: {q_contig.is_contiguous()}")
    print(f"Stride: {q_contig.stride()}")

    try:
        output_contig = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q_contig, k_contig, v_contig
        )
        has_nan = torch.isnan(output_contig).any().item()
        has_inf = torch.isinf(output_contig).any().item()
        print(f"Result: NaN={has_nan}, Inf={has_inf}")

        if not has_nan and not has_inf:
            print("✅ PASS: Contiguous copies work correctly")
            print("   This confirms the issue is with stride handling")
        else:
            print("❌ FAIL: Even contiguous copies fail")

    except Exception as e:
        print(f"❌ FAIL: Exception - {e}")

    print("\n" + "=" * 50)
    print("Summary:")
    print("The tests show that non-contiguous tensors (from permute)")
    print("produce NaN values or cause memory faults, confirming the")
    print("need for proper stride handling in the Metal kernels.")


if __name__ == "__main__":
    test_non_contiguous_issue()
