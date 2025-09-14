#!/usr/bin/env python3
"""
Debug script for investigating NaN outputs in Metal SDPA backend.
Author: bghira
"""

import torch
import numpy as np
import metal_sdpa_extension


def print_tensor_stats(tensor, name):
    """Print detailed tensor statistics"""
    flat = tensor.flatten()
    print(f"{name} stats:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Is contiguous: {tensor.is_contiguous()}")
    print(f"  Min: {flat.min().item():.6f}")
    print(f"  Max: {flat.max().item():.6f}")
    print(f"  Mean: {flat.mean().item():.6f}")
    print(f"  Std: {flat.std().item():.6f}")
    print(f"  Has NaN: {torch.isnan(flat).any().item()}")
    print(f"  Has Inf: {torch.isinf(flat).any().item()}")
    print(f"  Sample values: {flat[:5].tolist()}")
    print()


def test_simple_case():
    """Test with the simplest possible case"""
    print("=== Testing Simple Case (2x2) ===")

    # Very simple 2x2 matrices
    q = torch.randn(2, 2, dtype=torch.float32)
    k = torch.randn(2, 2, dtype=torch.float32)
    v = torch.randn(2, 2, dtype=torch.float32)

    # Normalize to reasonable values
    q = q / np.sqrt(2)  # scale by sqrt(head_dim)
    k = k / np.sqrt(2)
    v = v / np.sqrt(2)

    print_tensor_stats(q, "Query")
    print_tensor_stats(k, "Key")
    print_tensor_stats(v, "Value")

    print("Calling Metal SDPA...")
    try:
        output = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,  # Let it compute default scale
            enable_gqa=False,
        )
        print_tensor_stats(output, "Output")

        # Compare with PyTorch reference
        print("Computing PyTorch reference...")
        ref_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        print_tensor_stats(ref_output, "Reference")

        if torch.isnan(output).any():
            print("❌ Output contains NaN!")
        else:
            print("✅ Output is valid")

    except Exception as e:
        print(f"Error: {e}")


def test_tensor_layout():
    """Test different tensor layouts and contiguity"""
    print("=== Testing Tensor Layout ===")

    # Test with different layouts
    for layout_name, q, k, v in [
        ("Contiguous", torch.randn(4, 8), torch.randn(4, 8), torch.randn(4, 8)),
        ("Transposed", torch.randn(8, 4).T, torch.randn(8, 4).T, torch.randn(8, 4).T),
    ]:
        print(f"\n--- {layout_name} Layout ---")
        print(f"Q contiguous: {q.is_contiguous()}")
        print(f"K contiguous: {k.is_contiguous()}")
        print(f"V contiguous: {v.is_contiguous()}")

        try:
            output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
            print(f"Result: {'NaN' if torch.isnan(output).any() else 'Valid'}")
        except Exception as e:
            print(f"Error: {e}")


def test_with_known_values():
    """Test with carefully constructed known values"""
    print("=== Testing with Known Values ===")

    # Identity-like matrices that should produce predictable results
    q = torch.eye(4, dtype=torch.float32)
    k = torch.eye(4, dtype=torch.float32)
    v = torch.ones(4, 4, dtype=torch.float32)

    print_tensor_stats(q, "Identity Query")
    print_tensor_stats(k, "Identity Key")
    print_tensor_stats(v, "Ones Value")

    try:
        output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
        print_tensor_stats(output, "Output")

        # With identity Q,K and ones V, we expect uniform attention weights
        # leading to outputs that are sums of V rows
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        print_tensor_stats(expected, "Expected")

    except Exception as e:
        print(f"Error: {e}")


def main():
    print("Metal Flash Attention NaN Debug")
    print("Author: bghira")
    print("=" * 50)

    # Check basic availability
    if not metal_sdpa_extension.is_metal_available():
        print("❌ Metal not available")
        return

    print("✅ Metal available")
    version = metal_sdpa_extension.get_version()
    print(f"MFA Version: {version}")
    print()

    test_simple_case()
    test_tensor_layout()
    test_with_known_values()


if __name__ == "__main__":
    main()
