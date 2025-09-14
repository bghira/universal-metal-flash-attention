#!/usr/bin/env python3
"""
Test demonstrating the scale factor issue and fix.
Author: bghira
"""

import torch
import torch.nn.functional as F
import metal_sdpa_extension
import numpy as np


def test_scale_factor_limitation():
    """Test that demonstrates the scale factor limitation and validates the fix"""
    print("Scale Factor Limitation Test")
    print("Author: bghira")
    print("=" * 70)

    if not metal_sdpa_extension.is_metal_available():
        print("❌ Metal not available")
        return

    # Use 8x8 tensors for clear scale factor demonstration
    seq_len, head_dim = 8, 8
    expected_scale = 1.0 / np.sqrt(head_dim)

    torch.manual_seed(42)
    q = torch.randn(seq_len, head_dim, dtype=torch.float32)
    k = torch.randn(seq_len, head_dim, dtype=torch.float32)
    v = torch.randn(seq_len, head_dim, dtype=torch.float32)

    print(f"Test configuration:")
    print(f"  Tensor shape: {q.shape}")
    print(f"  Expected scale (1/√{head_dim}): {expected_scale:.6f}")
    print()

    # Test different scale factors
    test_scales = [
        ("Correct scale", expected_scale),
        ("Wrong scale (0.5)", 0.5),
        ("Wrong scale (0.25)", 0.25),
        ("Wrong scale (1.0)", 1.0),
    ]

    results = []

    for name, scale in test_scales:
        print(f"--- {name}: {scale:.6f} ---")

        try:
            # Metal implementation (with validation warnings)
            metal_output = metal_sdpa_extension.metal_scaled_dot_product_attention(
                q, k, v, scale=scale, is_causal=False
            )

            # PyTorch reference
            torch_output = F.scaled_dot_product_attention(
                q, k, v, scale=scale, is_causal=False
            )

            # Calculate difference
            diff = torch.abs(metal_output - torch_output).max().item()

            print(
                f"  Metal output range: [{metal_output.min().item():.3f}, {metal_output.max().item():.3f}]"
            )
            print(
                f"  PyTorch output range: [{torch_output.min().item():.3f}, {torch_output.max().item():.3f}]"
            )
            print(f"  Max absolute difference: {diff:.2e}")

            if diff < 1e-5:
                print(
                    f"  ✅ Outputs match (Metal correctly using {expected_scale:.6f})"
                )
                status = "MATCH"
            else:
                print(f"  ❌ Outputs differ (Metal ignoring scale parameter)")
                status = "DIFFER"

            results.append((name, scale, diff, status))

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append((name, scale, float("inf"), "ERROR"))

        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("The MFA library has a hardcoded scale factor and ignores the parameter.")
    print(
        "This is why PyTorch integration only works correctly when using 1/√head_dim."
    )
    print()
    print("Results:")
    for name, scale, diff, status in results:
        print(f"  {name:20s} (scale={scale:.3f}): {status}")

    print()
    print("✅ Fix implemented: Added validation warnings to identify scale mismatches")
    print("📋 TODO: Modify MFA core library to accept custom scale factors")


def test_documentation_example():
    """Test the recommended usage pattern"""
    print(f"\n{'='*70}")
    print("RECOMMENDED USAGE")
    print("=" * 70)

    # This is how users should call the function to avoid issues
    head_dim = 16
    q = torch.randn(32, head_dim, dtype=torch.float32)
    k = torch.randn(32, head_dim, dtype=torch.float32)
    v = torch.randn(32, head_dim, dtype=torch.float32)

    # Always use the default scale or explicitly pass 1/√head_dim
    recommended_scale = 1.0 / np.sqrt(head_dim)

    print(f"Recommended usage for head_dim={head_dim}:")
    print(f"  scale = 1.0 / np.sqrt({head_dim}) = {recommended_scale:.6f}")
    print()

    try:
        # This will work correctly
        output = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q, k, v, scale=recommended_scale, is_causal=False
        )
        print(f"✅ Success with recommended scale")
        print(f"   Output shape: {output.shape}")
        print(
            f"   Output range: [{output.min().item():.3f}, {output.max().item():.3f}]"
        )

        # Compare with PyTorch for verification
        torch_output = F.scaled_dot_product_attention(
            q, k, v, scale=recommended_scale, is_causal=False
        )
        diff = torch.abs(output - torch_output).max().item()
        print(f"   Difference from PyTorch: {diff:.2e}")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    test_scale_factor_limitation()
    test_documentation_example()
