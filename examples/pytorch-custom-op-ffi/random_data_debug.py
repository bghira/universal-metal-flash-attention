#!/usr/bin/env python3
"""
Debug random data handling to find the root cause of correctness issues.
Author: bghira
"""

import metal_sdpa_extension
import numpy as np
import torch
import torch.nn.functional as F


def test_random_data_sizes():
    """Test with random data at different sizes to isolate the issue"""
    print("Random Data Debug")
    print("Author: bghira")
    print("=" * 70)

    if not metal_sdpa_extension.is_metal_available():
        print("❌ Metal not available")
        return

    # Test different sizes
    test_configs = [
        (4, 4),  # Same as Swift test
        (8, 8),  # Slightly larger
        (16, 16),  # Medium size
        (32, 32),  # Larger
    ]

    for seq_len, head_dim in test_configs:
        print(f"\n{'='*50}")
        print(f"Testing size: seq_len={seq_len}, head_dim={head_dim}")
        print(f"{'='*50}")

        # Generate random data
        torch.manual_seed(42)  # For reproducibility
        q = torch.randn(seq_len, head_dim, dtype=torch.float32)
        k = torch.randn(seq_len, head_dim, dtype=torch.float32)
        v = torch.randn(seq_len, head_dim, dtype=torch.float32)

        scale = 1.0 / np.sqrt(head_dim)

        print(f"Input tensors:")
        print(f"  Q range: [{q.min().item():.3f}, {q.max().item():.3f}]")
        print(f"  K range: [{k.min().item():.3f}, {k.max().item():.3f}]")
        print(f"  V range: [{v.min().item():.3f}, {v.max().item():.3f}]")
        print(f"  Scale: {scale:.6f}")

        try:
            # Metal implementation
            metal_output = metal_sdpa_extension.metal_scaled_dot_product_attention(
                q, k, v, scale=scale, is_causal=False
            )

            # PyTorch reference
            torch_output = F.scaled_dot_product_attention(
                q, k, v, scale=scale, is_causal=False
            )

            # Check for NaN/Inf
            metal_has_nan = torch.isnan(metal_output).any()
            metal_has_inf = torch.isinf(metal_output).any()
            torch_has_nan = torch.isnan(torch_output).any()
            torch_has_inf = torch.isinf(torch_output).any()

            print(f"Results:")
            print(f"  Metal - NaN: {metal_has_nan}, Inf: {metal_has_inf}")
            print(f"  PyTorch - NaN: {torch_has_nan}, Inf: {torch_has_inf}")

            if not (metal_has_nan or metal_has_inf or torch_has_nan or torch_has_inf):
                # Calculate difference
                diff = torch.abs(metal_output - torch_output).max().item()
                relative_diff = diff / torch.abs(torch_output).max().item()

                print(f"  Max absolute diff: {diff:.2e}")
                print(f"  Max relative diff: {relative_diff:.2e}")

                # Compare statistics
                metal_mean = metal_output.mean().item()
                torch_mean = torch_output.mean().item()
                metal_std = metal_output.std().item()
                torch_std = torch_output.std().item()

                print(f"  Metal mean: {metal_mean:.6f}, std: {metal_std:.6f}")
                print(f"  PyTorch mean: {torch_mean:.6f}, std: {torch_std:.6f}")

                if diff < 1e-5:
                    print("  ✅ Close match")
                elif diff < 1e-3:
                    print("  ⚠️ Acceptable difference")
                else:
                    print("  ❌ Large difference")

                    # Print some actual values for debugging
                    print(f"  Metal output (first 4x4):")
                    print(f"  {metal_output[:4, :4]}")
                    print(f"  PyTorch output (first 4x4):")
                    print(f"  {torch_output[:4, :4]}")
            else:
                print("  ❌ Contains NaN or Inf")

        except Exception as e:
            print(f"  ❌ Error: {e}")


def test_edge_cases():
    """Test edge cases that might cause issues"""
    print(f"\n{'='*70}")
    print("Edge Cases Debug")

    test_cases = [
        ("Very small values", torch.full((4, 4), 1e-6, dtype=torch.float32)),
        ("Very large values", torch.full((4, 4), 1e3, dtype=torch.float32)),
        ("Mixed positive/negative", torch.randn(4, 4) * 10),
        ("All zeros", torch.zeros(4, 4, dtype=torch.float32)),
        ("Identity-like", torch.eye(4, dtype=torch.float32)),
    ]

    for name, tensor in test_cases:
        print(f"\n--- {name} ---")
        print(f"Tensor range: [{tensor.min().item():.2e}, {tensor.max().item():.2e}]")

        try:
            metal_output = metal_sdpa_extension.metal_scaled_dot_product_attention(
                tensor, tensor, tensor, scale=0.5, is_causal=False
            )
            torch_output = F.scaled_dot_product_attention(
                tensor, tensor, tensor, scale=0.5, is_causal=False
            )

            metal_has_issues = (
                torch.isnan(metal_output).any() or torch.isinf(metal_output).any()
            )
            torch_has_issues = (
                torch.isnan(torch_output).any() or torch.isinf(torch_output).any()
            )

            if not (metal_has_issues or torch_has_issues):
                diff = torch.abs(metal_output - torch_output).max().item()
                print(f"✅ Max diff: {diff:.2e}")
            else:
                print(
                    f"❌ Metal issues: {metal_has_issues}, PyTorch issues: {torch_has_issues}"
                )

        except Exception as e:
            print(f"❌ Error: {e}")


def test_parameter_sensitivity():
    """Test sensitivity to different parameters"""
    print(f"\n{'='*70}")
    print("Parameter Sensitivity Debug")

    torch.manual_seed(123)
    q = torch.randn(8, 8, dtype=torch.float32)
    k = torch.randn(8, 8, dtype=torch.float32)
    v = torch.randn(8, 8, dtype=torch.float32)

    # Test different scales
    scales = [0.1, 0.25, 0.35355, 0.5, 1.0, 2.0]  # 0.35355 ≈ 1/√8

    for scale in scales:
        print(f"\n--- Scale: {scale:.5f} ---")

        try:
            metal_output = metal_sdpa_extension.metal_scaled_dot_product_attention(
                q, k, v, scale=scale, is_causal=False
            )
            torch_output = F.scaled_dot_product_attention(
                q, k, v, scale=scale, is_causal=False
            )

            diff = torch.abs(metal_output - torch_output).max().item()
            print(f"Max diff: {diff:.2e}")

            if diff > 1e-3:
                print(f"❌ Large difference at scale {scale}")
                print(
                    f"Metal range: [{metal_output.min().item():.3f}, {metal_output.max().item():.3f}]"
                )
                print(
                    f"PyTorch range: [{torch_output.min().item():.3f}, {torch_output.max().item():.3f}]"
                )

        except Exception as e:
            print(f"❌ Error with scale {scale}: {e}")


def main():
    test_random_data_sizes()
    test_edge_cases()
    test_parameter_sensitivity()


if __name__ == "__main__":
    main()
