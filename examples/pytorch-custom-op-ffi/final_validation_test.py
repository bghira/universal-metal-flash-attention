#!/usr/bin/env python3
"""
Final validation test confirming the scale factor fix.
Author: bghira
"""

import torch
import torch.nn.functional as F
import metal_sdpa_extension
import numpy as np


def test_scale_factor_fix():
    """Comprehensive test that validates the scale factor fix"""
    print("Final Scale Factor Fix Validation")
    print("Author: bghira")
    print("=" * 70)

    if not metal_sdpa_extension.is_metal_available():
        print("❌ Metal not available")
        return False

    all_passed = True
    test_configs = [
        (4, 4, [0.25, 0.5, 0.75]),  # Small tensors, various scales
        (8, 8, [0.1, 0.35355, 1.0]),  # Medium tensors, including 1/√8
        (16, 16, [0.25, 0.5]),  # Larger tensors
    ]

    for seq_len, head_dim, scales in test_configs:
        print(f"\n--- Testing {seq_len}x{head_dim} tensors ---")

        # Use same random seed for reproducibility
        torch.manual_seed(42)
        q = torch.randn(seq_len, head_dim, dtype=torch.float32)
        k = torch.randn(seq_len, head_dim, dtype=torch.float32)
        v = torch.randn(seq_len, head_dim, dtype=torch.float32)

        for scale in scales:
            try:
                # Metal implementation
                metal_out = metal_sdpa_extension.metal_scaled_dot_product_attention(
                    q, k, v, scale=scale, is_causal=False
                )

                # PyTorch reference
                torch_out = F.scaled_dot_product_attention(
                    q, k, v, scale=scale, is_causal=False
                )

                # Check for NaN/Inf
                if torch.isnan(metal_out).any() or torch.isinf(metal_out).any():
                    print(f"  ❌ Scale {scale:.3f}: Metal output has NaN/Inf")
                    all_passed = False
                    continue

                # Calculate difference
                diff = torch.abs(metal_out - torch_out).max().item()

                if diff < 1e-5:
                    print(f"  ✅ Scale {scale:.3f}: Max diff {diff:.2e}")
                else:
                    print(f"  ❌ Scale {scale:.3f}: Max diff {diff:.2e} (too large)")
                    all_passed = False

            except Exception as e:
                print(f"  ❌ Scale {scale:.3f}: Error - {e}")
                all_passed = False

    return all_passed


def test_backward_compatibility():
    """Test that existing code still works without specifying scale"""
    print(f"\n{'='*70}")
    print("Backward Compatibility Test")

    try:
        q = torch.randn(8, 16, dtype=torch.float32)
        k = torch.randn(8, 16, dtype=torch.float32)
        v = torch.randn(8, 16, dtype=torch.float32)

        # Test without explicit scale (should default to 1/√head_dim)
        metal_out = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

        # Test with explicit default scale
        expected_scale = 1.0 / np.sqrt(16)
        metal_out_explicit = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q, k, v, scale=expected_scale
        )

        diff = torch.abs(metal_out - metal_out_explicit).max().item()

        if diff < 1e-7:
            print("✅ Backward compatibility maintained")
            return True
        else:
            print(f"❌ Backward compatibility broken: diff {diff:.2e}")
            return False

    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        return False


def main():
    scale_test_passed = test_scale_factor_fix()
    compat_test_passed = test_backward_compatibility()

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")

    if scale_test_passed and compat_test_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Scale factor issue has been successfully fixed")
        print("✅ PyTorch integration now supports arbitrary scale factors")
        print("✅ Backward compatibility is maintained")
        print("✅ MFA core library now accepts custom scale factors")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
