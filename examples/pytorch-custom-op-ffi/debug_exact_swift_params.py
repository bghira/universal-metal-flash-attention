#!/usr/bin/env python3
"""
Test PyTorch integration with EXACT Swift test parameters.
Author: bghira
"""

import metal_sdpa_extension
import numpy as np
import torch
import torch.nn.functional as F


def test_exact_swift_params():
    """Test with exactly the same parameters as Swift test"""
    print("Testing PyTorch Integration with Exact Swift Test Parameters")
    print("Author: bghira")
    print("=" * 70)

    if not metal_sdpa_extension.is_metal_available():
        print("❌ Metal not available")
        return

    # EXACT same parameters as Swift test
    seq_len = 4
    head_dim = 4

    # Create tensors with ALL 1.0s (exactly like Swift test)
    q = torch.ones(seq_len, head_dim, dtype=torch.float32)
    k = torch.ones(seq_len, head_dim, dtype=torch.float32)
    v = torch.ones(seq_len, head_dim, dtype=torch.float32)

    print(f"Input tensors: All 1.0s, shape {q.shape}")
    print(f"Scale calculation: 1.0 / sqrt({head_dim}) = {1.0 / np.sqrt(head_dim)}")

    # Test with same scale as Swift
    scale = 1.0 / np.sqrt(head_dim)

    try:
        # Metal implementation with explicit scale
        metal_output = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q, k, v, scale=scale, is_causal=False
        )

        print(f"Metal output shape: {metal_output.shape}")
        print(f"Metal output (first 10): {metal_output.flatten()[:10].tolist()}")

        # Check for NaN/Inf
        has_nan = torch.isnan(metal_output).any()
        has_inf = torch.isinf(metal_output).any()

        print(f"Has NaN: {has_nan}")
        print(f"Has Inf: {has_inf}")

        # Expected result for attention with all 1.0s:
        # Q*K^T produces seq_len x seq_len matrix of all head_dim (4.0)
        # After scaling by 1/sqrt(head_dim) = 0.5, we get 2.0 everywhere
        # Softmax of all 2.0s = uniform distribution (1/seq_len each)
        # Output = uniform_weights * V = (1/seq_len) * sum(V_rows) = (1/4) * 4 * [1,1,1,1] = [1,1,1,1]

        expected_uniform = 1.0 / seq_len
        expected_output = (
            torch.ones_like(v) * expected_uniform * seq_len
        )  # Should be all 1.0s

        print(
            f"Expected output (theoretical): {expected_output.flatten()[:10].tolist()}"
        )

        # Compare with PyTorch reference
        torch_output = F.scaled_dot_product_attention(
            q, k, v, scale=scale, is_causal=False
        )
        print(f"PyTorch output (first 10): {torch_output.flatten()[:10].tolist()}")

        # Calculate differences
        metal_vs_torch = torch.abs(metal_output - torch_output).max().item()
        metal_vs_expected = torch.abs(metal_output - expected_output).max().item()

        print(f"\nDifference Analysis:")
        print(f"Metal vs PyTorch: {metal_vs_torch:.2e}")
        print(f"Metal vs Expected: {metal_vs_expected:.2e}")

        if metal_vs_torch < 1e-5:
            print("✅ Metal matches PyTorch reference")
        else:
            print("❌ Metal differs from PyTorch reference")

        if metal_vs_expected < 1e-5:
            print("✅ Metal matches theoretical expectation")
        else:
            print("⚠️ Metal differs from theoretical expectation")

        return metal_vs_torch < 1e-5

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_tensor_layout_debug():
    """Debug tensor layout issues"""
    print(f"\n{'='*70}")
    print("Tensor Layout Debug")

    seq_len, head_dim = 4, 4

    # Test different tensor layouts
    layouts = [
        ("C-contiguous", torch.ones(seq_len, head_dim, dtype=torch.float32)),
        ("Fortran-like", torch.ones(head_dim, seq_len, dtype=torch.float32).T),
        ("Strided", torch.ones(seq_len * 2, head_dim, dtype=torch.float32)[::2, :]),
    ]

    for name, tensor in layouts:
        print(f"\n--- {name} ---")
        print(f"Is contiguous: {tensor.is_contiguous()}")
        print(f"Stride: {tensor.stride()}")
        print(f"Storage offset: {tensor.storage_offset()}")

        try:
            output = metal_sdpa_extension.metal_scaled_dot_product_attention(
                tensor, tensor, tensor
            )
            print(f"Result: ✅ Success, output sum = {output.sum().item():.4f}")
        except Exception as e:
            print(f"Result: ❌ Error - {e}")


def test_precision_mapping():
    """Test precision type mapping"""
    print(f"\n{'='*70}")
    print("Precision Mapping Debug")

    dtypes = [
        (torch.float32, "FP32"),
        (torch.float16, "FP16"),
        (torch.bfloat16, "BF16"),
    ]

    for dtype, name in dtypes:
        print(f"\n--- {name} ---")

        try:
            q = torch.ones(4, 4, dtype=dtype)
            k = torch.ones(4, 4, dtype=dtype)
            v = torch.ones(4, 4, dtype=dtype)

            output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

            has_nan = torch.isnan(output).any()
            print(
                f"Result: {'❌ NaN' if has_nan else '✅ Valid'}, output sum = {output.sum().item():.4f}"
            )

        except Exception as e:
            print(f"Result: ❌ Error - {e}")


def main():
    """Main debug function"""
    success = test_exact_swift_params()
    test_tensor_layout_debug()
    test_precision_mapping()

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    if success:
        print("✅ Basic functionality works with Swift test parameters")
        print("The issue may be in random data handling or parameter differences")
    else:
        print("❌ Even Swift test parameters fail in PyTorch integration")
        print("There's a fundamental issue in the FFI layer or parameter mapping")


if __name__ == "__main__":
    main()
