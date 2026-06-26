#!/usr/bin/env python3
"""
Minimal Reproducible Examples (MREs) for BF16 precision issues in UMFA.

This test file isolates the exact point where bf16 handling fails by testing:
1. Basic bf16 tensor pass-through
2. Simple attention computation with bf16 vs fp32
3. BF16 dtype preservation through FFI boundary
4. Numerical differences between bf16 and fp32 outputs

Each test provides clear PASS/FAIL status to help identify where bf16 processing breaks down.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add the python package to path
sys.path.insert(0, str(Path(__file__).parent / "python"))

try:
    import metal_sdpa_extension

    METAL_AVAILABLE = True
    print("✓ metal_sdpa_extension imported successfully")
except ImportError as e:
    METAL_AVAILABLE = False
    print(f"✗ metal_sdpa_extension import failed: {e}")


def print_test_header(test_name):
    """Print formatted test header."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print("=" * 60)


def print_test_result(passed, message=""):
    """Print test result with clear PASS/FAIL status."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status}: {message}")
    return passed


def create_test_tensors(
    batch_size=1, num_heads=1, seq_len=8, head_dim=64, dtype=torch.float32
):
    """Create small test tensors for attention computation."""
    # Use small values to avoid overflow in bf16
    scale = 0.1

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype) * scale
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype) * scale
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype) * scale

    return q, k, v


def test_1_bf16_tensor_passthrough():
    """
    Test 1: Basic bf16 tensor pass-through test (just pass bf16 data through FFI and back)

    This test verifies the most basic bf16 handling - can we create bf16 tensors
    and pass them through the Metal SDPA function without crashing?
    """
    print_test_header("Test 1: BF16 Tensor Pass-through")

    if not METAL_AVAILABLE:
        return print_test_result(False, "Metal extension not available")

    if not hasattr(torch, "bfloat16"):
        return print_test_result(False, "BFloat16 not available in this PyTorch build")

    try:
        # Create minimal bf16 tensors
        q, k, v = create_test_tensors(dtype=torch.bfloat16)

        print(f"Input dtypes: q={q.dtype}, k={k.dtype}, v={v.dtype}")
        print(f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")

        # Attempt to pass through Metal SDPA
        output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

        # Check basic properties
        dtype_preserved = output.dtype == torch.bfloat16
        shape_correct = output.shape == q.shape
        no_nans = not torch.isnan(output).any()
        no_infs = not torch.isinf(output).any()

        print(f"Output dtype: {output.dtype} (preserved: {dtype_preserved})")
        print(f"Output shape: {output.shape} (correct: {shape_correct})")
        print(f"Contains NaN: {torch.isnan(output).any()}")
        print(f"Contains Inf: {torch.isinf(output).any()}")

        passed = dtype_preserved and shape_correct and no_nans and no_infs
        return print_test_result(
            passed,
            (
                "BF16 tensors passed through FFI successfully"
                if passed
                else "BF16 tensor pass-through failed"
            ),
        )

    except Exception as e:
        return print_test_result(False, f"Exception during bf16 pass-through: {e}")


def test_2_bf16_vs_fp32_attention():
    """
    Test 2: Simple attention computation with bf16 vs fp32 (small tensors, like 1x8x64x64)

    This test compares bf16 and fp32 attention computation to verify that:
    1. BF16 computation completes without errors
    2. Results are numerically reasonable compared to FP32
    """
    print_test_header("Test 2: BF16 vs FP32 Attention Computation")

    if not METAL_AVAILABLE:
        return print_test_result(False, "Metal extension not available")

    if not hasattr(torch, "bfloat16"):
        return print_test_result(False, "BFloat16 not available in this PyTorch build")

    try:
        # Create test data in fp32 first
        q_fp32, k_fp32, v_fp32 = create_test_tensors(
            batch_size=1, num_heads=8, seq_len=64, head_dim=64, dtype=torch.float32
        )

        # Convert to bf16
        q_bf16 = q_fp32.to(torch.bfloat16)
        k_bf16 = k_fp32.to(torch.bfloat16)
        v_bf16 = v_fp32.to(torch.bfloat16)

        print(f"FP32 input range: [{q_fp32.min():.4f}, {q_fp32.max():.4f}]")
        print(f"BF16 input range: [{q_bf16.min():.4f}, {q_bf16.max():.4f}]")

        # Compute attention in both precisions
        output_fp32 = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q_fp32, k_fp32, v_fp32
        )
        output_bf16 = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q_bf16, k_bf16, v_bf16
        )

        # Check basic properties
        fp32_valid = (
            not torch.isnan(output_fp32).any() and not torch.isinf(output_fp32).any()
        )
        bf16_valid = (
            not torch.isnan(output_bf16).any() and not torch.isinf(output_bf16).any()
        )

        print(f"FP32 output valid: {fp32_valid}")
        print(f"BF16 output valid: {bf16_valid}")
        print(f"FP32 output range: [{output_fp32.min():.4f}, {output_fp32.max():.4f}]")
        print(f"BF16 output range: [{output_bf16.min():.4f}, {output_bf16.max():.4f}]")

        # Compare numerical results (convert bf16 to fp32 for comparison)
        output_bf16_as_fp32 = output_bf16.to(torch.float32)
        abs_diff = torch.abs(output_fp32 - output_bf16_as_fp32)
        rel_diff = abs_diff / (torch.abs(output_fp32) + 1e-8)

        max_abs_diff = abs_diff.max().item()
        max_rel_diff = rel_diff.max().item()
        mean_abs_diff = abs_diff.mean().item()

        print(f"Max absolute difference: {max_abs_diff:.6f}")
        print(f"Max relative difference: {max_rel_diff:.4f}")
        print(f"Mean absolute difference: {mean_abs_diff:.6f}")

        # Reasonable thresholds for bf16 vs fp32 comparison
        abs_threshold = 1e-2  # BF16 has limited precision
        rel_threshold = 0.1  # 10% relative difference is acceptable for BF16

        diff_reasonable = max_abs_diff < abs_threshold and max_rel_diff < rel_threshold

        passed = fp32_valid and bf16_valid and diff_reasonable
        return print_test_result(
            passed,
            (
                "BF16 vs FP32 comparison passed"
                if passed
                else f"BF16 vs FP32 comparison failed (abs_diff={max_abs_diff:.6f}, rel_diff={max_rel_diff:.4f})"
            ),
        )

    except Exception as e:
        return print_test_result(
            False, f"Exception during bf16 vs fp32 comparison: {e}"
        )


def test_3_bf16_dtype_preservation():
    """
    Test 3: Check if bf16 tensors maintain their dtype through the FFI boundary

    This test specifically validates that:
    1. Input bf16 tensors remain bf16 after moving through FFI
    2. Output tensors have the expected bf16 dtype
    3. No unexpected dtype conversions occur
    """
    print_test_header("Test 3: BF16 Dtype Preservation Through FFI")

    if not METAL_AVAILABLE:
        return print_test_result(False, "Metal extension not available")

    if not hasattr(torch, "bfloat16"):
        return print_test_result(False, "BFloat16 not available in this PyTorch build")

    try:
        # Create bf16 tensors
        q, k, v = create_test_tensors(dtype=torch.bfloat16)

        # Verify input dtypes
        input_dtypes_correct = (
            q.dtype == torch.bfloat16
            and k.dtype == torch.bfloat16
            and v.dtype == torch.bfloat16
        )

        print(f"Input dtypes correct: {input_dtypes_correct}")
        print(f"Q dtype: {q.dtype}")
        print(f"K dtype: {k.dtype}")
        print(f"V dtype: {v.dtype}")

        if not input_dtypes_correct:
            return print_test_result(
                False, "Input tensors do not have correct bf16 dtype"
            )

        # Pass through Metal SDPA
        output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

        # Check output dtype
        output_dtype_correct = output.dtype == torch.bfloat16
        print(f"Output dtype: {output.dtype}")
        print(f"Output dtype correct: {output_dtype_correct}")

        # Check that data values are preserved (not converted to different precision)
        # We do this by checking that the values still have bf16-like precision
        # BF16 has 7 bits of mantissa, so we expect some precision loss compared to fp32

        # Convert back to fp32 and check that values are in reasonable range
        output_fp32 = output.to(torch.float32)
        values_reasonable = (
            torch.isfinite(output_fp32).all() and output_fp32.abs().max() < 100.0
        )  # Reasonable range

        print(f"Output values reasonable: {values_reasonable}")
        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

        passed = input_dtypes_correct and output_dtype_correct and values_reasonable
        return print_test_result(
            passed,
            (
                "BF16 dtype preserved through FFI"
                if passed
                else "BF16 dtype preservation failed"
            ),
        )

    except Exception as e:
        return print_test_result(
            False, f"Exception during dtype preservation test: {e}"
        )


def test_4_bf16_fp32_numerical_differences():
    """
    Test 4: Compare numerical differences between bf16 and fp32 outputs

    This test provides detailed numerical analysis of bf16 vs fp32 differences to:
    1. Quantify the precision loss when using bf16
    2. Identify any systematic errors or bias in bf16 computation
    3. Validate that differences are within expected ranges for bf16 precision
    """
    print_test_header("Test 4: BF16 vs FP32 Numerical Analysis")

    if not METAL_AVAILABLE:
        return print_test_result(False, "Metal extension not available")

    if not hasattr(torch, "bfloat16"):
        return print_test_result(False, "BFloat16 not available in this PyTorch build")

    try:
        # Test with multiple tensor sizes and configurations
        test_configs = [
            (1, 1, 32, 32),  # Small test
            (1, 4, 64, 64),  # Medium test
            (2, 8, 128, 64),  # Larger test
        ]

        all_passed = True

        for batch_size, num_heads, seq_len, head_dim in test_configs:
            print(
                f"\nTesting config: batch={batch_size}, heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}"
            )

            # Create test tensors in fp32
            q_fp32, k_fp32, v_fp32 = create_test_tensors(
                batch_size, num_heads, seq_len, head_dim, torch.float32
            )

            # Convert to bf16
            q_bf16 = q_fp32.to(torch.bfloat16)
            k_bf16 = k_fp32.to(torch.bfloat16)
            v_bf16 = v_fp32.to(torch.bfloat16)

            # Compute attention in both precisions
            output_fp32 = metal_sdpa_extension.metal_scaled_dot_product_attention(
                q_fp32, k_fp32, v_fp32
            )
            output_bf16 = metal_sdpa_extension.metal_scaled_dot_product_attention(
                q_bf16, k_bf16, v_bf16
            )

            # Convert bf16 output to fp32 for numerical comparison
            output_bf16_as_fp32 = output_bf16.to(torch.float32)

            # Compute various error metrics
            abs_error = torch.abs(output_fp32 - output_bf16_as_fp32)
            rel_error = abs_error / (torch.abs(output_fp32) + 1e-8)

            # Statistics
            max_abs_error = abs_error.max().item()
            mean_abs_error = abs_error.mean().item()
            std_abs_error = abs_error.std().item()

            max_rel_error = rel_error.max().item()
            mean_rel_error = rel_error.mean().item()

            # Check for systematic bias
            signed_error = output_fp32 - output_bf16_as_fp32
            mean_bias = signed_error.mean().item()

            print(f"  Max absolute error: {max_abs_error:.6f}")
            print(f"  Mean absolute error: {mean_abs_error:.6f}")
            print(f"  Std absolute error: {std_abs_error:.6f}")
            print(f"  Max relative error: {max_rel_error:.4f}")
            print(f"  Mean relative error: {mean_rel_error:.4f}")
            print(f"  Mean bias: {mean_bias:.6f}")

            # Validate error is within acceptable ranges for bf16
            # BF16 has ~3 decimal digits of precision, so we expect some error
            abs_error_ok = max_abs_error < 1e-2  # 0.01 absolute error
            rel_error_ok = max_rel_error < 0.2  # 20% relative error (generous for bf16)
            bias_ok = abs(mean_bias) < 1e-3  # No systematic bias

            config_passed = abs_error_ok and rel_error_ok and bias_ok
            all_passed = all_passed and config_passed

            status = "✓" if config_passed else "✗"
            print(
                f"  {status} Config result: abs_err_ok={abs_error_ok}, rel_err_ok={rel_error_ok}, bias_ok={bias_ok}"
            )

        return print_test_result(
            all_passed,
            (
                "Numerical differences within acceptable ranges"
                if all_passed
                else "Numerical differences exceed acceptable thresholds"
            ),
        )

    except Exception as e:
        return print_test_result(False, f"Exception during numerical analysis: {e}")


def main():
    """Run all bf16 precision isolation tests."""
    print("BF16 Precision Isolation Test Suite")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"BFloat16 available: {hasattr(torch, 'bfloat16')}")
    print(f"Metal extension available: {METAL_AVAILABLE}")

    if not METAL_AVAILABLE:
        print("\n⚠️  Metal extension not available - tests will be skipped")
        print("   Make sure the extension is built and available in PYTHONPATH")
        return

    # Run all tests
    test_results = []

    test_results.append(test_1_bf16_tensor_passthrough())
    test_results.append(test_2_bf16_vs_fp32_attention())
    test_results.append(test_3_bf16_dtype_preservation())
    test_results.append(test_4_bf16_fp32_numerical_differences())

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print("=" * 60)

    passed_count = sum(test_results)
    total_count = len(test_results)

    print(f"Tests passed: {passed_count}/{total_count}")

    if passed_count == total_count:
        print("✓ ALL TESTS PASSED - BF16 handling appears to be working correctly")
    else:
        print("✗ SOME TESTS FAILED - BF16 handling has issues that need investigation")
        print("\nRecommendations:")
        print("1. Check the failed tests above for specific error messages")
        print("2. Run test_bf16_ffi_conversion.py for deeper FFI boundary analysis")
        print("3. Examine the Metal kernel implementations for bf16 support")
        print(
            "4. Check if bf16 tensors are being inadvertently converted to other types"
        )


if __name__ == "__main__":
    main()
