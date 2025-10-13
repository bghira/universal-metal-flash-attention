#!/usr/bin/env python3
"""
BF16 FFI Conversion Tests for UMFA.

This test file specifically focuses on the FFI boundary to isolate conversion issues:
1. torch_dtype_to_mfa_dtype conversion for bf16
2. ensure_contiguous_cpu function with bf16 tensors
3. Whether bf16 values are being inadvertently converted to fp32 or other types
4. Direct FFI function call testing to bypass higher-level abstractions

These tests help identify if the issue is in:
- PyTorch tensor creation/handling
- C++ FFI boundary dtype mapping
- Metal kernel dtype specification
- Data corruption during conversion
"""

import ctypes
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


def inspect_tensor_memory(tensor, label=""):
    """Inspect the actual memory contents of a tensor to detect conversions."""
    print(f"\nTensor inspection - {label}:")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Shape: {tensor.shape}")
    print(f"  Device: {tensor.device}")
    print(f"  Is contiguous: {tensor.is_contiguous()}")
    print(f"  Element size: {tensor.element_size()} bytes")
    print(f"  Storage size: {tensor.storage().size()}")

    # Print a few raw values
    flat = tensor.flatten()
    print(f"  First 5 values: {flat[:5].tolist()}")
    print(f"  Value range: [{tensor.min():.6f}, {tensor.max():.6f}]")

    # For bf16, check if values look like they've been converted
    if tensor.dtype == torch.bfloat16:
        # Convert to fp32 to see precision
        as_fp32 = tensor.to(torch.float32)
        # Check if values have bf16-like precision (limited mantissa)
        print(f"  As FP32 range: [{as_fp32.min():.6f}, {as_fp32.max():.6f}]")


def test_1_torch_dtype_to_mfa_dtype_mapping():
    """
    Test 1: torch_dtype_to_mfa_dtype conversion for bf16

    This test verifies that the C++ function torch_dtype_to_mfa_dtype
    correctly maps torch::kBFloat16 to MFA_PRECISION_BF16.
    """
    print_test_header("Test 1: torch_dtype_to_mfa_dtype Mapping")

    if not METAL_AVAILABLE:
        return print_test_result(False, "Metal extension not available")

    if not hasattr(torch, "bfloat16"):
        return print_test_result(False, "BFloat16 not available in this PyTorch build")

    try:
        # Test the dtype mapping by creating tensors and checking what dtype the backend reports
        dtypes_to_test = [
            (torch.float32, "float32"),
            (torch.float16, "float16"),
            (torch.bfloat16, "bfloat16"),
        ]

        all_mappings_correct = True

        for torch_dtype, name in dtypes_to_test:
            print(f"\nTesting {name} ({torch_dtype}) mapping:")

            # Create a small tensor of this dtype
            tensor = torch.randn(2, 2, 4, 4, dtype=torch_dtype) * 0.1

            print(f"  Created tensor with dtype: {tensor.dtype}")
            print(f"  Tensor dtype name: {tensor.dtype}")

            # Try to process it through Metal SDPA to see if dtype mapping works
            try:
                q = k = v = tensor
                output = metal_sdpa_extension.metal_scaled_dot_product_attention(
                    q, k, v
                )

                # Check if output has the expected dtype
                dtype_preserved = output.dtype == torch_dtype
                print(f"  Output dtype: {output.dtype}")
                print(f"  Dtype preserved: {dtype_preserved}")

                if not dtype_preserved:
                    print(f"  ✗ DTYPE MAPPING ISSUE: {torch_dtype} -> {output.dtype}")
                    all_mappings_correct = False
                else:
                    print(f"  ✓ Dtype mapping correct for {name}")

            except Exception as e:
                print(f"  ✗ Exception with {name}: {e}")
                # Check if the error is dtype-related
                if "dtype" in str(e).lower() or "precision" in str(e).lower():
                    print(f"    -> This looks like a dtype mapping error")
                    all_mappings_correct = False

        return print_test_result(
            all_mappings_correct,
            (
                "All dtype mappings correct"
                if all_mappings_correct
                else "Dtype mapping issues detected"
            ),
        )

    except Exception as e:
        return print_test_result(False, f"Exception during dtype mapping test: {e}")


def test_2_ensure_contiguous_cpu_bf16():
    """
    Test 2: ensure_contiguous_cpu function with bf16 tensors

    This test verifies that the ensure_contiguous_cpu function in the C++ backend
    correctly handles bf16 tensors without converting them to other dtypes.
    """
    print_test_header("Test 2: ensure_contiguous_cpu with BF16")

    if not METAL_AVAILABLE:
        return print_test_result(False, "Metal extension not available")

    if not hasattr(torch, "bfloat16"):
        return print_test_result(False, "BFloat16 not available in this PyTorch build")

    try:
        # Create bf16 tensors in various configurations
        test_configs = [
            ("contiguous_cpu", lambda: torch.randn(2, 4, 8, 16, dtype=torch.bfloat16)),
            (
                "non_contiguous_cpu",
                lambda: torch.randn(2, 8, 4, 16, dtype=torch.bfloat16).permute(
                    0, 2, 1, 3
                ),
            ),
            (
                "contiguous_mps",
                lambda: (
                    torch.randn(2, 4, 8, 16, dtype=torch.bfloat16).to("mps")
                    if torch.backends.mps.is_available()
                    else None
                ),
            ),
        ]

        all_passed = True

        for config_name, tensor_factory in test_configs:
            print(f"\nTesting {config_name}:")

            try:
                tensor = tensor_factory()
                if tensor is None:
                    print("  Skipped (MPS not available)")
                    continue

                inspect_tensor_memory(tensor, f"Input {config_name}")

                # The ensure_contiguous_cpu function is called internally by Metal SDPA
                # We test it indirectly by passing the tensor through the attention function
                q = k = v = tensor

                # Before calling attention, check tensor properties
                orig_dtype = tensor.dtype
                orig_device = tensor.device
                orig_contiguous = tensor.is_contiguous()

                print(
                    f"  Original: dtype={orig_dtype}, device={orig_device}, contiguous={orig_contiguous}"
                )

                # Call attention (which internally calls ensure_contiguous_cpu)
                output = metal_sdpa_extension.metal_scaled_dot_product_attention(
                    q, k, v
                )

                inspect_tensor_memory(output, f"Output {config_name}")

                # Check that dtype was preserved through the ensure_contiguous_cpu process
                dtype_preserved = output.dtype == orig_dtype
                values_reasonable = torch.isfinite(output).all()

                config_passed = dtype_preserved and values_reasonable
                all_passed = all_passed and config_passed

                print(
                    f"  Result: dtype_preserved={dtype_preserved}, values_reasonable={values_reasonable}"
                )
                status = "✓" if config_passed else "✗"
                print(f"  {status} {config_name} test")

            except Exception as e:
                print(f"  ✗ Exception in {config_name}: {e}")
                all_passed = False

        return print_test_result(
            all_passed,
            (
                "ensure_contiguous_cpu preserves bf16 correctly"
                if all_passed
                else "ensure_contiguous_cpu has bf16 issues"
            ),
        )

    except Exception as e:
        return print_test_result(
            False, f"Exception during ensure_contiguous_cpu test: {e}"
        )


def test_3_bf16_value_corruption_detection():
    """
    Test 3: Whether bf16 values are being inadvertently converted to fp32 or other types

    This test creates known bf16 values and tracks them through the FFI boundary
    to detect if they're being corrupted or converted.
    """
    print_test_header("Test 3: BF16 Value Corruption Detection")

    if not METAL_AVAILABLE:
        return print_test_result(False, "Metal extension not available")

    if not hasattr(torch, "bfloat16"):
        return print_test_result(False, "BFloat16 not available in this PyTorch build")

    try:
        # Create bf16 tensors with known values that are representable exactly in bf16
        # BF16 has 1 sign bit, 8 exponent bits, 7 mantissa bits

        # Test values that should be exactly representable in bf16
        test_values = [
            1.0,  # Exactly representable
            0.5,  # Exactly representable
            0.25,  # Exactly representable
            1.5,  # Exactly representable
            -1.0,  # Exactly representable
            0.0,  # Exactly representable
        ]

        print("Testing with known bf16-representable values:")

        # Create a tensor with these exact values
        # Make it into a valid attention tensor shape
        base_tensor = torch.tensor(test_values, dtype=torch.bfloat16)

        # Expand to proper attention tensor shape
        q = base_tensor.view(1, 1, 2, 3).expand(1, 4, 8, 16)
        k = q.clone()
        v = q.clone()

        print(f"Input value sample: {q.flatten()[:6].tolist()}")
        inspect_tensor_memory(q, "Input with known values")

        # Store original values for comparison
        original_flat = q.flatten()
        original_as_fp32 = original_flat.to(torch.float32)

        # Process through Metal SDPA
        output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

        inspect_tensor_memory(output, "Output")

        # Check if output values are reasonable for a bf16 computation
        output_as_fp32 = output.to(torch.float32)

        # The output won't be exactly the same due to attention computation,
        # but we can check for signs of corruption:

        # 1. Check that values are finite
        finite_check = torch.isfinite(output).all()
        print(f"All output values finite: {finite_check}")

        # 2. Check that the range is reasonable (attention shouldn't blow up)
        reasonable_range = output.abs().max() < 10.0
        print(
            f"Output in reasonable range: {reasonable_range} (max abs: {output.abs().max():.4f})"
        )

        # 3. Check for bf16-like precision patterns
        # If values were converted to fp32 and back, we might see precision artifacts
        output_converted_back = (
            output.to(torch.float32).to(torch.bfloat16).to(torch.float32)
        )
        precision_preserved = torch.allclose(
            output_as_fp32, output_converted_back, rtol=1e-6
        )
        print(f"BF16 precision patterns preserved: {precision_preserved}")

        # 4. Check that dtype is still bf16
        dtype_correct = output.dtype == torch.bfloat16
        print(f"Output dtype correct: {dtype_correct}")

        all_checks_passed = (
            finite_check and reasonable_range and precision_preserved and dtype_correct
        )

        return print_test_result(
            all_checks_passed,
            (
                "No bf16 value corruption detected"
                if all_checks_passed
                else "BF16 value corruption detected"
            ),
        )

    except Exception as e:
        return print_test_result(False, f"Exception during value corruption test: {e}")


def test_4_direct_ffi_dtype_behavior():
    """
    Test 4: Direct FFI function call testing to bypass higher-level abstractions

    This test attempts to isolate FFI behavior by testing lower-level functions
    and examining the exact data flow through the FFI boundary.
    """
    print_test_header("Test 4: Direct FFI Dtype Behavior")

    if not METAL_AVAILABLE:
        return print_test_result(False, "Metal extension not available")

    if not hasattr(torch, "bfloat16"):
        return print_test_result(False, "BFloat16 not available in this PyTorch build")

    try:
        # Check if we can access lower-level FFI functions
        extension_attrs = dir(metal_sdpa_extension)
        print("Available extension functions:")
        for attr in sorted(extension_attrs):
            if not attr.startswith("_"):
                print(f"  {attr}")

        # Test with a minimal example that goes directly through the FFI
        print(f"\nTesting direct FFI path:")

        # Create minimal bf16 tensors
        q = torch.tensor([[[[1.0, 0.5]]]], dtype=torch.bfloat16)
        k = torch.tensor([[[[0.5, 1.0]]]], dtype=torch.bfloat16)
        v = torch.tensor([[[[0.25, 0.75]]]], dtype=torch.bfloat16)

        print(f"Minimal input shapes: q={q.shape}, k={k.shape}, v={v.shape}")
        print(f"Input dtypes: q={q.dtype}, k={k.dtype}, v={v.dtype}")
        print(f"Input values:")
        print(f"  q: {q.flatten().tolist()}")
        print(f"  k: {k.flatten().tolist()}")
        print(f"  v: {v.flatten().tolist()}")

        # Check memory layout before FFI call
        print(f"\nMemory layout before FFI:")
        print(
            f"  q contiguous: {q.is_contiguous()}, storage size: {q.storage().size()}"
        )
        print(
            f"  k contiguous: {k.is_contiguous()}, storage size: {k.storage().size()}"
        )
        print(
            f"  v contiguous: {v.is_contiguous()}, storage size: {v.storage().size()}"
        )

        # Call the main function
        try:
            output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

            print(f"\nFFI call successful!")
            print(f"Output shape: {output.shape}")
            print(f"Output dtype: {output.dtype}")
            print(f"Output values: {output.flatten().tolist()}")
            print(f"Output contiguous: {output.is_contiguous()}")

            # Verify no unexpected conversions happened
            input_dtype_preserved = (
                q.dtype == torch.bfloat16
            )  # Input shouldn't be modified
            output_dtype_correct = output.dtype == torch.bfloat16
            shape_preserved = output.shape == q.shape
            values_finite = torch.isfinite(output).all()

            print(f"\nValidation:")
            print(f"  Input dtype preserved: {input_dtype_preserved}")
            print(f"  Output dtype correct: {output_dtype_correct}")
            print(f"  Shape preserved: {shape_preserved}")
            print(f"  Values finite: {values_finite}")

            all_good = (
                input_dtype_preserved
                and output_dtype_correct
                and shape_preserved
                and values_finite
            )

            return print_test_result(
                all_good,
                (
                    "Direct FFI dtype behavior correct"
                    if all_good
                    else "Direct FFI dtype behavior issues detected"
                ),
            )

        except Exception as ffi_e:
            print(f"\nFFI call failed: {ffi_e}")

            # Analyze the error to see if it's dtype-related
            error_str = str(ffi_e).lower()
            dtype_related = any(
                keyword in error_str
                for keyword in ["dtype", "precision", "type", "bf16", "bfloat"]
            )

            print(f"Error appears dtype-related: {dtype_related}")

            if dtype_related:
                return print_test_result(False, f"FFI dtype-related error: {ffi_e}")
            else:
                return print_test_result(False, f"FFI non-dtype error: {ffi_e}")

    except Exception as e:
        return print_test_result(False, f"Exception during direct FFI test: {e}")


def test_5_bf16_tensor_round_trip():
    """
    Test 5: BF16 tensor round-trip testing

    This test creates bf16 tensors, converts them through various paths that
    the FFI might use, and checks for data corruption.
    """
    print_test_header("Test 5: BF16 Tensor Round-trip")

    if not hasattr(torch, "bfloat16"):
        return print_test_result(False, "BFloat16 not available in this PyTorch build")

    try:
        # Create a bf16 tensor with known values
        original_values = [1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 1.5, -1.5]
        original = torch.tensor(original_values, dtype=torch.bfloat16)

        print(f"Original BF16 values: {original.tolist()}")

        # Test various conversion paths that might happen in FFI

        # Path 1: BF16 -> CPU (should be no-op if already on CPU)
        cpu_version = original.cpu()
        cpu_match = torch.equal(original, cpu_version)
        print(f"BF16 CPU conversion: {cpu_match}")

        # Path 2: BF16 -> contiguous (should be no-op if already contiguous)
        contiguous_version = original.contiguous()
        contiguous_match = torch.equal(original, contiguous_version)
        print(f"BF16 contiguous conversion: {contiguous_match}")

        # Path 3: BF16 -> FP32 -> BF16 (this might happen inadvertently in FFI)
        fp32_intermediate = original.to(torch.float32)
        back_to_bf16 = fp32_intermediate.to(torch.bfloat16)
        roundtrip_match = torch.equal(original, back_to_bf16)
        print(f"BF16 -> FP32 -> BF16 round-trip: {roundtrip_match}")

        if not roundtrip_match:
            diff = original.to(torch.float32) - back_to_bf16.to(torch.float32)
            print(f"  Round-trip differences: {diff.abs().max().item():.8f}")

        # Path 4: BF16 data_ptr() and reconstruction
        # This tests if the raw memory representation is preserved
        try:
            original_bytes = original.numpy().tobytes()
            reconstructed = torch.frombuffer(original_bytes, dtype=torch.bfloat16)

            # Reshape to original shape
            if reconstructed.shape != original.shape:
                reconstructed = reconstructed.view(original.shape)

            memory_match = torch.equal(original, reconstructed)
            print(f"BF16 memory round-trip: {memory_match}")
        except Exception as e:
            print(f"BF16 memory round-trip failed: {e}")
            memory_match = False

        # Overall assessment
        all_roundtrips_good = (
            cpu_match and contiguous_match and roundtrip_match and memory_match
        )

        return print_test_result(
            all_roundtrips_good,
            (
                "All BF16 round-trips successful"
                if all_roundtrips_good
                else "Some BF16 round-trips failed - potential conversion issues"
            ),
        )

    except Exception as e:
        return print_test_result(False, f"Exception during round-trip test: {e}")


def main():
    """Run all BF16 FFI conversion tests."""
    print("BF16 FFI Conversion Test Suite")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"BFloat16 available: {hasattr(torch, 'bfloat16')}")
    print(f"Metal extension available: {METAL_AVAILABLE}")

    if torch.backends.mps.is_available():
        print(f"MPS backend available: {torch.backends.mps.is_available()}")
    else:
        print("MPS backend not available")

    if not METAL_AVAILABLE:
        print("\n⚠️  Metal extension not available - tests will be skipped")
        print("   Make sure the extension is built and available in PYTHONPATH")

    # Run all tests
    test_results = []

    test_results.append(test_1_torch_dtype_to_mfa_dtype_mapping())
    test_results.append(test_2_ensure_contiguous_cpu_bf16())
    test_results.append(test_3_bf16_value_corruption_detection())
    test_results.append(test_4_direct_ffi_dtype_behavior())
    test_results.append(test_5_bf16_tensor_round_trip())

    # Summary
    print(f"\n{'='*60}")
    print("FFI CONVERSION TEST SUMMARY")
    print("=" * 60)

    passed_count = sum(test_results)
    total_count = len(test_results)

    print(f"Tests passed: {passed_count}/{total_count}")

    if passed_count == total_count:
        print("✓ ALL FFI CONVERSION TESTS PASSED")
        print("  BF16 FFI boundary appears to be working correctly")
    else:
        print("✗ SOME FFI CONVERSION TESTS FAILED")
        print("\nDiagnostic Summary:")
        print("1. If torch_dtype_to_mfa_dtype mapping failed:")
        print("   -> Check C++ backend torch_dtype_to_mfa_dtype function")
        print("   -> Verify MFA_PRECISION_BF16 enum value is correct")
        print("2. If ensure_contiguous_cpu failed:")
        print("   -> Check ensure_contiguous_cpu function for dtype preservation")
        print("   -> Look for inadvertent .to() calls that change dtype")
        print("3. If value corruption was detected:")
        print("   -> Check Metal kernel implementations for bf16 support")
        print("   -> Verify buffer creation preserves bf16 data")
        print("4. If direct FFI behavior failed:")
        print("   -> Check FFI function signatures and parameter passing")
        print("   -> Verify Metal buffer creation with correct precision")
        print("5. If round-trip tests failed:")
        print("   -> Check for unnecessary dtype conversions in FFI pipeline")


if __name__ == "__main__":
    main()
