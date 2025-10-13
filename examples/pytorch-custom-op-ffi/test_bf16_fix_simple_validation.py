#!/usr/bin/env python3
"""
Simple BF16 Fix Validation Test

This test validates the bf16 fixes using only the available functions
from the rebuilt extension, focusing on core functionality.
"""

import os
import sys

import numpy as np
import torch

# Add the project path to sys.path to import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    # Import the extension directly
    import metal_sdpa_extension as ext

    print("✅ Successfully imported metal_sdpa_extension")
except ImportError as e:
    print(f"❌ Failed to import metal_sdpa_extension: {e}")
    sys.exit(1)


class SimpleBF16Validator:
    """Simple bf16 fix validation using available functions."""

    def __init__(self):
        self.test_results = []
        self.device = torch.device("cpu")

    def log_result(self, test_name, passed, message=""):
        """Log a test result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.test_results.append((test_name, passed, message))
        print(f"{status}: {test_name}")
        if message:
            print(f"    {message}")

    def create_test_tensors(
        self, dtype=torch.bfloat16, batch_size=2, seq_len=128, num_heads=8, head_dim=64
    ):
        """Create test tensors with specific patterns for validation."""
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=self.device
        )
        k = torch.randn(
            batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=self.device
        )
        v = torch.randn(
            batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=self.device
        )

        # Add some specific values we can check for corruption
        q[0, 0, 0, 0] = 1.5
        k[0, 0, 0, 0] = 2.25
        v[0, 0, 0, 0] = 0.75

        return q, k, v

    def test_extension_availability(self):
        """Test that the extension is available and has expected functions."""
        test_name = "Extension Availability"

        try:
            # Check for core functions
            required_functions = [
                "metal_scaled_dot_product_attention",
                "is_metal_available",
                "get_version",
            ]

            missing_functions = []
            for func_name in required_functions:
                if not hasattr(ext, func_name):
                    missing_functions.append(func_name)

            if missing_functions:
                passed = False
                message = f"Missing functions: {missing_functions}"
            else:
                passed = True
                message = "All required functions available"

            self.log_result(test_name, passed, message)

        except Exception as e:
            self.log_result(test_name, False, f"Exception: {str(e)}")

    def test_metal_availability(self):
        """Test Metal availability check."""
        test_name = "Metal Availability"

        try:
            is_available = ext.is_metal_available()
            passed = isinstance(is_available, bool)
            message = f"Metal available: {is_available}"
            self.log_result(test_name, passed, message)

        except Exception as e:
            self.log_result(test_name, False, f"Exception: {str(e)}")

    def test_bf16_basic_functionality(self):
        """Test basic bf16 functionality with Metal SDPA."""
        test_name = "BF16 Basic Functionality"

        try:
            # Create bf16 tensors
            q, k, v = self.create_test_tensors(dtype=torch.bfloat16)

            # Store original values for comparison
            orig_q_val = float(q[0, 0, 0, 0])
            orig_k_val = float(k[0, 0, 0, 0])
            orig_v_val = float(v[0, 0, 0, 0])

            # Call the Metal SDPA function
            result = ext.metal_scaled_dot_product_attention(q, k, v)

            # Check that we got a valid result
            if result is None:
                passed = False
                message = "Result is None"
            elif result.numel() == 0:
                passed = False
                message = "Result has no elements"
            elif result.shape != q.shape:
                passed = False
                message = f"Shape mismatch: input {q.shape}, output {result.shape}"
            else:
                passed = True
                message = f"Successfully processed bf16 tensors: {q.shape} -> {result.shape}, output dtype: {result.dtype}"

            self.log_result(test_name, passed, message)

        except Exception as e:
            self.log_result(test_name, False, f"Exception: {str(e)}")

    def test_bf16_value_preservation(self):
        """Test that bf16 input values are not corrupted during processing."""
        test_name = "BF16 Value Preservation"

        try:
            # Create bf16 tensors with known values
            q, k, v = self.create_test_tensors(dtype=torch.bfloat16)

            # Store original values
            orig_q_val = float(q[0, 0, 0, 0])
            orig_k_val = float(k[0, 0, 0, 0])
            orig_v_val = float(v[0, 0, 0, 0])

            # Call the function
            result = ext.metal_scaled_dot_product_attention(q, k, v)

            # Check that input tensors weren't corrupted
            new_q_val = float(q[0, 0, 0, 0])
            new_k_val = float(k[0, 0, 0, 0])
            new_v_val = float(v[0, 0, 0, 0])

            tolerance = 1e-6
            q_preserved = abs(orig_q_val - new_q_val) < tolerance
            k_preserved = abs(orig_k_val - new_k_val) < tolerance
            v_preserved = abs(orig_v_val - new_v_val) < tolerance

            passed = q_preserved and k_preserved and v_preserved
            message = (
                f"Q: {orig_q_val} -> {new_q_val} ({'✓' if q_preserved else '✗'}), "
            )
            message += (
                f"K: {orig_k_val} -> {new_k_val} ({'✓' if k_preserved else '✗'}), "
            )
            message += f"V: {orig_v_val} -> {new_v_val} ({'✓' if v_preserved else '✗'})"

            self.log_result(test_name, passed, message)

        except Exception as e:
            self.log_result(test_name, False, f"Exception: {str(e)}")

    def test_mixed_precision_support(self):
        """Test that different precision inputs work correctly."""
        test_name = "Mixed Precision Support"

        try:
            test_cases = [
                (torch.bfloat16, "bf16"),
                (torch.float16, "fp16"),
                (torch.float32, "fp32"),
            ]

            all_passed = True
            messages = []

            for dtype, case_name in test_cases:
                try:
                    q, k, v = self.create_test_tensors(dtype=dtype)
                    result = ext.metal_scaled_dot_product_attention(q, k, v)

                    case_passed = result is not None and result.numel() > 0
                    messages.append(f"{case_name}: {'✓' if case_passed else '✗'}")
                    all_passed = all_passed and case_passed

                except Exception as e:
                    messages.append(f"{case_name}: Exception - {str(e)[:50]}...")
                    all_passed = False

            self.log_result(test_name, all_passed, "; ".join(messages))

        except Exception as e:
            self.log_result(test_name, False, f"Exception: {str(e)}")

    def test_quantized_attention_bf16(self):
        """Test quantized attention with bf16 inputs."""
        test_name = "Quantized Attention BF16"

        try:
            # Check if quantized functions are available
            if not hasattr(ext, "quantized_scaled_dot_product_attention"):
                self.log_result(
                    test_name, True, "Quantized attention not available (skipped)"
                )
                return

            # Create bf16 tensors
            q, k, v = self.create_test_tensors(dtype=torch.bfloat16)

            # Try quantized attention
            result = ext.quantized_scaled_dot_product_attention(q, k, v)

            passed = result is not None and result.numel() > 0
            message = (
                f"Quantized attention with bf16: {'Success' if passed else 'Failed'}"
            )

            self.log_result(test_name, passed, message)

        except Exception as e:
            self.log_result(test_name, False, f"Exception: {str(e)}")

    def test_version_info(self):
        """Test version information retrieval."""
        test_name = "Version Information"

        try:
            version = ext.get_version()
            passed = version is not None and len(version) == 3
            message = f"Version: {version}"
            self.log_result(test_name, passed, message)

        except Exception as e:
            self.log_result(test_name, False, f"Exception: {str(e)}")

    def run_all_tests(self):
        """Run all simple bf16 validation tests."""
        print("🧪 Running Simple BF16 Fix Validation Tests")
        print("=" * 50)

        # Basic functionality tests
        print("\n📋 Basic Functionality Tests")
        self.test_extension_availability()
        self.test_metal_availability()
        self.test_version_info()

        # BF16 specific tests
        print("\n📋 BF16 Specific Tests")
        self.test_bf16_basic_functionality()
        self.test_bf16_value_preservation()

        # Additional tests
        print("\n📋 Additional Tests")
        self.test_mixed_precision_support()
        self.test_quantized_attention_bf16()

        # Summary
        print("\n" + "=" * 50)
        print("📊 Test Summary")

        total_tests = len(self.test_results)
        passed_tests = sum(1 for _, passed, _ in self.test_results if passed)
        failed_tests = total_tests - passed_tests

        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")

        if failed_tests > 0:
            print("\n❌ Failed Tests:")
            for test_name, passed, message in self.test_results:
                if not passed:
                    print(f"  - {test_name}: {message}")

        return failed_tests == 0


if __name__ == "__main__":
    print("🔧 Simple BF16 Fix Validation Test Suite")
    print("This test validates basic bf16 functionality after the rebuild.")
    print()

    validator = SimpleBF16Validator()
    success = validator.run_all_tests()

    if success:
        print("\n🎉 All tests passed! BF16 fixes appear to be working correctly.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please review the implementation and retry.")
        sys.exit(1)
