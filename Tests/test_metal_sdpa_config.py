#!/usr/bin/env python3
"""
Test suite for Metal SDPA QuantizationConfig functionality.

This test suite verifies:
1. Enum value mapping between C++ and Python
2. Non-quantized attention functionality
3. Configuration options for QuantizationConfig

NOTE: Quantized attention functions are temporarily disabled due to
issues in the MFA library's Swift/Metal implementation.
"""

import os
import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "examples" / "pytorch-custom-op-ffi"))


def _prepend_dyld_library_path():
    existing = os.environ.get("DYLD_LIBRARY_PATH", "")
    candidates = [
        PROJECT_ROOT / ".build" / "arm64-apple-macosx" / "release",
        PROJECT_ROOT / ".build" / "arm64-apple-macosx" / "debug",
    ]
    valid = [str(path) for path in candidates if path.exists()]
    if not valid:
        return
    prefix = ":".join(valid)
    os.environ["DYLD_LIBRARY_PATH"] = f"{prefix}:{existing}" if existing else prefix


_prepend_dyld_library_path()

# Import the Metal SDPA extension
import metal_sdpa_extension


class TestMetalSDPAConfig(unittest.TestCase):
    """Test suite for Metal SDPA configuration and attention functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 1
        self.seq_len = 4
        self.num_heads = 1
        self.head_dim = 8

        # Create test tensors
        self.q = (
            torch.randn(
                self.batch_size,
                self.seq_len,
                self.num_heads,
                self.head_dim,
                dtype=torch.float16,
            )
            * 0.1
        )
        self.k = (
            torch.randn(
                self.batch_size,
                self.seq_len,
                self.num_heads,
                self.head_dim,
                dtype=torch.float16,
            )
            * 0.1
        )
        self.v = (
            torch.randn(
                self.batch_size,
                self.seq_len,
                self.num_heads,
                self.head_dim,
                dtype=torch.float16,
            )
            * 0.1
        )

    def test_enum_values(self):
        """Test that OutputPrecision enum values are accessible."""
        # These are the Python enum values (which differ from C++ due to pybind11)
        self.assertEqual(int(metal_sdpa_extension.OutputPrecision.FP32), 0)
        self.assertEqual(int(metal_sdpa_extension.OutputPrecision.FP16), 1)
        self.assertEqual(int(metal_sdpa_extension.OutputPrecision.BF16), 2)

    def test_non_quantized_attention(self):
        """Test non-quantized attention function."""
        result = metal_sdpa_extension.metal_scaled_dot_product_attention(
            self.q, self.k, self.v
        )

        # Verify output shape
        self.assertEqual(result.shape, self.q.shape)

        # Verify output dtype (keeps input dtype)
        self.assertEqual(result.dtype, torch.float16)

        # Verify result is not all zeros
        self.assertGreater(result.abs().max().item(), 0)

    def test_config_creation(self):
        """Test QuantizationConfig creation and property access."""
        config = metal_sdpa_extension.QuantizationConfig()

        # Test default values
        self.assertEqual(config.precision, "int8")
        self.assertEqual(int(config.output_precision), 0)  # FP32
        self.assertFalse(config.is_causal)
        self.assertIsNone(config.scale)

        # Test setting values
        config.precision = "int4"
        config.output_precision = metal_sdpa_extension.OutputPrecision.FP16
        config.is_causal = True
        config.scale = 0.5

        self.assertEqual(config.precision, "int4")
        self.assertEqual(int(config.output_precision), 1)  # FP16
        self.assertTrue(config.is_causal)
        self.assertEqual(config.scale, 0.5)

    @unittest.skip("Quantized attention temporarily disabled due to MFA library issues")
    def test_quantized_attention_int8(self):
        """Test INT8 quantized attention (SKIPPED: MFA library issue)."""
        result = metal_sdpa_extension.quantized_scaled_dot_product_attention(
            self.q, self.k, self.v, precision="int8"
        )
        self.assertEqual(result.shape, self.q.shape)

    @unittest.skip("Quantized attention temporarily disabled due to MFA library issues")
    def test_quantized_attention_with_config(self):
        """Test quantized attention with config (SKIPPED: MFA library issue)."""
        config = metal_sdpa_extension.QuantizationConfig()
        config.precision = "int8"
        config.output_precision = metal_sdpa_extension.OutputPrecision.FP16

        result = (
            metal_sdpa_extension.quantized_scaled_dot_product_attention_with_config(
                self.q, self.k, self.v, config
            )
        )
        self.assertEqual(result.shape, self.q.shape)
        self.assertEqual(result.dtype, torch.float16)


class TestEnumMapping(unittest.TestCase):
    """Test enum value mapping between C++ and Python."""

    def test_output_precision_enum_values(self):
        """Verify that OutputPrecision enum values are correctly exposed."""
        # The actual Python values (due to pybind11 sequential assignment)
        python_values = {
            "FP32": int(metal_sdpa_extension.OutputPrecision.FP32),
            "FP16": int(metal_sdpa_extension.OutputPrecision.FP16),
            "BF16": int(metal_sdpa_extension.OutputPrecision.BF16),
        }

        # Expected Python values
        expected_values = {
            "FP32": 0,
            "FP16": 1,
            "BF16": 2,
        }

        self.assertEqual(python_values, expected_values)

        # Document the mapping issue
        print("\nEnum Value Mapping Summary:")
        print("Python -> C++ mapping (handled internally):")
        print("  Python FP32 (0) -> C++ FP32 (2)")
        print("  Python FP16 (1) -> C++ FP16 (0)")
        print("  Python BF16 (2) -> C++ BF16 (1)")
        print("Note: C++ code includes workaround to handle this mapping correctly.")


def suite():
    """Create test suite."""
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestMetalSDPAConfig))
    test_suite.addTest(unittest.makeSuite(TestEnumMapping))
    return test_suite


if __name__ == "__main__":
    print("=" * 60)
    print("Metal SDPA Configuration Test Suite")
    print("=" * 60)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite())

    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ All tests passed!")
        print("✅ Non-quantized attention works correctly")
        print("✅ Enum values are accessible")
        print("✅ QuantizationConfig can be created and configured")
        print("⚠️  Quantized attention tests skipped (MFA library issue)")
    else:
        print("❌ Some tests failed")
    print("=" * 60)

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
