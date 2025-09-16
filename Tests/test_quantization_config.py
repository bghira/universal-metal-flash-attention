#!/usr/bin/env python3
"""
Red-Green Test for QuantizationConfig with Output Precision
Tests that FP16, BF16, and FP32 output precision options work correctly
with the quantized attention function.
"""

import os
import sys
import unittest

import torch

# Add the PyTorch extension path
sys.path.append(
    "/Users/kash/src/universal-metal-flash-attention/examples/pytorch-custom-op-ffi"
)

# Import the Metal SDPA extension
import metal_sdpa_extension


class TestQuantizationConfig(unittest.TestCase):
    """Test suite for QuantizationConfig with output precision options."""

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

    def test_output_precision_fp16(self):
        """Test that FP16 output precision produces correct dtype."""
        config = metal_sdpa_extension.QuantizationConfig()
        config.precision = "int8"
        config.output_precision = metal_sdpa_extension.OutputPrecision.FP16

        result = (
            metal_sdpa_extension.quantized_scaled_dot_product_attention_with_config(
                self.q, self.k, self.v, config
            )
        )

        # Check shape is correct
        expected_shape = (self.batch_size, self.seq_len, self.num_heads, self.head_dim)
        self.assertEqual(
            result.shape,
            expected_shape,
            f"Expected shape {expected_shape}, got {result.shape}",
        )

        # Check dtype is FP16
        self.assertEqual(
            result.dtype,
            torch.float16,
            f"Expected dtype torch.float16, got {result.dtype}",
        )

        # Check values are reasonable (not NaN, not zero)
        self.assertFalse(torch.isnan(result).any(), "Output contains NaN values")
        self.assertFalse((result == 0).all(), "Output is all zeros")

    def test_output_precision_bf16(self):
        """Test that BF16 output precision produces correct dtype."""
        config = metal_sdpa_extension.QuantizationConfig()
        config.precision = "int8"
        config.output_precision = metal_sdpa_extension.OutputPrecision.BF16

        result = (
            metal_sdpa_extension.quantized_scaled_dot_product_attention_with_config(
                self.q, self.k, self.v, config
            )
        )

        # Check shape is correct
        expected_shape = (self.batch_size, self.seq_len, self.num_heads, self.head_dim)
        self.assertEqual(
            result.shape,
            expected_shape,
            f"Expected shape {expected_shape}, got {result.shape}",
        )

        # Check dtype is BF16
        self.assertEqual(
            result.dtype,
            torch.bfloat16,
            f"Expected dtype torch.bfloat16, got {result.dtype}",
        )

        # Check values are reasonable (not NaN, not zero)
        self.assertFalse(torch.isnan(result).any(), "Output contains NaN values")
        self.assertFalse((result == 0).all(), "Output is all zeros")

    def test_output_precision_fp32(self):
        """Test that FP32 output precision produces correct dtype."""
        config = metal_sdpa_extension.QuantizationConfig()
        config.precision = "int8"
        config.output_precision = metal_sdpa_extension.OutputPrecision.FP32

        result = (
            metal_sdpa_extension.quantized_scaled_dot_product_attention_with_config(
                self.q, self.k, self.v, config
            )
        )

        # Check shape is correct
        expected_shape = (self.batch_size, self.seq_len, self.num_heads, self.head_dim)
        self.assertEqual(
            result.shape,
            expected_shape,
            f"Expected shape {expected_shape}, got {result.shape}",
        )

        # Check dtype is FP32
        self.assertEqual(
            result.dtype,
            torch.float32,
            f"Expected dtype torch.float32, got {result.dtype}",
        )

        # Check values are reasonable (not NaN, not zero)
        self.assertFalse(torch.isnan(result).any(), "Output contains NaN values")
        self.assertFalse((result == 0).all(), "Output is all zeros")

    def test_int4_quantization_with_fp16_output(self):
        """Test INT4 quantization with FP16 output precision."""
        config = metal_sdpa_extension.QuantizationConfig()
        config.precision = "int4"
        config.output_precision = metal_sdpa_extension.OutputPrecision.FP16

        result = (
            metal_sdpa_extension.quantized_scaled_dot_product_attention_with_config(
                self.q, self.k, self.v, config
            )
        )

        # Check shape and dtype
        expected_shape = (self.batch_size, self.seq_len, self.num_heads, self.head_dim)
        self.assertEqual(result.shape, expected_shape)
        self.assertEqual(result.dtype, torch.float16)

        # Check values
        self.assertFalse(torch.isnan(result).any())
        self.assertFalse((result == 0).all())

    def test_causal_masking_with_config(self):
        """Test that causal masking works with config."""
        config = metal_sdpa_extension.QuantizationConfig()
        config.precision = "int8"
        config.is_causal = True
        config.output_precision = metal_sdpa_extension.OutputPrecision.FP32

        result = (
            metal_sdpa_extension.quantized_scaled_dot_product_attention_with_config(
                self.q, self.k, self.v, config
            )
        )

        # Check basic properties
        self.assertEqual(result.dtype, torch.float32)
        self.assertFalse(torch.isnan(result).any())

        # With causal masking, later positions shouldn't attend to future positions
        # This is a basic smoke test - more detailed causal testing would require
        # comparing against a reference implementation

    def test_custom_scale_with_config(self):
        """Test that custom scale parameter works."""
        config = metal_sdpa_extension.QuantizationConfig()
        config.precision = "int8"
        config.scale = 0.5  # Custom scale
        config.output_precision = metal_sdpa_extension.OutputPrecision.FP32

        result = (
            metal_sdpa_extension.quantized_scaled_dot_product_attention_with_config(
                self.q, self.k, self.v, config
            )
        )

        # Check basic properties
        self.assertEqual(result.dtype, torch.float32)
        self.assertFalse(torch.isnan(result).any())
        self.assertFalse((result == 0).all())

    def test_enum_values_are_correct(self):
        """Test that enum values map correctly to expected values."""
        # Due to pybind11 limitations, Python enum values might not match C++
        # But we can at least verify they exist and are distinct
        fp16_val = int(metal_sdpa_extension.OutputPrecision.FP16)
        bf16_val = int(metal_sdpa_extension.OutputPrecision.BF16)
        fp32_val = int(metal_sdpa_extension.OutputPrecision.FP32)

        # Check they are distinct
        self.assertNotEqual(fp16_val, bf16_val)
        self.assertNotEqual(fp16_val, fp32_val)
        self.assertNotEqual(bf16_val, fp32_val)

        print(f"\nEnum values: FP16={fp16_val}, BF16={bf16_val}, FP32={fp32_val}")


def run_tests():
    """Run the test suite."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestQuantizationConfig)

    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return 0 if all tests pass, 1 otherwise
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
