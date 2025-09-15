#!/usr/bin/env python3
"""
Test suite for the scale factor fix in Metal Flash Attention PyTorch integration.
Author: bghira
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

# Handle import gracefully for CI environments without Metal
try:
    import metal_sdpa_extension

    METAL_AVAILABLE = metal_sdpa_extension.is_metal_available()
except ImportError:
    metal_sdpa_extension = None
    METAL_AVAILABLE = False


class TestScaleFactorFix:
    """Test suite for scale factor functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        if not METAL_AVAILABLE:
            pytest.skip("Metal not available")

    @pytest.mark.gpu
    @pytest.mark.correctness
    @pytest.mark.parametrize(
        "seq_len,head_dim",
        [
            (4, 4),
            (8, 8),
            (16, 16),
            (32, 32),
        ],
    )
    @pytest.mark.parametrize("scale", [0.1, 0.25, 0.35355, 0.5, 1.0])
    def test_scale_factor_correctness(self, seq_len, head_dim, scale):
        """Test that custom scale factors work correctly."""
        torch.manual_seed(42)
        q = torch.randn(seq_len, head_dim, dtype=torch.float32)
        k = torch.randn(seq_len, head_dim, dtype=torch.float32)
        v = torch.randn(seq_len, head_dim, dtype=torch.float32)

        # Metal implementation
        metal_output = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q, k, v, scale=scale, is_causal=False
        )

        # PyTorch reference
        torch_output = F.scaled_dot_product_attention(
            q, k, v, scale=scale, is_causal=False
        )

        # Check for NaN/Inf
        assert not torch.isnan(metal_output).any(), "Metal output contains NaN"
        assert not torch.isinf(metal_output).any(), "Metal output contains Inf"

        # Check correctness
        diff = torch.abs(metal_output - torch_output).max().item()
        assert diff < 1e-5, f"Metal vs PyTorch difference too large: {diff:.2e}"

    @pytest.mark.gpu
    @pytest.mark.correctness
    def test_backward_compatibility(self):
        """Test that existing code still works without explicit scale."""
        q = torch.randn(8, 16, dtype=torch.float32)
        k = torch.randn(8, 16, dtype=torch.float32)
        v = torch.randn(8, 16, dtype=torch.float32)

        # Without explicit scale (should default to 1/√head_dim)
        metal_output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

        # With explicit default scale
        expected_scale = 1.0 / np.sqrt(16)
        metal_output_explicit = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q, k, v, scale=expected_scale
        )

        diff = torch.abs(metal_output - metal_output_explicit).max().item()
        assert diff < 1e-7, f"Backward compatibility broken: {diff:.2e}"

    @pytest.mark.gpu
    @pytest.mark.correctness
    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ],
    )
    def test_data_type_support(self, dtype):
        """Test that scale factor fix works with different data types."""
        q = torch.ones(4, 4, dtype=dtype)
        k = torch.ones(4, 4, dtype=dtype)
        v = torch.ones(4, 4, dtype=dtype)

        scale = 0.5  # Custom scale factor

        # This should work without hanging or errors
        output = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q, k, v, scale=scale, is_causal=False
        )

        assert (
            output.dtype == dtype
        ), f"Output dtype mismatch: {output.dtype} != {dtype}"
        assert not torch.isnan(output).any(), f"Output contains NaN for {dtype}"

    @pytest.mark.gpu
    @pytest.mark.unit
    def test_extreme_scale_values(self):
        """Test behavior with extreme scale values."""
        q = torch.randn(4, 4, dtype=torch.float32)
        k = torch.randn(4, 4, dtype=torch.float32)
        v = torch.randn(4, 4, dtype=torch.float32)

        # Very small scale
        output_small = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q, k, v, scale=1e-6, is_causal=False
        )
        assert not torch.isnan(output_small).any(), "Small scale produces NaN"

        # Larger scale (should work but may have numerical differences)
        output_large = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q, k, v, scale=1.5, is_causal=False
        )
        assert not torch.isnan(output_large).any(), "Large scale produces NaN"

    @pytest.mark.gpu
    @pytest.mark.integration
    def test_swift_test_parameters(self):
        """Test with exact parameters from Swift test suite."""
        # EXACT same parameters as Swift test
        seq_len = 4
        head_dim = 4

        # Create tensors with ALL 1.0s (exactly like Swift test)
        q = torch.ones(seq_len, head_dim, dtype=torch.float32)
        k = torch.ones(seq_len, head_dim, dtype=torch.float32)
        v = torch.ones(seq_len, head_dim, dtype=torch.float32)

        # Test with same scale as Swift
        scale = 1.0 / np.sqrt(head_dim)

        # Metal implementation
        metal_output = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q, k, v, scale=scale, is_causal=False
        )

        # PyTorch reference
        torch_output = F.scaled_dot_product_attention(
            q, k, v, scale=scale, is_causal=False
        )

        # Expected theoretical output (all 1.0s)
        expected_output = torch.ones_like(v)

        # Check all comparisons
        metal_vs_torch = torch.abs(metal_output - torch_output).max().item()
        metal_vs_expected = torch.abs(metal_output - expected_output).max().item()

        assert metal_vs_torch < 1e-5, f"Metal vs PyTorch: {metal_vs_torch:.2e}"
        assert metal_vs_expected < 1e-5, f"Metal vs expected: {metal_vs_expected:.2e}"


class TestTensorLayouts:
    """Test different tensor layouts and configurations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        if not METAL_AVAILABLE:
            pytest.skip("Metal not available")

    @pytest.mark.gpu
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "layout_type,tensor_factory",
        [
            ("contiguous", lambda s, h: torch.ones(s, h, dtype=torch.float32)),
            ("non_contiguous", lambda s, h: torch.ones(h, s, dtype=torch.float32).T),
            ("strided", lambda s, h: torch.ones(s * 2, h, dtype=torch.float32)[::2, :]),
        ],
    )
    def test_tensor_layouts(self, layout_type, tensor_factory):
        """Test different tensor memory layouts."""
        seq_len, head_dim = 4, 4
        tensor = tensor_factory(seq_len, head_dim)

        # Should work regardless of layout
        output = metal_sdpa_extension.metal_scaled_dot_product_attention(
            tensor, tensor, tensor
        )

        assert output.shape == tensor.shape, f"Shape mismatch for {layout_type}"
        assert not torch.isnan(output).any(), f"NaN in output for {layout_type}"


@pytest.mark.skipif(not METAL_AVAILABLE, reason="Metal not available")
class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.gpu
    @pytest.mark.unit
    def test_device_availability(self):
        """Test device availability detection."""
        assert metal_sdpa_extension.is_metal_available(), "Metal should be available"

    @pytest.mark.gpu
    @pytest.mark.unit
    def test_version_info(self):
        """Test version information retrieval."""
        major, minor, patch = metal_sdpa_extension.get_version()
        assert isinstance(major, int), "Major version should be int"
        assert isinstance(minor, int), "Minor version should be int"
        assert isinstance(patch, int), "Patch version should be int"
        assert major >= 1, "Major version should be >= 1"

    @pytest.mark.gpu
    @pytest.mark.unit
    def test_error_handling(self):
        """Test proper error handling for invalid inputs."""
        # Test with mismatched tensor shapes
        q = torch.randn(4, 4, dtype=torch.float32)
        k = torch.randn(5, 4, dtype=torch.float32)  # Different seq_len
        v = torch.randn(4, 4, dtype=torch.float32)

        with pytest.raises(RuntimeError):
            metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
