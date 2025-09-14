#!/usr/bin/env python3
"""
Basic tests for Universal Metal Flash Attention Python bindings.

Tests core functionality, error handling, and API compatibility.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Add the source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import umfa
from umfa import MFAError


class TestBasicFunctionality:
    """Test basic MFA functionality."""

    def test_metal_availability(self):
        """Test Metal availability check."""
        # This should work on Apple Silicon Macs
        is_available = umfa.is_metal_available()
        assert isinstance(is_available, bool)

        if not is_available:
            pytest.skip("Metal not available on this system")

    def test_version_info(self):
        """Test version information retrieval."""
        major, minor, patch = umfa.get_version()
        assert isinstance(major, int) and major >= 0
        assert isinstance(minor, int) and minor >= 0
        assert isinstance(patch, int) and patch >= 0

    def test_context_creation(self):
        """Test MFA context creation and cleanup."""
        if not umfa.is_metal_available():
            pytest.skip("Metal not available")

        # Test direct creation
        ctx = umfa.MFAContext()
        assert ctx
        ctx.close()

        # Test context manager
        with umfa.MFAContext() as ctx:
            assert ctx

        # Test convenience function
        ctx = umfa.create_context()
        assert ctx
        ctx.close()

    def test_basic_attention(self):
        """Test basic attention computation."""
        if not umfa.is_metal_available():
            pytest.skip("Metal not available")

        # Small test case
        seq_len, head_dim = 32, 64
        q = np.random.randn(seq_len, head_dim).astype(np.float16)
        k = np.random.randn(seq_len, head_dim).astype(np.float16)
        v = np.random.randn(seq_len, head_dim).astype(np.float16)

        with umfa.MFAContext() as ctx:
            output = umfa.flash_attention_forward(ctx, q, k, v)

        # Check output properties
        assert output.shape == q.shape
        assert output.dtype == q.dtype
        assert not np.isnan(output).any()
        assert np.isfinite(output).all()

    def test_convenience_function(self):
        """Test convenience attention function."""
        if not umfa.is_metal_available():
            pytest.skip("Metal not available")

        seq_len, head_dim = 32, 64
        q = np.random.randn(seq_len, head_dim).astype(np.float16)
        k = np.random.randn(seq_len, head_dim).astype(np.float16)
        v = np.random.randn(seq_len, head_dim).astype(np.float16)

        output = umfa.attention(q, k, v)

        assert output.shape == q.shape
        assert output.dtype == q.dtype
        assert not np.isnan(output).any()


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_shapes(self):
        """Test handling of invalid tensor shapes."""
        if not umfa.is_metal_available():
            pytest.skip("Metal not available")

        q = np.random.randn(32, 64).astype(np.float16)
        k = np.random.randn(16, 64).astype(np.float16)  # Wrong seq_len
        v = np.random.randn(32, 64).astype(np.float16)

        with pytest.raises(ValueError):
            umfa.attention(q, k, v)

    def test_invalid_dimensions(self):
        """Test handling of invalid tensor dimensions."""
        if not umfa.is_metal_available():
            pytest.skip("Metal not available")

        # 1D tensors should fail
        q = np.random.randn(64).astype(np.float16)
        k = np.random.randn(64).astype(np.float16)
        v = np.random.randn(64).astype(np.float16)

        with pytest.raises(ValueError):
            umfa.attention(q, k, v)

    def test_non_contiguous_arrays(self):
        """Test handling of non-contiguous arrays."""
        if not umfa.is_metal_available():
            pytest.skip("Metal not available")

        seq_len, head_dim = 32, 64
        q_large = np.random.randn(seq_len, head_dim * 2).astype(np.float16)
        q = q_large[:, ::2]  # Non-contiguous

        k = np.random.randn(seq_len, head_dim).astype(np.float16)
        v = np.random.randn(seq_len, head_dim).astype(np.float16)

        with umfa.MFAContext() as ctx:
            with pytest.raises(ValueError):
                umfa.flash_attention_forward(ctx, q, k, v)

    def test_invalid_precision(self):
        """Test handling of invalid precision specifications."""
        if not umfa.is_metal_available():
            pytest.skip("Metal not available")

        q = np.random.randn(32, 64).astype(np.float16)
        k = np.random.randn(32, 64).astype(np.float16)
        v = np.random.randn(32, 64).astype(np.float16)

        with pytest.raises(ValueError):
            umfa.attention(q, k, v, input_precision="invalid")


class TestPrecisions:
    """Test different precision modes."""

    @pytest.mark.parametrize("precision", ["fp16", "fp32"])
    def test_precision_modes(self, precision):
        """Test different precision modes."""
        if not umfa.is_metal_available():
            pytest.skip("Metal not available")

        dtype = np.float16 if precision == "fp16" else np.float32
        seq_len, head_dim = 32, 64

        q = np.random.randn(seq_len, head_dim).astype(dtype)
        k = np.random.randn(seq_len, head_dim).astype(dtype)
        v = np.random.randn(seq_len, head_dim).astype(dtype)

        output = umfa.attention(q, k, v, input_precision=precision)

        assert output.shape == q.shape
        assert not np.isnan(output).any()

    def test_mixed_precision(self):
        """Test mixed precision computation."""
        if not umfa.is_metal_available():
            pytest.skip("Metal not available")

        seq_len, head_dim = 32, 64
        q = np.random.randn(seq_len, head_dim).astype(np.float32)
        k = np.random.randn(seq_len, head_dim).astype(np.float32)
        v = np.random.randn(seq_len, head_dim).astype(np.float32)

        output = umfa.attention(
            q,
            k,
            v,
            input_precision="fp32",
            intermediate_precision="fp16",
            output_precision="fp32",
        )

        assert output.shape == q.shape
        assert output.dtype == np.float32


class TestCausalMask:
    """Test causal masking functionality."""

    def test_causal_vs_non_causal(self):
        """Test difference between causal and non-causal attention."""
        if not umfa.is_metal_available():
            pytest.skip("Metal not available")

        seq_len, head_dim = 32, 64
        q = np.random.randn(seq_len, head_dim).astype(np.float16)
        k = np.random.randn(seq_len, head_dim).astype(np.float16)
        v = np.random.randn(seq_len, head_dim).astype(np.float16)

        output_normal = umfa.attention(q, k, v, causal=False)
        output_causal = umfa.attention(q, k, v, causal=True)

        # Outputs should be different
        assert not np.allclose(output_normal, output_causal, rtol=1e-3)

        # Both should be valid
        assert not np.isnan(output_normal).any()
        assert not np.isnan(output_causal).any()


class TestMemoryManagement:
    """Test memory management and resource cleanup."""

    def test_context_cleanup(self):
        """Test proper cleanup of contexts."""
        if not umfa.is_metal_available():
            pytest.skip("Metal not available")

        # Create and destroy many contexts
        for _ in range(10):
            ctx = umfa.MFAContext()
            assert ctx
            ctx.close()

    def test_buffer_cleanup(self):
        """Test proper cleanup of buffers."""
        if not umfa.is_metal_available():
            pytest.skip("Metal not available")

        seq_len, head_dim = 32, 64
        q = np.random.randn(seq_len, head_dim).astype(np.float16)

        with umfa.MFAContext() as ctx:
            # Create and destroy many buffers
            for _ in range(10):
                buf = umfa.MFABuffer(ctx, q)
                assert buf
                buf.close()


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
