#!/usr/bin/env python3
"""
Test suite for Metal SDPA backend integration.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pytest
import sys
from pathlib import Path

# Add the python package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

try:
    from pytorch_custom_op_ffi import (
        register_metal_sdpa_backend,
        unregister_metal_sdpa_backend,
        use_metal_sdpa,
        is_metal_sdpa_available,
        MetalSDPAContext,
    )

    BACKEND_AVAILABLE = True
except ImportError as e:
    BACKEND_AVAILABLE = False
    import_error = e


class TestMetalSDPABackend:
    """Test Metal SDPA backend functionality."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        # Unregister backend before each test to ensure clean state
        if BACKEND_AVAILABLE:
            try:
                unregister_metal_sdpa_backend()
            except:
                pass

        yield

        # Cleanup after test
        if BACKEND_AVAILABLE:
            try:
                unregister_metal_sdpa_backend()
            except:
                pass

    def test_backend_availability(self):
        """Test that backend is available on compatible systems."""
        if not BACKEND_AVAILABLE:
            pytest.skip(f"Backend not available: {import_error}")

        # Should be available on macOS with Metal
        if sys.platform == "darwin":
            assert is_metal_sdpa_available(), "Metal SDPA should be available on macOS"
        else:
            assert (
                not is_metal_sdpa_available()
            ), "Metal SDPA should not be available on non-macOS"

    def test_backend_registration(self):
        """Test backend registration and unregistration."""
        if not BACKEND_AVAILABLE or not is_metal_sdpa_available():
            pytest.skip("Backend not available")

        # Test registration
        register_metal_sdpa_backend()
        assert torch.backends.metal_sdpa.enabled

        # Test unregistration
        unregister_metal_sdpa_backend()
        assert not torch.backends.metal_sdpa.enabled

    def test_context_manager(self):
        """Test context manager usage."""
        if not BACKEND_AVAILABLE or not is_metal_sdpa_available():
            pytest.skip("Backend not available")

        seq_len, head_dim = 128, 64
        q = torch.randn(seq_len, head_dim, dtype=torch.float16)
        k = torch.randn(seq_len, head_dim, dtype=torch.float16)
        v = torch.randn(seq_len, head_dim, dtype=torch.float16)

        # Test context manager
        with use_metal_sdpa():
            # This should work without errors
            pass

    def test_metal_sdpa_context(self):
        """Test MetalSDPAContext class."""
        if not BACKEND_AVAILABLE or not is_metal_sdpa_available():
            pytest.skip("Backend not available")

        seq_len, head_dim = 128, 64
        q = torch.randn(seq_len, head_dim, dtype=torch.float16)
        k = torch.randn(seq_len, head_dim, dtype=torch.float16)
        v = torch.randn(seq_len, head_dim, dtype=torch.float16)

        with MetalSDPAContext() as ctx:
            # Test direct call (bypasses PyTorch dispatcher)
            output = ctx.direct_call(q, k, v, is_causal=True)
            assert output.shape == q.shape
            assert output.dtype == q.dtype

    def test_sdpa_correctness(self):
        """Test SDPA correctness against PyTorch reference."""
        if not BACKEND_AVAILABLE or not is_metal_sdpa_available():
            pytest.skip("Backend not available")

        seq_len, head_dim = 64, 32  # Smaller size for quick test
        q = torch.randn(seq_len, head_dim, dtype=torch.float16)
        k = torch.randn(seq_len, head_dim, dtype=torch.float16)
        v = torch.randn(seq_len, head_dim, dtype=torch.float16)

        # Get PyTorch reference
        torch_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Get Metal output
        with MetalSDPAContext() as ctx:
            metal_output = ctx.direct_call(q, k, v, is_causal=True)

        # Compare (allow some tolerance due to precision differences)
        max_diff = torch.abs(torch_output - metal_output).max().item()
        assert max_diff < 1e-2, f"Outputs differ by {max_diff}"

    def test_different_dtypes(self):
        """Test different input dtypes."""
        if not BACKEND_AVAILABLE or not is_metal_sdpa_available():
            pytest.skip("Backend not available")

        seq_len, head_dim = 64, 32
        dtypes = [torch.float16, torch.float32]

        for dtype in dtypes:
            q = torch.randn(seq_len, head_dim, dtype=dtype)
            k = torch.randn(seq_len, head_dim, dtype=dtype)
            v = torch.randn(seq_len, head_dim, dtype=dtype)

            with MetalSDPAContext() as ctx:
                output = ctx.direct_call(q, k, v, is_causal=True)
                assert output.dtype == dtype

    def test_causal_masking(self):
        """Test causal masking functionality."""
        if not BACKEND_AVAILABLE or not is_metal_sdpa_available():
            pytest.skip("Backend not available")

        seq_len, head_dim = 32, 16
        q = torch.randn(seq_len, head_dim, dtype=torch.float16)
        k = torch.randn(seq_len, head_dim, dtype=torch.float16)
        v = torch.randn(seq_len, head_dim, dtype=torch.float16)

        with MetalSDPAContext() as ctx:
            # Test both causal and non-causal
            causal_output = ctx.direct_call(q, k, v, is_causal=True)
            non_causal_output = ctx.direct_call(q, k, v, is_causal=False)

            assert causal_output.shape == non_causal_output.shape
            # Outputs should be different due to masking
            diff = torch.abs(causal_output - non_causal_output).max().item()
            assert diff > 1e-3, "Causal and non-causal outputs should be different"


def test_import():
    """Test that imports work correctly."""
    if not BACKEND_AVAILABLE:
        print(f"Import failed: {import_error}")
        return

    # Basic import test
    from pytorch_custom_op_ffi import (
        register_metal_sdpa_backend,
        is_metal_sdpa_available,
    )

    print(f"Metal available: {is_metal_sdpa_available()}")


def benchmark_performance():
    """Benchmark Metal SDPA vs PyTorch SDPA."""
    if not BACKEND_AVAILABLE or not is_metal_sdpa_available():
        print("Backend not available for benchmarking")
        return

    import time

    print("\n🚀 Performance Benchmark: Metal SDPA vs PyTorch SDPA")
    print("=" * 60)

    configs = [
        (128, 64),
        (256, 64),
        (512, 64),
    ]

    for seq_len, head_dim in configs:
        print(f"\nTesting seq_len={seq_len}, head_dim={head_dim}")

        q = torch.randn(seq_len, head_dim, dtype=torch.float16)
        k = torch.randn(seq_len, head_dim, dtype=torch.float16)
        v = torch.randn(seq_len, head_dim, dtype=torch.float16)

        # Warm up
        for _ in range(5):
            _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        with MetalSDPAContext() as ctx:
            for _ in range(5):
                _ = ctx.direct_call(q, k, v, is_causal=True)

        # Benchmark PyTorch
        torch_times = []
        for _ in range(20):
            start = time.time()
            torch_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            torch_times.append(time.time() - start)

        # Benchmark Metal
        metal_times = []
        with MetalSDPAContext() as ctx:
            for _ in range(20):
                start = time.time()
                metal_output = ctx.direct_call(q, k, v, is_causal=True)
                metal_times.append(time.time() - start)

        torch_mean = np.mean(torch_times) * 1000
        metal_mean = np.mean(metal_times) * 1000
        speedup = torch_mean / metal_mean

        # Verify correctness
        max_diff = torch.abs(torch_output - metal_output).max().item()

        print(f"  PyTorch SDPA:  {torch_mean:.2f}ms")
        print(f"  Metal SDPA:    {metal_mean:.2f}ms")
        print(f"  Speedup:       {speedup:.2f}x")
        print(f"  Max diff:      {max_diff:.6f}")
        print(f"  Status:        {'✅ PASS' if max_diff < 1e-2 else '❌ FAIL'}")


if __name__ == "__main__":
    # Run basic tests
    test_import()

    if BACKEND_AVAILABLE and is_metal_sdpa_available():
        benchmark_performance()
    else:
        print("Skipping benchmark - backend not available")

    # Run pytest if available
    try:
        import pytest

        pytest.main([__file__, "-v"])
    except ImportError:
        print("pytest not available, run: pip install pytest")
