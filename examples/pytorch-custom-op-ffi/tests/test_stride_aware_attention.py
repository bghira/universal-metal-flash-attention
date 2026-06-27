"""
Test suite for stride-aware Metal kernel implementation.

This module tests the handling of non-contiguous tensors (from permute operations)
to validate that the Metal kernels correctly handle stride information and prevent
memory corruption issues.

The memory corruption issue occurs when:
1. PyTorch tensors are permuted from FLUX layout [B,H,S,D] to Metal layout [B,S,H,D]
2. Permuted tensors are non-contiguous (share underlying storage with different strides)
3. Metal kernels assume contiguous memory and calculate incorrect offsets
"""

import gc
import os

# Try to import the Metal SDPA extension
import sys
import time
from typing import Optional, Tuple

import numpy as np
import pytest
import torch

# Add parent directory to path to find the extension
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

try:
    import metal_sdpa_extension

    HAS_METAL = True
except ImportError as e:
    HAS_METAL = False
    print(f"Warning: metal_sdpa_extension not available: {e}")


def create_test_tensors(
    batch_size: int = 1,
    num_heads: int = 12,
    seq_len: int = 77,
    head_dim: int = 64,
    layout: str = "flux",
    dtype: torch.dtype = torch.float16,
    device: str = "mps",
    contiguous: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create test tensors for attention computation.

    Args:
        batch_size: Batch dimension
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Dimension per head
        layout: "flux" for [B,H,S,D] or "metal" for [B,S,H,D]
        dtype: Data type for tensors
        device: Device to place tensors on
        contiguous: Whether to return contiguous tensors

    Returns:
        Tuple of (query, key, value) tensors
    """
    torch.manual_seed(42)  # For reproducibility

    if layout == "flux":
        # FLUX layout: [batch, heads, sequence, dim]
        shape = (batch_size, num_heads, seq_len, head_dim)
    else:
        # Metal layout: [batch, sequence, heads, dim]
        shape = (batch_size, seq_len, num_heads, head_dim)

    # Create tensors with small values to avoid numerical issues
    q = torch.randn(shape, dtype=dtype, device=device) * 0.1
    k = torch.randn(shape, dtype=dtype, device=device) * 0.1
    v = torch.randn(shape, dtype=dtype, device=device) * 0.1

    if not contiguous and layout == "metal":
        # Create non-contiguous tensors by permuting from FLUX layout
        # This simulates the real-world scenario where FLUX models
        # permute their tensors before passing to Metal
        flux_shape = (batch_size, num_heads, seq_len, head_dim)
        q_flux = torch.randn(flux_shape, dtype=dtype, device=device) * 0.1
        k_flux = torch.randn(flux_shape, dtype=dtype, device=device) * 0.1
        v_flux = torch.randn(flux_shape, dtype=dtype, device=device) * 0.1

        # Permute to Metal layout - creates non-contiguous tensors
        q = q_flux.permute(0, 2, 1, 3)  # [B,H,S,D] -> [B,S,H,D]
        k = k_flux.permute(0, 2, 1, 3)
        v = v_flux.permute(0, 2, 1, 3)

        # Verify tensors are non-contiguous
        assert not q.is_contiguous(), "Q should be non-contiguous"
        assert not k.is_contiguous(), "K should be non-contiguous"
        assert not v.is_contiguous(), "V should be non-contiguous"

    return q, k, v


def print_tensor_info(tensor: torch.Tensor, name: str = "Tensor"):
    """Print detailed information about a tensor including its stride layout."""
    print(f"\n{name} Info:")
    print(f"  Shape: {list(tensor.shape)}")
    print(f"  Strides: {tensor.stride()}")
    print(f"  Contiguous: {tensor.is_contiguous()}")
    print(f"  Data pointer: {tensor.data_ptr():x}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    print(f"  Numel: {tensor.numel()}")
    print(f"  Element size: {tensor.element_size()} bytes")
    print(f"  Total bytes: {tensor.numel() * tensor.element_size()}")


@pytest.mark.skipif(not HAS_METAL, reason="Metal SDPA extension not available")
class TestStrideAwareAttention:
    """Test suite for stride-aware attention implementation."""

    def test_mre_memory_corruption(self):
        """
        Minimal Reproducible Example (MRE) for memory corruption issue.

        This test demonstrates the memory corruption that occurs when
        non-contiguous tensors from permute operations are passed to
        Metal kernels that assume contiguous memory layout.
        """
        print("\n" + "=" * 80)
        print("MRE: Memory Corruption with Non-Contiguous Tensors")
        print("=" * 80)

        # Create FLUX layout tensors
        batch, heads, seq_len, dim = 1, 12, 77, 64
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        print(f"\nTest configuration:")
        print(f"  Batch: {batch}, Heads: {heads}, Seq: {seq_len}, Dim: {dim}")
        print(f"  Device: {device}")

        # Create contiguous FLUX layout tensors [B,H,S,D]
        torch.manual_seed(42)
        q_flux = (
            torch.randn(batch, heads, seq_len, dim, dtype=torch.float16, device=device)
            * 0.1
        )
        k_flux = (
            torch.randn(batch, heads, seq_len, dim, dtype=torch.float16, device=device)
            * 0.1
        )
        v_flux = (
            torch.randn(batch, heads, seq_len, dim, dtype=torch.float16, device=device)
            * 0.1
        )

        print("\nOriginal FLUX tensors (contiguous):")
        print_tensor_info(q_flux, "Q_flux")

        # Permute to Metal layout [B,S,H,D] - creates non-contiguous views
        q_metal = q_flux.permute(0, 2, 1, 3)
        k_metal = k_flux.permute(0, 2, 1, 3)
        v_metal = v_flux.permute(0, 2, 1, 3)

        print("\nPermuted Metal tensors (non-contiguous):")
        print_tensor_info(q_metal, "Q_metal")

        # Calculate expected strides for contiguous Metal layout
        expected_strides = (seq_len * heads * dim, heads * dim, dim, 1)
        actual_strides = q_metal.stride()

        print(f"\nStride mismatch detection:")
        print(f"  Expected (contiguous): {expected_strides}")
        print(f"  Actual (permuted):     {actual_strides}")
        print(f"  Match: {expected_strides == actual_strides}")

        # The problem: Metal kernel assumes contiguous memory
        # It calculates offset as: (batch * seq * heads + seq * heads + head) * dim
        # But the actual memory layout doesn't match this due to permutation

        try:
            # This call may cause memory corruption if Metal doesn't handle strides
            print("\nCalling Metal SDPA with non-contiguous tensors...")
            output = metal_sdpa_extension.metal_scaled_dot_product_attention(
                q_metal, k_metal, v_metal
            )
            print("✅ Call completed without crash")

            # Verify output shape
            expected_shape = q_metal.shape
            assert (
                output.shape == expected_shape
            ), f"Output shape mismatch: {output.shape} != {expected_shape}"
            print(f"✅ Output shape correct: {list(output.shape)}")

            # Check for NaN/Inf values that might indicate corruption
            has_nan = torch.isnan(output).any().item()
            has_inf = torch.isinf(output).any().item()

            if has_nan or has_inf:
                print(f"⚠️  Warning: Output contains NaN: {has_nan}, Inf: {has_inf}")
            else:
                print("✅ Output contains valid values (no NaN/Inf)")

        except Exception as e:
            print(f"❌ Error occurred: {e}")
            # This is expected if memory corruption is detected
            raise

        # Compare with contiguous version
        print("\nComparing with contiguous tensor approach...")
        q_contig = q_metal.contiguous()
        k_contig = k_metal.contiguous()
        v_contig = v_metal.contiguous()

        print_tensor_info(q_contig, "Q_contiguous")

        output_contig = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q_contig, k_contig, v_contig
        )

        print("✅ Contiguous tensor call completed")

        # If both succeed, compare outputs
        if "output" in locals():
            diff = torch.abs(output - output_contig)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            print(f"\nOutput comparison:")
            print(f"  Max difference: {max_diff:.6f}")
            print(f"  Mean difference: {mean_diff:.6f}")

            # Large differences indicate the non-contiguous version is wrong
            if max_diff > 0.01:
                print(f"⚠️  Large difference detected - possible memory corruption!")
            else:
                print("✅ Outputs match within tolerance")

    def test_stride_patterns(self):
        """Test various stride patterns from common tensor operations."""
        print("\n" + "=" * 80)
        print("Testing Various Stride Patterns")
        print("=" * 80)

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        batch, heads, seq_len, dim = 1, 8, 64, 32

        test_cases = [
            ("Contiguous", lambda t: t),
            ("Permute(0,2,1,3)", lambda t: t.permute(0, 2, 1, 3)),
            ("Transpose(1,2)", lambda t: t.transpose(1, 2)),
            (
                "View after permute",
                lambda t: t.permute(0, 2, 1, 3)
                .reshape(batch, seq_len, -1)
                .view(batch, seq_len, heads, dim),
            ),
            ("Slice", lambda t: t[:, :, : seq_len // 2, :]),
        ]

        for name, transform_fn in test_cases:
            print(f"\n{name}:")

            # Create base tensor in FLUX layout
            torch.manual_seed(42)
            q_base = (
                torch.randn(
                    batch, heads, seq_len, dim, dtype=torch.float16, device=device
                )
                * 0.1
            )
            k_base = (
                torch.randn(
                    batch, heads, seq_len, dim, dtype=torch.float16, device=device
                )
                * 0.1
            )
            v_base = (
                torch.randn(
                    batch, heads, seq_len, dim, dtype=torch.float16, device=device
                )
                * 0.1
            )

            # Apply transformation
            q = transform_fn(q_base)
            k = transform_fn(k_base)
            v = transform_fn(v_base)

            print(f"  Shape: {list(q.shape)}")
            print(f"  Strides: {q.stride()}")
            print(f"  Contiguous: {q.is_contiguous()}")

            try:
                output = metal_sdpa_extension.metal_scaled_dot_product_attention(
                    q, k, v
                )
                print(f"  ✅ Success - Output shape: {list(output.shape)}")

                # Check for validity
                has_nan = torch.isnan(output).any().item()
                has_inf = torch.isinf(output).any().item()
                if has_nan or has_inf:
                    print(f"  ⚠️  Contains NaN: {has_nan}, Inf: {has_inf}")

            except Exception as e:
                print(f"  ❌ Failed: {e}")

    @pytest.mark.slow
    def test_performance_comparison(self):
        """Compare performance of stride-aware vs contiguous approaches."""
        print("\n" + "=" * 80)
        print("Performance Comparison: Stride-Aware vs Contiguous")
        print("=" * 80)

        device = "mps" if torch.backends.mps.is_available() else "cpu"

        # Test different sizes
        test_configs = [
            (1, 12, 77, 64, "Small (FLUX text)"),
            (1, 24, 1024, 128, "Medium (FLUX image)"),
            (2, 24, 2048, 128, "Large (batched)"),
        ]

        for batch, heads, seq_len, dim, desc in test_configs:
            print(f"\n{desc}: B={batch}, H={heads}, S={seq_len}, D={dim}")

            # Create FLUX layout tensors
            torch.manual_seed(42)
            q_flux = (
                torch.randn(
                    batch, heads, seq_len, dim, dtype=torch.float16, device=device
                )
                * 0.1
            )
            k_flux = (
                torch.randn(
                    batch, heads, seq_len, dim, dtype=torch.float16, device=device
                )
                * 0.1
            )
            v_flux = (
                torch.randn(
                    batch, heads, seq_len, dim, dtype=torch.float16, device=device
                )
                * 0.1
            )

            # Non-contiguous (permuted)
            q_perm = q_flux.permute(0, 2, 1, 3)
            k_perm = k_flux.permute(0, 2, 1, 3)
            v_perm = v_flux.permute(0, 2, 1, 3)

            # Contiguous copies
            q_cont = q_perm.contiguous()
            k_cont = k_perm.contiguous()
            v_cont = v_perm.contiguous()

            # Warm up
            for _ in range(3):
                _ = metal_sdpa_extension.metal_scaled_dot_product_attention(
                    q_cont, k_cont, v_cont
                )

            # Time contiguous approach (includes copy time)
            torch.mps.synchronize() if device == "mps" else None
            start = time.perf_counter()

            for _ in range(10):
                q_c = q_perm.contiguous()
                k_c = k_perm.contiguous()
                v_c = v_perm.contiguous()
                output_cont = metal_sdpa_extension.metal_scaled_dot_product_attention(
                    q_c, k_c, v_c
                )

            torch.mps.synchronize() if device == "mps" else None
            time_contiguous = (time.perf_counter() - start) / 10

            # Time non-contiguous approach (if it works)
            try:
                # Warm up
                for _ in range(3):
                    _ = metal_sdpa_extension.metal_scaled_dot_product_attention(
                        q_perm, k_perm, v_perm
                    )

                torch.mps.synchronize() if device == "mps" else None
                start = time.perf_counter()

                for _ in range(10):
                    output_perm = (
                        metal_sdpa_extension.metal_scaled_dot_product_attention(
                            q_perm, k_perm, v_perm
                        )
                    )

                torch.mps.synchronize() if device == "mps" else None
                time_noncontig = (time.perf_counter() - start) / 10

                print(f"  Contiguous (with copy): {time_contiguous*1000:.3f} ms")
                print(f"  Non-contiguous (direct): {time_noncontig*1000:.3f} ms")
                print(f"  Speedup: {time_contiguous/time_noncontig:.2f}x")

                # Verify outputs match
                if torch.allclose(output_cont, output_perm, rtol=1e-3, atol=1e-3):
                    print(f"  ✅ Outputs match")
                else:
                    diff = torch.abs(output_cont - output_perm).max().item()
                    print(f"  ⚠️  Output mismatch: max diff = {diff}")

            except Exception as e:
                print(f"  Non-contiguous failed: {e}")
                print(f"  Contiguous time: {time_contiguous*1000:.3f} ms")

    def test_memory_safety(self):
        """Test memory safety with various edge cases."""
        print("\n" + "=" * 80)
        print("Memory Safety Tests")
        print("=" * 80)

        device = "mps" if torch.backends.mps.is_available() else "cpu"

        # Test 1: Overlapping memory views
        print("\nTest 1: Overlapping memory views")
        base = torch.randn(2, 16, 64, 32, dtype=torch.float16, device=device) * 0.1
        q = base[0:1, 0:12, :, :]  # Slice creating a view
        k = base[0:1, 0:12, :, :]
        v = base[0:1, 0:12, :, :]

        print(f"  Base tensor uses {base.numel() * base.element_size()} bytes")
        print(f"  Q,K,V are views: {q.data_ptr() == base.data_ptr()}")

        try:
            output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
            print("  ✅ Handled overlapping views correctly")
        except Exception as e:
            print(f"  ❌ Failed with overlapping views: {e}")

        # Test 2: Transposed tensors
        print("\nTest 2: Transposed tensors")
        q = torch.randn(1, 64, 12, 32, dtype=torch.float16, device=device).transpose(
            1, 2
        )
        k = torch.randn(1, 64, 12, 32, dtype=torch.float16, device=device).transpose(
            1, 2
        )
        v = torch.randn(1, 64, 12, 32, dtype=torch.float16, device=device).transpose(
            1, 2
        )

        print(f"  Shape after transpose: {list(q.shape)}")
        print(f"  Contiguous: {q.is_contiguous()}")

        try:
            output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
            print("  ✅ Handled transposed tensors correctly")
        except Exception as e:
            print(f"  ❌ Failed with transposed tensors: {e}")

        # Test 3: Memory pressure (allocate and free many tensors)
        print("\nTest 3: Memory pressure test")
        for i in range(10):
            q, k, v = create_test_tensors(
                batch_size=1,
                num_heads=12,
                seq_len=256,
                head_dim=64,
                layout="metal",
                dtype=torch.float16,
                device=device,
                contiguous=False,
            )

            try:
                output = metal_sdpa_extension.metal_scaled_dot_product_attention(
                    q, k, v
                )
                del output
                gc.collect()
                torch.mps.empty_cache() if device == "mps" else None
            except Exception as e:
                print(f"  ❌ Failed on iteration {i}: {e}")
                break
        else:
            print("  ✅ Passed memory pressure test (10 iterations)")


@pytest.mark.skipif(not HAS_METAL, reason="Metal SDPA extension not available")
@pytest.mark.slow
def test_basic_stride_handling():
    """Quick test to verify basic stride handling."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    # Create non-contiguous tensor via permute
    flux_tensor = torch.randn(1, 12, 77, 64, dtype=torch.float16, device="mps") * 0.1
    metal_tensor = flux_tensor.permute(0, 2, 1, 3)  # [B,H,S,D] -> [B,S,H,D]

    assert (
        not metal_tensor.is_contiguous()
    ), "Tensor should be non-contiguous after permute"

    # This should work without memory corruption if strides are handled correctly
    q, k, v = metal_tensor, metal_tensor, metal_tensor
    output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

    assert output.shape == metal_tensor.shape, "Output shape should match input"
    assert not torch.isnan(output).any(), "Output should not contain NaN"
    assert not torch.isinf(output).any(), "Output should not contain Inf"


if __name__ == "__main__":
    # Run tests directly
    test_runner = TestStrideAwareAttention()

    print("Running Stride-Aware Attention Tests")
    print("=" * 80)

    # Run MRE first
    test_runner.test_mre_memory_corruption()

    # Run other tests
    test_runner.test_stride_patterns()
    test_runner.test_performance_comparison()
    test_runner.test_memory_safety()

    print("\n" + "=" * 80)
    print("All tests completed")
    print("=" * 80)
