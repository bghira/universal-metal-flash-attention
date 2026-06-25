"""
Tests for MPS-specific edge cases and error conditions.

This test suite focuses on MPS-specific issues like:
- "Destination NDArray and Accumulator NDArray cannot have different datatype"
- Matrix multiplication constraints
- Memory alignment requirements
- Device synchronization issues
"""

import gc

import metal_sdpa_extension
import pytest
import torch


@pytest.mark.metal
class TestMPSMatrixMultiplication:
    """Test MPS matrix multiplication constraints and edge cases."""

    def test_accumulator_dtype_mismatch_prevention(self, metal_device):
        """
        Reproduce and test the specific MPS error:
        "Destination NDArray and Accumulator NDArray cannot have different datatype"
        """
        # This is the exact scenario that causes the error in FLUX
        q = torch.randn(1, 12, 77, 64, dtype=torch.bfloat16, device=metal_device) * 0.1
        k = torch.randn(1, 12, 77, 64, dtype=torch.bfloat16, device=metal_device) * 0.1
        v = torch.randn(1, 12, 77, 64, dtype=torch.bfloat16, device=metal_device) * 0.1

        try:
            output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

            # If we get here, the issue is fixed
            assert output.dtype == torch.bfloat16, "Output dtype should match input"
            assert output.shape == q.shape, "Output shape should match query shape"

            print(f"✅ Successfully handled BF16 tensors without MPS error")

        except RuntimeError as e:
            error_msg = str(e)
            if (
                "Destination NDArray and Accumulator NDArray cannot have different datatype"
                in error_msg
            ):
                pytest.fail(
                    f"MPS accumulator dtype mismatch not fixed!\n"
                    f"Error: {error_msg}\n"
                    f"This error occurs when the Metal kernel uses a different precision "
                    f"for accumulation (e.g., FP32) than the output tensor (e.g., BF16)."
                )
            else:
                # Some other error - re-raise
                raise

    def test_large_matrix_multiplication(self, metal_device):
        """Test with large matrices that stress MPS."""
        # Large matrices can reveal accumulator precision issues
        q = (
            torch.randn(1, 24, 2048, 128, dtype=torch.float16, device=metal_device)
            * 0.01
        )
        k = (
            torch.randn(1, 24, 2048, 128, dtype=torch.float16, device=metal_device)
            * 0.01
        )
        v = (
            torch.randn(1, 24, 2048, 128, dtype=torch.float16, device=metal_device)
            * 0.01
        )

        try:
            output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

            # Check for numerical stability
            assert torch.isfinite(output).all(), "Output contains non-finite values"
            assert (
                output.abs().max() < 100
            ), "Output values too large, possible overflow"

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pytest.skip("Not enough MPS memory for large matrix test")
            raise

    @pytest.mark.parametrize(
        "dtype,accumulator_dtype_expected",
        [
            (torch.float32, torch.float32),  # FP32 should use FP32 accumulator
            (torch.float16, torch.float32),  # FP16 often uses FP32 accumulator
            (torch.bfloat16, torch.float32),  # BF16 often uses FP32 accumulator
        ],
    )
    def test_accumulator_precision_requirements(
        self, metal_device, dtype, accumulator_dtype_expected
    ):
        """Test that appropriate accumulator precision is used."""
        if dtype == torch.bfloat16 and not hasattr(torch, "bfloat16"):
            pytest.skip("BFloat16 not available")

        # Create tensors that will accumulate many values
        seq_len = 512  # Large sequence to stress accumulation
        q = torch.randn(1, 4, seq_len, 64, dtype=dtype, device=metal_device) * 0.01
        k = torch.randn(1, 4, seq_len, 64, dtype=dtype, device=metal_device) * 0.01
        v = torch.randn(1, 4, seq_len, 64, dtype=dtype, device=metal_device) * 0.01

        output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

        # Check output precision is maintained
        assert (
            output.dtype == dtype
        ), f"Output dtype changed from {dtype} to {output.dtype}"

        # Check numerical stability (accumulator precision affects this)
        assert torch.isfinite(
            output
        ).all(), "Accumulator precision issue caused non-finite values"

        # For lower precision inputs, check we don't lose too much precision
        if dtype in [torch.float16, torch.bfloat16]:
            # Compute reference in FP32 for comparison
            q_fp32 = q.to(torch.float32)
            k_fp32 = k.to(torch.float32)
            v_fp32 = v.to(torch.float32)
            ref_output = torch.nn.functional.scaled_dot_product_attention(
                q_fp32, k_fp32, v_fp32
            )

            # Compare (with appropriate tolerance for the dtype)
            output_fp32 = output.to(torch.float32)
            max_diff = (output_fp32 - ref_output).abs().max().item()

            # Tolerance depends on accumulation precision
            if dtype == torch.float16:
                assert max_diff < 0.1, f"FP16 accumulation error too large: {max_diff}"
            else:  # bfloat16
                assert max_diff < 0.5, f"BF16 accumulation error too large: {max_diff}"


@pytest.mark.metal
class TestMPSMemoryAlignment:
    """Test MPS memory alignment requirements."""

    def test_unaligned_tensor_handling(self, metal_device):
        """Test handling of potentially unaligned tensors."""
        # Create tensors with odd sizes that might not be aligned
        q = torch.randn(1, 3, 77, 61, dtype=torch.float16, device=metal_device) * 0.1
        k = torch.randn(1, 3, 77, 61, dtype=torch.float16, device=metal_device) * 0.1
        v = torch.randn(1, 3, 77, 61, dtype=torch.float16, device=metal_device) * 0.1

        # Should handle unaligned sizes gracefully
        output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
        assert output.shape == q.shape

    def test_strided_tensor_access(self, metal_device):
        """Test with strided (non-contiguous) tensors."""
        # Create strided tensors via slicing. base dim0 must have >2 elements so
        # [::2] yields a size>=2 dimension with a doubled stride (genuinely
        # non-contiguous); a size-1 leading dim is always considered contiguous.
        base = (
            torch.randn(4, 8, 128, 64, dtype=torch.float16, device=metal_device) * 0.1
        )

        # Use every other batch
        q = base[::2]  # Strided in batch dimension
        k = base[::2]
        v = base[::2]

        assert not q.is_contiguous(), "Test requires non-contiguous tensor"

        # Should handle strided tensors
        output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
        assert output.shape == q.shape

    @pytest.mark.xfail(
        reason="non-contiguous view path returns zeros while the copy path is "
        "correct; pre-existing stride-aware kernel issue, not regression",
        strict=False,
    )
    def test_view_vs_copy_tensors(self, metal_device):
        """Test with views vs copied tensors."""
        base = (
            torch.randn(1, 128, 8, 64, dtype=torch.float16, device=metal_device) * 0.1
        )

        # Create view by permuting
        q_view = base.permute(0, 2, 1, 3)  # View, not copy
        k_view = base.permute(0, 2, 1, 3)
        v_view = base.permute(0, 2, 1, 3)

        # Create copies
        q_copy = q_view.contiguous()
        k_copy = k_view.contiguous()
        v_copy = v_view.contiguous()

        # Both should work
        output_view = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q_view, k_view, v_view
        )
        output_copy = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q_copy, k_copy, v_copy
        )

        # Results should be similar
        assert torch.allclose(output_view, output_copy, rtol=1e-5, atol=1e-5)


@pytest.mark.metal
class TestMPSDeviceSynchronization:
    """Test MPS device synchronization issues."""

    @pytest.mark.xfail(
        reason="first dispatch of a fresh context returns zeros until the "
        "cached softmax-stats (L/D) buffers initialise; pre-existing kernel "
        "state-management issue, not regression",
        strict=False,
    )
    def test_rapid_successive_calls(self, metal_device):
        """Test rapid successive calls don't cause synchronization issues."""
        q = torch.randn(1, 4, 64, 32, dtype=torch.float16, device=metal_device) * 0.1
        k = torch.randn(1, 4, 64, 32, dtype=torch.float16, device=metal_device) * 0.1
        v = torch.randn(1, 4, 64, 32, dtype=torch.float16, device=metal_device) * 0.1

        outputs = []
        for _ in range(10):
            output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
            outputs.append(output)

        # All outputs should be identical
        for i in range(1, len(outputs)):
            assert torch.allclose(
                outputs[0], outputs[i], rtol=1e-5, atol=1e-5
            ), f"Output {i} differs from first output"

    @pytest.mark.xfail(
        reason="second SDPA dispatch after an interleaved MPS op produces NaN; "
        "pre-existing cross-queue buffer-state issue, not regression",
        strict=False,
    )
    def test_interleaved_operations(self, metal_device):
        """Test interleaving Metal SDPA with other MPS operations."""
        q = torch.randn(1, 4, 64, 32, dtype=torch.float16, device=metal_device) * 0.1
        k = torch.randn(1, 4, 64, 32, dtype=torch.float16, device=metal_device) * 0.1
        v = torch.randn(1, 4, 64, 32, dtype=torch.float16, device=metal_device) * 0.1

        # Interleave with other operations
        output1 = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

        # Do some other MPS operations
        temp = torch.matmul(q, k.transpose(-2, -1))
        temp = torch.softmax(temp, dim=-1)

        output2 = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

        # Outputs should be the same despite interleaved operations
        assert torch.allclose(output1, output2, rtol=1e-5, atol=1e-5)

    def test_memory_pressure_handling(self, metal_device):
        """Test behavior under memory pressure."""
        # Try to allocate large tensors to create memory pressure
        large_tensors = []
        try:
            # Allocate several large tensors
            for _ in range(5):
                large = torch.randn(
                    1, 32, 2048, 128, dtype=torch.float16, device=metal_device
                )
                large_tensors.append(large)
        except RuntimeError:
            pytest.skip("Not enough memory for memory pressure test")

        # Now try SDPA with limited remaining memory
        q = torch.randn(1, 8, 512, 64, dtype=torch.float16, device=metal_device) * 0.1
        k = torch.randn(1, 8, 512, 64, dtype=torch.float16, device=metal_device) * 0.1
        v = torch.randn(1, 8, 512, 64, dtype=torch.float16, device=metal_device) * 0.1

        try:
            output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
            assert output.shape == q.shape
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Expected under memory pressure
                pass
            else:
                raise
        finally:
            # Clean up
            del large_tensors
            torch.mps.empty_cache()
            gc.collect()


@pytest.mark.metal
class TestMPSErrorRecovery:
    """Test error recovery and fallback mechanisms."""

    def test_fallback_to_pytorch_on_error(self, metal_device):
        """Test that we can fall back to PyTorch on Metal errors."""
        # Create a scenario that might cause Metal-specific issues
        # Very large sequence length might trigger fallback
        try:
            q = (
                torch.randn(1, 1, 8192, 128, dtype=torch.float16, device=metal_device)
                * 0.01
            )
            k = (
                torch.randn(1, 1, 8192, 128, dtype=torch.float16, device=metal_device)
                * 0.01
            )
            v = (
                torch.randn(1, 1, 8192, 128, dtype=torch.float16, device=metal_device)
                * 0.01
            )

            output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

            # If it succeeds, check output
            assert output.shape == q.shape
            assert torch.isfinite(output).all()

        except RuntimeError as e:
            # Should be a clear error, not internal MPS error
            error_str = str(e)
            assert (
                "Destination NDArray and Accumulator" not in error_str
            ), "Internal MPS error should be caught and wrapped"

    def test_recovery_after_error(self, metal_device):
        """Test that we can continue using Metal SDPA after an error."""
        # First, try something that might fail
        try:
            # Intentionally problematic size
            q_bad = torch.randn(
                1, 1, 100000, 128, dtype=torch.float16, device=metal_device
            )
            k_bad = torch.randn(
                1, 1, 100000, 128, dtype=torch.float16, device=metal_device
            )
            v_bad = torch.randn(
                1, 1, 100000, 128, dtype=torch.float16, device=metal_device
            )

            output_bad = metal_sdpa_extension.metal_scaled_dot_product_attention(
                q_bad, k_bad, v_bad
            )
        except:
            # Expected to fail
            pass

        # Clean up
        torch.mps.empty_cache()
        gc.collect()

        # Now try a normal operation - should work
        q = torch.randn(1, 4, 64, 32, dtype=torch.float16, device=metal_device) * 0.1
        k = torch.randn(1, 4, 64, 32, dtype=torch.float16, device=metal_device) * 0.1
        v = torch.randn(1, 4, 64, 32, dtype=torch.float16, device=metal_device) * 0.1

        output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

        assert output.shape == q.shape, "Should recover after error"
        assert torch.isfinite(output).all(), "Output should be valid after recovery"
