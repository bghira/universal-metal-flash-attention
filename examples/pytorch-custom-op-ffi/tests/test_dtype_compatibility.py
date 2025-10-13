"""
Tests for dtype compatibility and conversions in Metal SDPA.

This test suite specifically addresses dtype mismatches that can cause
MPS errors like "Destination NDArray and Accumulator NDArray cannot have
different datatype".
"""

import metal_sdpa_extension
import pytest
import torch


@pytest.mark.metal
@pytest.mark.dtype
class TestDtypeCompatibility:
    """Test dtype handling and compatibility."""

    def test_bfloat16_compatibility(self, metal_device, create_test_tensors):
        """Test BFloat16 tensor processing - critical for FLUX."""
        if not hasattr(torch, "bfloat16"):
            pytest.skip("BFloat16 not available")

        # Create BF16 tensors like FLUX uses
        q, k, v = create_test_tensors(
            batch_size=1,
            num_heads=12,
            seq_len=77,
            head_dim=64,
            dtype=torch.bfloat16,
            device=metal_device,
            layout="flux",
        )

        # This should not crash with MPS dtype mismatch
        try:
            output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

            # Check output properties
            assert (
                output.dtype == torch.bfloat16
            ), f"Expected BF16 output, got {output.dtype}"
            assert (
                output.shape == q.shape
            ), f"Output shape mismatch: {output.shape} vs {q.shape}"
            assert not torch.isnan(output).any(), "Output contains NaN"
            assert not torch.isinf(output).any(), "Output contains Inf"

        except RuntimeError as e:
            if "Destination NDArray and Accumulator NDArray" in str(e):
                pytest.fail(f"MPS dtype mismatch error: {e}")
            raise

    def test_float16_compatibility(self, metal_device, create_test_tensors):
        """Test Float16 tensor processing."""
        q, k, v = create_test_tensors(
            batch_size=1,
            num_heads=8,
            seq_len=128,
            head_dim=64,
            dtype=torch.float16,
            device=metal_device,
            layout="flux",
        )

        output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

        assert (
            output.dtype == torch.float16
        ), f"Expected FP16 output, got {output.dtype}"
        assert output.shape == q.shape, f"Output shape mismatch"
        assert not torch.isnan(output).any(), "Output contains NaN"

    def test_float32_compatibility(self, metal_device, create_test_tensors):
        """Test Float32 tensor processing."""
        q, k, v = create_test_tensors(
            batch_size=1,
            num_heads=4,
            seq_len=64,
            head_dim=64,
            dtype=torch.float32,
            device=metal_device,
            layout="flux",
        )

        output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

        assert (
            output.dtype == torch.float32
        ), f"Expected FP32 output, got {output.dtype}"
        assert output.shape == q.shape, f"Output shape mismatch"
        assert not torch.isnan(output).any(), "Output contains NaN"

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
    def test_dtype_preservation(self, metal_device, dtype):
        """Test that output dtype matches input dtype."""
        if dtype == torch.bfloat16 and not hasattr(torch, "bfloat16"):
            pytest.skip("BFloat16 not available")

        # Small tensors for quick testing
        q = torch.randn(1, 1, 32, 32, dtype=dtype, device=metal_device) * 0.1
        k = torch.randn(1, 1, 32, 32, dtype=dtype, device=metal_device) * 0.1
        v = torch.randn(1, 1, 32, 32, dtype=dtype, device=metal_device) * 0.1

        output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

        assert (
            output.dtype == dtype
        ), f"Dtype not preserved: input={dtype}, output={output.dtype}"

    def test_mixed_precision_error_handling(self, metal_device):
        """Test handling of mixed precision inputs (should either work or fail gracefully)."""
        q = torch.randn(1, 4, 64, 64, dtype=torch.float32, device=metal_device) * 0.1
        k = torch.randn(1, 4, 64, 64, dtype=torch.float16, device=metal_device) * 0.1
        v = torch.randn(1, 4, 64, 64, dtype=torch.float16, device=metal_device) * 0.1

        # This should either:
        # 1. Work with automatic type promotion
        # 2. Fail with a clear error message (not MPS internal error)
        try:
            output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
            # If it works, output should be the higher precision dtype
            assert output.dtype in [torch.float32, q.dtype]
        except (RuntimeError, TypeError) as e:
            # Should give a clear error, not MPS internal assertion
            assert "Destination NDArray and Accumulator" not in str(
                e
            ), f"Got MPS internal error instead of clear message: {e}"

    @pytest.mark.flux
    def test_flux_typical_dtypes(self, metal_device, create_test_tensors):
        """Test with FLUX's typical tensor configurations."""
        # FLUX typically uses BF16 for transformer blocks
        configurations = [
            # Text encoder attention
            (1, 12, 77, 64, torch.bfloat16),
            # Image transformer attention
            (1, 24, 1536, 128, torch.bfloat16),
            # Larger batch
            (2, 24, 1536, 128, torch.bfloat16),
        ]

        for batch, heads, seq_len, head_dim, dtype in configurations:
            if dtype == torch.bfloat16 and not hasattr(torch, "bfloat16"):
                continue

            q, k, v = create_test_tensors(
                batch_size=batch,
                num_heads=heads,
                seq_len=seq_len,
                head_dim=head_dim,
                dtype=dtype,
                device=metal_device,
                layout="flux",
            )

            try:
                output = metal_sdpa_extension.metal_scaled_dot_product_attention(
                    q, k, v
                )

                assert (
                    output.dtype == dtype
                ), f"Dtype mismatch for config {configurations}"
                assert (
                    output.shape == q.shape
                ), f"Shape mismatch for config {configurations}"
                assert not torch.isnan(
                    output
                ).any(), f"NaN in output for config {configurations}"

            except RuntimeError as e:
                if "Destination NDArray and Accumulator" in str(e):
                    pytest.fail(
                        f"MPS dtype error for FLUX config {configurations}: {e}"
                    )
                raise

    def test_accumulator_dtype_consistency(self, metal_device):
        """
        Test that accumulator dtype is consistent with output dtype.
        This is the core issue causing the MPS error.
        """
        # Test configurations that are likely to cause accumulator issues
        configs = [
            (torch.bfloat16, "BFloat16"),
            (torch.float16, "Float16"),
            (torch.float32, "Float32"),
        ]

        for dtype, name in configs:
            if dtype == torch.bfloat16 and not hasattr(torch, "bfloat16"):
                continue

            with self.subTest(dtype=name):
                # Create tensors that require accumulation
                q = torch.randn(1, 8, 256, 64, dtype=dtype, device=metal_device) * 0.1
                k = torch.randn(1, 8, 256, 64, dtype=dtype, device=metal_device) * 0.1
                v = torch.randn(1, 8, 256, 64, dtype=dtype, device=metal_device) * 0.1

                try:
                    output = metal_sdpa_extension.metal_scaled_dot_product_attention(
                        q, k, v
                    )

                    # Successful execution means accumulator dtype was handled correctly
                    assert (
                        output.dtype == dtype
                    ), f"Output dtype {output.dtype} doesn't match input {dtype}"

                    # Additional sanity checks
                    assert output.shape == q.shape
                    assert torch.isfinite(
                        output
                    ).all(), "Output contains non-finite values"

                except RuntimeError as e:
                    error_str = str(e)
                    if (
                        "Destination NDArray and Accumulator NDArray cannot have different datatype"
                        in error_str
                    ):
                        pytest.fail(
                            f"Accumulator dtype mismatch for {name}: {error_str}\n"
                            f"This indicates the Metal kernel is using a different dtype for "
                            f"accumulation than the output tensor."
                        )
                    else:
                        # Re-raise other errors
                        raise

    @pytest.mark.parametrize(
        "q_dtype,kv_dtype",
        [
            (torch.float32, torch.float16),
            (torch.float16, torch.float32),
            (torch.bfloat16, torch.float16),
            (torch.float16, torch.bfloat16),
        ],
    )
    def test_query_key_value_dtype_mixing(self, metal_device, q_dtype, kv_dtype):
        """Test different dtype combinations for Q, K, V."""
        if q_dtype == torch.bfloat16 or kv_dtype == torch.bfloat16:
            if not hasattr(torch, "bfloat16"):
                pytest.skip("BFloat16 not available")

        q = torch.randn(1, 4, 64, 32, dtype=q_dtype, device=metal_device) * 0.1
        k = torch.randn(1, 4, 64, 32, dtype=kv_dtype, device=metal_device) * 0.1
        v = torch.randn(1, 4, 64, 32, dtype=kv_dtype, device=metal_device) * 0.1

        # Should either handle mixed dtypes or fail with clear message
        try:
            output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
            # If successful, check output dtype is reasonable
            assert output.dtype in [
                q_dtype,
                kv_dtype,
                torch.float32,
            ], f"Unexpected output dtype: {output.dtype}"
        except (RuntimeError, TypeError) as e:
            # Check for clear error message, not internal MPS error
            error_str = str(e)
            assert (
                "Destination NDArray and Accumulator" not in error_str
            ), f"Internal MPS error instead of clear dtype mismatch message: {error_str}"
            # Expected to have a message about dtype mismatch
            assert any(
                word in error_str.lower() for word in ["dtype", "type", "mismatch"]
            ), f"Error message doesn't clearly indicate dtype issue: {error_str}"


class TestDtypeConversions:
    """Test automatic dtype conversions and promotions."""

    @pytest.mark.metal
    def test_cpu_to_mps_dtype_preservation(self, metal_device):
        """Test that dtypes are preserved when moving from CPU to MPS."""
        dtypes = [torch.float32, torch.float16]
        if hasattr(torch, "bfloat16"):
            dtypes.append(torch.bfloat16)

        for dtype in dtypes:
            # Create on CPU first
            q_cpu = torch.randn(1, 2, 32, 32, dtype=dtype) * 0.1
            k_cpu = torch.randn(1, 2, 32, 32, dtype=dtype) * 0.1
            v_cpu = torch.randn(1, 2, 32, 32, dtype=dtype) * 0.1

            # Move to MPS
            q_mps = q_cpu.to(metal_device)
            k_mps = k_cpu.to(metal_device)
            v_mps = v_cpu.to(metal_device)

            # Check dtypes are preserved
            assert q_mps.dtype == dtype
            assert k_mps.dtype == dtype
            assert v_mps.dtype == dtype

            # Run through Metal SDPA
            output = metal_sdpa_extension.metal_scaled_dot_product_attention(
                q_mps, k_mps, v_mps
            )

            assert (
                output.dtype == dtype
            ), f"Output dtype {output.dtype} != input {dtype}"

    @pytest.mark.metal
    def test_non_contiguous_tensor_dtypes(self, metal_device):
        """Test dtype handling with non-contiguous tensors."""
        dtype = torch.float16

        # Create tensors and make them non-contiguous via permutation
        q = (
            torch.randn(1, 32, 4, 32, dtype=dtype, device=metal_device).permute(
                0, 2, 1, 3
            )
            * 0.1
        )
        k = (
            torch.randn(1, 32, 4, 32, dtype=dtype, device=metal_device).permute(
                0, 2, 1, 3
            )
            * 0.1
        )
        v = (
            torch.randn(1, 32, 4, 32, dtype=dtype, device=metal_device).permute(
                0, 2, 1, 3
            )
            * 0.1
        )

        assert not q.is_contiguous()
        assert not k.is_contiguous()
        assert not v.is_contiguous()

        output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

        assert output.dtype == dtype, f"Non-contiguous tensor dtype not preserved"
        assert output.shape == q.shape
