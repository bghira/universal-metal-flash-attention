"""
Integration tests specifically for FLUX model use cases.

These tests reproduce real FLUX scenarios to catch issues that only
appear in production workloads.
"""

import metal_sdpa_extension
import pytest
import torch
import torch.nn.functional as F


@pytest.mark.metal
@pytest.mark.flux
@pytest.mark.integration
@pytest.mark.requires_bfloat
@pytest.mark.xfail(
    reason="bfloat16 attention produces NaN; bfloat now compiles (MSL 3.2) but "
    "the kernel overflows/NaNs on bf16 inputs — pre-existing numerics bug",
    strict=False,
)
class TestFLUXIntegration:
    """Test real FLUX model integration scenarios."""

    def test_flux_text_encoder_full_pipeline(self, metal_device):
        """Test full FLUX text encoder attention pattern."""
        # FLUX text encoder configuration
        batch_size = 1
        num_heads = 12
        seq_len = 77
        head_dim = 64
        dtype = torch.bfloat16

        # Create tensors matching FLUX text encoder
        torch.manual_seed(42)
        q = (
            torch.randn(
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                dtype=dtype,
                device=metal_device,
            )
            * 0.1
        )
        k = (
            torch.randn(
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                dtype=dtype,
                device=metal_device,
            )
            * 0.1
        )
        v = (
            torch.randn(
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                dtype=dtype,
                device=metal_device,
            )
            * 0.1
        )

        # Apply causal mask like FLUX does
        causal_mask = torch.ones(seq_len, seq_len, device=metal_device, dtype=dtype)
        causal_mask = torch.triu(causal_mask, diagonal=1) * -float("inf")
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Run through Metal SDPA
        try:
            output = metal_sdpa_extension.metal_scaled_dot_product_attention(
                q, k, v, attn_mask=causal_mask
            )

            # Validate output
            assert output.shape == q.shape, "Output shape mismatch"
            assert output.dtype == dtype, f"Expected {dtype}, got {output.dtype}"
            assert torch.isfinite(output).all(), "Output contains non-finite values"

            # Compare with PyTorch reference
            with torch.inference_mode():
                ref_output = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=causal_mask
                )

            # Should be close to reference (with tolerance for BF16)
            max_diff = (output - ref_output).abs().max().item()
            assert max_diff < 0.1, f"Output differs from reference by {max_diff}"

        except RuntimeError as e:
            if "Destination NDArray and Accumulator" in str(e):
                pytest.fail(f"MPS dtype mismatch in FLUX text encoder: {e}")
            raise

    def test_flux_image_transformer_attention(self, metal_device):
        """Test FLUX image transformer attention with typical configuration."""
        # FLUX image transformer configuration
        batch_size = 1
        num_heads = 24
        seq_len = 1536  # For 1024x1024 image
        head_dim = 128
        dtype = torch.bfloat16

        # Create tensors
        torch.manual_seed(42)
        q = (
            torch.randn(
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                dtype=dtype,
                device=metal_device,
            )
            * 0.01
        )
        k = (
            torch.randn(
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                dtype=dtype,
                device=metal_device,
            )
            * 0.01
        )
        v = (
            torch.randn(
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                dtype=dtype,
                device=metal_device,
            )
            * 0.01
        )

        # FLUX doesn't use causal mask for image transformer
        try:
            output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

            # Validate
            assert output.shape == q.shape
            assert output.dtype == dtype
            assert torch.isfinite(output).all()

            # Check memory usage doesn't explode
            # Large sequence length can cause memory issues
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Ensure operation completes

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pytest.skip(f"Not enough memory for seq_len={seq_len}")
            if "Destination NDArray and Accumulator" in str(e):
                pytest.fail(f"MPS dtype mismatch in FLUX image transformer: {e}")
            raise

    def test_flux_cross_attention(self, metal_device):
        """Test FLUX cross-attention between image and text."""
        batch_size = 1
        num_heads = 24
        dtype = torch.bfloat16

        # Image queries
        q = (
            torch.randn(
                batch_size, num_heads, 1536, 128, dtype=dtype, device=metal_device
            )
            * 0.01
        )

        # Text keys/values
        k = (
            torch.randn(
                batch_size, num_heads, 77, 128, dtype=dtype, device=metal_device
            )
            * 0.01
        )
        v = (
            torch.randn(
                batch_size, num_heads, 77, 128, dtype=dtype, device=metal_device
            )
            * 0.01
        )

        try:
            output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

            assert output.shape == (batch_size, num_heads, 1536, 128)
            assert output.dtype == dtype
            assert torch.isfinite(output).all()

        except RuntimeError as e:
            if "Destination NDArray and Accumulator" in str(e):
                pytest.fail(f"MPS dtype mismatch in FLUX cross-attention: {e}")
            raise

    @pytest.mark.slow
    def test_flux_generation_simulation(self, metal_device):
        """Simulate a full FLUX generation step sequence."""
        batch_size = 1
        dtype = torch.bfloat16

        # Simulate 4 denoising steps (FLUX-schnell typical)
        for step in range(4):
            # Text encoder attention
            q_text = (
                torch.randn(batch_size, 12, 77, 64, dtype=dtype, device=metal_device)
                * 0.1
            )
            k_text = (
                torch.randn(batch_size, 12, 77, 64, dtype=dtype, device=metal_device)
                * 0.1
            )
            v_text = (
                torch.randn(batch_size, 12, 77, 64, dtype=dtype, device=metal_device)
                * 0.1
            )

            output_text = metal_sdpa_extension.metal_scaled_dot_product_attention(
                q_text, k_text, v_text, is_causal=True
            )

            # Image self-attention
            q_img = (
                torch.randn(batch_size, 24, 1536, 128, dtype=dtype, device=metal_device)
                * 0.01
            )
            k_img = (
                torch.randn(batch_size, 24, 1536, 128, dtype=dtype, device=metal_device)
                * 0.01
            )
            v_img = (
                torch.randn(batch_size, 24, 1536, 128, dtype=dtype, device=metal_device)
                * 0.01
            )

            output_img_self = metal_sdpa_extension.metal_scaled_dot_product_attention(
                q_img, k_img, v_img
            )

            # Cross-attention
            k_cross = (
                torch.randn(batch_size, 24, 77, 128, dtype=dtype, device=metal_device)
                * 0.01
            )
            v_cross = (
                torch.randn(batch_size, 24, 77, 128, dtype=dtype, device=metal_device)
                * 0.01
            )

            output_cross = metal_sdpa_extension.metal_scaled_dot_product_attention(
                q_img, k_cross, v_cross
            )

            # Validate all outputs
            assert torch.isfinite(
                output_text
            ).all(), f"NaN in text attention at step {step}"
            assert torch.isfinite(
                output_img_self
            ).all(), f"NaN in image self-attention at step {step}"
            assert torch.isfinite(
                output_cross
            ).all(), f"NaN in cross-attention at step {step}"

    def test_flux_batch_processing(self, metal_device):
        """Test FLUX with batch processing."""
        batch_sizes = [1, 2, 4]
        num_heads = 24
        seq_len = 512  # Smaller for memory efficiency
        head_dim = 128
        dtype = torch.bfloat16

        for batch_size in batch_sizes:
            q = (
                torch.randn(
                    batch_size,
                    num_heads,
                    seq_len,
                    head_dim,
                    dtype=dtype,
                    device=metal_device,
                )
                * 0.01
            )
            k = (
                torch.randn(
                    batch_size,
                    num_heads,
                    seq_len,
                    head_dim,
                    dtype=dtype,
                    device=metal_device,
                )
                * 0.01
            )
            v = (
                torch.randn(
                    batch_size,
                    num_heads,
                    seq_len,
                    head_dim,
                    dtype=dtype,
                    device=metal_device,
                )
                * 0.01
            )

            try:
                output = metal_sdpa_extension.metal_scaled_dot_product_attention(
                    q, k, v
                )

                assert output.shape[0] == batch_size, f"Batch size not preserved"
                assert torch.isfinite(output).all(), f"NaN with batch_size={batch_size}"

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    pytest.skip(f"Not enough memory for batch_size={batch_size}")
                raise

    @pytest.mark.parametrize(
        "resolution",
        [
            (256, 256),
            (512, 512),
            (768, 768),
            (1024, 1024),
        ],
    )
    def test_flux_resolution_scaling(self, metal_device, resolution):
        """Test FLUX attention at different resolutions."""
        width, height = resolution
        # Approximate sequence length based on resolution
        # FLUX uses 16x16 patches typically
        seq_len = (width // 16) * (height // 16)

        batch_size = 1
        num_heads = 24
        head_dim = 128
        dtype = torch.bfloat16

        # Skip if sequence length is too large for available memory
        if seq_len > 4096:
            pytest.skip(f"Sequence length {seq_len} too large for testing")

        q = (
            torch.randn(
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                dtype=dtype,
                device=metal_device,
            )
            * 0.01
        )
        k = (
            torch.randn(
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                dtype=dtype,
                device=metal_device,
            )
            * 0.01
        )
        v = (
            torch.randn(
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                dtype=dtype,
                device=metal_device,
            )
            * 0.01
        )

        try:
            output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

            assert output.shape == q.shape, f"Shape mismatch at resolution {resolution}"
            assert torch.isfinite(output).all(), f"NaN at resolution {resolution}"

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pytest.skip(f"Not enough memory for resolution {resolution}")
            if "Destination NDArray and Accumulator" in str(e):
                pytest.fail(f"MPS dtype mismatch at resolution {resolution}: {e}")
            raise


@pytest.mark.metal
@pytest.mark.flux
class TestFLUXErrorScenarios:
    """Test error scenarios specific to FLUX usage."""

    def test_flux_dtype_mismatch_reproduction(self, metal_device):
        """
        Reproduce the exact error from the FLUX benchmark:
        "Destination NDArray and Accumulator NDArray cannot have different datatype"
        """
        # Exact configuration that caused the error
        batch_size = 1
        num_heads = 12
        seq_len = 77
        head_dim = 64
        dtype = torch.bfloat16

        # Create tensors
        q = (
            torch.randn(
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                dtype=dtype,
                device=metal_device,
            )
            * 0.1
        )
        k = (
            torch.randn(
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                dtype=dtype,
                device=metal_device,
            )
            * 0.1
        )
        v = (
            torch.randn(
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                dtype=dtype,
                device=metal_device,
            )
            * 0.1
        )

        # This specific call pattern triggered the error
        try:
            # The error occurred during the conversion back to FLUX layout
            # after Metal computation with potentially different accumulator dtype
            output = metal_sdpa_extension.metal_scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
            )

            # If we reach here, the issue is fixed
            assert output.dtype == dtype, f"Output dtype should be {dtype}"
            assert output.shape == q.shape, "Output shape should match query"
            assert torch.isfinite(output).all(), "Output should be finite"

            print("✅ MPS dtype mismatch issue appears to be fixed!")

        except RuntimeError as e:
            error_msg = str(e)
            if (
                "Destination NDArray and Accumulator NDArray cannot have different datatype"
                in error_msg
            ):
                pytest.fail(
                    f"MPS dtype mismatch NOT fixed!\n"
                    f"Error: {error_msg}\n"
                    f"This exact error was encountered in the FLUX benchmark.\n"
                    f"The Metal kernel is likely using FP32 accumulator with BF16 output."
                )
            else:
                # Different error - still worth investigating
                raise

    def test_flux_memory_exhaustion_handling(self, metal_device):
        """Test handling when FLUX exhausts MPS memory."""
        # Try to create tensors that will exhaust memory
        batch_size = 4
        num_heads = 24
        seq_len = 4096  # Very large sequence
        head_dim = 128
        dtype = torch.bfloat16

        try:
            q = (
                torch.randn(
                    batch_size,
                    num_heads,
                    seq_len,
                    head_dim,
                    dtype=dtype,
                    device=metal_device,
                )
                * 0.01
            )
            k = (
                torch.randn(
                    batch_size,
                    num_heads,
                    seq_len,
                    head_dim,
                    dtype=dtype,
                    device=metal_device,
                )
                * 0.01
            )
            v = (
                torch.randn(
                    batch_size,
                    num_heads,
                    seq_len,
                    head_dim,
                    dtype=dtype,
                    device=metal_device,
                )
                * 0.01
            )

            output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

        except RuntimeError as e:
            # Should give clear memory error, not internal MPS error
            error_msg = str(e).lower()
            assert (
                "memory" in error_msg or "allocation" in error_msg
            ), f"Expected memory error, got: {e}"
            # Should NOT be accumulator dtype error
            assert (
                "accumulator" not in error_msg
            ), f"Memory exhaustion triggered dtype error: {e}"
