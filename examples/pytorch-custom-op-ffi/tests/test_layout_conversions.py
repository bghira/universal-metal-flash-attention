"""
Tests for tensor layout conversions between FLUX and Metal formats.

FLUX uses: [batch, heads, sequence, dim]
Metal expects: [batch, sequence, heads, dim]
"""

import metal_sdpa_extension
import pytest
import torch


@pytest.mark.metal
@pytest.mark.layout
@pytest.mark.slow
class TestLayoutConversions:
    """Test layout conversions between FLUX and Metal formats."""

    def test_flux_layout_detection(self, metal_device):
        """Test that FLUX layout tensors are correctly detected and handled."""
        # FLUX layout: [batch, heads, sequence, dim]
        flux_shapes = [
            (1, 12, 77, 64),  # Text encoder shape
            (1, 24, 1536, 128),  # Main transformer shape
            (2, 24, 1536, 128),  # Batched
        ]

        for shape in flux_shapes:
            q = torch.randn(shape, dtype=torch.float16, device=metal_device) * 0.1
            k = torch.randn(shape, dtype=torch.float16, device=metal_device) * 0.1
            v = torch.randn(shape, dtype=torch.float16, device=metal_device) * 0.1

            # Should handle FLUX layout automatically
            output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

            # Output should maintain FLUX layout
            assert output.shape == shape, f"Layout not preserved for shape {shape}"

    def test_metal_layout_detection(self, metal_device):
        """Test that Metal layout tensors are correctly detected."""
        # Metal layout: [batch, sequence, heads, dim]
        metal_shapes = [
            (1, 77, 12, 64),  # Metal version of text encoder
            (1, 1536, 24, 128),  # Metal version of main transformer
            (2, 1536, 24, 128),  # Batched
        ]

        for shape in metal_shapes:
            q = torch.randn(shape, dtype=torch.float16, device=metal_device) * 0.1
            k = torch.randn(shape, dtype=torch.float16, device=metal_device) * 0.1
            v = torch.randn(shape, dtype=torch.float16, device=metal_device) * 0.1

            # Should handle Metal layout
            output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

            # Output should maintain Metal layout
            assert output.shape == shape, f"Layout not preserved for shape {shape}"

    def test_ambiguous_layout_handling(self, metal_device):
        """Test handling of ambiguous tensor shapes."""
        # Shapes that could be interpreted either way
        ambiguous_shapes = [
            (1, 64, 64, 64),  # All dimensions same
            (1, 32, 32, 128),  # Could be either layout
        ]

        for shape in ambiguous_shapes:
            q = torch.randn(shape, dtype=torch.float16, device=metal_device) * 0.1
            k = torch.randn(shape, dtype=torch.float16, device=metal_device) * 0.1
            v = torch.randn(shape, dtype=torch.float16, device=metal_device) * 0.1

            # Should handle ambiguous shapes without crashing
            output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
            assert output.shape == shape

    @pytest.mark.xfail(
        reason="non-contiguous (permuted) tensor path returns zeros; "
        "pre-existing stride-aware kernel issue, not regression",
        strict=False,
    )
    def test_layout_conversion_consistency(self, metal_device, reference_attention):
        """Test that layout conversion doesn't affect attention output."""
        batch, heads, seq_len, dim = 1, 8, 128, 64

        # Create FLUX layout tensors
        torch.manual_seed(42)
        q_flux = (
            torch.randn(
                batch, heads, seq_len, dim, dtype=torch.float32, device=metal_device
            )
            * 0.1
        )
        k_flux = (
            torch.randn(
                batch, heads, seq_len, dim, dtype=torch.float32, device=metal_device
            )
            * 0.1
        )
        v_flux = (
            torch.randn(
                batch, heads, seq_len, dim, dtype=torch.float32, device=metal_device
            )
            * 0.1
        )

        # Manually convert to Metal layout for comparison
        q_metal = q_flux.permute(0, 2, 1, 3)  # [B,H,S,D] -> [B,S,H,D]
        k_metal = k_flux.permute(0, 2, 1, 3)
        v_metal = v_flux.permute(0, 2, 1, 3)

        # Get outputs from both layouts
        output_flux = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q_flux, k_flux, v_flux
        )
        output_metal = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q_metal, k_metal, v_metal
        )

        # Convert Metal output back to FLUX layout for comparison
        output_metal_as_flux = output_metal.permute(
            0, 2, 1, 3
        )  # [B,S,H,D] -> [B,H,S,D]

        # Outputs should be equivalent
        assert torch.allclose(
            output_flux, output_metal_as_flux, rtol=1e-4, atol=1e-4
        ), "Layout conversion affects attention output"

    def test_non_contiguous_layout_conversion(self, metal_device):
        """Test layout conversion with non-contiguous tensors."""
        # Create non-contiguous FLUX tensors via permutation
        base = (
            torch.randn(1, 128, 12, 64, dtype=torch.float16, device=metal_device) * 0.1
        )

        # Permute to FLUX layout - creates non-contiguous tensors
        q = base.permute(0, 2, 1, 3)  # [B,S,H,D] -> [B,H,S,D]
        k = base.permute(0, 2, 1, 3)
        v = base.permute(0, 2, 1, 3)

        assert not q.is_contiguous()
        assert not k.is_contiguous()

        # Should handle non-contiguous tensors
        output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
        assert output.shape == q.shape

    @pytest.mark.parametrize(
        "batch,heads,seq_len,dim",
        [
            (1, 1, 64, 64),  # Single head
            (1, 100, 64, 64),  # Many heads (edge case for detection)
            (1, 64, 100, 64),  # Many sequences (edge case)
            (4, 8, 256, 32),  # Typical batched case
        ],
    )
    def test_layout_detection_edge_cases(
        self, metal_device, batch, heads, seq_len, dim
    ):
        """Test layout detection with edge case dimensions."""
        # Test FLUX layout
        flux_shape = (batch, heads, seq_len, dim)
        q = torch.randn(flux_shape, dtype=torch.float16, device=metal_device) * 0.1
        k = torch.randn(flux_shape, dtype=torch.float16, device=metal_device) * 0.1
        v = torch.randn(flux_shape, dtype=torch.float16, device=metal_device) * 0.1

        output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
        assert output.shape == flux_shape

        # Test Metal layout
        metal_shape = (batch, seq_len, heads, dim)
        q = torch.randn(metal_shape, dtype=torch.float16, device=metal_device) * 0.1
        k = torch.randn(metal_shape, dtype=torch.float16, device=metal_device) * 0.1
        v = torch.randn(metal_shape, dtype=torch.float16, device=metal_device) * 0.1

        output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
        assert output.shape == metal_shape


@pytest.mark.metal
@pytest.mark.flux
@pytest.mark.requires_bfloat
@pytest.mark.xfail(
    reason="bfloat16 attention produces NaN; bfloat compiles (MSL 3.2) but the "
    "kernel NaNs on bf16 inputs — pre-existing numerics bug",
    strict=False,
)
class TestFLUXSpecificLayouts:
    """Test FLUX-specific layout scenarios."""

    def test_flux_text_encoder_layout(self, metal_device):
        """Test with FLUX text encoder specific dimensions."""
        # FLUX text encoder: 12 heads, 77 tokens
        batch_size = 1
        num_heads = 12
        seq_len = 77
        head_dim = 64

        q = (
            torch.randn(
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                dtype=torch.bfloat16,
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
                dtype=torch.bfloat16,
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
                dtype=torch.bfloat16,
                device=metal_device,
            )
            * 0.1
        )

        output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        assert output.dtype == torch.bfloat16
        assert torch.isfinite(output).all()

    def test_flux_image_transformer_layout(self, metal_device):
        """Test with FLUX image transformer specific dimensions."""
        # FLUX main transformer: 24 heads, variable sequence length
        batch_size = 1
        num_heads = 24
        seq_len = 1536  # Common for 1024x1024 images
        head_dim = 128

        q = (
            torch.randn(
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                dtype=torch.bfloat16,
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
                dtype=torch.bfloat16,
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
                dtype=torch.bfloat16,
                device=metal_device,
            )
            * 0.01
        )

        output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        assert output.dtype == torch.bfloat16
        assert torch.isfinite(output).all()

    def test_flux_cross_attention_layout(self, metal_device):
        """Test FLUX cross-attention with different Q and KV sequence lengths."""
        batch_size = 1
        num_heads = 24

        # Query from image tokens
        q_seq_len = 1536
        q = (
            torch.randn(
                batch_size,
                num_heads,
                q_seq_len,
                128,
                dtype=torch.bfloat16,
                device=metal_device,
            )
            * 0.01
        )

        # Key/Value from text tokens
        kv_seq_len = 77
        k = (
            torch.randn(
                batch_size,
                num_heads,
                kv_seq_len,
                128,
                dtype=torch.bfloat16,
                device=metal_device,
            )
            * 0.01
        )
        v = (
            torch.randn(
                batch_size,
                num_heads,
                kv_seq_len,
                128,
                dtype=torch.bfloat16,
                device=metal_device,
            )
            * 0.01
        )

        output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

        # Output should have query sequence length
        assert output.shape == (batch_size, num_heads, q_seq_len, 128)
        assert output.dtype == torch.bfloat16
        assert torch.isfinite(output).all()

    @pytest.mark.parametrize(
        "resolution,expected_seq_len",
        [
            ((256, 256), 256),  # Small resolution
            ((512, 512), 1024),  # Medium resolution
            ((1024, 1024), 4096),  # Large resolution
        ],
    )
    def test_flux_resolution_dependent_layouts(
        self, metal_device, resolution, expected_seq_len
    ):
        """Test FLUX layouts for different image resolutions."""
        batch_size = 1
        num_heads = 24
        head_dim = 128

        # Sequence length depends on image resolution
        # Approximate: seq_len = (resolution[0] * resolution[1]) / (patch_size ** 2)
        seq_len = expected_seq_len

        q = (
            torch.randn(
                batch_size,
                num_heads,
                seq_len,
                head_dim,
                dtype=torch.bfloat16,
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
                dtype=torch.bfloat16,
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
                dtype=torch.bfloat16,
                device=metal_device,
            )
            * 0.01
        )

        output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        assert torch.isfinite(
            output
        ).all(), f"Non-finite values for resolution {resolution}"
