"""Numeric correctness tests for the MLA (Multi-Latent Attention) FFI.

The MLA path decompresses a compressed KV latent into full K and V via two
GEMMs:  K[M,N] = kv_latent[M,K] @ W_k[K,N],  V[M,N] = kv_latent[M,K] @ W_v[K,N]
with M = batch*seq, K = kv_latent_dim, N = num_heads*head_dim. Inputs/outputs
are FP16 with FP32 accumulation.

These tests load known weights through the Python FFI, run the decompression,
and compare against a torch reference computed from the same FP16 inputs.
"""

import metal_sdpa_extension as ext
import pytest
import torch

MLA_SHAPES = [
    # (batch, seq, heads, head_dim, kv_latent_dim)
    (1, 128, 8, 128, 512),
    (1, 512, 8, 128, 512),
    (1, 1024, 8, 128, 512),
    (2, 256, 8, 128, 512),
]


@pytest.mark.gpu
@pytest.mark.metal
@pytest.mark.parametrize("batch,seq,heads,head_dim,kv_latent_dim", MLA_SHAPES)
def test_mla_decompression_matches_reference(
    metal_device, batch, seq, heads, head_dim, kv_latent_dim
):
    torch.manual_seed(0)
    total_dim = heads * head_dim
    scale = 0.1

    wk = (
        torch.randn(kv_latent_dim, total_dim, device=metal_device, dtype=torch.float16)
        * scale
    )
    wv = (
        torch.randn(kv_latent_dim, total_dim, device=metal_device, dtype=torch.float16)
        * scale
    )
    kv_latent = (
        torch.randn(
            batch * seq, kv_latent_dim, device=metal_device, dtype=torch.float16
        )
        * scale
    )

    ctx = ext.MlaContext()
    ctx.load_weights(wk, wv)
    k, v = ctx.forward(kv_latent, batch, heads, seq, head_dim, kv_latent_dim)

    assert k.dtype == torch.float16, "K output must be FP16"
    assert v.dtype == torch.float16, "V output must be FP16"
    expected_shape = [batch * seq, total_dim]
    assert list(k.shape) == expected_shape, f"K shape {k.shape} != {expected_shape}"
    assert list(v.shape) == expected_shape, f"V shape {v.shape} != {expected_shape}"
    assert torch.isfinite(k).all(), "K output contains non-finite values"
    assert torch.isfinite(v).all(), "V output contains non-finite values"

    # Reference computed in FP32 from the same FP16-quantized inputs, then
    # rounded back to FP16 so the comparison isolates GEMM error from input
    # quantization.
    k_ref = (kv_latent.float() @ wk.float()).half()
    v_ref = (kv_latent.float() @ wv.float()).half()

    # FP16 storage + FP32 accumulation over K=512 terms: ~1e-2 observed.
    torch.testing.assert_close(k.cpu(), k_ref.cpu(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(v.cpu(), v_ref.cpu(), rtol=2e-2, atol=2e-2)


@pytest.mark.gpu
@pytest.mark.metal
def test_mla_output_is_written_in_place(metal_device):
    """Regression for the bug where the FFI returned the caller's never-written
    pre-allocated tensors. With the fix, the decompression must actually fill the
    outputs (non-zero, matching the reference), not return empty buffers."""
    torch.manual_seed(1)
    heads, head_dim, kv_latent_dim = 8, 128, 512
    total_dim = heads * head_dim
    seq = 64
    batch = 1

    wk = (
        torch.randn(kv_latent_dim, total_dim, device=metal_device, dtype=torch.float16)
        * 0.1
    )
    wv = (
        torch.randn(kv_latent_dim, total_dim, device=metal_device, dtype=torch.float16)
        * 0.1
    )
    kv_latent = (
        torch.randn(
            batch * seq, kv_latent_dim, device=metal_device, dtype=torch.float16
        )
        * 0.1
    )

    ctx = ext.MlaContext()
    ctx.load_weights(wk, wv)
    k, v = ctx.forward(kv_latent, batch, heads, seq, head_dim, kv_latent_dim)

    # Empty/unwritten outputs would be all zeros.
    assert (
        k.abs().sum().item() > 0.0
    ), "K output is empty (all zeros) — FFI write-back broken"
    assert (
        v.abs().sum().item() > 0.0
    ), "V output is empty (all zeros) — FFI write-back broken"

    k_ref = (kv_latent.float() @ wk.float()).half()
    torch.testing.assert_close(k.cpu(), k_ref.cpu(), rtol=2e-2, atol=2e-2)


@pytest.mark.gpu
@pytest.mark.metal
def test_mla_context_load_vs_random_weights(metal_device):
    """load_weights must override any prior random init with the supplied weights."""
    torch.manual_seed(2)
    heads, head_dim, kv_latent_dim = 4, 64, 256
    total_dim = heads * head_dim
    seq = 32
    batch = 1

    wk = (
        torch.randn(kv_latent_dim, total_dim, device=metal_device, dtype=torch.float16)
        * 0.1
    )
    wv = (
        torch.randn(kv_latent_dim, total_dim, device=metal_device, dtype=torch.float16)
        * 0.1
    )
    kv_latent = (
        torch.randn(
            batch * seq, kv_latent_dim, device=metal_device, dtype=torch.float16
        )
        * 0.1
    )

    ctx = ext.MlaContext()
    ctx.init_random_weights(heads, head_dim, kv_latent_dim)
    ctx.load_weights(wk, wv)
    k, v = ctx.forward(kv_latent, batch, heads, seq, head_dim, kv_latent_dim)

    k_ref = (kv_latent.float() @ wk.float()).half()
    torch.testing.assert_close(k.cpu(), k_ref.cpu(), rtol=2e-2, atol=2e-2)
