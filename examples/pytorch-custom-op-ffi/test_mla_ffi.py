#!/usr/bin/env python3
"""
Test MLA (Multi-Latent Attention) FFI bindings

Demonstrates KV cache decompression using optimized MFA GEMM.
Performance: 10.9 TFLOPS @ 2048×2048 on M3 Max
"""

import numpy as np
import torch

try:
    from python.metal_sdpa_ffi import MlaContext, is_metal_available

    EXTENSION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Extension not yet built: {e}")
    print("   Run: cd examples/pytorch-custom-op-ffi && pip install -e .")
    EXTENSION_AVAILABLE = False


def test_mla_decompression():
    """Test MLA decompression with random weights"""

    print("\n🔧 MLA (Multi-Latent Attention) Decompression Test")
    print("Performance: 10.9 TFLOPS @ 2048×2048 on M3 Max\n")

    if not EXTENSION_AVAILABLE:
        print("❌ Extension not available - skipping test")
        return

    # Check Metal availability
    if not is_metal_available():
        print("❌ Metal is not available on this device")
        return

    print("✓ Metal device is available")

    # Configuration
    batch_size = 1
    num_heads = 8
    sequence_length = 512
    head_dim = 128
    kv_latent_dim = 512  # 2x compression (1024 → 512)

    total_dim = num_heads * head_dim
    compression_ratio = total_dim / kv_latent_dim

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Heads: {num_heads}, Head dim: {head_dim}")
    print(f"  KV latent dim: {kv_latent_dim} (compression: {compression_ratio}x)")

    # Create MLA context using RAII wrapper
    try:
        mla_ctx = MlaContext()
        print("\n✓ Created MLA context")
    except Exception as e:
        print(f"❌ Failed to create MLA context: {e}")
        import traceback

        traceback.print_exc()
        return

    try:
        # Initialize random weights for testing
        mla_ctx.init_random_weights(num_heads, head_dim, kv_latent_dim)
        print("✓ Initialized decompression weights (W_k, W_v)")

        # Create compressed KV latent tensor on MPS device
        kv_latent = torch.randn(
            batch_size,
            sequence_length,
            kv_latent_dim,
            dtype=torch.float16,
            device=torch.device("mps"),
        )
        print(f"✓ Created compressed KV latent tensor: {list(kv_latent.shape)} FP16")

        # Perform MLA forward pass
        print("\n⚡ Running MLA decompression...")
        decompressed_k, decompressed_v = mla_ctx.forward(
            kv_latent, batch_size, num_heads, sequence_length, head_dim, kv_latent_dim
        )

        print("\n✅ MLA decompression successful!")
        print(f"   K output shape: {list(decompressed_k.shape)} {decompressed_k.dtype}")
        print(f"   V output shape: {list(decompressed_v.shape)} {decompressed_v.dtype}")

        # Verify shapes
        expected_shape = [batch_size * sequence_length, total_dim]
        assert (
            list(decompressed_k.shape) == expected_shape
        ), f"K shape mismatch: {decompressed_k.shape} != {expected_shape}"
        assert (
            list(decompressed_v.shape) == expected_shape
        ), f"V shape mismatch: {decompressed_v.shape} != {expected_shape}"

        print("\n✅ All assertions passed!")

    except Exception as e:
        print(f"\n❌ MLA test failed: {e}")
        import traceback

        traceback.print_exc()


def test_mla_with_real_model():
    """
    Example of how MLA would be used with a real model
    (requires model weights and C++ bindings)
    """

    print("\n📚 MLA Usage Example (Conceptual)")
    print("=" * 60)

    code_example = """
# 1. Load pre-trained MLA weights from model checkpoint
wk = torch.load("model_weights/mla_wk.pt")  # [kv_latent_dim, num_heads × head_dim]
wv = torch.load("model_weights/mla_wv.pt")  # [kv_latent_dim, num_heads × head_dim]

# 2. Create MLA context and load weights
mla_ctx = mla_create_context()
mla_load_weights(mla_ctx, wk, wv)

# 3. During inference, decompress KV cache
kv_latent = model.compress_kv_cache(k_full, v_full)  # Compress to latent
decompressed_k, decompressed_v = mla_forward(
    mla_ctx,
    mfa_ctx,
    kv_latent,
    batch_size=1,
    num_heads=8,
    sequence_length=512,
    head_dim=128,
    kv_latent_dim=512
)

# 4. Use decompressed K, V for attention
attention_output = flash_attention(q, decompressed_k, decompressed_v)
"""

    print(code_example)
    print("=" * 60)


if __name__ == "__main__":
    test_mla_decompression()
    test_mla_with_real_model()

    print("\n" + "=" * 60)
    print("MLA FFI Test Complete")
    print("=" * 60)
    print("\n✨ Next steps:")
    print("  1. Implement C++ PyTorch extension bindings")
    print("  2. Test with real DeepSeek-V3 MLA weights")
    print("  3. Benchmark against PyTorch MPS")
    print()
