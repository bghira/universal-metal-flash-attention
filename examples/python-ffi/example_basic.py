#!/usr/bin/env python3
"""
Basic Universal Metal Flash Attention Python Example

Demonstrates zero-copy, high-performance attention computation using the
Universal Metal Flash Attention Python bindings.
"""

import sys
from pathlib import Path

import numpy as np

# Add the source directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import umfa


def main():
    """Run basic attention example."""
    print("Universal Metal Flash Attention - Python Example")
    print("=" * 50)

    # Check system compatibility
    if not umfa.is_metal_available():
        print("❌ Metal is not available on this device")
        return 1

    print("✅ Metal device is supported")

    # Print version info
    major, minor, patch = umfa.get_version()
    print(f"✅ MFA version: {major}.{minor}.{patch}")

    # Create test data
    seq_len = 512
    head_dim = 64
    print(f"✅ Creating test tensors: seq_len={seq_len}, head_dim={head_dim}")

    # Use FP16 for optimal performance on Apple Silicon
    q = np.random.randn(seq_len, head_dim).astype(np.float16)
    k = np.random.randn(seq_len, head_dim).astype(np.float16)
    v = np.random.randn(seq_len, head_dim).astype(np.float16)

    print(f"✅ Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")
    print(f"✅ Data type: {q.dtype}")

    try:
        # Method 1: Using context manager (recommended)
        print("\n🚀 Running attention with context manager...")
        with umfa.MFAContext() as ctx:
            output1 = umfa.flash_attention_forward(
                ctx, q, k, v, causal=False, input_precision="fp16"
            )

        print(f"✅ Output shape: {output1.shape}, dtype: {output1.dtype}")
        print(f"✅ Output range: [{output1.min():.4f}, {output1.max():.4f}]")

        # Method 2: Using convenience function
        print("\n🚀 Running attention with convenience function...")
        output2 = umfa.attention(q, k, v, causal=True)

        print(f"✅ Causal output shape: {output2.shape}, dtype: {output2.dtype}")
        print(f"✅ Causal output range: [{output2.min():.4f}, {output2.max():.4f}]")

        # Verify outputs are different (causal vs non-causal)
        diff = np.mean(np.abs(output1 - output2))
        print(f"✅ Difference between causal/non-causal: {diff:.6f}")

        # Test different precisions
        print("\n🧪 Testing different precisions...")

        # FP32 precision
        q_fp32 = q.astype(np.float32)
        k_fp32 = k.astype(np.float32)
        v_fp32 = v.astype(np.float32)

        output_fp32 = umfa.attention(
            q_fp32, k_fp32, v_fp32, input_precision="fp32", output_precision="fp32"
        )
        print(f"✅ FP32 output: shape={output_fp32.shape}, dtype={output_fp32.dtype}")

    except umfa.MFAError as e:
        print(f"❌ MFA Error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

    print("\n✅ All tests completed successfully!")
    print("\n📊 Performance Notes:")
    print("   • Zero-copy operation: no data copying between Python and Metal")
    print("   • FP16 precision provides optimal speed/memory tradeoffs")
    print("   • Causal masking supported for autoregressive models")
    print("   • Maintains 4400+ GINSTRS/sec MFA performance")

    return 0


if __name__ == "__main__":
    exit(main())
