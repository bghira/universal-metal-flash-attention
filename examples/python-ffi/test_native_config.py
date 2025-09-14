#!/usr/bin/env python3

import numpy as np
import sys
from pathlib import Path

# Add the source directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import umfa


def main():
    print("Test Native Configuration")
    print("=" * 35)

    # Check system compatibility
    if not umfa.is_metal_available():
        print("❌ Metal is not available on this device")
        return 1

    print("✅ Metal device is supported")

    # Use the same config as working native tests: seq=10, head=3
    seq_len = 10
    head_dim = 3
    print(f"✅ Creating test tensors: seq_len={seq_len}, head_dim={head_dim}")

    # Create random data like native tests
    np.random.seed(42)  # For reproducible results
    q = np.random.randn(seq_len, head_dim).astype(np.float32)
    k = np.random.randn(seq_len, head_dim).astype(np.float32)
    v = np.random.randn(seq_len, head_dim).astype(np.float32)

    print(f"✅ Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")
    print(
        f"✅ Input ranges: Q=[{q.min():.2f}, {q.max():.2f}], K=[{k.min():.2f}, {k.max():.2f}], V=[{v.min():.2f}, {v.max():.2f}]"
    )

    try:
        # Use FP32 precision like native tests
        print(f"\n🚀 Running attention with FP32...")
        with umfa.MFAContext() as ctx:
            output = umfa.flash_attention_forward(
                ctx, q, k, v, causal=False, input_precision="fp32"
            )

            print(f"✅ Output shape: {output.shape}, dtype: {output.dtype}")
            print(f"✅ Output range: [{output.min():.6f}, {output.max():.6f}]")
            print(f"✅ Output sample: {output.flat[:5]}")

            # Check for non-zero output
            has_non_zero = np.any(output != 0.0)
            print(f"✅ Has non-zero values: {has_non_zero}")

            if not has_non_zero:
                print("❌ ERROR: Output is all zeros!")
                return 1
            else:
                print("✅ SUCCESS: Output contains non-zero values!")

        # Test with causal masking too
        print(f"\n🚀 Testing causal masking...")
        output_causal = umfa.attention(q, k, v, causal=True)
        print(
            f"✅ Causal output range: [{output_causal.min():.6f}, {output_causal.max():.6f}]"
        )

        has_non_zero_causal = np.any(output_causal != 0.0)
        print(f"✅ Causal has non-zero values: {has_non_zero_causal}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
