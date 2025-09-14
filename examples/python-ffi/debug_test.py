#!/usr/bin/env python3

import numpy as np
import sys
from pathlib import Path

# Add the source directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import umfa


def main():
    print("Debug Test - Python FFI Issue")
    print("=" * 40)

    # Check system compatibility
    if not umfa.is_metal_available():
        print("❌ Metal is not available on this device")
        return 1

    print("✅ Metal device is supported")

    # Create simple test data like the working Swift test
    seq_len = 4
    head_dim = 4
    print(f"✅ Creating test tensors: seq_len={seq_len}, head_dim={head_dim}")

    # Use initialized data (like Swift test) instead of random
    # Test with FP32 first (like the working Swift test)
    q = np.ones((seq_len, head_dim), dtype=np.float32)
    k = np.ones((seq_len, head_dim), dtype=np.float32)
    v = np.ones((seq_len, head_dim), dtype=np.float32)

    print(f"✅ Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")
    print(f"✅ Data type: {q.dtype}")
    print(
        f"✅ Input data (first few elements): Q={q.flat[:5]}, K={k.flat[:5]}, V={v.flat[:5]}"
    )

    try:
        # Test with FP32 precision (like working Swift test)
        print("\n🚀 Running attention with FP32 precision...")
        output = umfa.attention(
            q, k, v, causal=False, input_precision="fp32", output_precision="fp32"
        )

        print(f"✅ Output shape: {output.shape}, dtype: {output.dtype}")
        print(f"✅ Output range: [{output.min():.4f}, {output.max():.4f}]")
        print(f"✅ Output data (first few elements): {output.flat[:10]}")

        # Check for non-zero output
        has_non_zero = np.any(output != 0.0)
        print(f"✅ Has non-zero values: {has_non_zero}")

        if not has_non_zero:
            print("❌ ERROR: Output is all zeros!")
            return 1
        else:
            print("✅ SUCCESS: Output contains non-zero values!")

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
