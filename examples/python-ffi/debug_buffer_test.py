#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np

# Add the source directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import umfa


def main():
    print("Buffer Debug Test")
    print("=" * 30)

    # Check system compatibility
    if not umfa.is_metal_available():
        print("❌ Metal is not available on this device")
        return 1

    print("✅ Metal device is supported")

    # Create very simple test data
    seq_len = 2
    head_dim = 2
    print(f"✅ Creating test tensors: seq_len={seq_len}, head_dim={head_dim}")

    # Use very simple known values
    q = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    k = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    v = np.array([[9.0, 10.0], [11.0, 12.0]], dtype=np.float32)

    print(f"✅ Input data:")
    print(f"  Q = {q}")
    print(f"  K = {k}")
    print(f"  V = {v}")

    # Test buffer creation and data access
    try:
        with umfa.MFAContext() as ctx:
            # Create buffers and check if data is preserved
            q_buf = umfa.MFABuffer(ctx, q)
            k_buf = umfa.MFABuffer(ctx, k)
            v_buf = umfa.MFABuffer(ctx, v)

            print("✅ Buffers created successfully")

            # Try to read back the data from buffers
            # Note: This is a low-level test to check data integrity
            q_ptr = q_buf.contents_ptr()

            # Check if we can access buffer contents
            print(f"✅ Q buffer pointer: {q_ptr}")

            # Test basic attention
            print(f"\n🚀 Running attention...")
            output = umfa.flash_attention_forward(
                ctx,
                q,
                k,
                v,
                causal=False,
                input_precision="fp32",
                intermediate_precision="fp32",
            )

            print(f"✅ Output shape: {output.shape}, dtype: {output.dtype}")
            print(f"✅ Output data: {output}")
            print(f"✅ Output range: [{output.min():.4f}, {output.max():.4f}]")

            # Check for non-zero output
            has_non_zero = np.any(output != 0.0)
            print(f"✅ Has non-zero values: {has_non_zero}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
