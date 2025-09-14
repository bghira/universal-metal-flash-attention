#!/usr/bin/env python3
"""
Simplified test with Swift parameters to isolate hanging issue.
Author: bghira
"""

import torch
import metal_sdpa_extension
import numpy as np


def main():
    print("Simple Swift Parameters Test")
    print("Author: bghira")
    print("=" * 50)

    if not metal_sdpa_extension.is_metal_available():
        print("❌ Metal not available")
        return

    print("✅ Metal available")

    # EXACT same parameters as Swift test
    seq_len = 4
    head_dim = 4
    scale = 1.0 / np.sqrt(head_dim)

    print(f"Creating {seq_len}x{head_dim} tensors with scale {scale:.6f}")

    # Create tensors with ALL 1.0s (exactly like Swift test)
    q = torch.ones(seq_len, head_dim, dtype=torch.float32)
    k = torch.ones(seq_len, head_dim, dtype=torch.float32)
    v = torch.ones(seq_len, head_dim, dtype=torch.float32)

    print("Calling metal_scaled_dot_product_attention...")

    try:
        # This is where it might be hanging
        metal_output = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q, k, v, scale=scale, is_causal=False
        )

        print("✅ Call succeeded!")
        print(f"Output shape: {metal_output.shape}")
        print(f"Output: {metal_output}")
        print(f"Has NaN: {torch.isnan(metal_output).any()}")
        print(f"Has Inf: {torch.isinf(metal_output).any()}")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
