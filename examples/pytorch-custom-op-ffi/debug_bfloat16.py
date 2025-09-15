#!/usr/bin/env python3
"""
Test bfloat16 specifically to see if it causes hanging.
Author: bghira
"""

import metal_sdpa_extension
import torch


def test_bfloat16():
    print("Testing BFloat16...")

    if not metal_sdpa_extension.is_metal_available():
        print("❌ Metal not available")
        return

    try:
        print("Creating BFloat16 tensors...")
        q = torch.ones(4, 4, dtype=torch.bfloat16)
        k = torch.ones(4, 4, dtype=torch.bfloat16)
        v = torch.ones(4, 4, dtype=torch.bfloat16)

        print("Calling metal_scaled_dot_product_attention with BFloat16...")
        output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

        print(f"✅ BFloat16 test succeeded! Output dtype: {output.dtype}")
        print(f"Output shape: {output.shape}")
        print(f"Has NaN: {torch.isnan(output).any()}")

    except Exception as e:
        print(f"❌ BFloat16 test failed: {e}")


if __name__ == "__main__":
    test_bfloat16()
