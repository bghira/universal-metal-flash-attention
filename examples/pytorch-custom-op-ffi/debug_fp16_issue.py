#!/usr/bin/env python3
"""
Debug FP16 hanging issue.
Author: bghira
"""

import metal_sdpa_extension
import torch


def test_fp16_simple():
    print("Testing FP16 issue...")

    if not metal_sdpa_extension.is_metal_available():
        print("❌ Metal not available")
        return

    try:
        print("Creating FP16 tensors...")
        q = torch.ones(4, 4, dtype=torch.float16)
        k = torch.ones(4, 4, dtype=torch.float16)
        v = torch.ones(4, 4, dtype=torch.float16)

        print("Calling metal_scaled_dot_product_attention with FP16...")
        output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

        print(f"✅ FP16 test succeeded! Output dtype: {output.dtype}")
        print(f"Output shape: {output.shape}")
        print(f"Has NaN: {torch.isnan(output).any()}")

    except Exception as e:
        print(f"❌ FP16 test failed: {e}")


if __name__ == "__main__":
    test_fp16_simple()
