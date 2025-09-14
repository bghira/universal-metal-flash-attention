#!/usr/bin/env python3
"""
Minimal debug to isolate the hanging issue.
Author: bghira
"""

import torch
import metal_sdpa_extension


def main():
    print("Minimal Debug Test")
    print("Author: bghira")

    if not metal_sdpa_extension.is_metal_available():
        print("❌ Metal not available")
        return

    print("✅ Metal available")

    # Simplest possible test
    print("Creating tiny 2x2 tensors...")
    q = torch.ones(2, 2, dtype=torch.float32)
    k = torch.ones(2, 2, dtype=torch.float32)
    v = torch.ones(2, 2, dtype=torch.float32)

    print("Calling metal_scaled_dot_product_attention...")
    try:
        output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
        print(f"✅ Success! Output shape: {output.shape}")
        print(f"Output: {output}")
        print(f"Has NaN: {torch.isnan(output).any()}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
