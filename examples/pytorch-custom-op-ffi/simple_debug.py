#!/usr/bin/env python3
"""
Simple debug test for investigating hanging.
Author: bghira
"""

import torch
import metal_sdpa_extension

print("Testing simple 2x2 case...")
q = torch.randn(2, 2, dtype=torch.float32)
k = torch.randn(2, 2, dtype=torch.float32)
v = torch.randn(2, 2, dtype=torch.float32)

print("Calling Metal SDPA...")
try:
    output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
    print("Simple case completed successfully")
except Exception as e:
    print(f"Simple case failed: {e}")

print("\nTesting larger 32x16 case...")
q = torch.randn(32, 16, dtype=torch.float32)
k = torch.randn(32, 16, dtype=torch.float32)
v = torch.randn(32, 16, dtype=torch.float32)

print("Calling Metal SDPA with larger tensors...")
try:
    output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
    print("Large case completed successfully")
    print(f"Output shape: {output.shape}")
    print(f"Has NaN: {torch.isnan(output).any()}")
except Exception as e:
    print(f"Large case failed: {e}")
