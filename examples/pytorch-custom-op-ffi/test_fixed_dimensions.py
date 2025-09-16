#!/usr/bin/env python3
"""Test the dimension fix directly"""

import metal_sdpa_extension
import torch
from metal_sdpa_extension import quantized_scaled_dot_product_attention_per_block

print("Testing dimension fix with FLUX-like tensors")

# FLUX tensor shape: [batch, num_heads, seq_len, head_dim]
batch = 1
num_heads = 24
seq_len = 1536
head_dim = 128

query = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
key = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
value = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.bfloat16)

print(f"Input shape: {list(query.shape)}")
print(f"Expected: batch={batch}, heads={num_heads}, seq={seq_len}, dim={head_dim}")

try:
    result = quantized_scaled_dot_product_attention_per_block(
        query, key, value, 128, 64, 64, "int8", False, None
    )
    print(f"✅ SUCCESS! Output shape: {list(result.shape)}")
    print(f"   Has NaN: {torch.isnan(result).any()}")
    print(f"   Has Inf: {torch.isinf(result).any()}")
except Exception as e:
    print(f"❌ FAILED: {e}")
