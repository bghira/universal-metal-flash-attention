#!/usr/bin/env python3
"""
Debug test for unified quantized_scaled_dot_product_attention_with_config
Minimal test to check if per-block functionality works
"""

import metal_sdpa_extension
import torch
from metal_sdpa_extension import (
    OutputPrecision,
    QuantizationConfig,
    quantized_scaled_dot_product_attention_with_config,
)

print("🔍 Debug Test: Unified Per-Block Quantization")
print("=" * 50)

# Small test tensors
batch_size = 1
seq_len = 512  # Smaller than FLUX to test stability
num_heads = 8
head_dim = 64

query = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16)
key = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16)
value = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16)

print(f"Test dimensions: B={batch_size}, S={seq_len}, H={num_heads}, D={head_dim}")
print()

# Test 1: Per-block with INT8, FP32 output (should be most stable)
print("Test 1: Per-block INT8 + FP32 output")
print("-" * 30)
config = QuantizationConfig()
config.precision = "int8"
config.output_precision = OutputPrecision.FP32
config.q_block_size = 128
config.k_block_size = 64
config.v_block_size = 64

try:
    result = quantized_scaled_dot_product_attention_with_config(
        query, key, value, config
    )
    has_nan = torch.isnan(result).any()
    has_inf = torch.isinf(result).any()
    print(f"✅ Result shape: {list(result.shape)}")
    print(f"   Has NaN: {has_nan}")
    print(f"   Has Inf: {has_inf}")
    print(f"   Output dtype: {result.dtype}")
    print(f"   Output range: [{result.min():.6f}, {result.max():.6f}]")
except Exception as e:
    print(f"❌ Failed: {e}")

print()

# Test 2: Compare with tensor-wise (should be unstable)
print("Test 2: Tensor-wise INT8 + FP32 output (unstable)")
print("-" * 30)
config_tensor = QuantizationConfig()
config_tensor.precision = "int8"
config_tensor.output_precision = OutputPrecision.FP32
config_tensor.q_block_size = 0  # Disable per-block
config_tensor.k_block_size = 0
config_tensor.v_block_size = 0

try:
    result_tensor = quantized_scaled_dot_product_attention_with_config(
        query, key, value, config_tensor
    )
    has_nan = torch.isnan(result_tensor).any()
    has_inf = torch.isinf(result_tensor).any()
    print(f"✅ Result shape: {list(result_tensor.shape)}")
    print(f"   Has NaN: {has_nan}")
    print(f"   Has Inf: {has_inf}")
    print(f"   Output dtype: {result_tensor.dtype}")
    if not has_nan and not has_inf:
        print(
            f"   Output range: [{result_tensor.min():.6f}, {result_tensor.max():.6f}]"
        )
except Exception as e:
    print(f"❌ Failed: {e}")

print()

# Test 3: Per-block with INT4, BF16 output
print("Test 3: Per-block INT4 + BF16 output")
print("-" * 30)
config_int4 = QuantizationConfig()
config_int4.precision = "int4"
config_int4.output_precision = OutputPrecision.BF16
config_int4.q_block_size = 128
config_int4.k_block_size = 64
config_int4.v_block_size = 64

try:
    result_int4 = quantized_scaled_dot_product_attention_with_config(
        query, key, value, config_int4
    )
    has_nan = torch.isnan(result_int4).any()
    has_inf = torch.isinf(result_int4).any()
    print(f"✅ Result shape: {list(result_int4.shape)}")
    print(f"   Has NaN: {has_nan}")
    print(f"   Has Inf: {has_inf}")
    print(f"   Output dtype: {result_int4.dtype}")
    if not has_nan and not has_inf:
        print(f"   Output range: [{result_int4.min():.6f}, {result_int4.max():.6f}]")
except Exception as e:
    print(f"❌ Failed: {e}")

print()
print("=" * 50)
print("🎉 Debug complete - checking if unified config works at all")
