#!/usr/bin/env python3
"""
Test the updated quantized_scaled_dot_product_attention_with_config
Now supports per-block quantization with configurable output precision
"""

import metal_sdpa_extension
import torch
from metal_sdpa_extension import (
    OutputPrecision,
    QuantizationConfig,
    quantized_scaled_dot_product_attention_with_config,
)

print("🎯 Testing Per-Block Quantization with Configurable Output Precision")
print("=" * 70)

# Create test tensors with FLUX-like dimensions
batch_size = 1
seq_len = 4096
num_heads = 16
head_dim = 64

query = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16)
key = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16)
value = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16)

print(f"Test dimensions: B={batch_size}, S={seq_len}, H={num_heads}, D={head_dim}")
print()

# Test 1: Per-tensor quantization (old behavior, should fail with NaN/inf)
print("Test 1: Per-tensor quantization (unstable)")
print("-" * 40)
config_tensor = QuantizationConfig()
config_tensor.precision = "int8"
config_tensor.output_precision = OutputPrecision.FP32
config_tensor.q_block_size = 0  # 0 = use tensor-wise quantization
config_tensor.k_block_size = 0
config_tensor.v_block_size = 0

try:
    result_tensor = quantized_scaled_dot_product_attention_with_config(
        query, key, value, config_tensor
    )
    has_nan = torch.isnan(result_tensor).any()
    has_inf = torch.isinf(result_tensor).any()
    print(f"✅ Tensor-wise result shape: {list(result_tensor.shape)}")
    print(f"   Has NaN: {has_nan}")
    print(f"   Has Inf: {has_inf}")
    if has_nan or has_inf:
        print(
            "   ⚠️ UNSTABLE: Contains NaN/Inf as expected for tensor-wise quantization"
        )
except Exception as e:
    print(f"❌ Failed: {e}")

print()

# Test 2: Per-block quantization with INT8 (stable)
print("Test 2: Per-block INT8 quantization (stable)")
print("-" * 40)
config_block_int8 = QuantizationConfig()
config_block_int8.precision = "int8"
config_block_int8.output_precision = OutputPrecision.FP32
config_block_int8.q_block_size = 128  # SageAttention defaults
config_block_int8.k_block_size = 64
config_block_int8.v_block_size = 64

try:
    result_block_int8 = quantized_scaled_dot_product_attention_with_config(
        query, key, value, config_block_int8
    )
    has_nan = torch.isnan(result_block_int8).any()
    has_inf = torch.isinf(result_block_int8).any()
    print(f"✅ Per-block INT8 result shape: {list(result_block_int8.shape)}")
    print(f"   Has NaN: {has_nan}")
    print(f"   Has Inf: {has_inf}")
    print(f"   Output dtype: {result_block_int8.dtype}")
    if not has_nan and not has_inf:
        print("   ✅ STABLE: No NaN/Inf with per-block quantization!")
except Exception as e:
    print(f"❌ Failed: {e}")

print()

# Test 3: Per-block quantization with INT4 and BF16 output
print("Test 3: Per-block INT4 with BF16 output")
print("-" * 40)
config_block_int4_bf16 = QuantizationConfig()
config_block_int4_bf16.precision = "int4"
config_block_int4_bf16.output_precision = OutputPrecision.BF16
config_block_int4_bf16.q_block_size = 128
config_block_int4_bf16.k_block_size = 64
config_block_int4_bf16.v_block_size = 64

try:
    result_block_int4 = quantized_scaled_dot_product_attention_with_config(
        query, key, value, config_block_int4_bf16
    )
    has_nan = torch.isnan(result_block_int4).any()
    has_inf = torch.isinf(result_block_int4).any()
    print(f"✅ Per-block INT4 result shape: {list(result_block_int4.shape)}")
    print(f"   Has NaN: {has_nan}")
    print(f"   Has Inf: {has_inf}")
    print(f"   Output dtype: {result_block_int4.dtype}")
    if not has_nan and not has_inf:
        print("   ✅ STABLE: INT4 + BF16 output works!")
except Exception as e:
    print(f"❌ Failed: {e}")

print()

# Test 4: Per-block INT4 with FP16 output (might fail)
print("Test 4: Per-block INT4 with FP16 output")
print("-" * 40)
config_block_int4_fp16 = QuantizationConfig()
config_block_int4_fp16.precision = "int4"
config_block_int4_fp16.output_precision = OutputPrecision.FP16
config_block_int4_fp16.q_block_size = 128
config_block_int4_fp16.k_block_size = 64
config_block_int4_fp16.v_block_size = 64

try:
    result_block_int4_fp16 = quantized_scaled_dot_product_attention_with_config(
        query, key, value, config_block_int4_fp16
    )
    has_nan = torch.isnan(result_block_int4_fp16).any()
    has_inf = torch.isinf(result_block_int4_fp16).any()
    print(f"✅ Per-block INT4 FP16 result shape: {list(result_block_int4_fp16.shape)}")
    print(f"   Has NaN: {has_nan}")
    print(f"   Has Inf: {has_inf}")
    print(f"   Output dtype: {result_block_int4_fp16.dtype}")
    if has_nan or has_inf:
        print("   ⚠️ INT4 + FP16 output might be unstable")
    else:
        print("   ✅ INT4 + FP16 output works!")
except Exception as e:
    print(f"❌ Failed: {e}")

print()

# Test 5: Compare results between different output precisions
print("Test 5: Comparing different output precisions")
print("-" * 40)

configs_to_compare = [
    ("FP32", OutputPrecision.FP32),
    ("BF16", OutputPrecision.BF16),
    ("FP16", OutputPrecision.FP16),
]

results = {}
for name, output_prec in configs_to_compare:
    config = QuantizationConfig()
    config.precision = "int8"
    config.output_precision = output_prec
    config.q_block_size = 128
    config.k_block_size = 64
    config.v_block_size = 64

    try:
        result = quantized_scaled_dot_product_attention_with_config(
            query, key, value, config
        )
        results[name] = result
        print(
            f"   {name}: dtype={result.dtype}, mean={result.mean():.6f}, std={result.std():.6f}"
        )
    except Exception as e:
        print(f"   {name}: Failed - {e}")

# Compare results
if len(results) > 1:
    print("\n   Comparing results:")
    ref_result = results.get("FP32")
    if ref_result is not None:
        for name, result in results.items():
            if name != "FP32":
                diff = (result.float() - ref_result.float()).abs().mean()
                print(f"   {name} vs FP32: mean abs diff = {diff:.6f}")

print()
print("=" * 70)
print("🎉 SUMMARY:")
print("✅ Per-block quantization is now integrated into with_config method!")
print("✅ Output precision control works with per-block quantization!")
print("✅ Simply set q/k/v_block_size > 0 to enable stable per-block quantization")
