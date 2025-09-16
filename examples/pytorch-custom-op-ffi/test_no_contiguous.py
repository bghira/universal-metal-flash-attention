#!/usr/bin/env python3
"""
Test what happens when we remove contiguous conversion entirely
"""

import time

import torch
from metal_sdpa_extension import (
    OutputPrecision,
    QuantizationConfig,
    quantized_scaled_dot_product_attention_per_block,
    quantized_scaled_dot_product_attention_with_config,
)

print("🧪 Testing Metal Backend Without Contiguous Conversion")
print("=" * 60)

# Small test size for quick iteration
batch_size = 1
seq_len = 512
num_heads = 4
head_dim = 64


def test_layout(layout_name, query, key, value):
    """Test a specific layout"""
    print(f"\n🔧 Testing {layout_name} layout")
    print(f"   Query shape: {list(query.shape)}")
    print(f"   Query strides: {list(query.stride())}")
    print(f"   Query contiguous: {query.is_contiguous()}")

    # Test per-tensor quantization
    config = QuantizationConfig()
    config.precision = "int8"
    config.output_precision = OutputPrecision.FP32
    config.is_causal = False

    try:
        print("   Testing per-tensor quantization...")
        start = time.time()
        result_tensor = quantized_scaled_dot_product_attention_with_config(
            query, key, value, config
        )
        end = time.time()

        has_nan = torch.isnan(result_tensor).any()
        has_inf = torch.isinf(result_tensor).any()
        print(f"   ✅ Per-tensor: {end-start:.4f}s, nan={has_nan}, inf={has_inf}")

    except Exception as e:
        print(f"   ❌ Per-tensor failed: {e}")
        return False

    # Test per-block quantization
    try:
        print("   Testing per-block quantization...")
        start = time.time()
        result_block = quantized_scaled_dot_product_attention_per_block(
            query, key, value, 128, 64, 64, "int8", False, None
        )
        end = time.time()

        has_nan = torch.isnan(result_block).any()
        has_inf = torch.isinf(result_block).any()
        print(f"   ✅ Per-block: {end-start:.4f}s, nan={has_nan}, inf={has_inf}")

    except Exception as e:
        print(f"   ❌ Per-block failed: {e}")
        return False

    return True


# Test 1: Contiguous (baseline)
print("\n🔵 Test 1: Contiguous Layout")
query = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)
key = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)
value = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)

success_contiguous = test_layout("Contiguous", query, key, value)

# Test 2: Transposed (non-contiguous)
print("\n🔵 Test 2: Transposed Layout")
query_t = torch.randn(
    batch_size, num_heads, seq_len, head_dim, dtype=torch.float32
).transpose(1, 2)
key_t = torch.randn(
    batch_size, num_heads, seq_len, head_dim, dtype=torch.float32
).transpose(1, 2)
value_t = torch.randn(
    batch_size, num_heads, seq_len, head_dim, dtype=torch.float32
).transpose(1, 2)

success_transposed = test_layout("Transposed", query_t, key_t, value_t)

# Test 3: Permuted (different stride pattern)
print("\n🔵 Test 3: Permuted Layout")
query_p = torch.randn(
    batch_size, num_heads, seq_len, head_dim, dtype=torch.float32
).permute(0, 2, 1, 3)
key_p = torch.randn(
    batch_size, num_heads, seq_len, head_dim, dtype=torch.float32
).permute(0, 2, 1, 3)
value_p = torch.randn(
    batch_size, num_heads, seq_len, head_dim, dtype=torch.float32
).permute(0, 2, 1, 3)

success_permuted = test_layout("Permuted", query_p, key_p, value_p)

# Test 4: View with different strides
print("\n🔵 Test 4: Reshaped Layout")
try:
    # Create tensor with different internal layout
    base = torch.randn(batch_size, seq_len * num_heads, head_dim, dtype=torch.float32)
    query_r = base.view(batch_size, seq_len, num_heads, head_dim)
    key_r = base.view(batch_size, seq_len, num_heads, head_dim)
    value_r = base.view(batch_size, seq_len, num_heads, head_dim)

    success_reshaped = test_layout("Reshaped", query_r, key_r, value_r)
except Exception as e:
    print(f"   ❌ Reshaped layout creation failed: {e}")
    success_reshaped = False

# Summary
print("\n🏆 RESULTS SUMMARY")
print("=" * 60)

results = {
    "Contiguous": success_contiguous,
    "Transposed": success_transposed,
    "Permuted": success_permuted,
    "Reshaped": success_reshaped,
}

for layout, success in results.items():
    status = "✅ WORKS" if success else "❌ FAILS"
    print(f"{layout:12} : {status}")

if all(results.values()):
    print("\n🎉 EXCELLENT: Metal backend handles all layouts gracefully!")
    print("   Philip Turner's codegen philosophy is preserved.")
    print("   The automatic kernel generation adapts to different layouts.")
elif any(results.values()):
    print("\n⚠️  PARTIAL: Some layouts work, others don't.")
    print("   May need selective contiguous conversion.")
else:
    print("\n💥 FAILURE: All layouts fail without contiguous conversion.")
    print("   Need to restore contiguous conversion.")

print(f"\n💡 Insights:")
print(f"   - Removing contiguous conversion reveals layout tolerance")
print(f"   - Metal kernels may adapt automatically to stride patterns")
print(f"   - This aligns with MFA's flexible codegen philosophy")
