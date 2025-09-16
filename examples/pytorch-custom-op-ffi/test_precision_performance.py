#!/usr/bin/env python3

import time

import torch
from metal_sdpa_extension import (
    OutputPrecision,
    QuantizationConfig,
    quantized_scaled_dot_product_attention_with_config,
)

print("🚀 Testing Precision Performance Impact on Quantized Attention")
print("================================================================")

# Test configuration for a single step performance measurement
batch_size = 1
seq_len_q = 4096  # Typical sequence length for attention layers
seq_len_kv = 4096
num_heads = 16  # Multi-head attention
head_dim = 64

device = torch.device("cpu")

# Create test tensors simulating FLUX attention dimensions
query = torch.randn(
    batch_size, seq_len_q, num_heads, head_dim, dtype=torch.float32, device=device
)
key = torch.randn(
    batch_size, seq_len_kv, num_heads, head_dim, dtype=torch.float32, device=device
)
value = torch.randn(
    batch_size, seq_len_kv, num_heads, head_dim, dtype=torch.float32, device=device
)

print(f"Test Configuration:")
print(f"  Batch Size: {batch_size}")
print(f"  Sequence Length: Q={seq_len_q}, KV={seq_len_kv}")
print(f"  Number of Heads: {num_heads}")
print(f"  Head Dimension: {head_dim}")
print(f"  Input shapes: Q={query.shape}, K={key.shape}, V={value.shape}")
print()


def test_precision(precision_name, output_precision, rounds=3):
    print(f"🧪 Testing {precision_name} Output Precision")
    print("-" * 50)

    # Create quantization config
    quant_config = QuantizationConfig()
    quant_config.precision = "int8"
    quant_config.output_precision = output_precision
    quant_config.is_causal = False

    times = []

    # Warmup run
    print("🔥 Warmup run...")
    try:
        _ = quantized_scaled_dot_product_attention_with_config(
            query, key, value, quant_config
        )
        print("✅ Warmup completed")
    except Exception as e:
        print(f"❌ Warmup failed: {e}")
        return None

    # Benchmark runs
    for i in range(rounds):
        print(f"⏱️  Run {i+1}/{rounds}...")
        start_time = time.time()

        try:
            output = quantized_scaled_dot_product_attention_with_config(
                query, key, value, quant_config
            )
            end_time = time.time()

            run_time = end_time - start_time
            times.append(run_time)

            print(f"   Time: {run_time:.4f}s")
            print(f"   Output shape: {output.shape}")
            print(f"   Output dtype: {output.dtype}")

        except Exception as e:
            print(f"❌ Run {i+1} failed: {e}")
            return None

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"📊 Results for {precision_name}:")
    print(f"   Average time: {avg_time:.4f}s")
    print(f"   Best time:    {min_time:.4f}s")
    print(f"   Worst time:   {max_time:.4f}s")
    print()

    return avg_time


# Test different output precisions
print("Starting precision performance comparison...")
print()

fp32_time = test_precision("FP32", OutputPrecision.FP32)
fp16_time = test_precision("FP16", OutputPrecision.FP16)
bf16_time = test_precision("BF16", OutputPrecision.BF16)

print("🏆 PERFORMANCE COMPARISON")
print("=" * 50)

if fp32_time:
    print(f"FP32: {fp32_time:.4f}s (baseline)")

if fp16_time:
    if fp32_time:
        speedup = fp32_time / fp16_time
        print(f"FP16: {fp16_time:.4f}s ({speedup:.2f}x speedup vs FP32)")
    else:
        print(f"FP16: {fp16_time:.4f}s")

if bf16_time:
    if fp32_time:
        speedup = fp32_time / bf16_time
        print(f"BF16: {bf16_time:.4f}s ({speedup:.2f}x speedup vs FP32)")
    else:
        print(f"BF16: {bf16_time:.4f}s")

print()
print("🎯 Expected: FP16/BF16 should be significantly faster than FP32")
print("💡 This shows the performance impact of intermediate precision choices")
