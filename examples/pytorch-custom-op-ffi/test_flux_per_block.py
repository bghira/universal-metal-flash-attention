#!/usr/bin/env python3
"""
Enhanced FLUX test with per-block quantization
"""

import time

import numpy as np
import torch
from metal_sdpa_extension import (
    quantized_scaled_dot_product_attention_per_block,  # New function
)
from metal_sdpa_extension import (
    OutputPrecision,
    QuantizationConfig,
    quantized_scaled_dot_product_attention_with_config,
)

print("🔥 FLUX Per-Block Quantization Test")
print("=" * 60)

# FLUX-style attention dimensions
batch_size = 1
seq_len = 4096  # FLUX typical sequence length
num_heads = 16  # FLUX attention heads
head_dim = 64  # Head dimension

# Create test tensors
query = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)
key = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)
value = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)

print(f"FLUX Configuration:")
print(f"  Sequence Length: {seq_len}")
print(f"  Number of Heads: {num_heads}")
print(f"  Head Dimension: {head_dim}")
print(f"  Total Parameters: {seq_len * num_heads * head_dim * 3:,}")


def benchmark_method(name, method_func, *args, rounds=3):
    """Benchmark a quantization method"""
    print(f"\n🧪 Testing {name}")
    print("-" * 40)

    times = []

    # Warmup
    try:
        _ = method_func(*args)
        print("✅ Warmup completed")
    except Exception as e:
        print(f"❌ {name} failed: {e}")
        return None

    # Benchmark
    for i in range(rounds):
        start = time.time()
        try:
            output = method_func(*args)
            end = time.time()

            run_time = end - start
            times.append(run_time)

            # Check for NaN/inf
            has_nan = torch.isnan(output).any()
            has_inf = torch.isinf(output).any()
            max_val = output.abs().max().item()

            print(
                f"  Run {i+1}: {run_time:.4f}s, max={max_val:.6f}, nan={has_nan}, inf={has_inf}"
            )

        except Exception as e:
            print(f"❌ Run {i+1} failed: {e}")
            return None

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"📊 {name} Results:")
    print(f"   Average: {avg_time:.4f}s ± {std_time:.4f}s")
    print(f"   Best:    {min(times):.4f}s")

    return avg_time


# Test 1: Per-tensor quantization (current)
config_tensor = QuantizationConfig()
config_tensor.precision = "int8"
config_tensor.output_precision = OutputPrecision.FP32
config_tensor.is_causal = False

tensor_time = benchmark_method(
    "Per-Tensor Quantization (Current)",
    quantized_scaled_dot_product_attention_with_config,
    query,
    key,
    value,
    config_tensor,
)

# Test 2: Per-block quantization (new) - SageAttention style
block_time = benchmark_method(
    "Per-Block Quantization (New - SageAttention)",
    quantized_scaled_dot_product_attention_per_block,
    query,
    key,
    value,
    128,  # q_block_size
    64,  # k_block_size
    64,  # v_block_size
    "int8",
    False,  # is_causal
    None,  # scale
)

# Test 3: Different block sizes
block_time_small = benchmark_method(
    "Per-Block Quantization (Smaller Blocks)",
    quantized_scaled_dot_product_attention_per_block,
    query,
    key,
    value,
    64,  # q_block_size (smaller)
    32,  # k_block_size (smaller)
    32,  # v_block_size (smaller)
    "int8",
    False,
    None,
)

# Summary
print("\n🏆 FLUX PERFORMANCE SUMMARY")
print("=" * 60)

if tensor_time:
    print(f"Per-Tensor (Current):     {tensor_time:.4f}s")

if block_time:
    print(f"Per-Block (SageAttention): {block_time:.4f}s")
    if tensor_time:
        improvement = (tensor_time - block_time) / tensor_time * 100
        print(f"  → Improvement: {improvement:+.1f}%")

if block_time_small:
    print(f"Per-Block (Small blocks): {block_time_small:.4f}s")
    if tensor_time:
        improvement = (tensor_time - block_time_small) / tensor_time * 100
        print(f"  → Improvement: {improvement:+.1f}%")

print("\n💡 Expected Results:")
print("  - Per-block should fix NaN/inf issues at higher scales")
print("  - Per-block may be slightly slower due to more complex kernel")
print("  - But per-block should be more numerically stable")
