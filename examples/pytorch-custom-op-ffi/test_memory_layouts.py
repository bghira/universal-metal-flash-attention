#!/usr/bin/env python3
"""
Test the impact of different PyTorch memory layouts on our quantized attention
"""

import time

import numpy as np
import torch
from metal_sdpa_extension import (
    OutputPrecision,
    QuantizationConfig,
    quantized_scaled_dot_product_attention_per_block,
    quantized_scaled_dot_product_attention_with_config,
)

print("🧠 Memory Layout Impact Analysis")
print("=" * 60)

# FLUX-style attention dimensions
batch_size = 1
seq_len = 2048  # Slightly smaller for faster testing
num_heads = 16
head_dim = 64


def analyze_tensor_layout(tensor, name):
    """Analyze and print tensor layout information"""
    print(f"📊 {name} Layout Analysis:")
    print(f"   Shape: {list(tensor.shape)}")
    print(f"   Strides: {list(tensor.stride())}")
    print(f"   Is contiguous: {tensor.is_contiguous()}")
    print(f"   Memory format: {tensor.memory_format}")
    print(f"   Storage offset: {tensor.storage_offset()}")
    print(f"   Data ptr: {tensor.data_ptr()}")
    print()


def create_tensors_with_layout(layout_name):
    """Create tensors with specific memory layout"""
    print(f"🔧 Creating tensors with {layout_name} layout...")

    if layout_name == "contiguous":
        # Standard contiguous layout: (batch, seq_len, num_heads, head_dim)
        query = torch.randn(
            batch_size, seq_len, num_heads, head_dim, dtype=torch.float32
        )
        key = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)
        value = torch.randn(
            batch_size, seq_len, num_heads, head_dim, dtype=torch.float32
        )

    elif layout_name == "channels_last":
        # Channels last: optimized for (N, H, W, C) -> (batch, seq_len, head_dim, num_heads)
        # Note: channels_last is typically for 4D tensors like images (N,C,H,W) -> (N,H,W,C)
        # For attention, we can try: (batch, seq_len, num_heads, head_dim) -> (batch, seq_len, head_dim, num_heads)
        query = torch.randn(
            batch_size, seq_len, head_dim, num_heads, dtype=torch.float32
        ).contiguous(memory_format=torch.channels_last)
        key = torch.randn(
            batch_size, seq_len, head_dim, num_heads, dtype=torch.float32
        ).contiguous(memory_format=torch.channels_last)
        value = torch.randn(
            batch_size, seq_len, head_dim, num_heads, dtype=torch.float32
        ).contiguous(memory_format=torch.channels_last)

        # Convert back to expected shape but preserve layout
        query = query.permute(0, 1, 3, 2)  # (batch, seq_len, num_heads, head_dim)
        key = key.permute(0, 1, 3, 2)
        value = value.permute(0, 1, 3, 2)

    elif layout_name == "transposed":
        # Create with different stride pattern
        query = torch.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=torch.float32
        ).transpose(1, 2)
        key = torch.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=torch.float32
        ).transpose(1, 2)
        value = torch.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=torch.float32
        ).transpose(1, 2)

    elif layout_name == "head_first":
        # Layout optimized for head-wise operations: (batch, num_heads, seq_len, head_dim)
        query = torch.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=torch.float32
        )
        key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float32)
        value = torch.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=torch.float32
        )

        # Convert to expected shape (batch, seq_len, num_heads, head_dim)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

    else:
        raise ValueError(f"Unknown layout: {layout_name}")

    return query, key, value


def benchmark_layout(layout_name, rounds=3):
    """Benchmark a specific memory layout"""
    print(f"\n🧪 Testing {layout_name} layout")
    print("-" * 40)

    # Create tensors
    query, key, value = create_tensors_with_layout(layout_name)

    # Analyze layouts
    analyze_tensor_layout(query, "Query")
    analyze_tensor_layout(key, "Key")
    analyze_tensor_layout(value, "Value")

    # Test per-tensor quantization
    config = QuantizationConfig()
    config.precision = "int8"
    config.output_precision = OutputPrecision.FP32
    config.is_causal = False

    print(f"⏱️ Benchmarking per-tensor quantization...")
    times_tensor = []

    for i in range(rounds):
        start = time.time()
        try:
            result = quantized_scaled_dot_product_attention_with_config(
                query, key, value, config
            )
            end = time.time()
            times_tensor.append(end - start)

            # Check results
            has_nan = torch.isnan(result).any()
            has_inf = torch.isinf(result).any()
            print(f"   Run {i+1}: {end-start:.4f}s, nan={has_nan}, inf={has_inf}")

        except Exception as e:
            print(f"   Run {i+1}: FAILED - {e}")
            return None, None

    # Test per-block quantization
    print(f"⏱️ Benchmarking per-block quantization...")
    times_block = []

    for i in range(rounds):
        start = time.time()
        try:
            result = quantized_scaled_dot_product_attention_per_block(
                query, key, value, 128, 64, 64, "int8", False, None
            )
            end = time.time()
            times_block.append(end - start)

            # Check results
            has_nan = torch.isnan(result).any()
            has_inf = torch.isinf(result).any()
            print(f"   Run {i+1}: {end-start:.4f}s, nan={has_nan}, inf={has_inf}")

        except Exception as e:
            print(f"   Run {i+1}: FAILED - {e}")
            return np.mean(times_tensor), None

    avg_tensor = np.mean(times_tensor)
    avg_block = np.mean(times_block)

    print(f"📈 Results for {layout_name}:")
    print(f"   Per-tensor: {avg_tensor:.4f}s ± {np.std(times_tensor):.4f}s")
    print(f"   Per-block:  {avg_block:.4f}s ± {np.std(times_block):.4f}s")

    return avg_tensor, avg_block


# Test different layouts
layouts = ["contiguous", "channels_last", "transposed", "head_first"]
results = {}

for layout in layouts:
    try:
        tensor_time, block_time = benchmark_layout(layout)
        results[layout] = {"tensor": tensor_time, "block": block_time}
    except Exception as e:
        print(f"❌ {layout} layout failed: {e}")
        results[layout] = {"tensor": None, "block": None}

# Summary
print("\n🏆 MEMORY LAYOUT PERFORMANCE SUMMARY")
print("=" * 60)

baseline_tensor = results.get("contiguous", {}).get("tensor")
baseline_block = results.get("contiguous", {}).get("block")

for layout, times in results.items():
    tensor_time = times["tensor"]
    block_time = times["block"]

    print(f"\n{layout.upper()} Layout:")
    if tensor_time is not None:
        if baseline_tensor:
            improvement = (baseline_tensor - tensor_time) / baseline_tensor * 100
            print(
                f"   Per-tensor: {tensor_time:.4f}s ({improvement:+.1f}% vs contiguous)"
            )
        else:
            print(f"   Per-tensor: {tensor_time:.4f}s")
    else:
        print(f"   Per-tensor: FAILED")

    if block_time is not None:
        if baseline_block:
            improvement = (baseline_block - block_time) / baseline_block * 100
            print(
                f"   Per-block:  {block_time:.4f}s ({improvement:+.1f}% vs contiguous)"
            )
        else:
            print(f"   Per-block:  {block_time:.4f}s")
    else:
        print(f"   Per-block:  FAILED")

print(f"\n💡 Analysis:")
print(f"   - Our implementation converts all tensors to contiguous format")
print(f"   - Layout optimizations might not show benefits due to this conversion")
print(f"   - Consider modifying the backend to preserve beneficial layouts")
print(f"   - channels_last might help with cache locality for certain operations")
