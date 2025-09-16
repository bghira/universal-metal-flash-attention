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
seq_len = 1024  # Smaller for faster testing
num_heads = 8
head_dim = 64


def analyze_tensor_layout(tensor, name):
    """Analyze and print tensor layout information"""
    print(f"📊 {name} Layout Analysis:")
    print(f"   Shape: {list(tensor.shape)}")
    print(f"   Strides: {list(tensor.stride())}")
    print(f"   Is contiguous: {tensor.is_contiguous()}")
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

    elif layout_name == "transposed":
        # Create with different stride pattern via transpose
        query = torch.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=torch.float32
        ).transpose(1, 2)
        key = torch.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=torch.float32
        ).transpose(1, 2)
        value = torch.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=torch.float32
        ).transpose(1, 2)

    elif layout_name == "head_optimized":
        # Layout where heads are contiguous: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads, head_dim)
        query = torch.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=torch.float32
        ).permute(0, 2, 1, 3)
        key = torch.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=torch.float32
        ).permute(0, 2, 1, 3)
        value = torch.randn(
            batch_size, num_heads, seq_len, head_dim, dtype=torch.float32
        ).permute(0, 2, 1, 3)

    elif layout_name == "dim_interleaved":
        # Create tensors where head_dim is not the innermost dimension
        query = torch.randn(
            batch_size, head_dim, seq_len, num_heads, dtype=torch.float32
        ).permute(0, 2, 3, 1)
        key = torch.randn(
            batch_size, head_dim, seq_len, num_heads, dtype=torch.float32
        ).permute(0, 2, 3, 1)
        value = torch.randn(
            batch_size, head_dim, seq_len, num_heads, dtype=torch.float32
        ).permute(0, 2, 3, 1)

    else:
        raise ValueError(f"Unknown layout: {layout_name}")

    return query, key, value


def benchmark_layout(layout_name, rounds=3):
    """Benchmark a specific memory layout"""
    print(f"\n🧪 Testing {layout_name} layout")
    print("-" * 40)

    # Create tensors
    query, key, value = create_tensors_with_layout(layout_name)

    # Analyze layouts - just query for brevity
    analyze_tensor_layout(query, "Query")

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


def analyze_memory_access_patterns():
    """Analyze how our current implementation handles memory access"""
    print(f"\n🔍 MEMORY ACCESS PATTERN ANALYSIS")
    print("=" * 60)

    # Create test tensors with different layouts
    layouts_to_test = ["contiguous", "transposed", "head_optimized", "dim_interleaved"]

    for layout in layouts_to_test:
        print(f"\n{layout.upper()} Layout:")
        query, key, value = create_tensors_with_layout(layout)

        # Calculate memory footprint and access patterns
        q_stride = query.stride()
        expected_access_pattern = []

        # For attention, we typically access:
        # - Q: [batch, seq_q, heads, head_dim]
        # - K: [batch, seq_k, heads, head_dim]
        # - V: [batch, seq_k, heads, head_dim]

        print(f"   Query strides: {q_stride}")
        print(f"   Head-wise access cost: {q_stride[2]} elements between heads")
        print(
            f"   Seq-wise access cost: {q_stride[1]} elements between sequence positions"
        )
        print(f"   Feature-wise access cost: {q_stride[3]} elements between features")

        # Estimate cache efficiency
        if q_stride == [1048576, 1024, 64, 1]:  # Contiguous
            print(f"   ✅ Optimal for head_dim iteration (stride=1)")
        elif q_stride[3] == 1:
            print(f"   ✅ Good for head_dim iteration (stride=1)")
        else:
            print(f"   ⚠️  Sub-optimal for head_dim iteration (stride={q_stride[3]})")


# Test different layouts
layouts = ["contiguous", "transposed", "head_optimized", "dim_interleaved"]
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

# Analyze memory access patterns
analyze_memory_access_patterns()

print(f"\n💡 Key Findings:")
print(f"   - Our C++ backend forces contiguous layout via ensure_contiguous_cpu()")
print(f"   - Any layout optimizations are lost during this conversion")
print(f"   - To leverage layouts, we'd need to modify the backend")
print(f"   - Benefits could include: better cache locality, reduced memory copies")
print(f"   - For quantization: layout could impact quantization block efficiency")
