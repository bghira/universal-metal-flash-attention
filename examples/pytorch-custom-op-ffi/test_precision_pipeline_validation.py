#!/usr/bin/env python3

import os
import sys
import time

import numpy as np
import torch
from metal_sdpa_extension import (
    OutputPrecision,
    QuantizationConfig,
    quantized_scaled_dot_product_attention_with_config,
)

# No path manipulation needed when running from the correct directory


print("🧪 RED-GREEN TEST: Precision Pipeline Validation")
print("=" * 60)
print("🎯 Goal: Validate that outputPrecision affects ALL intermediate computations")
print("   NOT just final buffer interpretation!")
print()

# Test configuration designed to amplify precision differences
batch_size = 1
seq_len_q = 512  # Larger to amplify precision effects
seq_len_kv = 512
num_heads = 8  # Multi-head to stress intermediate calculations
head_dim = 64

device = torch.device("cpu")

# Create test tensors with specific patterns to amplify precision differences
# Use values that will show clear differences between FP32 and FP16 precision
torch.manual_seed(42)  # Reproducible results

# Create inputs with small but significant differences that FP16 vs FP32 will handle differently
query = (
    torch.randn(
        batch_size, seq_len_q, num_heads, head_dim, dtype=torch.float32, device=device
    )
    * 0.1
)
key = (
    torch.randn(
        batch_size, seq_len_kv, num_heads, head_dim, dtype=torch.float32, device=device
    )
    * 0.1
)
value = (
    torch.randn(
        batch_size, seq_len_kv, num_heads, head_dim, dtype=torch.float32, device=device
    )
    * 0.1
)

# Add small precision-sensitive values that should show clear differences
query += torch.full_like(query, 1e-4)  # Small values that FP16 will round differently
key += torch.full_like(key, 1e-4)
value += torch.full_like(value, 1e-4)

print(f"Test Configuration:")
print(f"  Batch Size: {batch_size}")
print(f"  Sequence Length: Q={seq_len_q}, KV={seq_len_kv}")
print(f"  Number of Heads: {num_heads}")
print(f"  Head Dimension: {head_dim}")
print(f"  Input shapes: Q={query.shape}, K={key.shape}, V={value.shape}")
print(f"  Input value ranges: Q=[{query.min():.6f}, {query.max():.6f}]")
print()


def run_precision_test(precision_name, output_precision, rounds=3):
    """Run attention with specific output precision and measure both performance and numerical differences"""
    print(f"🔬 Testing {precision_name} Output Precision")
    print("-" * 40)

    # Create quantization config
    quant_config = QuantizationConfig()
    quant_config.precision = "int8"
    quant_config.output_precision = output_precision
    quant_config.is_causal = False

    print(
        f"Config: precision={quant_config.precision}, output_precision={output_precision}"
    )

    times = []
    outputs = []

    # Warmup
    print("🔥 Warmup...")
    try:
        warmup_output = quantized_scaled_dot_product_attention_with_config(
            query, key, value, quant_config
        )
        print(f"   Warmup output dtype: {warmup_output.dtype}")
        print(
            f"   Warmup output range: [{warmup_output.min():.6f}, {warmup_output.max():.6f}]"
        )
    except Exception as e:
        print(f"❌ Warmup failed: {e}")
        return None, None

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
            outputs.append(output.clone())

            print(f"   Time: {run_time:.4f}s")

        except Exception as e:
            print(f"❌ Run {i+1} failed: {e}")
            return None, None

    avg_time = sum(times) / len(times)
    final_output = outputs[-1]

    print(f"📊 {precision_name} Results:")
    print(f"   Average time: {avg_time:.4f}s")
    print(f"   Output dtype: {final_output.dtype}")
    print(f"   Output shape: {final_output.shape}")
    print(
        f"   Output stats: min={final_output.min():.8f}, max={final_output.max():.8f}, mean={final_output.mean():.8f}"
    )
    print(f"   Output std: {final_output.std():.8f}")
    print()

    return avg_time, final_output


# RED PHASE: Run with different output precisions
print("🔴 RED PHASE: Testing different output precisions")
print("=" * 60)

fp32_time, fp32_output = run_precision_test("FP32", OutputPrecision.FP32)
fp16_time, fp16_output = run_precision_test("FP16", OutputPrecision.FP16)
bf16_time, bf16_output = run_precision_test("BF16", OutputPrecision.BF16)

# GREEN PHASE: Analyze results for evidence of precision pipeline effects
print("🟢 GREEN PHASE: Analyzing precision pipeline effects")
print("=" * 60)

if fp32_output is not None and fp16_output is not None:
    # Calculate numerical differences
    fp32_vs_fp16_diff = torch.abs(fp32_output - fp16_output.to(fp32_output.dtype))
    max_diff = fp32_vs_fp16_diff.max().item()
    mean_diff = fp32_vs_fp16_diff.mean().item()

    print(f"📈 FP32 vs FP16 Numerical Differences:")
    print(f"   Max absolute difference: {max_diff:.8f}")
    print(f"   Mean absolute difference: {mean_diff:.8f}")
    print(
        f"   Relative difference: {mean_diff / fp32_output.abs().mean().item() * 100:.4f}%"
    )

    # Check if differences are significant enough to indicate intermediate precision effects
    if max_diff > 1e-6:
        print("✅ SIGNIFICANT DIFFERENCES DETECTED!")
        print("   This suggests outputPrecision IS affecting intermediate calculations")
    else:
        print("❌ NO SIGNIFICANT DIFFERENCES!")
        print(
            "   This suggests outputPrecision is NOT affecting intermediate calculations"
        )
        print("   Only final buffer interpretation is being changed")
    print()

if fp32_output is not None and bf16_output is not None:
    # Calculate numerical differences for BF16
    fp32_vs_bf16_diff = torch.abs(fp32_output - bf16_output.to(fp32_output.dtype))
    max_diff = fp32_vs_bf16_diff.max().item()
    mean_diff = fp32_vs_bf16_diff.mean().item()

    print(f"📈 FP32 vs BF16 Numerical Differences:")
    print(f"   Max absolute difference: {max_diff:.8f}")
    print(f"   Mean absolute difference: {mean_diff:.8f}")
    print(
        f"   Relative difference: {mean_diff / fp32_output.abs().mean().item() * 100:.4f}%"
    )

    if max_diff > 1e-6:
        print("✅ SIGNIFICANT DIFFERENCES DETECTED!")
        print("   This suggests outputPrecision IS affecting intermediate calculations")
    else:
        print("❌ NO SIGNIFICANT DIFFERENCES!")
        print(
            "   This suggests outputPrecision is NOT affecting intermediate calculations"
        )
    print()

# Performance analysis
print("🚀 Performance Analysis:")
print("-" * 30)

if fp32_time and fp16_time:
    speedup = fp32_time / fp16_time
    print(f"FP32: {fp32_time:.4f}s (baseline)")
    print(f"FP16: {fp16_time:.4f}s ({speedup:.2f}x speedup)")

    if speedup > 1.5:
        print("✅ SIGNIFICANT SPEEDUP!")
        print("   This suggests intermediate computations are actually using FP16")
    elif speedup > 1.1:
        print("⚠️  MODEST SPEEDUP")
        print("   Some intermediate effects, but may not be comprehensive")
    else:
        print("❌ NO MEANINGFUL SPEEDUP!")
        print("   Suggests intermediate computations are still using FP32")

if fp32_time and bf16_time:
    speedup = fp32_time / bf16_time
    print(f"BF16: {bf16_time:.4f}s ({speedup:.2f}x speedup)")

    if speedup > 1.5:
        print("✅ SIGNIFICANT SPEEDUP!")
        print("   This suggests intermediate computations are actually using BF16")
    elif speedup > 1.1:
        print("⚠️  MODEST SPEEDUP")
        print("   Some intermediate effects, but may not be comprehensive")
    else:
        print("❌ NO MEANINGFUL SPEEDUP!")
        print("   Suggests intermediate computations are still using FP32")

print()
print("🎯 CONCLUSION:")
print("=" * 40)
print("Expected for RED-GREEN PASS:")
print("✅ Significant numerical differences (>1e-6) between precisions")
print("✅ Significant performance speedup (>1.5x) for FP16/BF16")
print("✅ This would indicate outputPrecision affects intermediate calculations")
print()
print("Expected for RED-GREEN FAIL:")
print("❌ Minimal numerical differences (<1e-6)")
print("❌ Minimal performance improvement (<1.1x)")
print("❌ This would indicate outputPrecision only affects final buffer interpretation")
print()
print("💡 If this test FAILS, we need to fix the precision pipeline to ensure")
print("   outputPrecision affects ALL compute intermediaries and shuffled buffers!")
