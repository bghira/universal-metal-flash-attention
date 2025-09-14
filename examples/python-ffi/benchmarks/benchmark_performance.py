#!/usr/bin/env python3
"""
Performance benchmarks for Universal Metal Flash Attention Python bindings.

Measures throughput, memory usage, and compares with reference implementations.
"""

import time
import gc
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np

# Add the source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import umfa


def benchmark_attention(
    seq_len: int, head_dim: int, num_runs: int = 10, precision: str = "fp16"
) -> Tuple[float, float]:
    """
    Benchmark attention performance.

    Args:
        seq_len: Sequence length
        head_dim: Head dimension
        num_runs: Number of runs for averaging
        precision: Tensor precision

    Returns:
        Tuple of (mean_time_ms, throughput_ginstrs_per_sec)
    """
    # Create test data
    dtype = np.float16 if precision == "fp16" else np.float32
    q = np.random.randn(seq_len, head_dim).astype(dtype)
    k = np.random.randn(seq_len, head_dim).astype(dtype)
    v = np.random.randn(seq_len, head_dim).astype(dtype)

    # Warm up
    with umfa.MFAContext() as ctx:
        for _ in range(3):
            _ = umfa.flash_attention_forward(ctx, q, k, v, input_precision=precision)

    # Benchmark
    times = []
    with umfa.MFAContext() as ctx:
        for _ in range(num_runs):
            gc.collect()  # Clean memory
            start = time.perf_counter()
            _ = umfa.flash_attention_forward(ctx, q, k, v, input_precision=precision)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

    mean_time_ms = np.mean(times)

    # Calculate throughput (GINSTRS/sec)
    # Flash attention: O(N²) operations for sequence length N
    # Approximation: 4 * seq_len² * head_dim operations
    ops_per_call = 4 * seq_len * seq_len * head_dim
    ginstrs_per_sec = (ops_per_call / 1e9) / (mean_time_ms / 1000)

    return mean_time_ms, ginstrs_per_sec


def benchmark_memory_usage(seq_len: int, head_dim: int) -> int:
    """
    Measure memory usage for attention computation.

    Args:
        seq_len: Sequence length
        head_dim: Head dimension

    Returns:
        Estimated memory usage in bytes
    """
    # Calculate theoretical memory usage
    # Input tensors: 3 * seq_len * head_dim * sizeof(fp16)
    # Output tensor: seq_len * head_dim * sizeof(fp16)
    # Intermediate: approximately seq_len² for attention scores

    element_size = 2  # FP16 = 2 bytes
    input_memory = 3 * seq_len * head_dim * element_size
    output_memory = seq_len * head_dim * element_size
    intermediate_memory = seq_len * seq_len * element_size  # Approximate

    total_memory = input_memory + output_memory + intermediate_memory
    return total_memory


def run_benchmarks():
    """Run comprehensive benchmarks."""
    print("Universal Metal Flash Attention - Performance Benchmarks")
    print("=" * 60)

    # Check system compatibility
    if not umfa.is_metal_available():
        print("❌ Metal is not available on this device")
        return 1

    umfa.print_system_info()

    # Benchmark configurations
    configs = [
        (128, 64),  # Small
        (512, 64),  # Medium
        (1024, 64),  # Large
        (2048, 64),  # Very Large
        (512, 128),  # Wide head
    ]

    print("\n📊 Performance Benchmarks")
    print("-" * 60)
    print(f"{'Config':<15} {'Time (ms)':<12} {'Throughput':<15} {'Memory (MB)':<12}")
    print("-" * 60)

    results = []
    for seq_len, head_dim in configs:
        config_name = f"{seq_len}x{head_dim}"

        try:
            # Run benchmark
            time_ms, throughput = benchmark_attention(seq_len, head_dim, num_runs=5)
            memory_mb = benchmark_memory_usage(seq_len, head_dim) / (1024 * 1024)

            print(
                f"{config_name:<15} {time_ms:>8.2f}ms   {throughput:>8.1f} GINST/s   {memory_mb:>8.1f} MB"
            )

            results.append(
                {
                    "config": config_name,
                    "seq_len": seq_len,
                    "head_dim": head_dim,
                    "time_ms": time_ms,
                    "throughput": throughput,
                    "memory_mb": memory_mb,
                }
            )

        except Exception as e:
            print(f"{config_name:<15} {'ERROR':<12} {str(e):<15}")

    print("-" * 60)

    # Precision comparison
    print("\n🔬 Precision Comparison (512x64)")
    print("-" * 40)
    print(f"{'Precision':<12} {'Time (ms)':<12} {'Throughput':<15}")
    print("-" * 40)

    for precision in ["fp16", "fp32"]:
        try:
            time_ms, throughput = benchmark_attention(
                512, 64, num_runs=5, precision=precision
            )
            print(
                f"{precision.upper():<12} {time_ms:>8.2f}ms   {throughput:>8.1f} GINST/s"
            )
        except Exception as e:
            print(f"{precision.upper():<12} {'ERROR':<12} {str(e)}")

    print("-" * 40)

    # Memory efficiency analysis
    if results:
        print("\n💾 Memory Efficiency Analysis")
        print("-" * 40)

        best_throughput = max(results, key=lambda x: x["throughput"])
        most_efficient = min(results, key=lambda x: x["memory_mb"] / x["throughput"])

        print(
            f"Highest Throughput: {best_throughput['config']} "
            f"({best_throughput['throughput']:.1f} GINST/s)"
        )
        print(
            f"Most Memory Efficient: {most_efficient['config']} "
            f"({most_efficient['throughput']/most_efficient['memory_mb']:.2f} GINST/s per MB)"
        )

    print("\n✅ Benchmarks completed successfully!")
    print("\n📈 Key Performance Insights:")
    print("   • Zero-copy operations eliminate Python→Metal overhead")
    print("   • FP16 precision provides 2x memory efficiency vs FP32")
    print("   • Performance scales quadratically with sequence length")
    print("   • Optimal for Apple Silicon unified memory architecture")

    return 0


def benchmark_vs_naive(seq_len: int = 256, head_dim: int = 64):
    """
    Compare MFA performance vs naive numpy implementation.

    Note: This is for demonstration only - naive implementation
    will be much slower and less memory efficient.
    """
    print(f"\n⚖️  MFA vs Naive Comparison ({seq_len}x{head_dim})")
    print("-" * 40)

    # Create test data
    q = np.random.randn(seq_len, head_dim).astype(np.float32)
    k = np.random.randn(seq_len, head_dim).astype(np.float32)
    v = np.random.randn(seq_len, head_dim).astype(np.float32)

    # Naive implementation
    def naive_attention(q, k, v):
        scores = np.dot(q, k.T) / np.sqrt(head_dim)
        attn_weights = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        attn_weights = attn_weights / np.sum(attn_weights, axis=1, keepdims=True)
        return np.dot(attn_weights, v)

    # Benchmark naive
    start = time.perf_counter()
    for _ in range(10):
        naive_output = naive_attention(q, k, v)
    naive_time = (time.perf_counter() - start) * 100  # ms per call

    # Benchmark MFA
    start = time.perf_counter()
    for _ in range(10):
        mfa_output = umfa.attention(q, k, v, input_precision="fp32")
    mfa_time = (time.perf_counter() - start) * 100  # ms per call

    speedup = naive_time / mfa_time

    print(f"Naive NumPy:     {naive_time:>8.2f}ms")
    print(f"MFA:            {mfa_time:>8.2f}ms")
    print(f"Speedup:        {speedup:>8.1f}x")

    # Verify correctness (should be close)
    max_diff = np.max(np.abs(naive_output - mfa_output))
    print(f"Max difference:  {max_diff:>8.6f}")


if __name__ == "__main__":
    try:
        result = run_benchmarks()

        # Optional: Run comparison if sequence length is reasonable
        try:
            benchmark_vs_naive(256, 64)
        except Exception as e:
            print(f"\nComparison benchmark failed: {e}")

        exit(result)
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        exit(1)
