#!/usr/bin/env python3

import time
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "examples/python-ffi/src"))

import umfa


def benchmark_metal_fa():
    print("🚀 Metal Flash Attention Performance Test")
    print("=" * 50)

    # Test configuration similar to PyTorch benchmark
    seq_len = 512
    head_dim = 64

    # Create test data (FP16 like PyTorch benchmark)
    q = np.random.randn(seq_len, head_dim).astype(np.float16)
    k = np.random.randn(seq_len, head_dim).astype(np.float16)
    v = np.random.randn(seq_len, head_dim).astype(np.float16)

    print(f"✅ Testing seq_len={seq_len}, head_dim={head_dim}")

    with umfa.MFAContext() as ctx:
        # Warmup
        for _ in range(5):
            _ = umfa.flash_attention_forward(
                ctx, q, k, v, causal=True, input_precision="fp16"
            )

        # Benchmark
        times = []
        for i in range(20):
            start = time.time()
            output = umfa.flash_attention_forward(
                ctx, q, k, v, causal=True, input_precision="fp16"
            )
            times.append(time.time() - start)

            if i == 0:
                print(
                    f"✅ First output range: [{output.min():.4f}, {output.max():.4f}]"
                )

        mean_time = np.mean(times) * 1000  # ms
        std_time = np.std(times) * 1000
        min_time = np.min(times) * 1000

        print(f"📊 Wall-clock time: {mean_time:.2f} ± {std_time:.2f} ms")
        print(f"📊 Min time: {min_time:.2f} ms")
        print(
            f"📊 GFLOPS estimate: {(2 * seq_len * seq_len * head_dim) / (mean_time * 1e6):.1f}"
        )


if __name__ == "__main__":
    benchmark_metal_fa()
