"""
Performance benchmarks for stride-aware vs contiguous tensor handling.

This module benchmarks the performance impact of handling non-contiguous
tensors directly versus the traditional approach of making them contiguous.
"""

import gc
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import metal_sdpa_extension

    HAS_METAL = True
except ImportError:
    HAS_METAL = False
    print("Warning: metal_sdpa_extension not available")


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    name: str
    batch_size: int
    num_heads: int
    seq_len: int
    head_dim: int
    time_ms: float
    memory_bytes: int
    is_contiguous: bool
    success: bool
    error: Optional[str] = None
    speedup: Optional[float] = None


class StrideBenchmark:
    """Benchmark suite for stride-aware attention performance."""

    def __init__(
        self, device: str = "mps", warmup_runs: int = 3, benchmark_runs: int = 10
    ):
        self.device = device if torch.backends.mps.is_available() else "cpu"
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results: List[BenchmarkResult] = []

    def measure_memory(self) -> int:
        """Measure current GPU memory usage."""
        if self.device == "mps":
            # MPS doesn't have direct memory query, estimate from tensors
            return 0  # Placeholder
        elif self.device.startswith("cuda"):
            return torch.cuda.memory_allocated()
        return 0

    def benchmark_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, name: str = "unnamed"
    ) -> Tuple[float, bool, Optional[str]]:
        """
        Benchmark a single attention operation.

        Returns:
            Tuple of (time_ms, success, error_message)
        """
        try:
            # Warmup
            for _ in range(self.warmup_runs):
                _ = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

            # Synchronize
            if self.device == "mps":
                torch.mps.synchronize()
            elif self.device.startswith("cuda"):
                torch.cuda.synchronize()

            # Benchmark
            start_time = time.perf_counter()
            for _ in range(self.benchmark_runs):
                output = metal_sdpa_extension.metal_scaled_dot_product_attention(
                    q, k, v
                )

            # Synchronize
            if self.device == "mps":
                torch.mps.synchronize()
            elif self.device.startswith("cuda"):
                torch.cuda.synchronize()

            elapsed_time = (time.perf_counter() - start_time) / self.benchmark_runs
            elapsed_ms = elapsed_time * 1000

            # Validate output
            if torch.isnan(output).any() or torch.isinf(output).any():
                return elapsed_ms, False, "Output contains NaN or Inf"

            return elapsed_ms, True, None

        except Exception as e:
            return 0.0, False, str(e)

    def run_configuration_benchmark(
        self,
        batch_size: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
    ) -> Dict[str, BenchmarkResult]:
        """Run benchmarks for a specific configuration."""
        print(
            f"\nBenchmarking B={batch_size}, H={num_heads}, S={seq_len}, D={head_dim}"
        )

        results = {}

        # Create base tensors in FLUX layout [B,H,S,D]
        torch.manual_seed(42)
        flux_shape = (batch_size, num_heads, seq_len, head_dim)
        q_flux = torch.randn(flux_shape, dtype=dtype, device=self.device) * 0.1
        k_flux = torch.randn(flux_shape, dtype=dtype, device=self.device) * 0.1
        v_flux = torch.randn(flux_shape, dtype=dtype, device=self.device) * 0.1

        # 1. Benchmark with contiguous FLUX layout
        print("  Testing contiguous FLUX layout...")
        time_ms, success, error = self.benchmark_attention(
            q_flux, k_flux, v_flux, "FLUX contiguous"
        )
        results["flux_contiguous"] = BenchmarkResult(
            name="FLUX contiguous",
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            time_ms=time_ms,
            memory_bytes=q_flux.numel() * q_flux.element_size() * 3,
            is_contiguous=True,
            success=success,
            error=error,
        )

        # 2. Create non-contiguous Metal layout via permute
        q_metal_perm = q_flux.permute(0, 2, 1, 3)  # [B,H,S,D] -> [B,S,H,D]
        k_metal_perm = k_flux.permute(0, 2, 1, 3)
        v_metal_perm = v_flux.permute(0, 2, 1, 3)

        print("  Testing non-contiguous Metal layout (permuted)...")
        time_ms, success, error = self.benchmark_attention(
            q_metal_perm, k_metal_perm, v_metal_perm, "Metal non-contiguous"
        )
        results["metal_noncontig"] = BenchmarkResult(
            name="Metal non-contiguous",
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            time_ms=time_ms,
            memory_bytes=q_metal_perm.numel() * q_metal_perm.element_size() * 3,
            is_contiguous=False,
            success=success,
            error=error,
        )

        # 3. Benchmark with explicit contiguous() call
        print("  Testing Metal layout with contiguous() call...")

        # Time the full operation including contiguous() call
        start_time = time.perf_counter()
        for _ in range(self.benchmark_runs):
            q_cont = q_metal_perm.contiguous()
            k_cont = k_metal_perm.contiguous()
            v_cont = v_metal_perm.contiguous()
            output = metal_sdpa_extension.metal_scaled_dot_product_attention(
                q_cont, k_cont, v_cont
            )

        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device.startswith("cuda"):
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start_time) / self.benchmark_runs * 1000

        results["metal_contiguous"] = BenchmarkResult(
            name="Metal with contiguous()",
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            time_ms=elapsed_ms,
            memory_bytes=q_metal_perm.numel()
            * q_metal_perm.element_size()
            * 6,  # Double memory
            is_contiguous=True,
            success=True,
            error=None,
        )

        # Calculate speedups
        if results["metal_contiguous"].success and results["metal_noncontig"].success:
            speedup = (
                results["metal_contiguous"].time_ms / results["metal_noncontig"].time_ms
            )
            results["metal_noncontig"].speedup = speedup

        return results

    def run_full_benchmark_suite(self):
        """Run complete benchmark suite with various configurations."""
        print("=" * 80)
        print("Stride-Aware Attention Performance Benchmarks")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Warmup runs: {self.warmup_runs}")
        print(f"Benchmark runs: {self.benchmark_runs}")

        # Test configurations
        configurations = [
            # (batch, heads, seq_len, head_dim, description)
            (1, 12, 77, 64, "FLUX text encoder"),
            (1, 24, 256, 128, "FLUX small image"),
            (1, 24, 1024, 128, "FLUX medium image"),
            (2, 24, 1024, 128, "FLUX batched medium"),
            (1, 24, 2048, 128, "FLUX large image"),
            (4, 32, 512, 96, "Large multi-head"),
            (1, 8, 4096, 64, "Long sequence"),
        ]

        all_results = []

        for batch, heads, seq_len, dim, desc in configurations:
            print(f"\n{desc}:")
            config_results = self.run_configuration_benchmark(
                batch, heads, seq_len, dim
            )

            # Print summary for this configuration
            for key, result in config_results.items():
                if result.success:
                    print(f"  {result.name}: {result.time_ms:.3f} ms")
                    if result.speedup:
                        print(f"    Speedup vs contiguous: {result.speedup:.2f}x")
                else:
                    print(f"  {result.name}: FAILED - {result.error}")

            all_results.extend(config_results.values())

        self.results = all_results
        return all_results

    def print_summary(self):
        """Print a summary of benchmark results."""
        print("\n" + "=" * 80)
        print("Benchmark Summary")
        print("=" * 80)

        if not self.results:
            print("No results to display")
            return

        # Group results by configuration
        configs = {}
        for r in self.results:
            key = (r.batch_size, r.num_heads, r.seq_len, r.head_dim)
            if key not in configs:
                configs[key] = []
            configs[key].append(r)

        # Calculate statistics
        total_speedup = []
        memory_saved = []

        for config, results in configs.items():
            batch, heads, seq, dim = config
            print(f"\nConfig: B={batch}, H={heads}, S={seq}, D={dim}")

            # Find contiguous and non-contiguous results
            contig = next((r for r in results if "contiguous()" in r.name), None)
            noncontig = next((r for r in results if "non-contiguous" in r.name), None)

            if contig and noncontig and contig.success and noncontig.success:
                speedup = contig.time_ms / noncontig.time_ms
                mem_save = contig.memory_bytes - noncontig.memory_bytes

                print(f"  Time saved: {contig.time_ms - noncontig.time_ms:.3f} ms")
                print(f"  Speedup: {speedup:.2f}x")
                print(f"  Memory saved: {mem_save / 1024 / 1024:.2f} MB")

                total_speedup.append(speedup)
                memory_saved.append(mem_save)

        if total_speedup:
            print("\n" + "-" * 40)
            print("Overall Statistics:")
            print(f"  Average speedup: {np.mean(total_speedup):.2f}x")
            print(f"  Min speedup: {np.min(total_speedup):.2f}x")
            print(f"  Max speedup: {np.max(total_speedup):.2f}x")
            print(f"  Total memory saved: {sum(memory_saved) / 1024 / 1024:.2f} MB")

    def save_results(self, filename: str = "stride_benchmark_results.json"):
        """Save benchmark results to JSON file."""
        if not self.results:
            print("No results to save")
            return

        data = {
            "timestamp": datetime.now().isoformat(),
            "device": self.device,
            "warmup_runs": self.warmup_runs,
            "benchmark_runs": self.benchmark_runs,
            "results": [asdict(r) for r in self.results],
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\nResults saved to {filename}")


def run_memory_allocation_benchmark():
    """Benchmark memory allocation overhead for contiguous() calls."""
    print("\n" + "=" * 80)
    print("Memory Allocation Overhead Benchmark")
    print("=" * 80)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    sizes = [
        (1, 12, 77, 64, "Small"),
        (1, 24, 1024, 128, "Medium"),
        (2, 24, 2048, 128, "Large"),
    ]

    for batch, heads, seq, dim, desc in sizes:
        print(f"\n{desc}: B={batch}, H={heads}, S={seq}, D={dim}")

        # Create non-contiguous tensor
        flux = torch.randn(batch, heads, seq, dim, dtype=torch.float16, device=device)
        metal = flux.permute(0, 2, 1, 3)  # Non-contiguous

        # Measure contiguous() overhead
        num_iterations = 100

        # Warmup
        for _ in range(10):
            _ = metal.contiguous()

        # Benchmark contiguous() call
        torch.mps.synchronize() if device == "mps" else None
        start = time.perf_counter()
        for _ in range(num_iterations):
            cont = metal.contiguous()
        torch.mps.synchronize() if device == "mps" else None
        elapsed = (time.perf_counter() - start) / num_iterations * 1000

        memory_size = cont.numel() * cont.element_size() / 1024 / 1024  # MB

        print(f"  contiguous() time: {elapsed:.3f} ms")
        print(f"  Memory allocated: {memory_size:.2f} MB")
        print(f"  Throughput: {memory_size / (elapsed/1000):.0f} MB/s")

        # Clean up
        del flux, metal, cont
        gc.collect()
        torch.mps.empty_cache() if device == "mps" else None


def run_stride_pattern_benchmark():
    """Benchmark different stride patterns and their performance impact."""
    print("\n" + "=" * 80)
    print("Stride Pattern Performance Impact")
    print("=" * 80)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    benchmark = StrideBenchmark(device=device, warmup_runs=3, benchmark_runs=10)

    # Base configuration
    batch, heads, seq_len, dim = 1, 12, 256, 64

    patterns = [
        ("Contiguous", lambda t: t),
        ("Permute(0,2,1,3)", lambda t: t.permute(0, 2, 1, 3)),
        ("Transpose(1,2)", lambda t: t.transpose(1, 2)),
        ("Narrow (slice)", lambda t: t[:, :, : seq_len // 2, :]),
        ("Stride-2 slice", lambda t: t[:, :, ::2, :]),
    ]

    print(f"Base shape: B={batch}, H={heads}, S={seq_len}, D={dim}")

    for name, transform in patterns:
        print(f"\n{name}:")

        # Create base tensors
        torch.manual_seed(42)
        q = (
            torch.randn(batch, heads, seq_len, dim, dtype=torch.float16, device=device)
            * 0.1
        )
        k = (
            torch.randn(batch, heads, seq_len, dim, dtype=torch.float16, device=device)
            * 0.1
        )
        v = (
            torch.randn(batch, heads, seq_len, dim, dtype=torch.float16, device=device)
            * 0.1
        )

        # Apply transformation
        q_t = transform(q)
        k_t = transform(k)
        v_t = transform(v)

        print(f"  Shape: {list(q_t.shape)}")
        print(f"  Stride: {q_t.stride()}")
        print(f"  Contiguous: {q_t.is_contiguous()}")

        # Benchmark
        time_ms, success, error = benchmark.benchmark_attention(q_t, k_t, v_t)

        if success:
            print(f"  Time: {time_ms:.3f} ms")

            # Compare with contiguous version
            q_c = q_t.contiguous()
            k_c = k_t.contiguous()
            v_c = v_t.contiguous()
            time_cont_ms, _, _ = benchmark.benchmark_attention(q_c, k_c, v_c)

            overhead = (time_ms - time_cont_ms) / time_cont_ms * 100
            print(f"  Contiguous time: {time_cont_ms:.3f} ms")
            print(f"  Overhead: {overhead:+.1f}%")
        else:
            print(f"  Failed: {error}")


if __name__ == "__main__":
    if not HAS_METAL:
        print("Metal SDPA extension not available. Exiting.")
        exit(1)

    # Run main benchmark suite
    benchmark = StrideBenchmark()
    benchmark.run_full_benchmark_suite()
    benchmark.print_summary()
    benchmark.save_results()

    # Run additional specialized benchmarks
    run_memory_allocation_benchmark()
    run_stride_pattern_benchmark()

    print("\n" + "=" * 80)
    print("All benchmarks completed")
    print("=" * 80)
