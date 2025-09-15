#!/usr/bin/env python3
"""
Optimized FLUX.1-Schnell benchmark with warmup and pipeline reuse
"""
import gc
import sys
import time
from pathlib import Path

import psutil
import torch
from diffusers import FluxPipeline

# Add pytorch-custom-op-ffi build to path
sys.path.append(
    str(
        Path(__file__).parent
        / "pytorch-custom-op-ffi"
        / "build"
        / "lib.macosx-15.0-arm64-cpython-312"
    )
)

try:
    import metal_sdpa_extension

    METAL_PYTORCH_AVAILABLE = True
    print("✅ Metal PyTorch Custom Op available")
except ImportError as e:
    METAL_PYTORCH_AVAILABLE = False
    print(f"❌ Metal PyTorch Custom Op not available: {e}")
    sys.exit(1)


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def benchmark_metal_pytorch_optimized():
    """Optimized benchmark for Metal PyTorch Custom Op with warmup"""
    print(f"\n{'='*60}")
    print(f"🚀 FLUX.1-Schnell Metal PyTorch Custom Op (Optimized with Warmup)")
    print(f"{'='*60}")

    # Load the pipeline once
    print("📥 Loading FLUX.1-Schnell pipeline...")
    start_time = time.time()
    load_start_memory = get_memory_usage()

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
    )
    pipe.to("mps")

    # Register Metal backend
    metal_sdpa_extension.register_backend()
    print("🔧 Metal PyTorch Custom Op backend registered")

    load_time = time.time() - start_time
    load_memory = get_memory_usage() - load_start_memory

    print(f"✅ Pipeline loaded and moved to MPS")
    print(f"⏱️  Pipeline load time: {load_time:.2f}s")
    print(f"💾 Memory used for loading: {load_memory:.1f} MB")

    prompt = "A futuristic cityscape with flying cars and neon lights, highly detailed digital art"

    # Warmup run
    print("🔥 Performing warmup run...")
    warmup_start = time.time()
    with torch.inference_mode():
        warmup_image = pipe(
            prompt=prompt,
            guidance_scale=0.0,
            num_inference_steps=4,
            height=1024,
            width=1024,
            generator=torch.Generator("cpu").manual_seed(
                999
            ),  # Different seed for warmup
        ).images[0]

    warmup_time = time.time() - warmup_start
    print(f"⏱️  Warmup generation time: {warmup_time:.2f}s")
    warmup_image.save("flux_schnell_warmup.png")
    print("📸 Warmup image saved")

    # Main benchmark runs (multiple for consistency)
    print(f"\n🎯 Running benchmark generations...")
    generation_times = []

    for i in range(3):  # Run 3 times for consistent results
        print(f"🎨 Benchmark run {i+1}/3...")

        start_memory = get_memory_usage()
        start_time = time.time()

        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                guidance_scale=0.0,
                num_inference_steps=4,
                height=1024,
                width=1024,
                generator=torch.Generator("cpu").manual_seed(
                    42 + i
                ),  # Vary seed slightly
            ).images[0]

        generation_time = time.time() - start_time
        generation_times.append(generation_time)
        peak_memory = get_memory_usage() - start_memory

        print(f"   ⏱️  Run {i+1} time: {generation_time:.2f}s")

        if i == 0:  # Save first benchmark image and measure memory
            image.save("flux_schnell_metal_pytorch_optimized.png")
            print(f"   💾 Peak memory usage: {peak_memory:.1f} MB")
            benchmark_memory = peak_memory

    # Calculate statistics
    avg_time = sum(generation_times) / len(generation_times)
    min_time = min(generation_times)
    max_time = max(generation_times)

    print(f"\n{'='*60}")
    print(f"📊 PERFORMANCE RESULTS")
    print(f"{'='*60}")
    print(f"Pipeline load time:     {load_time:.2f}s")
    print(f"Warmup generation:      {warmup_time:.2f}s")
    print(f"Average generation:     {avg_time:.2f}s")
    print(f"Best generation:        {min_time:.2f}s")
    print(f"Worst generation:       {max_time:.2f}s")
    print(f"Load memory usage:      {load_memory:.1f} MB")
    print(f"Peak memory usage:      {benchmark_memory:.1f} MB")

    print(f"\n🚀 Our optimized multi-head attention delivered:")
    print(f"   • Consistent sub-{avg_time:.0f}s generation times")
    print(f"   • Metal GPU acceleration with warmup benefits")
    print(f"   • Efficient memory usage: {benchmark_memory:.0f}MB peak")

    # Cleanup
    metal_sdpa_extension.unregister_backend()
    del pipe
    torch.mps.empty_cache()
    gc.collect()

    return {
        "load_time": load_time,
        "warmup_time": warmup_time,
        "avg_generation_time": avg_time,
        "min_generation_time": min_time,
        "max_generation_time": max_time,
        "load_memory": load_memory,
        "peak_memory": benchmark_memory,
    }


if __name__ == "__main__":
    print("🔥 FLUX.1-Schnell Optimized Benchmark with Multi-Head Attention")
    print("=" * 70)
    print("✅ Running on MPS (Apple Silicon)")
    print(f"🔧 PyTorch version: {torch.__version__}")

    if METAL_PYTORCH_AVAILABLE:
        results = benchmark_metal_pytorch_optimized()
        print(
            f"\n✨ Benchmark completed! Check flux_schnell_metal_pytorch_optimized.png"
        )
    else:
        print("❌ Cannot run benchmark - Metal PyTorch Custom Op not available")
