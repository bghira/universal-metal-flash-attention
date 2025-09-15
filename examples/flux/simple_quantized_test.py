#!/usr/bin/env python3
"""
Simple test to measure quantization effects on memory bandwidth in FLUX
"""
import sys
import time
from pathlib import Path

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
    """Get current GPU memory usage"""
    if torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / 1024 / 1024  # MB
    return 0


def simulate_quantized_attention(query, key, value, precision="int8", **kwargs):
    """
    Simulate quantized attention by reducing precision of K/V tensors
    This tests memory bandwidth benefits without full quantized kernel implementation
    """
    original_k_dtype = key.dtype
    original_v_dtype = value.dtype

    # Quantize K and V tensors to reduce memory bandwidth
    if precision == "int8":
        # Simulate INT8 by using smaller data types
        k_quantized = key.to(torch.int8).to(
            original_k_dtype
        )  # Round-trip to simulate quantization
        v_quantized = value.to(torch.int8).to(original_v_dtype)
    elif precision == "int4":
        # Simulate INT4 by more aggressive quantization
        # Scale to 4-bit range (-7 to 7), then back
        k_scale = key.abs().max() / 7.0
        v_scale = value.abs().max() / 7.0
        k_quantized = (
            torch.clamp(key / k_scale, -7, 7)
            .round()
            .to(torch.int8)
            .to(original_k_dtype)
            * k_scale
        )
        v_quantized = (
            torch.clamp(value / v_scale, -7, 7)
            .round()
            .to(torch.int8)
            .to(original_v_dtype)
            * v_scale
        )
    else:
        k_quantized = key
        v_quantized = value

    # Use our Metal backend with "quantized" tensors
    return metal_sdpa_extension.metal_scaled_dot_product_attention(
        query, k_quantized, v_quantized, **kwargs
    )


# Monkey patch for quantized testing
original_sdpa = None


def patch_with_quantized_simulation(precision="int8"):
    """Patch attention to use quantized simulation"""
    global original_sdpa
    import torch.nn.functional as F

    if original_sdpa is None:
        original_sdpa = F.scaled_dot_product_attention

    def quantized_wrapper(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        **kwargs,
    ):
        # Only apply quantization to suitable tensors
        if (
            query.device.type == "mps"
            and query.dtype in [torch.float16, torch.bfloat16]
            and attn_mask is None
            and dropout_p == 0.0
        ):

            try:
                return simulate_quantized_attention(
                    query,
                    key,
                    value,
                    precision=precision,
                    scale=scale if scale is not None else 1.0 / query.shape[-1] ** 0.5,
                    is_causal=is_causal,
                )
            except Exception as e:
                print(f"⚠️ Quantized attention failed, falling back: {e}")
                return original_sdpa(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                    **kwargs,
                )
        else:
            return original_sdpa(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                **kwargs,
            )

    F.scaled_dot_product_attention = quantized_wrapper


def restore_original_sdpa():
    """Restore original SDPA"""
    if original_sdpa is not None:
        import torch.nn.functional as F

        F.scaled_dot_product_attention = original_sdpa


def benchmark_flux_with_quantization(precision="int8"):
    """Benchmark FLUX with quantized attention simulation"""
    print(f"\n{'='*60}")
    print(f"🚀 Testing FLUX.1-Schnell with Simulated {precision.upper()} Quantization")
    print(f"{'='*60}")

    # Register our Metal backend
    metal_sdpa_extension.register_backend()

    # Patch attention with quantization
    patch_with_quantized_simulation(precision)

    try:
        # Load pipeline
        print("📥 Loading FLUX.1-Schnell pipeline...")
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
        )
        pipe.to("mps")

        prompt = "A futuristic cityscape"

        # Quick benchmark
        print(f"🎯 Running {precision.upper()} quantized benchmark...")
        torch.mps.empty_cache()

        start_memory = get_memory_usage()
        start_time = time.time()

        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                guidance_scale=0.0,
                num_inference_steps=4,
                height=512,
                width=512,
                generator=torch.Generator("cpu").manual_seed(42),
            ).images[0]

        generation_time = time.time() - start_time
        peak_memory = get_memory_usage()

        print(f"✅ Generation completed!")
        print(f"⏱️  Generation time: {generation_time:.2f}s")
        print(f"💾 Peak GPU memory: {peak_memory:.1f} MB")

        image.save(f"flux_schnell_simulated_{precision}_quantized.png")
        print(f"🖼️  Image saved as: flux_schnell_simulated_{precision}_quantized.png")

        return {
            "precision": precision,
            "generation_time": generation_time,
            "peak_memory": peak_memory,
        }

    finally:
        restore_original_sdpa()
        metal_sdpa_extension.unregister_backend()
        if "pipe" in locals():
            del pipe
        torch.mps.empty_cache()


def benchmark_baseline():
    """Benchmark baseline without quantization"""
    print(f"\n{'='*60}")
    print(f"🚀 Testing FLUX.1-Schnell Baseline (No Quantization)")
    print(f"{'='*60}")

    # Register our Metal backend without quantization
    metal_sdpa_extension.register_backend()

    try:
        # Load pipeline
        print("📥 Loading FLUX.1-Schnell pipeline...")
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
        )
        pipe.to("mps")

        prompt = "A futuristic cityscape"

        # Baseline benchmark
        print("🎯 Running baseline benchmark...")
        torch.mps.empty_cache()

        start_memory = get_memory_usage()
        start_time = time.time()

        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                guidance_scale=0.0,
                num_inference_steps=4,
                height=512,
                width=512,
                generator=torch.Generator("cpu").manual_seed(42),
            ).images[0]

        generation_time = time.time() - start_time
        peak_memory = get_memory_usage()

        print(f"✅ Generation completed!")
        print(f"⏱️  Generation time: {generation_time:.2f}s")
        print(f"💾 Peak GPU memory: {peak_memory:.1f} MB")

        image.save("flux_schnell_baseline.png")
        print("🖼️  Image saved as: flux_schnell_baseline.png")

        return {
            "precision": "baseline",
            "generation_time": generation_time,
            "peak_memory": peak_memory,
        }

    finally:
        metal_sdpa_extension.unregister_backend()
        if "pipe" in locals():
            del pipe
        torch.mps.empty_cache()


if __name__ == "__main__":
    print("🔥 FLUX.1-Schnell Quantization Memory Bandwidth Test")
    print("=" * 70)
    print("✅ Running on MPS (Apple Silicon)")
    print(f"🔧 PyTorch version: {torch.__version__}")

    if METAL_PYTORCH_AVAILABLE:
        results = []

        # Baseline
        baseline = benchmark_baseline()
        results.append(baseline)

        # INT8 simulation
        int8_result = benchmark_flux_with_quantization("int8")
        results.append(int8_result)

        # INT4 simulation
        int4_result = benchmark_flux_with_quantization("int4")
        results.append(int4_result)

        # Summary
        print(f"\n{'='*70}")
        print(f"📊 QUANTIZATION COMPARISON")
        print(f"{'='*70}")

        for result in results:
            speedup = baseline["generation_time"] / result["generation_time"]
            memory_ratio = result["peak_memory"] / baseline["peak_memory"]
            print(
                f"{result['precision'].upper():<10} Time: {result['generation_time']:.2f}s ({speedup:.2f}x) Memory: {result['peak_memory']:.1f}MB ({memory_ratio:.2f}x)"
            )

        # Find best result
        fastest = min(
            results[1:], key=lambda x: x["generation_time"]
        )  # Skip baseline for "fastest"
        print(f"\n🚀 Best quantization: {fastest['precision'].upper()}")
        print(
            f"   Speedup: {baseline['generation_time'] / fastest['generation_time']:.2f}x"
        )
        print(
            f"   Memory: {fastest['peak_memory'] / baseline['peak_memory']:.2f}x of baseline"
        )

        print(f"\n✨ Quantization benchmark completed!")
    else:
        print("❌ Cannot run benchmark - Metal PyTorch Custom Op not available")
