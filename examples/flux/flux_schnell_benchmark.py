#!/usr/bin/env python3
"""
FLUX.1-Schnell Comprehensive Metal SDPA Benchmark

This benchmark tests the performance of FLUX.1-Schnell with:
1. PyTorch Vanilla SDPA (baseline)
2. Metal UMFA with BF16 (vanilla Universal Metal Flash Attention)
3. Metal UMFA with INT8 quantization
4. Metal UMFA with INT4 quantization

Tested at resolutions: 256x256, 512x512, 1024x1024

Requirements:
    pip install diffusers torch transformers accelerate safetensors

Usage:
    python flux_schnell_benchmark.py

Outputs are saved to examples/flux/output/ with performance metrics.
"""

import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import psutil
import torch
import torch.nn.functional as F

# Resolve repository root dynamically
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _prepend_dyld_library_path(paths) -> None:
    """Prepend paths to DYLD_LIBRARY_PATH if they exist."""
    existing = os.environ.get("DYLD_LIBRARY_PATH", "")
    valid_paths = [str(path) for path in paths if path.exists()]
    if not valid_paths:
        return
    prefix = ":".join(valid_paths)
    if existing:
        os.environ["DYLD_LIBRARY_PATH"] = f"{prefix}:{existing}"
    else:
        os.environ["DYLD_LIBRARY_PATH"] = prefix


_prepend_dyld_library_path(
    [
        PROJECT_ROOT / ".build" / "arm64-apple-macosx" / "release",
        PROJECT_ROOT / ".build" / "arm64-apple-macosx" / "debug",
    ]
)

# Ensure repository modules (PyTorch custom op, Python bindings) are importable
sys.path.insert(0, str(PROJECT_ROOT / "examples" / "pytorch-custom-op-ffi"))


def _maybe_add_venv_site_packages() -> None:
    venv_root = Path(os.environ.get("VIRTUAL_ENV", PROJECT_ROOT / ".venv"))
    if not venv_root.exists():
        return
    for site_packages in venv_root.glob("lib/python*/site-packages"):
        if site_packages.is_dir():
            sys.path.insert(0, str(site_packages))
            break


_maybe_add_venv_site_packages()

# Try to import Metal SDPA extension
try:
    # First try to build the extension if needed
    build_dir = project_root / "examples" / "pytorch-custom-op-ffi" / "build"
    if build_dir.exists():
        # Add build directory to path
        import glob

        lib_dirs = glob.glob(str(build_dir / "lib.*"))
        if lib_dirs:
            sys.path.insert(0, lib_dirs[0])

    import metal_sdpa_extension

    METAL_PYTORCH_AVAILABLE = True
    print("✅ Metal PyTorch Custom Op available")

    # Check for quantization support
    HAS_QUANTIZATION = hasattr(
        metal_sdpa_extension, "quantized_scaled_dot_product_attention"
    )
    if HAS_QUANTIZATION:
        print("✅ Quantization support available")
    else:
        print("⚠️ Quantization support not available in current build")
        print("   To enable: rebuild with quantization support")
except ImportError as e:
    METAL_PYTORCH_AVAILABLE = False
    HAS_QUANTIZATION = False
    print(f"❌ Metal PyTorch Custom Op not available: {e}")
    print("   To build: cd examples/pytorch-custom-op-ffi && python setup.py build")

try:
    from diffusers import FluxPipeline

    DIFFUSERS_AVAILABLE = True
    print("✅ Diffusers available")
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print(
        "❌ Diffusers not available - install with: pip install diffusers transformers accelerate"
    )


class BenchmarkConfig:
    """Configuration for a benchmark run"""

    def __init__(
        self, name: str, quantization: Optional[str] = None, use_metal: bool = False
    ):
        self.name = name
        self.quantization = quantization  # None, 'int8', 'int4'
        self.use_metal = use_metal
        self.metrics: Dict[str, Any] = {}


def create_metal_sdpa_wrapper(quantization_mode: Optional[str] = None):
    """Create a Metal SDPA wrapper with specified quantization"""
    if not METAL_PYTORCH_AVAILABLE:
        return None

    original_sdpa = F.scaled_dot_product_attention

    def metal_sdpa_wrapper(*args, **kwargs):
        try:
            # Extract arguments
            if len(args) < 3:
                return original_sdpa(*args, **kwargs)

            query, key, value = args[0], args[1], args[2]

            # Handle optional arguments
            attn_mask = kwargs.get("attn_mask")
            if attn_mask is None and len(args) > 3:
                attn_mask = args[3]

            dropout_p = kwargs.get("dropout_p", 0.0)
            if dropout_p == 0.0 and len(args) > 4:
                dropout_p = args[4]

            is_causal = kwargs.get("is_causal", False)
            if not is_causal and len(args) > 5:
                is_causal = args[5]

            scale = kwargs.get("scale")
            if scale is None and len(args) > 6:
                scale = args[6]

            enable_gqa = kwargs.get("enable_gqa", False)
        except Exception as e:
            print(f"Argument extraction failed: {e}")
            return original_sdpa(*args, **kwargs)

        # Check if we can use Metal SDPA
        if query.device.type != "mps" or not METAL_PYTORCH_AVAILABLE:
            return original_sdpa(*args, **kwargs)

        try:
            # Use quantized version if quantization mode is specified
            if quantization_mode and HAS_QUANTIZATION:
                # Use the simpler quantized_scaled_dot_product_attention function
                # which takes a precision string directly
                result = metal_sdpa_extension.quantized_scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    precision=quantization_mode,  # 'int4' or 'int8'
                    is_causal=is_causal,
                    scale=(
                        scale if scale is not None else (1.0 / (query.shape[-1] ** 0.5))
                    ),
                )
                return result
            else:
                # Use regular Metal SDPA without quantization
                result = metal_sdpa_extension.metal_scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=(
                        scale if scale is not None else (1.0 / (query.shape[-1] ** 0.5))
                    ),
                    enable_gqa=enable_gqa,
                )
                return result
        except Exception as e:
            print(f"Metal SDPA failed: {e}, falling back to PyTorch")
            return original_sdpa(*args, **kwargs)

    return metal_sdpa_wrapper


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def get_mps_memory_info():
    """Get MPS memory info if available"""
    if torch.backends.mps.is_available():
        return {
            "allocated": torch.mps.current_allocated_memory() / 1024 / 1024,
            "reserved": torch.mps.driver_allocated_memory() / 1024 / 1024,
        }
    return {"allocated": 0, "reserved": 0}


def patch_attention(quantization_mode: Optional[str] = None):
    """Patch PyTorch SDPA to use Metal with optional quantization"""
    if not METAL_PYTORCH_AVAILABLE:
        return None

    wrapper = create_metal_sdpa_wrapper(quantization_mode)
    if wrapper:
        original = F.scaled_dot_product_attention
        F.scaled_dot_product_attention = wrapper
        return original
    return None


def restore_attention(original_sdpa):
    """Restore original PyTorch SDPA"""
    if original_sdpa is not None:
        F.scaled_dot_product_attention = original_sdpa


def benchmark_flux_configuration(
    config: BenchmarkConfig, resolution: Tuple[int, int], prompt: str = None
) -> Dict[str, Any]:
    """Benchmark FLUX with a specific configuration and resolution"""

    if prompt is None:
        prompt = "A majestic castle on a hilltop at sunset, highly detailed digital art"

    width, height = resolution
    res_str = f"{width}x{height}"

    print(f"\n{'='*60}")
    print(f"🚀 Testing: {config.name} @ {res_str}")
    print(f"{'='*60}")

    if not DIFFUSERS_AVAILABLE:
        print("❌ Diffusers not available")
        return None

    # Clear GPU memory
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

    initial_memory = get_memory_usage()

    # Apply attention patch if using Metal
    original_sdpa = None
    if config.use_metal:
        original_sdpa = patch_attention(config.quantization)
        if original_sdpa is None:
            print(f"❌ Failed to patch attention for {config.name}")
            return None

    try:
        # Load FLUX.1-Schnell pipeline with FP32
        print("📥 Loading FLUX.1-Schnell pipeline (FP32...)")
        load_start = time.time()

        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            torch_dtype=torch.float32,  # Always use FP32 base weights
        )

        if torch.backends.mps.is_available():
            pipe = pipe.to("mps")
            print("✅ Pipeline moved to MPS")

        load_time = time.time() - load_start
        load_memory = get_memory_usage() - initial_memory

        print(f"⏱️  Pipeline load time: {load_time:.2f}s")
        print(f"💾 Memory after loading: {load_memory:.1f} MB")

        # Generate image
        print(f"🎨 Generating @ {res_str}: '{prompt}'")

        # Warm-up run (optional)
        # pipe(prompt=prompt, num_inference_steps=1, height=256, width=256,
        #      guidance_scale=0.0).images[0]

        generation_start = time.time()
        step_times = []

        # Custom callback to measure per-step timing
        def step_callback(_pipe, _step_index, _timestep, callback_kwargs):
            nonlocal step_times
            step_times.append(time.time())
            return callback_kwargs

        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                num_inference_steps=4,  # FLUX.1-Schnell optimized for 4 steps
                height=height,
                width=width,
                guidance_scale=0.0,  # No classifier-free guidance
                generator=torch.Generator().manual_seed(42),  # Reproducible
                callback_on_step_end=step_callback,
            ).images[0]

        generation_time = time.time() - generation_start
        peak_memory = get_memory_usage()
        peak_mps = get_mps_memory_info()

        # Calculate per-step times
        if len(step_times) > 1:
            per_step_times = [
                step_times[i + 1] - step_times[i] for i in range(len(step_times) - 1)
            ]
            avg_step_time = sum(per_step_times) / len(per_step_times)
        else:
            avg_step_time = generation_time / 4

        # Save image
        output_dir = PROJECT_ROOT / "examples" / "flux" / "output" / res_str
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = config.name.lower().replace(" ", "_")
        output_path = output_dir / f"{filename}.png"
        image.save(output_path)

        print(f"✅ Generation completed!")
        print(f"⏱️  Total time: {generation_time:.2f}s")
        print(f"⏱️  Avg per step: {avg_step_time:.3f}s")
        print(f"💾 Peak memory: {peak_memory:.1f} MB")
        print(f"💾 MPS allocated: {peak_mps['allocated']:.1f} MB")
        print(f"🖼️  Saved: {output_path}")

        # Collect metrics
        metrics = {
            "config": config.name,
            "resolution": res_str,
            "width": width,
            "height": height,
            "quantization": config.quantization,
            "load_time": load_time,
            "generation_time": generation_time,
            "avg_step_time": avg_step_time,
            "load_memory_mb": load_memory,
            "peak_memory_mb": peak_memory - initial_memory,
            "mps_allocated_mb": peak_mps["allocated"],
            "mps_reserved_mb": peak_mps["reserved"],
            "output_path": str(output_path),
        }

        # Clean up
        del pipe
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

        return metrics

    except Exception as e:
        print(f"❌ Error during {config.name} benchmark: {e}")
        import traceback

        traceback.print_exc()
        return None

    finally:
        # Restore original attention
        restore_attention(original_sdpa)


def main():
    print("🔥 FLUX.1-Schnell Comprehensive Metal SDPA Benchmark")
    print("=" * 60)

    if not torch.backends.mps.is_available():
        print("❌ MPS not available - this benchmark requires Apple Silicon")
        return

    print(f"✅ Running on MPS (Apple Silicon)")
    print(f"🔧 PyTorch version: {torch.__version__}")
    print(f"📁 Output directory: examples/flux/output/")

    # Define configurations
    configurations = [
        BenchmarkConfig("PyTorch Vanilla", quantization=None, use_metal=False),
        BenchmarkConfig("Metal UMFA BF16", quantization=None, use_metal=True),
    ]

    # Add quantization configs if available
    if HAS_QUANTIZATION:
        configurations.extend(
            [
                BenchmarkConfig("Metal UMFA INT8", quantization="int8", use_metal=True),
                BenchmarkConfig("Metal UMFA INT4", quantization="int4", use_metal=True),
            ]
        )
    else:
        print("⚠️ Quantization not available, skipping INT8/INT4 tests")

    # Define resolutions to test
    resolutions = [
        (256, 256),
        (512, 512),
        (1024, 1024),
    ]

    # Collect all results
    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run benchmarks
    for resolution in resolutions:
        print(f"\n{'='*60}")
        print(f"📐 Resolution: {resolution[0]}x{resolution[1]}")
        print(f"{'='*60}")

        for config in configurations:
            result = benchmark_flux_configuration(config, resolution)
            if result:
                all_results.append(result)

    # Save results to JSON
    if all_results:
        results_file = Path(f"examples/flux/output/benchmark_results_{timestamp}.json")
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n📊 Results saved to: {results_file}")

        # Print comparison table
        print("\n" + "=" * 60)
        print("📈 PERFORMANCE COMPARISON")
        print("=" * 60)

        # Group results by resolution
        for resolution in resolutions:
            res_str = f"{resolution[0]}x{resolution[1]}"
            res_results = [r for r in all_results if r["resolution"] == res_str]

            if not res_results:
                continue

            print(f"\n🔍 Resolution: {res_str}")
            print(
                f"{'Configuration':<20} {'Time (s)':<12} {'Speedup':<10} {'Memory (MB)':<12}"
            )
            print("-" * 54)

            # Find baseline (PyTorch Vanilla)
            baseline = next(
                (r for r in res_results if "Vanilla" in r["config"]), res_results[0]
            )

            for result in res_results:
                speedup = baseline["generation_time"] / result["generation_time"]
                speedup_str = f"{speedup:.2f}x" if speedup != 1.0 else "baseline"

                print(
                    f"{result['config']:<20} "
                    f"{result['generation_time']:<8.2f}    "
                    f"{speedup_str:<10} "
                    f"{result['peak_memory_mb']:<8.1f}"
                )

        # Overall best performer
        best_result = min(all_results, key=lambda x: x["generation_time"])
        print(
            f"\n🏆 Best Overall: {best_result['config']} @ {best_result['resolution']}"
        )
        print(f"   Time: {best_result['generation_time']:.2f}s")
        print(f"   Memory: {best_result['peak_memory_mb']:.1f} MB")

    print("\n✨ Benchmark completed! Check examples/flux/output/ for generated images.")


if __name__ == "__main__":
    main()
