#!/usr/bin/env python3
"""
Quick FLUX benchmark for Metal SDPA - Tests one resolution with minimal steps
"""

import gc
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F

# Resolve repository root dynamically
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _prepend_dyld_library_path(paths) -> None:
    existing = os.environ.get("DYLD_LIBRARY_PATH", "")
    valid = [str(path) for path in paths if path.exists()]
    if not valid:
        return
    prefix = ":".join(valid)
    os.environ["DYLD_LIBRARY_PATH"] = f"{prefix}:{existing}" if existing else prefix


_prepend_dyld_library_path(
    [
        PROJECT_ROOT / ".build" / "arm64-apple-macosx" / "release",
        PROJECT_ROOT / ".build" / "arm64-apple-macosx" / "debug",
    ]
)

# Add the PyTorch custom op path to sys.path
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
    import metal_sdpa_extension

    METAL_PYTORCH_AVAILABLE = True
    print("✅ Metal PyTorch Custom Op available")
    HAS_QUANTIZATION = hasattr(
        metal_sdpa_extension, "quantized_scaled_dot_product_attention"
    )
    if HAS_QUANTIZATION:
        print("✅ Quantization support available")
except ImportError as e:
    METAL_PYTORCH_AVAILABLE = False
    HAS_QUANTIZATION = False
    print(f"❌ Metal PyTorch Custom Op not available: {e}")

try:
    from diffusers import FluxPipeline

    DIFFUSERS_AVAILABLE = True
    print("✅ Diffusers available")
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("❌ Diffusers not available")


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
            is_causal = kwargs.get("is_causal", False)
            scale = kwargs.get("scale")

        except Exception:
            return original_sdpa(*args, **kwargs)

        # Check if we can use Metal SDPA
        if query.device.type != "mps":
            return original_sdpa(*args, **kwargs)

        try:
            # Use quantized version if quantization mode is specified
            if quantization_mode and HAS_QUANTIZATION:
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
                    enable_gqa=False,
                )
                return result
        except Exception:
            # Silently fall back to PyTorch
            return original_sdpa(*args, **kwargs)

    return metal_sdpa_wrapper


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


def run_quick_benchmark():
    """Run a quick benchmark with 512x512 resolution and 2 steps"""

    if not DIFFUSERS_AVAILABLE:
        print("❌ Cannot run benchmark without diffusers")
        return

    print("\n" + "=" * 60)
    print("🚀 FLUX Quick Benchmark - 512x512, 2 steps")
    print("=" * 60)

    configs = [
        ("PyTorch Vanilla", None),
        ("Metal UMFA BF16", "bf16"),
    ]

    if HAS_QUANTIZATION:
        configs.extend(
            [
                ("Metal UMFA INT8", "int8"),
                ("Metal UMFA INT4", "int4"),
            ]
        )

    results = []

    for config_name, quantization in configs:
        print(f"\n📊 Testing: {config_name}")

        # Clear memory
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

        # Apply patch if using Metal
        original_sdpa = None
        if quantization == "bf16":
            original_sdpa = patch_attention(None)  # BF16 = no quantization
        elif quantization in ["int8", "int4"]:
            original_sdpa = patch_attention(quantization)

        try:
            # Load pipeline
            print("  Loading pipeline...")
            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
            )
            pipe = pipe.to("mps")

            # Generate with minimal steps
            print("  Generating image...")
            start_time = time.time()

            with torch.inference_mode():
                image = pipe(
                    prompt="A simple test",
                    num_inference_steps=2,  # Minimal steps
                    height=512,
                    width=512,
                    guidance_scale=0.0,
                    generator=torch.Generator().manual_seed(42),
                ).images[0]

            generation_time = time.time() - start_time

            # Save result
            output_dir = PROJECT_ROOT / "examples" / "flux" / "output" / "quick_test"
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = config_name.lower().replace(" ", "_") + ".png"
            image.save(output_dir / filename)

            print(f"  ✅ Time: {generation_time:.2f}s")
            results.append({"config": config_name, "time": generation_time})

            # Clean up
            del pipe
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results.append({"config": config_name, "time": None, "error": str(e)})

        finally:
            # Restore original attention
            restore_attention(original_sdpa)

    # Print summary
    print("\n" + "=" * 60)
    print("📈 RESULTS SUMMARY")
    print("=" * 60)

    baseline_time = None
    for result in results:
        if result.get("time"):
            if baseline_time is None:
                baseline_time = result["time"]
            speedup = baseline_time / result["time"] if baseline_time else 1.0
            print(f"{result['config']:<20} {result['time']:>8.2f}s  ({speedup:.2f}x)")
        else:
            print(f"{result['config']:<20} Failed")

    print("\n✅ Quick benchmark complete!")


if __name__ == "__main__":
    run_quick_benchmark()
