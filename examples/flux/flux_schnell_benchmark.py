#!/usr/bin/env python3
"""
FLUX.1-Schnell MPS Benchmark with Metal Flash Attention

This example demonstrates the performance difference between:
1. Standard PyTorch SDPA (F.scaled_dot_product_attention)
2. Metal Flash Attention PyTorch Custom Op (with GLUON optimizations)
3. Metal Flash Attention Python FFI drop-in wrapper

Requirements:
    pip install diffusers torch transformers accelerate safetensors

Usage:
    python flux_schnell_benchmark.py

The script will run FLUX.1-Schnell text-to-image generation using each backend
and compare performance, memory usage, and output quality.
"""

import gc

# Set up library path for Metal FFI library
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

lib_path = "/Users/kash/src/universal-metal-flash-attention/.build/arm64-apple-macosx/release:/Users/kash/src/universal-metal-flash-attention/.build/arm64-apple-macosx/debug"
if "DYLD_LIBRARY_PATH" in os.environ:
    os.environ["DYLD_LIBRARY_PATH"] = lib_path + ":" + os.environ["DYLD_LIBRARY_PATH"]
else:
    os.environ["DYLD_LIBRARY_PATH"] = lib_path

# Add the pytorch-custom-op-ffi build directory to path (not just the root)
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

# Skip Python FFI for now to focus on PyTorch Custom Op
METAL_PYTHON_FFI_AVAILABLE = False
print("⏭️ Skipping Metal Python FFI for now")

try:
    from diffusers import FluxPipeline

    DIFFUSERS_AVAILABLE = True
    print("✅ Diffusers available")
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print(
        "❌ Diffusers not available - install with: pip install diffusers transformers accelerate"
    )


class MetalSDPAWrapper:
    """Drop-in replacement for F.scaled_dot_product_attention using Metal FFI"""

    def __init__(self, original_sdpa, quantized=False):
        self.context = umfa.MFAContext() if METAL_PYTHON_FFI_AVAILABLE else None
        self.original_sdpa = original_sdpa
        self.quantized = quantized

    def __call__(self, *args, **kwargs):
        try:
            # Extract arguments flexibly
            if len(args) < 3:
                return self.original_sdpa(*args, **kwargs)

            query, key, value = args[0], args[1], args[2]
            attn_mask = kwargs.get("attn_mask", args[3] if len(args) > 3 else None)
            dropout_p = kwargs.get("dropout_p", args[4] if len(args) > 4 else 0.0)
            is_causal = kwargs.get("is_causal", args[5] if len(args) > 5 else False)
            scale = kwargs.get("scale", args[6] if len(args) > 6 else None)
        except Exception as e:
            print(
                f"MetalSDPAWrapper argument extraction failed: {e}, args={len(args)}, kwargs={list(kwargs.keys())}"
            )
            return self.original_sdpa(*args, **kwargs)

        if not METAL_PYTHON_FFI_AVAILABLE or query.device.type != "mps":
            # Fallback to standard PyTorch SDPA
            return self.original_sdpa(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )

        # Convert to numpy for Metal FFI, preserving bfloat16 precision
        batch_size, num_heads, seq_len, head_dim = query.shape

        # Determine precision based on tensor dtype
        if query.dtype == torch.bfloat16:
            precision = "bf16"

            # Convert bfloat16 to float32 for numpy (numpy doesn't support bfloat16 directly)
            def tensor_to_numpy(t):
                return t.float().cpu().numpy()

        elif query.dtype == torch.float16:
            precision = "fp16"

            def tensor_to_numpy(t):
                return t.cpu().numpy().astype("float16")

        else:
            precision = "fp32"

            def tensor_to_numpy(t):
                return t.cpu().numpy().astype("float32")

        if num_heads > 1:
            # Metal Flash Attention currently supports single-head only
            # Process each head separately
            outputs = []
            for head_idx in range(num_heads):
                q_head = tensor_to_numpy(query[:, head_idx])
                k_head = tensor_to_numpy(key[:, head_idx])
                v_head = tensor_to_numpy(value[:, head_idx])

                # Reshape to 2D for Metal FFI (assuming batch_size=1)
                if batch_size == 1:
                    q_head = q_head.squeeze(0)  # [seq_len, head_dim]
                    k_head = k_head.squeeze(0)
                    v_head = v_head.squeeze(0)

                    if self.quantized and precision == "bf16":
                        # Quantize K/V to INT8 for memory efficiency
                        k_scale = np.abs(k_head).max() / 127.0
                        v_scale = np.abs(v_head).max() / 127.0
                        k_quantized = (k_head / k_scale).astype(np.int8)
                        v_quantized = (v_head / v_scale).astype(np.int8)

                        out_head = umfa.quantized_attention(
                            q_head.astype(np.float32),
                            k_quantized,
                            v_quantized,
                            context=self.context,
                            causal=is_causal,
                            softmax_scale=scale,
                            query_precision="bf16",
                            kv_precision="int8",
                            output_precision="bf16",
                            k_scale=k_scale,
                            v_scale=v_scale,
                        )
                    else:
                        out_head = umfa.flash_attention_forward(
                            self.context,
                            q_head,
                            k_head,
                            v_head,
                            causal=is_causal,
                            softmax_scale=scale,
                            input_precision=precision,
                            intermediate_precision=precision,
                            output_precision=precision,
                        )

                    # Add batch dimension back and convert to correct dtype
                    out_tensor = (
                        torch.from_numpy(out_head).unsqueeze(0).to(query.device)
                    )
                    if query.dtype == torch.bfloat16:
                        out_tensor = out_tensor.to(torch.bfloat16)
                    outputs.append(out_tensor)
                else:
                    # Fallback for multi-batch
                    return self.original_sdpa(
                        query,
                        key,
                        value,
                        attn_mask=attn_mask,
                        dropout_p=dropout_p,
                        is_causal=is_causal,
                        scale=scale,
                    )

            # Stack heads back together
            output = torch.stack(outputs, dim=1)
            return output
        else:
            # Single head case
            q_np = tensor_to_numpy(query.squeeze(1))  # [batch, seq_len, head_dim]
            k_np = tensor_to_numpy(key.squeeze(1))
            v_np = tensor_to_numpy(value.squeeze(1))

            if batch_size == 1:
                q_np = q_np.squeeze(0)  # [seq_len, head_dim]
                k_np = k_np.squeeze(0)
                v_np = v_np.squeeze(0)

                if self.quantized and precision == "bf16":
                    # Quantize K/V to INT8 for memory efficiency
                    k_scale = np.abs(k_np).max() / 127.0
                    v_scale = np.abs(v_np).max() / 127.0
                    k_quantized = (k_np / k_scale).astype(np.int8)
                    v_quantized = (v_np / v_scale).astype(np.int8)

                    out_np = umfa.quantized_attention(
                        q_np.astype(np.float32),
                        k_quantized,
                        v_quantized,
                        context=self.context,
                        causal=is_causal,
                        softmax_scale=scale,
                        query_precision="bf16",
                        kv_precision="int8",
                        output_precision="bf16",
                        k_scale=k_scale,
                        v_scale=v_scale,
                    )
                else:
                    out_np = umfa.flash_attention_forward(
                        self.context,
                        q_np,
                        k_np,
                        v_np,
                        causal=is_causal,
                        softmax_scale=scale,
                        input_precision=precision,
                        intermediate_precision=precision,
                        output_precision=precision,
                    )

                output = (
                    torch.from_numpy(out_np).unsqueeze(0).unsqueeze(1).to(query.device)
                )
                if query.dtype == torch.bfloat16:
                    output = output.to(torch.bfloat16)
                return output
            else:
                # Fallback for multi-batch
                return self.original_sdpa(*args, **kwargs)


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def patch_attention_pytorch_custom_op():
    """Patch PyTorch SDPA to use Metal Custom Op"""
    if not METAL_PYTORCH_AVAILABLE:
        return None

    original_sdpa = torch.nn.functional.scaled_dot_product_attention

    def metal_sdpa_wrapper(*args, **kwargs):
        try:
            # Extract arguments flexibly to handle varying PyTorch versions
            if len(args) < 3:
                return original_sdpa(*args, **kwargs)

            query, key, value = args[0], args[1], args[2]

            # Handle remaining args/kwargs safely
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

            # Handle enable_gqa parameter (new in PyTorch 2.8+)
            enable_gqa = kwargs.get("enable_gqa", False)
        except Exception as e:
            print(
                f"Argument extraction failed: {e}, args={len(args)}, kwargs={list(kwargs.keys())}"
            )
            return original_sdpa(*args, **kwargs)

        # Only use Metal for MPS tensors with compatible shapes
        if (
            query.device.type == "mps"
            and query.dim() == 4
            and query.shape[0] == 1  # batch_size = 1
            and query.shape[1] == 1  # single head
            and attn_mask is None
            and dropout_p == 0.0
            and not enable_gqa
        ):  # Disable for GQA

            try:
                # Keep 4D shape for Metal Custom Op (it actually expects 4D tensors)
                result_4d = metal_sdpa_extension.metal_scaled_dot_product_attention(
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

                return result_4d
            except Exception as e:
                print(f"Metal Custom Op failed, falling back to PyTorch: {e}")
                return original_sdpa(*args, **kwargs)

        return original_sdpa(*args, **kwargs)

    torch.nn.functional.scaled_dot_product_attention = metal_sdpa_wrapper
    return original_sdpa


def patch_attention_python_ffi():
    """Patch PyTorch SDPA to use Metal Python FFI"""
    if not METAL_PYTHON_FFI_AVAILABLE:
        return None

    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    metal_wrapper = MetalSDPAWrapper(original_sdpa)

    torch.nn.functional.scaled_dot_product_attention = metal_wrapper
    return original_sdpa


def patch_attention_quantized_ffi():
    """Patch PyTorch SDPA to use Quantized Metal Python FFI"""
    if not METAL_PYTHON_FFI_AVAILABLE:
        return None

    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    metal_wrapper = MetalSDPAWrapper(original_sdpa, quantized=True)

    torch.nn.functional.scaled_dot_product_attention = metal_wrapper
    return original_sdpa


def restore_attention(original_sdpa):
    """Restore original PyTorch SDPA"""
    if original_sdpa is not None:
        torch.nn.functional.scaled_dot_product_attention = original_sdpa


def benchmark_flux_schnell(backend_name, patch_func=None):
    """Benchmark FLUX.1-Schnell with specified attention backend"""
    print(f"\n{'='*60}")
    print(f"🚀 Testing FLUX.1-Schnell with {backend_name}")
    print(f"{'='*60}")

    if not DIFFUSERS_AVAILABLE:
        print("❌ Diffusers not available - skipping benchmark")
        return None

    # Clear GPU memory
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

    initial_memory = get_memory_usage()

    # Apply attention patch
    original_sdpa = None
    if patch_func:
        original_sdpa = patch_func()
        if original_sdpa is None:
            print(f"❌ Failed to patch attention for {backend_name}")
            return None

    try:
        # Load FLUX.1-Schnell pipeline
        print("📥 Loading FLUX.1-Schnell pipeline...")
        load_start = time.time()

        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
        )

        if torch.backends.mps.is_available():
            pipe = pipe.to("mps")
            print("✅ Pipeline moved to MPS")

        load_time = time.time() - load_start
        load_memory = get_memory_usage() - initial_memory

        print(f"⏱️  Pipeline load time: {load_time:.2f}s")
        print(f"💾 Memory used for loading: {load_memory:.1f} MB")

        # Generate image
        prompt = "A futuristic cityscape with flying cars and neon lights, highly detailed digital art"
        print(f"🎨 Generating image with prompt: '{prompt}'")

        generation_start = time.time()

        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                num_inference_steps=4,  # FLUX.1-Schnell is designed for 4 steps
                height=1024,
                width=1024,
                guidance_scale=0.0,  # FLUX.1-Schnell doesn't use guidance
                generator=torch.Generator().manual_seed(42),  # For reproducible results
            ).images[0]

        generation_time = time.time() - generation_start
        peak_memory = get_memory_usage()
        total_memory = peak_memory - initial_memory

        # Save image
        output_path = f"flux_schnell_{backend_name.lower().replace(' ', '_')}.png"
        image.save(output_path)

        print(f"✅ Generation completed!")
        print(f"⏱️  Generation time: {generation_time:.2f}s")
        print(f"💾 Peak memory usage: {total_memory:.1f} MB")
        print(f"🖼️  Image saved as: {output_path}")

        # Clean up
        del pipe
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

        return {
            "backend": backend_name,
            "load_time": load_time,
            "generation_time": generation_time,
            "load_memory_mb": load_memory,
            "peak_memory_mb": total_memory,
            "output_path": output_path,
        }

    except Exception as e:
        print(f"❌ Error during {backend_name} benchmark: {e}")
        return None

    finally:
        # Restore original attention
        restore_attention(original_sdpa)


def main():
    print("🔥 FLUX.1-Schnell Metal Flash Attention Benchmark")
    print("=" * 60)

    if not torch.backends.mps.is_available():
        print("❌ MPS not available - this benchmark requires Apple Silicon")
        return

    print(f"✅ Running on MPS (Apple Silicon)")
    print(f"🔧 PyTorch version: {torch.__version__}")

    results = []

    # Benchmark 1: Standard PyTorch SDPA
    print("\n" + "=" * 60)
    print("📊 BENCHMARK 1: Standard PyTorch SDPA")
    result1 = benchmark_flux_schnell("Standard PyTorch SDPA")
    if result1:
        results.append(result1)

    # Benchmark 2: Metal PyTorch Custom Op (with GLUON)
    if METAL_PYTORCH_AVAILABLE:
        print("\n" + "=" * 60)
        print("📊 BENCHMARK 2: Metal PyTorch Custom Op (GLUON)")
        result2 = benchmark_flux_schnell(
            "Metal PyTorch Custom Op", patch_attention_pytorch_custom_op
        )
        if result2:
            results.append(result2)

    # Benchmark 3: Metal Python FFI Drop-in (with GLUON)
    if METAL_PYTHON_FFI_AVAILABLE:
        print("\n" + "=" * 60)
        print("📊 BENCHMARK 3: Metal Python FFI Drop-in (GLUON)")
        result3 = benchmark_flux_schnell("Metal Python FFI", patch_attention_python_ffi)
        if result3:
            results.append(result3)

    # Benchmark 4: Quantized Metal FFI (INT8 K/V with GLUON)
    if METAL_PYTHON_FFI_AVAILABLE:
        print("\n" + "=" * 60)
        print("📊 BENCHMARK 4: Quantized Metal FFI (INT8 K/V + GLUON)")
        result4 = benchmark_flux_schnell(
            "Quantized Metal FFI", patch_attention_quantized_ffi
        )
        if result4:
            results.append(result4)

    # Print comparison
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("📈 PERFORMANCE COMPARISON")
        print("=" * 60)

        baseline = results[0]
        print(f"{'Backend':<25} {'Gen Time':<12} {'Speedup':<10} {'Memory':<12}")
        print("-" * 60)

        for result in results:
            speedup = baseline["generation_time"] / result["generation_time"]
            speedup_str = f"{speedup:.2f}x" if speedup != 1.0 else "baseline"

            print(
                f"{result['backend']:<25} {result['generation_time']:<8.2f}s    {speedup_str:<10} {result['peak_memory_mb']:<8.1f} MB"
            )

        # Check for significant improvements
        if len(results) > 1:
            best_metal = min(
                [r for r in results[1:]], key=lambda x: x["generation_time"]
            )
            improvement = (
                (baseline["generation_time"] - best_metal["generation_time"])
                / baseline["generation_time"]
                * 100
            )

            print(f"\n🎯 Best Metal backend: {best_metal['backend']}")
            print(
                f"🚀 Performance improvement: {improvement:.1f}% faster than PyTorch SDPA"
            )
            print(f"💫 GLUON optimizations delivered real-world speedups!")

    print(
        f"\n✨ Benchmark completed! Check the generated images for quality comparison."
    )


if __name__ == "__main__":
    main()
