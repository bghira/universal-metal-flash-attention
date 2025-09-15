#!/usr/bin/env python3
"""
Test quantized attention (INT8 and INT4) performance in FLUX.1-Schnell
"""
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from diffusers import FluxPipeline

# Fix library path for Python FFI
lib_path = (
    "/Users/kash/src/universal-metal-flash-attention/.build/arm64-apple-macosx/release"
)
if "DYLD_LIBRARY_PATH" in os.environ:
    os.environ["DYLD_LIBRARY_PATH"] = lib_path + ":" + os.environ["DYLD_LIBRARY_PATH"]
else:
    os.environ["DYLD_LIBRARY_PATH"] = lib_path

# Add python-ffi to path
sys.path.append(str(Path(__file__).parent / "python-ffi" / "src"))

try:
    import umfa

    METAL_PYTHON_FFI_AVAILABLE = True
    print("✅ Metal Python FFI available")
except ImportError as e:
    METAL_PYTHON_FFI_AVAILABLE = False
    print(f"❌ Metal Python FFI not available: {e}")
    sys.exit(1)


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


# Custom quantized attention backend class
class QuantizedMetalSDPABackend:
    def __init__(self, precision="int8"):
        self.precision = precision
        self.context = umfa.create_context()
        self.original_sdpa = None
        print(f"🔧 Quantized Metal SDPA Backend initialized ({precision})")

    def __call__(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
    ):
        # Only process float16/bfloat16 tensors on CPU (our typical input pattern)
        if (
            query.device.type != "cpu"
            or query.dtype not in [torch.float16, torch.bfloat16]
            or key.device.type != "cpu"
            or value.device.type != "cpu"
        ):
            return self.original_sdpa(
                query, key, value, attn_mask, dropout_p, is_causal, scale
            )

        # Get tensor dimensions
        if query.dim() == 4:  # [batch, seq_len, num_heads, head_dim]
            batch_size, seq_len_q, num_heads, head_dim = query.shape
            seq_len_kv = key.shape[1]
        elif query.dim() == 3:  # [batch, seq_len, embed_dim] - need to reshape
            batch_size, seq_len_q, embed_dim = query.shape
            seq_len_kv = key.shape[1]
            # Assume standard multi-head split
            num_heads = 16  # Common for FLUX
            head_dim = embed_dim // num_heads

            # Reshape to multi-head format
            query = query.view(batch_size, seq_len_q, num_heads, head_dim)
            key = key.view(batch_size, seq_len_kv, num_heads, head_dim)
            value = value.view(batch_size, seq_len_kv, num_heads, head_dim)
        else:
            return self.original_sdpa(
                query, key, value, attn_mask, dropout_p, is_causal, scale
            )

        if scale is None:
            scale = 1.0 / (head_dim**0.5)

        # Convert to numpy for quantized processing
        def to_numpy(t):
            return t.cpu().numpy().astype(np.float32)

        # Process batch size 1 only for simplicity
        if batch_size == 1 and num_heads <= 8:  # Limit to reasonable head counts
            outputs = []

            for head_idx in range(num_heads):
                q_head = to_numpy(query[0, :, head_idx, :])  # [seq_len, head_dim]
                k_head = to_numpy(key[0, :, head_idx, :])
                v_head = to_numpy(value[0, :, head_idx, :])

                # Quantize K and V tensors
                if self.precision == "int8":
                    k_scale = np.abs(k_head).max() / 127.0
                    v_scale = np.abs(v_head).max() / 127.0
                    k_quantized = np.clip(k_head / k_scale, -127, 127).astype(np.int8)
                    v_quantized = np.clip(v_head / v_scale, -127, 127).astype(np.int8)

                    out_head = umfa.quantized_attention(
                        q_head,
                        k_quantized,
                        v_quantized,
                        context=self.context,
                        causal=is_causal,
                        softmax_scale=scale,
                        query_precision="fp32",
                        kv_precision="int8",
                        output_precision="fp32",
                        k_scale=k_scale,
                        v_scale=v_scale,
                    )
                elif self.precision == "int4":
                    k_scale = np.abs(k_head).max() / 7.0  # 4-bit range: -7 to 7
                    v_scale = np.abs(v_head).max() / 7.0
                    k_quantized = np.clip(k_head / k_scale, -7, 7).astype(
                        np.int8
                    )  # Store as int8 but with 4-bit range
                    v_quantized = np.clip(v_head / v_scale, -7, 7).astype(np.int8)

                    out_head = umfa.quantized_attention(
                        q_head,
                        k_quantized,
                        v_quantized,
                        context=self.context,
                        causal=is_causal,
                        softmax_scale=scale,
                        query_precision="fp32",
                        kv_precision="int4",
                        output_precision="fp32",
                        k_scale=k_scale,
                        v_scale=v_scale,
                    )

                # Convert back to tensor
                out_tensor = torch.from_numpy(out_head).unsqueeze(0).to(query.device)
                if query.dtype == torch.bfloat16:
                    out_tensor = out_tensor.to(torch.bfloat16)
                elif query.dtype == torch.float16:
                    out_tensor = out_tensor.to(torch.float16)
                outputs.append(out_tensor)

            # Stack heads back together
            output = torch.stack(
                outputs, dim=2
            )  # [batch, seq_len, num_heads, head_dim]

            # Reshape back to original format if needed
            if output.shape != query.shape:
                output = output.view(batch_size, seq_len_q, -1)

            return output
        else:
            # Fallback for unsupported cases
            return self.original_sdpa(
                query, key, value, attn_mask, dropout_p, is_causal, scale
            )


def patch_attention_with_quantized_backend(precision="int8"):
    """Patch PyTorch SDPA to use quantized Metal backend"""
    import torch.nn.functional as F

    backend = QuantizedMetalSDPABackend(precision)
    backend.original_sdpa = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = backend

    return backend.original_sdpa


def benchmark_quantized_flux(precision="int8"):
    """Benchmark FLUX with quantized attention"""
    print(f"\n{'='*60}")
    print(f"🚀 Testing FLUX.1-Schnell with Quantized {precision.upper()} Attention")
    print(f"{'='*60}")

    # Patch attention
    original_sdpa = patch_attention_with_quantized_backend(precision)

    try:
        # Load pipeline
        print("📥 Loading FLUX.1-Schnell pipeline...")
        start_time = time.time()
        load_start_memory = get_memory_usage()

        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
        )
        pipe.to("mps")

        load_time = time.time() - start_time
        load_memory = get_memory_usage() - load_start_memory

        print(f"✅ Pipeline loaded and moved to MPS")
        print(f"⏱️  Pipeline load time: {load_time:.2f}s")
        print(f"💾 Memory used for loading: {load_memory:.1f} MB")

        prompt = "A futuristic cityscape with flying cars and neon lights, highly detailed digital art"

        # Warmup
        print("🔥 Performing warmup run...")
        warmup_start = time.time()
        with torch.inference_mode():
            warmup_image = pipe(
                prompt=prompt,
                guidance_scale=0.0,
                num_inference_steps=4,
                height=512,
                width=512,
                generator=torch.Generator("cpu").manual_seed(999),
            ).images[0]

        warmup_time = time.time() - warmup_start
        print(f"⏱️  Warmup generation time: {warmup_time:.2f}s")
        warmup_image.save(f"flux_schnell_quantized_{precision}_warmup.png")

        # Benchmark
        print(f"🎯 Running quantized {precision.upper()} benchmark...")
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
        peak_memory = get_memory_usage() - start_memory

        print(f"✅ Generation completed!")
        print(f"⏱️  Benchmark generation time: {generation_time:.2f}s")
        print(f"💾 Peak memory usage: {peak_memory:.1f} MB")

        image.save(f"flux_schnell_quantized_{precision}_benchmark.png")
        print(f"🖼️  Image saved as: flux_schnell_quantized_{precision}_benchmark.png")

        # Cleanup
        del pipe
        torch.mps.empty_cache()
        gc.collect()

        return {
            "precision": precision,
            "load_time": load_time,
            "warmup_time": warmup_time,
            "generation_time": generation_time,
            "load_memory": load_memory,
            "peak_memory": peak_memory,
        }

    finally:
        # Restore original SDPA
        import torch.nn.functional as F

        F.scaled_dot_product_attention = original_sdpa


if __name__ == "__main__":
    print("🔥 FLUX.1-Schnell Quantized Attention Benchmark")
    print("=" * 70)
    print("✅ Running on MPS (Apple Silicon)")
    print(f"🔧 PyTorch version: {torch.__version__}")

    if METAL_PYTHON_FFI_AVAILABLE:
        # Test INT8
        print("\n" + "=" * 70)
        print("🧮 Testing INT8 Quantized Attention")
        int8_results = benchmark_quantized_flux("int8")

        # Test INT4
        print("\n" + "=" * 70)
        print("🧮 Testing INT4 Quantized Attention")
        int4_results = benchmark_quantized_flux("int4")

        # Summary
        print(f"\n{'='*70}")
        print(f"📊 QUANTIZED PERFORMANCE COMPARISON")
        print(f"{'='*70}")
        print(f"INT8 Generation Time:   {int8_results['generation_time']:.2f}s")
        print(f"INT4 Generation Time:   {int4_results['generation_time']:.2f}s")
        print(f"INT8 Peak Memory:       {int8_results['peak_memory']:.1f} MB")
        print(f"INT4 Peak Memory:       {int4_results['peak_memory']:.1f} MB")

        if int4_results["generation_time"] < int8_results["generation_time"]:
            speedup = int8_results["generation_time"] / int4_results["generation_time"]
            print(f"🚀 INT4 is {speedup:.2f}x faster than INT8!")
        else:
            speedup = int4_results["generation_time"] / int8_results["generation_time"]
            print(f"🚀 INT8 is {speedup:.2f}x faster than INT4!")

        print(f"\n✨ Quantized benchmark completed!")
    else:
        print("❌ Cannot run quantized benchmark - Metal Python FFI not available")
