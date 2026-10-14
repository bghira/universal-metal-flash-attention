#!/usr/bin/env python3
"""Profile quantized attention overhead at FLUX-like dimensions."""
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "examples" / "pytorch-custom-op-ffi"))

for p in [
    PROJECT_ROOT / ".build" / "arm64-apple-macosx" / "release",
    PROJECT_ROOT / ".build" / "arm64-apple-macosx" / "debug",
]:
    if p.exists():
        existing = os.environ.get("DYLD_LIBRARY_PATH", "")
        os.environ["DYLD_LIBRARY_PATH"] = f"{p}:{existing}" if existing else str(p)
        break

import metal_sdpa_extension as ext
import torch

device = torch.device("mps")
torch.mps.synchronize()

B, H, S, D = 1, 24, 4096, 128
q = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
k = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
v = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)

WARMUP = 5
ITERS = 20


def bench(fn, label):
    for _ in range(WARMUP):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.mps.synchronize()
    dt = (time.perf_counter() - t0) / ITERS
    print(f"  {label:50s} {dt*1000:8.1f} ms")
    return dt


print(f"Shape: [{B}, {H}, {S}, {D}]  (FLUX 1024x1024 image stream)")
print(f"Warmup={WARMUP}, Iters={ITERS}\n")

# 1. BF16 non-autograd (raw SDPA wrapper)
print("BF16 non-autograd (metal_scaled_dot_product_attention):")
t_bf16_raw = bench(
    lambda: ext.metal_scaled_dot_product_attention(
        q, k, v, is_causal=False, scale=1.0 / (D**0.5), enable_gqa=False
    ),
    "forward",
)

# 2. BF16 autograd
print("\nBF16 autograd (metal_flash_attention_autograd):")
t_bf16_ag = bench(
    lambda: ext.metal_flash_attention_autograd(
        q, k, v, is_causal=False, scale=1.0 / (D**0.5)
    ),
    "forward",
)

# 3. INT8 autograd
print("\nINT8 autograd (quantized_scaled_dot_product_attention):")
t_int8 = bench(
    lambda: ext.quantized_scaled_dot_product_attention(
        q, k, v, precision="int8", is_causal=False, scale=1.0 / (D**0.5)
    ),
    "forward",
)

# 4. INT4 autograd
print("\nINT4 autograd (quantized_scaled_dot_product_attention):")
t_int4 = bench(
    lambda: ext.quantized_scaled_dot_product_attention(
        q, k, v, precision="int4", is_causal=False, scale=1.0 / (D**0.5)
    ),
    "forward",
)

# 5. PyTorch native SDPA
print("\nPyTorch native SDPA:")
t_native = bench(
    lambda: torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=False, scale=1.0 / (D**0.5)
    ),
    "forward",
)

print("\n" + "=" * 70)
print(f"  {'Configuration':48s} {'ms':>8s} {'vs BF16':>8s}")
print("-" * 70)
for label, t in [
    ("PyTorch native SDPA", t_native),
    ("BF16 non-autograd", t_bf16_raw),
    ("BF16 autograd", t_bf16_ag),
    ("INT8 autograd", t_int8),
    ("INT4 autograd", t_int4),
]:
    delta = t - t_bf16_raw
    print(f"  {label:48s} {t*1000:8.1f} {delta*1000:+8.1f}")

print(f"\n  INT8 overhead vs BF16 autograd: {(t_int8-t_bf16_ag)*1000:+.1f} ms/call")
print(f"  INT4 overhead vs BF16 autograd: {(t_int4-t_bf16_ag)*1000:+.1f} ms/call")
print(f"  BF16 autograd overhead vs raw:  {(t_bf16_ag-t_bf16_raw)*1000:+.1f} ms/call")
print(f"\n  At 76 FLUX layers × 4 steps = 304 calls:")
print(f"    BF16 raw:     {t_bf16_raw*304:.1f}s")
print(f"    BF16 autograd:{t_bf16_ag*304:.1f}s")
print(f"    INT8:         {t_int8*304:.1f}s")
print(f"    INT4:         {t_int4*304:.1f}s")
