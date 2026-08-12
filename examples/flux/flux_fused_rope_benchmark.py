#!/usr/bin/env python3
"""
FLUX benchmark with fused RoPE + attention via UMFA's rope_scaled_dot_product_attention.

Patches apply_rotary_emb to a no-op (stores cos/sin for the SDPA wrapper),
then the SDPA wrapper calls UMFA's fused rope+attention path instead of
doing rotation + attention separately.
"""
import argparse
import gc
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _prepend_dyld(paths):
    existing = os.environ.get("DYLD_LIBRARY_PATH", "")
    valid = [str(p) for p in paths if p.exists()]
    if not valid:
        return
    prefix = ":".join(valid)
    os.environ["DYLD_LIBRARY_PATH"] = f"{prefix}:{existing}" if existing else prefix


_prepend_dyld(
    [
        PROJECT_ROOT / ".build" / "arm64-apple-macosx" / "release",
        PROJECT_ROOT / ".build" / "arm64-apple-macosx" / "debug",
    ]
)


def _setup_venv():
    venv = Path(os.environ.get("VIRTUAL_ENV", PROJECT_ROOT / ".venv"))
    if not venv.exists():
        return
    for sp in venv.glob("lib/python*/site-packages"):
        if sp.is_dir():
            sys.path.insert(0, str(sp))
            break


_setup_venv()
sys.path.insert(0, str(PROJECT_ROOT / "examples" / "pytorch-custom-op-ffi"))

import metal_sdpa_extension as ext

try:
    from diffusers import FluxPipeline
    from diffusers.models.embeddings import apply_rotary_emb

    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

# Thread-local storage for passing rotary tables from the patched
# apply_rotary_emb to the SDPA wrapper.
_rope_state = threading.local()
_original_apply_rotary_emb = apply_rotary_emb


def _noop_apply_rotary_emb(x, freqs_cis, **kwargs):
    """Store rotary tables and return x unmodified — rotation is fused into attention."""
    if isinstance(freqs_cis, (tuple, list)):
        _rope_state.cos = freqs_cis[0]
        _rope_state.sin = freqs_cis[1]
    else:
        _rope_state.cos = freqs_cis
        _rope_state.sin = None
    return x


def create_fused_rope_sdpa_wrapper():
    """SDPA wrapper that uses fused RoPE+attention when rotary tables are available."""
    original_sdpa = F.scaled_dot_product_attention

    def wrapper(*args, **kwargs):
        query = args[0] if args else kwargs.get("query")
        key = args[1] if len(args) > 1 else kwargs.get("key")
        value = args[2] if len(args) > 2 else kwargs.get("value")

        if query is None or query.device.type != "mps":
            return original_sdpa(*args, **kwargs)

        scale = kwargs.get("scale", 1.0 / (query.shape[-1] ** 0.5))
        is_causal = kwargs.get("is_causal", False)

        cos = getattr(_rope_state, "cos", None)
        sin = getattr(_rope_state, "sin", None)

        try:
            if cos is not None and sin is not None:
                # Fused RoPE + attention path.
                # UMFA expects BHSD; FLUX gives BSHD after unflatten.
                # transpose(1,2) is a view (last dim stays contiguous).
                needs_transpose = (
                    query.dim() == 4
                    and query.shape[1] != query.shape[2]
                    and query.shape[1] > query.shape[2]
                )
                # FLUX BSHD: [B, S, H, D] where S > H typically.
                # Detect BSHD vs BHSD by checking if dim 1 > dim 2 (S > H).
                if needs_transpose:
                    q_bhsd = query.transpose(1, 2).contiguous()
                    k_bhsd = key.transpose(1, 2).contiguous()
                    v_bhsd = value.transpose(1, 2).contiguous()
                else:
                    q_bhsd = query
                    k_bhsd = key
                    v_bhsd = value

                result = ext.rope_scaled_dot_product_attention(
                    q_bhsd,
                    k_bhsd,
                    v_bhsd,
                    cos,
                    sin,
                    is_causal=is_causal,
                    scale=scale,
                )

                if needs_transpose:
                    result = result.transpose(1, 2)

                # Clear stored rotary tables.
                _rope_state.cos = None
                _rope_state.sin = None
                return result
            else:
                # Regular UMFA path (no rotary).
                result = ext.metal_scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    is_causal=is_causal,
                    scale=scale,
                    enable_gqa=False,
                )
                return result
        except Exception:
            _rope_state.cos = None
            _rope_state.sin = None
            return original_sdpa(*args, **kwargs)

    return wrapper


def patch_attention(use_fused_rope: bool):
    """Patch PyTorch SDPA and apply_rotary_emb."""
    import diffusers.models.embeddings as emb_mod

    if use_fused_rope:
        # Replace apply_rotary_emb with no-op that stores tables.
        emb_mod.apply_rotary_emb = _noop_apply_rotary_emb
        wrapper = create_fused_rope_sdpa_wrapper()
    else:
        # Restore original apply_rotary_emb, use regular UMFA SDPA.
        emb_mod.apply_rotary_emb = _original_apply_rotary_emb

        def regular_wrapper(*args, **kwargs):
            q = args[0] if args else kwargs.get("query")
            if q is None or q.device.type != "mps":
                return F.scaled_dot_product_attention(*args, **kwargs)
            try:
                scale = kwargs.get("scale", 1.0 / (q.shape[-1] ** 0.5))
                return ext.metal_scaled_dot_product_attention(
                    q,
                    args[1] if len(args) > 1 else kwargs["key"],
                    args[2] if len(args) > 2 else kwargs["value"],
                    is_causal=kwargs.get("is_causal", False),
                    scale=scale,
                    enable_gqa=False,
                )
            except:
                return F.scaled_dot_product_attention(*args, **kwargs)

        wrapper = regular_wrapper

    original = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = wrapper
    return original


def restore_attention(original_sdpa):
    if original_sdpa is not None:
        F.scaled_dot_product_attention = original_sdpa
    import diffusers.models.embeddings as emb_mod

    emb_mod.apply_rotary_emb = _original_apply_rotary_emb


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="black-forest-labs/FLUX.1-schnell")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--guidance-scale", type=float, default=None)
    p.add_argument(
        "--precision",
        choices=["all", "vanilla", "bf16", "fused-rope", "bf16-norope"],
        default="all",
    )
    p.add_argument("--prompt", default="A simple test")
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


def run_benchmark(args):
    if not DIFFUSERS_AVAILABLE:
        print("Cannot run without diffusers")
        return

    guidance = args.guidance_scale or (0.0 if "schnell" in args.model.lower() else 3.5)

    print(f"\n{'='*60}")
    print(f"FLUX Fused-RoPE Benchmark — {args.width}x{args.height}, {args.steps} steps")
    print(f"{'='*60}")

    configs = [
        ("PyTorch Vanilla", "vanilla"),
        ("UMFA BF16 (separate RoPE)", "bf16"),
        ("UMFA BF16 (fused RoPE)", "fused-rope"),
    ]

    if args.precision != "all":
        configs = [(n, m) for n, m in configs if m == args.precision]

    print("\nLoading pipeline...")
    load_start = time.time()
    pipe = FluxPipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=args.local_files_only
    )
    pipe = pipe.to("mps")
    pipe.set_progress_bar_config(disable=True)
    print(f"Loaded in {time.time() - load_start:.1f}s")

    results = []

    for config_name, mode in configs:
        print(f"\nTesting: {config_name}")
        torch.mps.empty_cache()
        gc.collect()

        original_sdpa = None
        if mode == "vanilla":
            pass  # no patching
        elif mode == "bf16":
            original_sdpa = patch_attention(use_fused_rope=False)
        elif mode == "fused-rope":
            original_sdpa = patch_attention(use_fused_rope=True)

        try:
            start = time.time()
            with torch.inference_mode():
                image = pipe(
                    prompt=args.prompt,
                    num_inference_steps=args.steps,
                    height=args.height,
                    width=args.width,
                    guidance_scale=guidance,
                    generator=torch.Generator().manual_seed(42),
                ).images[0]

            dt = time.time() - start
            out_dir = PROJECT_ROOT / "examples" / "flux" / "output" / "fused_rope_test"
            out_dir.mkdir(parents=True, exist_ok=True)
            image.save(out_dir / f"{mode}.png")
            print(f"  Time: {dt:.2f}s")
            results.append({"config": config_name, "time": dt})

        except Exception as e:
            print(f"  Failed: {e}")
            results.append({"config": config_name, "time": None})

        finally:
            restore_attention(original_sdpa)
            torch.mps.empty_cache()
            gc.collect()

    # PSNR
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    import numpy as np
    from PIL import Image

    baseline_time = None
    for r in results:
        if r["time"]:
            if baseline_time is None:
                baseline_time = r["time"]
            speedup = baseline_time / r["time"]
            print(f"  {r['config']:<35s} {r['time']:>7.2f}s  ({speedup:.2f}x)")

    # PSNR vs vanilla
    ref_path = (
        PROJECT_ROOT
        / "examples"
        / "flux"
        / "output"
        / "fused_rope_test"
        / "vanilla.png"
    )
    if ref_path.exists():
        ref = np.array(Image.open(ref_path).convert("RGB"), dtype=np.float32)
        print("\n  PSNR vs PyTorch Vanilla:")
        for r in results:
            mode = [m for n, m in configs if n == r["config"]][0]
            path = (
                PROJECT_ROOT
                / "examples"
                / "flux"
                / "output"
                / "fused_rope_test"
                / f"{mode}.png"
            )
            if path.exists() and mode != "vanilla":
                img = np.array(Image.open(path).convert("RGB"), dtype=np.float32)
                mse = np.mean((ref - img) ** 2)
                psnr = float("inf") if mse == 0 else 10 * np.log10(255**2 / mse)
                print(f"    {r['config']:<35s} {psnr:.2f} dB")


if __name__ == "__main__":
    run_benchmark(parse_args())
