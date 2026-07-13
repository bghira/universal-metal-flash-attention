#!/usr/bin/env python3
"""Z-Image Turbo on Universal Metal Flash Attention.

Generates an image with TONGYI-MAI/Z-Image-Turbo (8 steps by default) on MPS.
Importing the Metal SDPA extension installs a process-wide ATen override for
scaled_dot_product_attention on MPS, so the transformer's attention runs
through UMFA without any monkeypatching. Pass --native to skip the extension
and run PyTorch's native MPS SDPA instead.
"""

import argparse
import os
import sys
import time
from pathlib import Path

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


def _maybe_add_venv_site_packages() -> None:
    venv_root = Path(os.environ.get("VIRTUAL_ENV", PROJECT_ROOT / ".venv"))
    if not venv_root.exists():
        return
    for site_packages in venv_root.glob("lib/python*/site-packages"):
        if site_packages.is_dir():
            sys.path.insert(0, str(site_packages))
            break


_maybe_add_venv_site_packages()

# Freshly built extension takes precedence over any stale installed copy
sys.path.insert(0, str(PROJECT_ROOT / "examples" / "pytorch-custom-op-ffi"))


def parse_args():
    parser = argparse.ArgumentParser(description="Z-Image Turbo via Metal SDPA")
    parser.add_argument(
        "--model", default="TONGYI-MAI/Z-Image-Turbo", help="Model id or local path"
    )
    parser.add_argument(
        "--prompt",
        default="a photo of a corgi wearing sunglasses on a beach",
        help="Prompt used for generation",
    )
    parser.add_argument("--steps", type=int, default=8, help="Inference steps")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=0.0,
        help="CFG scale. ZImagePipeline enables CFG when guidance_scale > 0; "
        "Turbo is guidance-distilled, so 0.0 (no CFG) is correct",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--native",
        action="store_true",
        help="Skip the Metal SDPA extension and use PyTorch's native MPS SDPA",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load from the local Hugging Face cache without network access",
    )
    parser.add_argument("--output", default=None, help="Output PNG path")
    return parser.parse_args()


def main():
    args = parse_args()

    import torch

    metal_ext = None
    if args.native:
        print("🐍 Native mode: Metal SDPA extension NOT loaded")
    else:
        import metal_sdpa_extension as metal_ext

        print(
            f"✅ Metal SDPA extension loaded (v{metal_ext.get_version()}) - "
            "ATen MPS override active"
        )

    from diffusers import ZImagePipeline

    print("📦 Loading Z-Image Turbo pipeline...")
    load_start = time.time()
    pipe = ZImagePipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        local_files_only=args.local_files_only,
    ).to("mps")
    print(f"📦 Pipeline loaded in {time.time() - load_start:.1f}s")

    if metal_ext is not None:
        metal_ext.reset_dispatch_stats()

    print(
        f"🎨 Generating {args.width}x{args.height}, {args.steps} steps, "
        f"guidance {args.guidance_scale}..."
    )
    start = time.time()
    with torch.inference_mode():
        image = pipe(
            prompt=args.prompt,
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator().manual_seed(args.seed),
        ).images[0]
    elapsed = time.time() - start

    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = PROJECT_ROOT / "examples" / "zimage" / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "native" if args.native else "umfa"
        out_path = out_dir / f"zimage_turbo_{suffix}_{args.steps}step.png"
    image.save(out_path)

    print(f"✅ Generated in {elapsed:.2f}s -> {out_path}")

    if metal_ext is not None:
        stats = metal_ext.get_dispatch_stats()
        print(f"📊 UMFA dispatch stats: {stats}")


if __name__ == "__main__":
    main()
