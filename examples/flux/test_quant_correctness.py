import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "examples" / "pytorch-custom-op-ffi"))
import metal_sdpa_extension as ext
import torch
import torch.nn.functional as F

device = torch.device("mps")
torch.manual_seed(42)
B, H, S, D = 1, 24, 1024, 128
q = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
k = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
v = torch.randn(B, H, S, D, device=device, dtype=torch.bfloat16)
ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=D**-0.5)

for prec in ("int8", "int4"):
    out = ext.quantized_scaled_dot_product_attention(
        q, k, v, precision=prec, is_causal=False, scale=D**-0.5
    )
    torch.mps.synchronize()
    err = (out.float() - ref).abs()
    psnr = 20 * torch.log10(ref.abs().max() / err.max())
    print(
        f"{prec}: max_abs_err={err.max().item():.4f}  mean_abs_err={err.mean().item():.5f}  psnr_vs_ref={psnr.item():.1f} dB  dtype={out.dtype}"
    )
