#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import numpy as np
import time
import sys
from pathlib import Path

# Add UMFA to path
sys.path.insert(0, str(Path(__file__).parent / "examples/python-ffi/src"))

import umfa


def torch_to_numpy_zero_copy(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to NumPy with zero-copy (when possible)."""
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor.detach().cpu().numpy()


def debug_pytorch_overhead():
    print("🔍 Debugging PyTorch Integration Overhead")
    print("=" * 50)

    # Test configuration
    seq_len = 512
    head_dim = 64

    # Create test tensors (FP16)
    q = torch.randn(seq_len, head_dim, dtype=torch.float16)
    k = torch.randn(seq_len, head_dim, dtype=torch.float16)
    v = torch.randn(seq_len, head_dim, dtype=torch.float16)

    print(f"✅ Created tensors: {q.shape}, dtype={q.dtype}")

    # Initialize MFA context once (reuse like PyTorch integration)
    context = umfa.MFAContext()

    # Warmup PyTorch
    for _ in range(3):
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # Warmup MFA
    q_np = torch_to_numpy_zero_copy(q)
    k_np = torch_to_numpy_zero_copy(k)
    v_np = torch_to_numpy_zero_copy(v)
    for _ in range(5):
        _ = umfa.flash_attention_forward(
            context, q_np, k_np, v_np, causal=True, input_precision="fp16"
        )

    total_times = []
    conversion_times = []
    mfa_times = []

    for _ in range(10):
        total_start = time.time()

        # Time the tensor conversion
        conv_start = time.time()
        q_np = torch_to_numpy_zero_copy(q)
        k_np = torch_to_numpy_zero_copy(k)
        v_np = torch_to_numpy_zero_copy(v)
        conv_time = time.time() - conv_start
        conversion_times.append(conv_time)

        # Time the MFA call
        mfa_start = time.time()
        output_np = umfa.flash_attention_forward(
            context, q_np, k_np, v_np, causal=True, input_precision="fp16"
        )
        mfa_time = time.time() - mfa_start
        mfa_times.append(mfa_time)

        total_time = time.time() - total_start
        total_times.append(total_time)

    # Convert back to torch (this part isn't timed in the benchmark)
    output_torch = torch.from_numpy(output_np)

    print(
        f"📊 Total time: {np.mean(total_times)*1000:.2f} ± {np.std(total_times)*1000:.2f} ms"
    )
    print(
        f"📊 Conversion time: {np.mean(conversion_times)*1000:.2f} ± {np.std(conversion_times)*1000:.2f} ms"
    )
    print(
        f"📊 MFA time: {np.mean(mfa_times)*1000:.2f} ± {np.std(mfa_times)*1000:.2f} ms"
    )
    print(
        f"📊 Conversion overhead: {np.mean(conversion_times)/np.mean(total_times)*100:.1f}%"
    )

    context.close()


if __name__ == "__main__":
    debug_pytorch_overhead()
