#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

# Add UMFA to path
sys.path.insert(0, str(Path(__file__).parent / "examples/python-ffi/src"))

try:
    import umfa

    print("✅ UMFA successfully imported")

    # Simple PyTorch tensor test
    seq_len, head_dim = 64, 32
    q = torch.randn(seq_len, head_dim, dtype=torch.float32)
    k = torch.randn(seq_len, head_dim, dtype=torch.float32)
    v = torch.randn(seq_len, head_dim, dtype=torch.float32)

    print(f"✅ Created tensors: {q.shape}, {k.shape}, {v.shape}")

    # Convert to numpy (zero-copy when possible)
    q_np = q.detach().numpy()
    k_np = k.detach().numpy()
    v_np = v.detach().numpy()

    print("✅ Converted to numpy arrays")

    # Use Metal Flash Attention
    with umfa.MFAContext() as ctx:
        output_np = umfa.flash_attention_forward(
            ctx, q_np, k_np, v_np, causal=False, input_precision="fp32"
        )

    # Convert back to PyTorch
    output_torch = torch.from_numpy(output_np)

    print(f"✅ Metal FA output: {output_torch.shape}")
    print(f"✅ Output range: [{output_torch.min():.4f}, {output_torch.max():.4f}]")
    print(f"✅ Has non-zero values: {torch.any(output_torch != 0.0)}")

    print("\n🎉 PyTorch + Metal Flash Attention integration successful!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
