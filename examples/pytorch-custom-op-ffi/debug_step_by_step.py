#!/usr/bin/env python3
"""
Step-by-step test to identify which function hangs.
Author: bghira
"""

import sys

import metal_sdpa_extension
import numpy as np
import torch
import torch.nn.functional as F


def test_exact_swift_params():
    """Test with exactly the same parameters as Swift test"""
    print("Step 1: Testing exact Swift parameters...")
    sys.stdout.flush()

    if not metal_sdpa_extension.is_metal_available():
        print("❌ Metal not available")
        return False

    # EXACT same parameters as Swift test
    seq_len = 4
    head_dim = 4

    # Create tensors with ALL 1.0s (exactly like Swift test)
    q = torch.ones(seq_len, head_dim, dtype=torch.float32)
    k = torch.ones(seq_len, head_dim, dtype=torch.float32)
    v = torch.ones(seq_len, head_dim, dtype=torch.float32)

    scale = 1.0 / np.sqrt(head_dim)

    try:
        # Metal implementation with explicit scale
        metal_output = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q, k, v, scale=scale, is_causal=False
        )

        # PyTorch reference
        torch_output = F.scaled_dot_product_attention(
            q, k, v, scale=scale, is_causal=False
        )

        # Calculate difference
        diff = torch.abs(metal_output - torch_output).max().item()

        if diff < 1e-5:
            print("✅ Step 1 PASSED")
            return True
        else:
            print(f"❌ Step 1 FAILED: diff {diff}")
            return False

    except Exception as e:
        print(f"❌ Step 1 ERROR: {e}")
        return False


def test_tensor_layout_debug():
    """Debug tensor layout issues"""
    print("Step 2: Testing tensor layouts...")
    sys.stdout.flush()

    seq_len, head_dim = 4, 4

    # Test different tensor layouts
    layouts = [
        ("C-contiguous", torch.ones(seq_len, head_dim, dtype=torch.float32)),
        ("Fortran-like", torch.ones(head_dim, seq_len, dtype=torch.float32).T),
    ]

    try:
        for name, tensor in layouts:
            print(f"  Testing {name}...")
            sys.stdout.flush()

            output = metal_sdpa_extension.metal_scaled_dot_product_attention(
                tensor, tensor, tensor
            )
            print(f"  ✅ {name} succeeded")

        print("✅ Step 2 PASSED")
        return True

    except Exception as e:
        print(f"❌ Step 2 ERROR: {e}")
        return False


def test_precision_mapping():
    """Test precision type mapping"""
    print("Step 3: Testing precision mapping...")
    sys.stdout.flush()

    dtypes = [
        (torch.float32, "FP32"),
        (torch.float16, "FP16"),
    ]

    try:
        for dtype, name in dtypes:
            print(f"  Testing {name}...")
            sys.stdout.flush()

            q = torch.ones(4, 4, dtype=dtype)
            k = torch.ones(4, 4, dtype=dtype)
            v = torch.ones(4, 4, dtype=dtype)

            output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

            has_nan = torch.isnan(output).any()
            print(f"  {'❌ NaN' if has_nan else '✅'} {name}")

        print("✅ Step 3 PASSED")
        return True

    except Exception as e:
        print(f"❌ Step 3 ERROR: {e}")
        return False


def main():
    print("Stepping through debug functions...")

    try:
        step1 = test_exact_swift_params()
        print("Moving to step 2...")
        sys.stdout.flush()

        step2 = test_tensor_layout_debug()
        print("Moving to step 3...")
        sys.stdout.flush()

        step3 = test_precision_mapping()
        print("All steps completed!")

        if step1 and step2 and step3:
            print("🎉 ALL STEPS PASSED!")
        else:
            print("❌ Some steps failed")

    except Exception as e:
        print(f"❌ Main function error: {e}")


if __name__ == "__main__":
    main()
