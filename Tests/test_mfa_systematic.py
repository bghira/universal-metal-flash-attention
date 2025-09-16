#!/usr/bin/env python3
"""Systematic tests for Metal Flash Attention implementation"""

import sys
from pathlib import Path

import numpy as np
import torch

# Add the extension to path
sys.path.insert(
    0, str(Path(__file__).parent / ".." / "examples" / "pytorch-custom-op-ffi")
)
import metal_sdpa_extension


def test_basic_shapes():
    """Test with progressively complex shapes"""
    print("=" * 60)
    print("TESTING BASIC SHAPES")
    print("=" * 60)

    test_cases = [
        # (batch, heads, seq_len, head_dim)
        (1, 1, 64, 64),  # Simplest case
        (1, 1, 128, 64),  # Different seq_len
        (1, 4, 64, 64),  # Multi-head
        (1, 8, 256, 64),  # Larger multi-head
        (1, 24, 1536, 128),  # FLUX shape
        (2, 24, 1536, 128),  # Batched FLUX
    ]

    for shape in test_cases:
        B, H, S, D = shape
        print(f"\nTesting shape: {shape}")

        # Create test tensors
        q = torch.randn(B, H, S, D, device="mps", dtype=torch.float16)
        k = torch.randn(B, H, S, D, device="mps", dtype=torch.float16)
        v = torch.randn(B, H, S, D, device="mps", dtype=torch.float16)

        try:
            # Test regular Metal SDPA
            result = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

            has_nan = torch.isnan(result).any().item()
            has_inf = torch.isinf(result).any().item()

            if has_nan or has_inf:
                print(f"  ❌ FAILED: NaN={has_nan}, Inf={has_inf}")
                # Debug info
                print(
                    f"     Result stats: min={result.min():.4f}, max={result.max():.4f}"
                )
            else:
                # Compare with native
                native = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                diff = (result - native).abs().max().item()

                if diff < 0.01:
                    print(f"  ✅ PASSED: max_diff={diff:.6f}")
                else:
                    print(f"  ⚠️  WARNING: large diff={diff:.6f}")
                    print(
                        f"     Metal: min={result.min():.4f}, max={result.max():.4f}, mean={result.mean():.4f}"
                    )
                    print(
                        f"     Native: min={native.min():.4f}, max={native.max():.4f}, mean={native.mean():.4f}"
                    )

        except Exception as e:
            print(f"  ❌ EXCEPTION: {e}")


def test_quantized_attention():
    """Test quantized attention with different configurations"""
    print("\n" + "=" * 60)
    print("TESTING QUANTIZED ATTENTION")
    print("=" * 60)

    # Simple test shape
    B, H, S, D = 1, 4, 256, 64

    q = torch.randn(B, H, S, D, device="mps", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="mps", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="mps", dtype=torch.float16)

    print(f"\nShape: [{B}, {H}, {S}, {D}]")

    # Test basic quantized attention
    print("\n1. Basic quantized attention:")
    try:
        result = metal_sdpa_extension.quantized_scaled_dot_product_attention(q, k, v)
        has_nan = torch.isnan(result).any().item()
        has_inf = torch.isinf(result).any().item()

        if has_nan or has_inf:
            print(f"  ❌ FAILED: NaN={has_nan}, Inf={has_inf}")
        else:
            print(f"  ✅ PASSED")
            print(
                f"     Stats: min={result.min():.4f}, max={result.max():.4f}, mean={result.mean():.4f}"
            )
    except Exception as e:
        print(f"  ❌ EXCEPTION: {e}")

    # Test with config
    print("\n2. Quantized with config:")
    try:
        config = metal_sdpa_extension.QuantizationConfig()
        config.query_precision = metal_sdpa_extension.FP32
        config.key_precision = metal_sdpa_extension.FP16
        config.value_precision = metal_sdpa_extension.FP16

        result = (
            metal_sdpa_extension.quantized_scaled_dot_product_attention_with_config(
                q, k, v, config
            )
        )

        has_nan = torch.isnan(result).any().item()
        has_inf = torch.isinf(result).any().item()

        if has_nan or has_inf:
            print(f"  ❌ FAILED: NaN={has_nan}, Inf={has_inf}")
        else:
            print(f"  ✅ PASSED")
            print(
                f"     Stats: min={result.min():.4f}, max={result.max():.4f}, mean={result.mean():.4f}"
            )
    except Exception as e:
        print(f"  ❌ EXCEPTION: {e}")


def test_edge_cases():
    """Test edge cases that might cause issues"""
    print("\n" + "=" * 60)
    print("TESTING EDGE CASES")
    print("=" * 60)

    # Test with very small values
    print("\n1. Very small values:")
    q = torch.randn(1, 1, 64, 64, device="mps", dtype=torch.float16) * 0.001
    k = torch.randn(1, 1, 64, 64, device="mps", dtype=torch.float16) * 0.001
    v = torch.randn(1, 1, 64, 64, device="mps", dtype=torch.float16) * 0.001

    try:
        result = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
        has_nan = torch.isnan(result).any().item()
        has_inf = torch.isinf(result).any().item()
        print(f"  NaN={has_nan}, Inf={has_inf}")
        if not (has_nan or has_inf):
            print(f"  ✅ Handles small values")
    except Exception as e:
        print(f"  ❌ Exception: {e}")

    # Test with large values
    print("\n2. Large values:")
    q = torch.randn(1, 1, 64, 64, device="mps", dtype=torch.float16) * 10
    k = torch.randn(1, 1, 64, 64, device="mps", dtype=torch.float16) * 10
    v = torch.randn(1, 1, 64, 64, device="mps", dtype=torch.float16) * 10

    try:
        result = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
        has_nan = torch.isnan(result).any().item()
        has_inf = torch.isinf(result).any().item()
        print(f"  NaN={has_nan}, Inf={has_inf}")
        if not (has_nan or has_inf):
            print(f"  ✅ Handles large values")
    except Exception as e:
        print(f"  ❌ Exception: {e}")

    # Test with zeros
    print("\n3. Zero tensors:")
    q = torch.zeros(1, 1, 64, 64, device="mps", dtype=torch.float16)
    k = torch.zeros(1, 1, 64, 64, device="mps", dtype=torch.float16)
    v = torch.randn(1, 1, 64, 64, device="mps", dtype=torch.float16)

    try:
        result = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
        has_nan = torch.isnan(result).any().item()
        has_inf = torch.isinf(result).any().item()
        print(f"  NaN={has_nan}, Inf={has_inf}")
        if not (has_nan or has_inf):
            print(f"  ✅ Handles zero Q/K")
    except Exception as e:
        print(f"  ❌ Exception: {e}")


def test_precision_configurations():
    """Test different precision configurations"""
    print("\n" + "=" * 60)
    print("TESTING PRECISION CONFIGURATIONS")
    print("=" * 60)

    B, H, S, D = 1, 4, 256, 64
    q = torch.randn(B, H, S, D, device="mps", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="mps", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="mps", dtype=torch.float16)

    configs = [
        (
            "FP32/FP32/FP32",
            metal_sdpa_extension.FP32,
            metal_sdpa_extension.FP32,
            metal_sdpa_extension.FP32,
        ),
        (
            "FP16/FP16/FP16",
            metal_sdpa_extension.FP16,
            metal_sdpa_extension.FP16,
            metal_sdpa_extension.FP16,
        ),
        (
            "FP32/FP16/FP16",
            metal_sdpa_extension.FP32,
            metal_sdpa_extension.FP16,
            metal_sdpa_extension.FP16,
        ),
    ]

    for name, q_prec, k_prec, v_prec in configs:
        print(f"\n{name}:")
        try:
            config = metal_sdpa_extension.QuantizationConfig()
            config.query_precision = q_prec
            config.key_precision = k_prec
            config.value_precision = v_prec
            config.output_precision = metal_sdpa_extension.FP32

            result = (
                metal_sdpa_extension.quantized_scaled_dot_product_attention_with_config(
                    q, k, v, config
                )
            )

            has_nan = torch.isnan(result).any().item()
            has_inf = torch.isinf(result).any().item()

            if has_nan or has_inf:
                print(f"  ❌ FAILED: NaN={has_nan}, Inf={has_inf}")
            else:
                print(f"  ✅ PASSED")
                print(
                    f"     Stats: min={result.min():.4f}, max={result.max():.4f}, mean={result.mean():.4f}"
                )

        except Exception as e:
            print(f"  ❌ EXCEPTION: {e}")


if __name__ == "__main__":
    print("METAL FLASH ATTENTION SYSTEMATIC TESTS")
    print("=" * 60)

    # Initialize
    print("Initializing Metal SDPA backend...")
    metal_sdpa_extension.register_backend()

    # Run tests
    test_basic_shapes()
    test_quantized_attention()
    test_edge_cases()
    test_precision_configurations()

    print("\n" + "=" * 60)
    print("TESTS COMPLETE")
    print("=" * 60)
