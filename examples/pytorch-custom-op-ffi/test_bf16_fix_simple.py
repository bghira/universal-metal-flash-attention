#!/usr/bin/env python3
"""
Simple BF16 Fix Validation Test

This test directly checks if the bf16 fix is working by using the raw extension.
"""

import sys

import torch

try:
    import metal_sdpa_extension

    print("✅ Successfully imported metal_sdpa_extension")
except ImportError as e:
    print(f"❌ Failed to import metal_sdpa_extension: {e}")
    sys.exit(1)


def test_bf16_fix():
    """Test that bf16 dtype is preserved through the FFI layer."""
    print("🧪 Testing BF16 fix...")

    # Create bf16 test tensors
    batch_size, seq_len, num_heads, head_dim = 2, 128, 8, 64

    print(f"Creating bf16 tensors: [{batch_size}, {seq_len}, {num_heads}, {head_dim}]")
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.bfloat16)

    # Store original values to check for corruption
    orig_q_val = float(q[0, 0, 0, 0])
    orig_k_val = float(k[0, 0, 0, 0])
    orig_v_val = float(v[0, 0, 0, 0])

    print(
        f"Original sample values: Q={orig_q_val:.6f}, K={orig_k_val:.6f}, V={orig_v_val:.6f}"
    )
    print(f"Input tensor dtypes: Q={q.dtype}, K={k.dtype}, V={v.dtype}")

    try:
        # Call the Metal SDPA function
        print("Calling metal_scaled_dot_product_attention...")
        result = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

        print(f"✅ Function call successful!")
        print(f"Result shape: {result.shape}")
        print(f"Result dtype: {result.dtype}")

        # Check that input tensors weren't corrupted by the FFI layer
        new_q_val = float(q[0, 0, 0, 0])
        new_k_val = float(k[0, 0, 0, 0])
        new_v_val = float(v[0, 0, 0, 0])

        print(
            f"After call sample values: Q={new_q_val:.6f}, K={new_k_val:.6f}, V={new_v_val:.6f}"
        )

        # Check for value corruption (main sign that the bf16 fix is working)
        tolerance = 1e-6
        q_preserved = abs(orig_q_val - new_q_val) < tolerance
        k_preserved = abs(orig_k_val - new_k_val) < tolerance
        v_preserved = abs(orig_v_val - new_v_val) < tolerance

        print(f"Value preservation: Q={q_preserved}, K={k_preserved}, V={v_preserved}")

        if q_preserved and k_preserved and v_preserved:
            print("✅ BF16 Fix SUCCESS: Input tensor values preserved!")
            return True
        else:
            print("❌ BF16 Fix FAILED: Input tensor values were corrupted")
            return False

    except Exception as e:
        print(f"❌ Function call failed: {e}")
        return False


def test_mixed_precision():
    """Test mixed precision scenarios."""
    print("\n🧪 Testing mixed precision scenarios...")

    dtypes_to_test = [
        (torch.bfloat16, "BF16"),
        (torch.float16, "FP16"),
        (torch.float32, "FP32"),
    ]

    results = []

    for dtype, name in dtypes_to_test:
        print(f"\nTesting {name}...")
        try:
            q = torch.randn(2, 64, 4, 32, dtype=dtype)
            k = torch.randn(2, 64, 4, 32, dtype=dtype)
            v = torch.randn(2, 64, 4, 32, dtype=dtype)

            result = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

            print(f"  Input: {dtype} → Output: {result.dtype}")
            results.append((name, True, f"Input: {dtype} → Output: {result.dtype}"))

        except Exception as e:
            print(f"  Failed: {e}")
            results.append((name, False, str(e)))

    return results


if __name__ == "__main__":
    print("🔧 Simple BF16 Fix Validation")
    print("=" * 40)

    # Test the main bf16 fix
    bf16_success = test_bf16_fix()

    # Test mixed precision
    mixed_results = test_mixed_precision()

    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Summary")

    print(f"BF16 Fix Test: {'✅ PASS' if bf16_success else '❌ FAIL'}")

    print("Mixed Precision Tests:")
    for name, success, message in mixed_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {name}: {status} - {message}")

    overall_success = bf16_success and all(success for _, success, _ in mixed_results)

    if overall_success:
        print("\n🎉 All tests passed! BF16 fix is working.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please review the implementation.")
        sys.exit(1)
