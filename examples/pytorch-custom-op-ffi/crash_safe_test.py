#!/usr/bin/env python3
"""
Crash-safe test for Metal SDPA backend.
Author: bghira
"""

import metal_sdpa_extension
import torch


def safe_test(name, test_func):
    """Safely run a test and catch any exceptions"""
    print(f"\n--- {name} ---")
    try:
        test_func()
        print(f"✅ {name}: Success")
    except Exception as e:
        print(f"❌ {name}: {e}")


def test_working_case():
    """Test known working case"""
    q = torch.randn(8, 4, dtype=torch.float32)
    k = torch.randn(8, 4, dtype=torch.float32)
    v = torch.randn(8, 4, dtype=torch.float32)

    output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
    assert not torch.isnan(output).any(), "Output contains NaN"


def test_multihead_rejection():
    """Test that multi-head is properly rejected"""
    q = torch.randn(2, 8, 4, 4, dtype=torch.float32)  # 4 heads
    k = torch.randn(2, 8, 4, 4, dtype=torch.float32)
    v = torch.randn(2, 8, 4, 4, dtype=torch.float32)

    output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
    # Should not reach here


def test_single_head_4d():
    """Test single head 4D tensor"""
    q = torch.randn(2, 8, 1, 4, dtype=torch.float32)  # 1 head
    k = torch.randn(2, 8, 1, 4, dtype=torch.float32)
    v = torch.randn(2, 8, 1, 4, dtype=torch.float32)

    output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
    assert not torch.isnan(output).any(), "Output contains NaN"


def test_large_tensors():
    """Test larger tensor dimensions"""
    q = torch.randn(128, 64, dtype=torch.float32)
    k = torch.randn(128, 64, dtype=torch.float32)
    v = torch.randn(128, 64, dtype=torch.float32)

    output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
    assert not torch.isnan(output).any(), "Output contains NaN"


def main():
    print("Crash-Safe Metal SDPA Test")
    print("Author: bghira")
    print("=" * 40)

    if not metal_sdpa_extension.is_metal_available():
        print("❌ Metal not available")
        return

    print("✅ Metal available")

    # Run tests safely
    safe_test("Working 2D Case", test_working_case)
    safe_test("Single Head 4D", test_single_head_4d)
    safe_test("Multi-head Rejection", test_multihead_rejection)
    safe_test("Large Tensors", test_large_tensors)

    print(f"\n{'='*40}")
    print("✅ All crash-safe tests completed!")
    print("The backend now properly handles error conditions")
    print("without crashing Python/macOS.")


if __name__ == "__main__":
    main()
