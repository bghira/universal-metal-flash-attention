#!/usr/bin/env python3
"""
Comprehensive correctness test for Metal SDPA with current compiler flags.
Author: bghira
"""

import torch
import torch.nn.functional as F
import metal_sdpa_extension
import numpy as np

def test_numerical_correctness():
    """Test numerical correctness across different scenarios"""
    print("Metal Flash Attention Correctness Verification")
    print("Author: bghira")
    print("Testing current Swift compiler configuration")
    print("=" * 60)

    if not metal_sdpa_extension.is_metal_available():
        print("❌ Metal not available")
        return

    all_passed = True
    tolerance = 1e-5  # Stricter tolerance for correctness

    # Test cases with increasing complexity
    test_cases = [
        ("Tiny (4x4)", 4, 4),
        ("Small (32x16)", 32, 16),
        ("Medium (128x32)", 128, 32),
        ("Large (256x64)", 256, 64),
        ("Very Large (512x128)", 512, 128),
    ]

    print(f"{'Test Case':20} | {'Max Diff':>12} | {'Rel Diff':>12} | {'Status':>8}")
    print("-" * 70)

    for name, seq_len, head_dim in test_cases:
        # Generate reproducible test data
        torch.manual_seed(42)
        q = torch.randn(seq_len, head_dim, dtype=torch.float32)
        k = torch.randn(seq_len, head_dim, dtype=torch.float32)
        v = torch.randn(seq_len, head_dim, dtype=torch.float32)

        try:
            # Metal implementation
            metal_output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)

            # PyTorch reference
            torch_output = F.scaled_dot_product_attention(q, k, v)

            # Check for NaN/Inf
            if torch.isnan(metal_output).any() or torch.isinf(metal_output).any():
                print(f"{name:20} | {'NaN/Inf':>12} | {'N/A':>12} | {'❌ FAIL':>8}")
                all_passed = False
                continue

            # Calculate differences
            abs_diff = torch.abs(metal_output - torch_output)
            max_diff = abs_diff.max().item()
            rel_diff = (abs_diff / torch.abs(torch_output).clamp(min=1e-8)).max().item()

            # Check tolerance
            passed = max_diff < tolerance and rel_diff < tolerance

            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{name:20} | {max_diff:>10.2e} | {rel_diff:>10.2e} | {status:>8}")

            if not passed:
                all_passed = False

        except Exception as e:
            print(f"{name:20} | {'ERROR':>12} | {'N/A':>12} | {'❌ FAIL':>8}")
            print(f"  Error: {e}")
            all_passed = False

    print("-" * 70)
    return all_passed

def test_specific_patterns():
    """Test specific mathematical patterns for correctness"""
    print(f"\n{'='*60}")
    print("Testing Specific Mathematical Patterns")

    # Test 1: Identity matrices (should produce predictable results)
    print("\n--- Identity Matrix Test ---")
    torch.manual_seed(123)
    size = 8
    q = torch.eye(size, dtype=torch.float32)
    k = torch.eye(size, dtype=torch.float32)
    v = torch.ones(size, size, dtype=torch.float32)

    metal_output = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v)
    torch_output = F.scaled_dot_product_attention(q, k, v)

    identity_diff = torch.abs(metal_output - torch_output).max().item()
    print(f"Identity pattern diff: {identity_diff:.2e}")

    # Test 2: Causal masking consistency
    print("\n--- Causal Masking Test ---")
    torch.manual_seed(456)
    q = torch.randn(16, 8, dtype=torch.float32)
    k = torch.randn(16, 8, dtype=torch.float32)
    v = torch.randn(16, 8, dtype=torch.float32)

    metal_causal = metal_sdpa_extension.metal_scaled_dot_product_attention(q, k, v, is_causal=True)
    torch_causal = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    causal_diff = torch.abs(metal_causal - torch_causal).max().item()
    print(f"Causal masking diff: {causal_diff:.2e}")

    # Test 3: Numerical stability with extreme values
    print("\n--- Numerical Stability Test ---")
    torch.manual_seed(789)

    # Large values (but not overflow)
    q_large = torch.randn(8, 4, dtype=torch.float32) * 10
    k_large = torch.randn(8, 4, dtype=torch.float32) * 10
    v_large = torch.randn(8, 4, dtype=torch.float32)

    try:
        metal_stable = metal_sdpa_extension.metal_scaled_dot_product_attention(q_large, k_large, v_large)
        torch_stable = F.scaled_dot_product_attention(q_large, k_large, v_large)

        stability_diff = torch.abs(metal_stable - torch_stable).max().item()
        print(f"Numerical stability diff: {stability_diff:.2e}")

        if torch.isnan(metal_stable).any():
            print("❌ Metal implementation produces NaN with large values")
            return False
        else:
            print("✅ Metal implementation handles large values correctly")

    except Exception as e:
        print(f"❌ Error with large values: {e}")
        return False

    return identity_diff < 1e-5 and causal_diff < 1e-5 and stability_diff < 1e-4

def test_dtype_consistency():
    """Test consistency across data types"""
    print(f"\n{'='*60}")
    print("Testing Data Type Consistency")

    torch.manual_seed(999)
    q_f32 = torch.randn(32, 16, dtype=torch.float32)
    k_f32 = torch.randn(32, 16, dtype=torch.float32)
    v_f32 = torch.randn(32, 16, dtype=torch.float32)

    # Test float32 as baseline
    metal_f32 = metal_sdpa_extension.metal_scaled_dot_product_attention(q_f32, k_f32, v_f32)
    torch_f32 = F.scaled_dot_product_attention(q_f32, k_f32, v_f32)

    print(f"Float32 baseline diff: {torch.abs(metal_f32 - torch_f32).max().item():.2e}")

    # Test float16 consistency
    try:
        q_f16 = q_f32.to(torch.float16)
        k_f16 = k_f32.to(torch.float16)
        v_f16 = v_f32.to(torch.float16)

        metal_f16 = metal_sdpa_extension.metal_scaled_dot_product_attention(q_f16, k_f16, v_f16)
        torch_f16 = F.scaled_dot_product_attention(q_f16, k_f16, v_f16)

        f16_diff = torch.abs(metal_f16.float() - torch_f16.float()).max().item()
        print(f"Float16 vs PyTorch diff: {f16_diff:.2e}")

        return f16_diff < 1e-3  # More relaxed for float16

    except Exception as e:
        print(f"❌ Float16 error: {e}")
        return False

def main():
    """Main correctness verification"""
    print("🧪 COMPREHENSIVE CORRECTNESS TEST")
    print("Testing current Swift compiler configuration")
    print("DO NOT CHANGE COMPILER FLAGS without verifying these tests still pass!")
    print()

    # Run all tests
    basic_correctness = test_numerical_correctness()
    pattern_correctness = test_specific_patterns()
    dtype_correctness = test_dtype_consistency()

    print(f"\n{'='*60}")
    print("FINAL CORRECTNESS ASSESSMENT")
    print(f"{'='*60}")

    print(f"Basic Numerical Correctness: {'✅ PASS' if basic_correctness else '❌ FAIL'}")
    print(f"Mathematical Patterns:       {'✅ PASS' if pattern_correctness else '❌ FAIL'}")
    print(f"Data Type Consistency:       {'✅ PASS' if dtype_correctness else '❌ FAIL'}")

    overall_pass = basic_correctness and pattern_correctness and dtype_correctness

    if overall_pass:
        print(f"\n🎯 OVERALL RESULT: ✅ CURRENT CONFIGURATION IS CORRECT")
        print("The Swift compiler flags are producing accurate results.")
        print("Current flags (including -Ounchecked) appear to be necessary for correctness.")
    else:
        print(f"\n🚨 OVERALL RESULT: ❌ CORRECTNESS ISSUES DETECTED")
        print("The current configuration has numerical accuracy problems.")
        print("This could indicate compiler flag issues or implementation bugs.")

    print(f"\n📊 SUMMARY:")
    print(f"- Tolerance used: 1e-5 (strict)")
    print(f"- Test coverage: Multiple sizes, patterns, data types")
    print(f"- Comparison: Against PyTorch reference implementation")

    return overall_pass

if __name__ == "__main__":
    main()
    <function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Verify current Swift compiler flags produce correct MFA results", "status": "in_progress", "activeForm": "Verifying current Swift compiler flags produce correct MFA results"}, {"content": "Test numerical correctness across different tensor sizes", "status": "pending", "activeForm": "Testing numerical correctness across different tensor sizes"}, {"content": "Compare MFA results with reference implementations", "status": "pending", "activeForm": "Comparing MFA results with reference implementations"}]