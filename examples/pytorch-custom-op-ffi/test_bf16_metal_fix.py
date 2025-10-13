#!/usr/bin/env python3
"""
BF16 Metal Fix Validation Test

This test validates the fix for BF16 NaN issues in the Metal accumulation pipeline.
The fix forces FP32 accumulation for BF16 inputs to prevent overflow/underflow
in complex attention patterns.

Test Categories:
1. Simple patterns that should work efficiently (baseline validation)
2. Complex patterns that previously produced NaNs (critical fix validation)
3. Performance comparison with fp32 baseline (performance impact analysis)
4. Edge cases and stress testing (robustness validation)
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add the python package to path
sys.path.insert(0, str(Path(__file__).parent / "python"))

try:
    import metal_sdpa_extension

    METAL_AVAILABLE = True
    print("✓ metal_sdpa_extension imported successfully")
except ImportError as e:
    METAL_AVAILABLE = False
    print(f"✗ metal_sdpa_extension import failed: {e}")


def print_test_header(test_name):
    """Print formatted test header."""
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print("=" * 80)


def print_test_result(passed, message=""):
    """Print test result with clear PASS/FAIL status."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status}: {message}")
    return passed


def create_simple_attention_pattern(
    batch_size=1, num_heads=1, seq_len=32, head_dim=64, dtype=torch.float32
):
    """Create simple, well-conditioned attention tensors for baseline validation."""
    torch.manual_seed(42)  # Reproducible results

    # Create normalized attention tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype) * 0.1
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype) * 0.1
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype) * 0.1

    # Normalize for numerical stability
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)

    return q, k, v


def create_complex_attention_pattern(
    batch_size=2, num_heads=8, seq_len=128, head_dim=64, dtype=torch.float32
):
    """Create complex attention patterns that previously caused NaN issues."""
    torch.manual_seed(123)  # Reproducible problematic patterns

    # Create base patterns
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)

    # Pattern 1: Large dynamic range that stresses accumulation
    q[:, :4, : seq_len // 4, :] *= 20.0  # Very large values
    k[:, 4:, seq_len // 2 :, :] *= 0.01  # Very small values

    # Pattern 2: High correlation patterns that produce large attention scores
    k[:, :, : seq_len // 3, :] = q[:, :, : seq_len // 3, :] * 3.0  # Strong correlation

    # Pattern 3: Outlier values that test overflow protection
    q[:, -1:, -1:, :10] *= 50.0  # Extreme outliers in last head

    return q, k, v


def create_flux_style_pattern(
    batch_size=1, num_heads=16, seq_len=256, head_dim=64, dtype=torch.float32
):
    """Create FLUX-style attention patterns that are known to be problematic."""
    torch.manual_seed(456)

    # FLUX models use specific attention patterns that caused the original NaN issues
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype) * 0.5
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype) * 0.5
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype) * 0.3

    # FLUX-specific patterns
    # 1. Cross-attention style correlations
    for head in range(0, num_heads, 4):
        k[:, head, : seq_len // 2, :] = q[:, head, : seq_len // 2, :] * 2.5

    # 2. Position-dependent scaling (common in diffusion models)
    pos_scale = torch.linspace(0.1, 2.0, seq_len).view(1, 1, seq_len, 1)
    q = q * pos_scale

    # 3. Feature dimension correlations
    v[:, :, :, : head_dim // 2] = v[:, :, :, head_dim // 2 :] * 1.5

    return q, k, v


def compute_pytorch_reference(q, k, v, scale=None):
    """Compute reference attention using PyTorch's implementation."""
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)

    # Manual computation for numerical analysis
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)

    return output


def analyze_attention_quality(output_metal, output_reference, test_name):
    """Analyze the quality of Metal attention output compared to reference."""
    if output_metal.dtype != output_reference.dtype:
        # Convert to common dtype for comparison
        output_metal = output_metal.to(output_reference.dtype)

    # Compute error metrics
    abs_diff = torch.abs(output_metal - output_reference)
    rel_diff = abs_diff / (torch.abs(output_reference) + 1e-8)

    metrics = {
        "max_abs_error": abs_diff.max().item(),
        "mean_abs_error": abs_diff.mean().item(),
        "max_rel_error": rel_diff.max().item(),
        "mean_rel_error": rel_diff.mean().item(),
        "l2_error": torch.norm(abs_diff).item(),
        "has_nan": torch.isnan(output_metal).any().item(),
        "has_inf": torch.isinf(output_metal).any().item(),
    }

    print(f"{test_name} Quality Metrics:")
    print(f"  Max absolute error: {metrics['max_abs_error']:.6f}")
    print(f"  Mean absolute error: {metrics['mean_abs_error']:.6f}")
    print(f"  Max relative error: {metrics['max_rel_error']:.4f}")
    print(f"  Mean relative error: {metrics['mean_rel_error']:.4f}")
    print(f"  L2 error: {metrics['l2_error']:.6f}")
    print(f"  Has NaN: {metrics['has_nan']}")
    print(f"  Has Inf: {metrics['has_inf']}")

    return metrics


def test_1_simple_pattern_baseline():
    """Test 1: Simple patterns should work efficiently and accurately."""
    print_test_header("Test 1: Simple Pattern Baseline (Post-Fix)")

    if not METAL_AVAILABLE:
        return print_test_result(False, "Metal extension not available")

    if not hasattr(torch, "bfloat16"):
        return print_test_result(False, "BFloat16 not available")

    try:
        # Test with simple pattern
        q_fp32, k_fp32, v_fp32 = create_simple_attention_pattern(dtype=torch.float32)
        q_bf16, k_bf16, v_bf16 = (
            q_fp32.to(torch.bfloat16),
            k_fp32.to(torch.bfloat16),
            v_fp32.to(torch.bfloat16),
        )

        print(f"Input shapes: {q_fp32.shape}")

        # Compute reference
        ref_output = compute_pytorch_reference(q_fp32, k_fp32, v_fp32)

        # Compute Metal outputs
        metal_fp32 = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q_fp32, k_fp32, v_fp32
        )
        metal_bf16 = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q_bf16, k_bf16, v_bf16
        )

        # Analyze quality
        fp32_metrics = analyze_attention_quality(metal_fp32, ref_output, "Metal FP32")
        bf16_metrics = analyze_attention_quality(
            metal_bf16.float(), ref_output, "Metal BF16"
        )

        # Simple patterns should have high accuracy and no NaN/Inf
        fp32_good = (
            not fp32_metrics["has_nan"]
            and not fp32_metrics["has_inf"]
            and fp32_metrics["max_abs_error"] < 0.001
        )
        bf16_good = (
            not bf16_metrics["has_nan"]
            and not bf16_metrics["has_inf"]
            and bf16_metrics["max_abs_error"] < 0.01
        )

        passed = fp32_good and bf16_good
        return print_test_result(
            passed,
            (
                "Simple patterns work correctly"
                if passed
                else f"Simple pattern issues: FP32_good={fp32_good}, BF16_good={bf16_good}"
            ),
        )

    except Exception as e:
        return print_test_result(False, f"Exception in simple pattern test: {e}")


def test_2_complex_pattern_fix_validation():
    """Test 2: Complex patterns that previously caused NaNs should now work."""
    print_test_header("Test 2: Complex Pattern Fix Validation")

    if not METAL_AVAILABLE:
        return print_test_result(False, "Metal extension not available")

    if not hasattr(torch, "bfloat16"):
        return print_test_result(False, "BFloat16 not available")

    try:
        # Test with complex pattern that previously caused NaNs
        q_fp32, k_fp32, v_fp32 = create_complex_attention_pattern(dtype=torch.float32)
        q_bf16, k_bf16, v_bf16 = (
            q_fp32.to(torch.bfloat16),
            k_fp32.to(torch.bfloat16),
            v_fp32.to(torch.bfloat16),
        )

        print(f"Input shapes: {q_fp32.shape}")
        print(f"Input ranges - Q: [{q_fp32.min():.3f}, {q_fp32.max():.3f}]")
        print(f"Input ranges - K: [{k_fp32.min():.3f}, {k_fp32.max():.3f}]")
        print(f"Input ranges - V: [{v_fp32.min():.3f}, {v_fp32.max():.3f}]")

        # Compute reference
        ref_output = compute_pytorch_reference(q_fp32, k_fp32, v_fp32)
        print(
            f"Reference output range: [{ref_output.min():.3f}, {ref_output.max():.3f}]"
        )

        # Test Metal outputs
        try:
            metal_fp32 = metal_sdpa_extension.metal_scaled_dot_product_attention(
                q_fp32, k_fp32, v_fp32
            )
            fp32_success = True
        except Exception as e:
            print(f"Metal FP32 failed: {e}")
            metal_fp32 = None
            fp32_success = False

        try:
            metal_bf16 = metal_sdpa_extension.metal_scaled_dot_product_attention(
                q_bf16, k_bf16, v_bf16
            )
            bf16_success = True
        except Exception as e:
            print(f"Metal BF16 failed: {e}")
            metal_bf16 = None
            bf16_success = False

        if not fp32_success or not bf16_success:
            return print_test_result(
                False,
                f"Metal computation failed: FP32={fp32_success}, BF16={bf16_success}",
            )

        # Analyze quality
        fp32_metrics = analyze_attention_quality(
            metal_fp32, ref_output, "Metal FP32 Complex"
        )
        bf16_metrics = analyze_attention_quality(
            metal_bf16.float(), ref_output, "Metal BF16 Complex"
        )

        # The critical test: BF16 should not produce NaN/Inf even on complex patterns
        bf16_fixed = not bf16_metrics["has_nan"] and not bf16_metrics["has_inf"]
        fp32_works = not fp32_metrics["has_nan"] and not fp32_metrics["has_inf"]

        if bf16_fixed and fp32_works:
            # Both work - check if quality is reasonable
            quality_acceptable = (
                bf16_metrics["max_abs_error"] < 1.0
            )  # More generous for complex patterns
            passed = quality_acceptable
            return print_test_result(
                passed,
                (
                    f"Complex pattern fix successful! Max error: {bf16_metrics['max_abs_error']:.6f}"
                    if passed
                    else f"Fix works but quality poor: {bf16_metrics['max_abs_error']:.6f}"
                ),
            )
        elif fp32_works and not bf16_fixed:
            return print_test_result(
                False, "❌ FIX FAILED: BF16 still produces NaN/Inf on complex patterns"
            )
        else:
            return print_test_result(
                False, f"Both failed: FP32_works={fp32_works}, BF16_fixed={bf16_fixed}"
            )

    except Exception as e:
        return print_test_result(False, f"Exception in complex pattern test: {e}")


def test_3_flux_pattern_validation():
    """Test 3: FLUX-style patterns that were specifically problematic."""
    print_test_header("Test 3: FLUX Pattern Validation")

    if not METAL_AVAILABLE:
        return print_test_result(False, "Metal extension not available")

    if not hasattr(torch, "bfloat16"):
        return print_test_result(False, "BFloat16 not available")

    try:
        # Test with FLUX-style pattern
        q_fp32, k_fp32, v_fp32 = create_flux_style_pattern(dtype=torch.float32)
        q_bf16, k_bf16, v_bf16 = (
            q_fp32.to(torch.bfloat16),
            k_fp32.to(torch.bfloat16),
            v_fp32.to(torch.bfloat16),
        )

        print(f"FLUX pattern shapes: {q_fp32.shape}")

        # Compute reference
        ref_output = compute_pytorch_reference(q_fp32, k_fp32, v_fp32)

        # Test Metal BF16 output
        metal_bf16 = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q_bf16, k_bf16, v_bf16
        )

        # Analyze quality
        bf16_metrics = analyze_attention_quality(
            metal_bf16.float(), ref_output, "Metal BF16 FLUX"
        )

        # FLUX patterns should work without NaN/Inf
        flux_fixed = not bf16_metrics["has_nan"] and not bf16_metrics["has_inf"]
        quality_reasonable = (
            bf16_metrics["max_abs_error"] < 2.0
        )  # FLUX patterns can be more challenging

        passed = flux_fixed and quality_reasonable
        return print_test_result(
            passed,
            (
                f"FLUX patterns work correctly! Max error: {bf16_metrics['max_abs_error']:.6f}"
                if passed
                else f"FLUX issues: fixed={flux_fixed}, quality={quality_reasonable}"
            ),
        )

    except Exception as e:
        return print_test_result(False, f"Exception in FLUX pattern test: {e}")


def test_4_performance_impact_analysis():
    """Test 4: Analyze performance impact of the FP32 accumulation fix."""
    print_test_header("Test 4: Performance Impact Analysis")

    if not METAL_AVAILABLE:
        return print_test_result(False, "Metal extension not available")

    if not hasattr(torch, "bfloat16"):
        return print_test_result(False, "BFloat16 not available")

    try:
        # Test multiple sizes to understand performance impact
        test_configs = [
            (1, 8, 128, 64),  # Small
            (2, 16, 256, 64),  # Medium
            (1, 24, 512, 80),  # Large
        ]

        performance_results = []

        for batch, heads, seq_len, head_dim in test_configs:
            print(
                f"\nTesting config: batch={batch}, heads={heads}, seq_len={seq_len}, head_dim={head_dim}"
            )

            # Create test data
            q_fp32, k_fp32, v_fp32 = create_simple_attention_pattern(
                batch, heads, seq_len, head_dim, torch.float32
            )
            q_bf16, k_bf16, v_bf16 = (
                q_fp32.to(torch.bfloat16),
                k_fp32.to(torch.bfloat16),
                v_fp32.to(torch.bfloat16),
            )

            # Warm up
            for _ in range(3):
                _ = metal_sdpa_extension.metal_scaled_dot_product_attention(
                    q_fp32, k_fp32, v_fp32
                )
                _ = metal_sdpa_extension.metal_scaled_dot_product_attention(
                    q_bf16, k_bf16, v_bf16
                )

            # Benchmark FP32
            torch.mps.synchronize() if torch.backends.mps.is_available() else None
            start_time = time.time()
            for _ in range(10):
                _ = metal_sdpa_extension.metal_scaled_dot_product_attention(
                    q_fp32, k_fp32, v_fp32
                )
            torch.mps.synchronize() if torch.backends.mps.is_available() else None
            fp32_time = (time.time() - start_time) / 10

            # Benchmark BF16 (now with FP32 accumulation)
            torch.mps.synchronize() if torch.backends.mps.is_available() else None
            start_time = time.time()
            for _ in range(10):
                _ = metal_sdpa_extension.metal_scaled_dot_product_attention(
                    q_bf16, k_bf16, v_bf16
                )
            torch.mps.synchronize() if torch.backends.mps.is_available() else None
            bf16_time = (time.time() - start_time) / 10

            speedup = fp32_time / bf16_time
            overhead = (bf16_time - fp32_time) / fp32_time * 100

            print(f"  FP32 time: {fp32_time*1000:.2f}ms")
            print(f"  BF16 time: {bf16_time*1000:.2f}ms")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Overhead: {overhead:+.1f}%")

            performance_results.append(
                {
                    "config": (batch, heads, seq_len, head_dim),
                    "fp32_time": fp32_time,
                    "bf16_time": bf16_time,
                    "speedup": speedup,
                    "overhead": overhead,
                }
            )

        # Analyze overall performance impact
        avg_speedup = np.mean([r["speedup"] for r in performance_results])
        avg_overhead = np.mean([r["overhead"] for r in performance_results])

        print(f"\n📊 PERFORMANCE SUMMARY:")
        print(f"  Average speedup: {avg_speedup:.2f}x")
        print(f"  Average overhead: {avg_overhead:+.1f}%")

        # The fix should still provide reasonable performance
        performance_acceptable = (
            avg_overhead < 50.0
        )  # Less than 50% overhead is acceptable

        passed = performance_acceptable
        return print_test_result(
            passed,
            (
                f"Performance impact acceptable: {avg_overhead:+.1f}% overhead"
                if passed
                else f"Performance impact too high: {avg_overhead:+.1f}% overhead"
            ),
        )

    except Exception as e:
        return print_test_result(False, f"Exception in performance test: {e}")


def test_5_edge_case_robustness():
    """Test 5: Edge cases and stress testing for robustness."""
    print_test_header("Test 5: Edge Case Robustness")

    if not METAL_AVAILABLE:
        return print_test_result(False, "Metal extension not available")

    if not hasattr(torch, "bfloat16"):
        return print_test_result(False, "BFloat16 not available")

    try:
        edge_cases = []

        # Case 1: Very small tensors
        q1 = torch.randn(1, 1, 2, 32, dtype=torch.bfloat16) * 0.1
        k1 = torch.randn(1, 1, 2, 32, dtype=torch.bfloat16) * 0.1
        v1 = torch.randn(1, 1, 2, 32, dtype=torch.bfloat16) * 0.1
        edge_cases.append(("tiny_tensors", q1, k1, v1))

        # Case 2: Single head, large sequence
        q2 = torch.randn(1, 1, 512, 64, dtype=torch.bfloat16) * 0.1
        k2 = torch.randn(1, 1, 512, 64, dtype=torch.bfloat16) * 0.1
        v2 = torch.randn(1, 1, 512, 64, dtype=torch.bfloat16) * 0.1
        edge_cases.append(("large_sequence", q2, k2, v2))

        # Case 3: Many heads, small sequence
        q3 = torch.randn(1, 32, 16, 64, dtype=torch.bfloat16) * 0.1
        k3 = torch.randn(1, 32, 16, 64, dtype=torch.bfloat16) * 0.1
        v3 = torch.randn(1, 32, 16, 64, dtype=torch.bfloat16) * 0.1
        edge_cases.append(("many_heads", q3, k3, v3))

        # Case 4: Zero values
        q4 = torch.zeros(1, 4, 32, 64, dtype=torch.bfloat16)
        k4 = torch.randn(1, 4, 32, 64, dtype=torch.bfloat16) * 0.01
        v4 = torch.randn(1, 4, 32, 64, dtype=torch.bfloat16) * 0.01
        edge_cases.append(("zero_query", q4, k4, v4))

        # Case 5: Near-zero values (underflow test)
        q5 = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16) * 1e-6
        k5 = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16) * 1e-6
        v5 = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16) * 1e-6
        edge_cases.append(("near_zero", q5, k5, v5))

        all_passed = True
        failed_cases = []

        for case_name, q, k, v in edge_cases:
            print(f"\nTesting edge case: {case_name}")
            print(f"  Shapes: Q{list(q.shape)}, K{list(k.shape)}, V{list(v.shape)}")

            try:
                output = metal_sdpa_extension.metal_scaled_dot_product_attention(
                    q, k, v
                )

                has_nan = torch.isnan(output).any().item()
                has_inf = torch.isinf(output).any().item()
                is_finite = torch.isfinite(output).all().item()

                print(
                    f"  Output valid: NaN={has_nan}, Inf={has_inf}, Finite={is_finite}"
                )

                if has_nan or has_inf or not is_finite:
                    all_passed = False
                    failed_cases.append(case_name)
                    print(f"  ❌ FAILED: Invalid output detected")
                else:
                    print(
                        f"  ✓ PASSED: Output range [{output.min():.6f}, {output.max():.6f}]"
                    )

            except Exception as e:
                print(f"  ❌ EXCEPTION: {e}")
                all_passed = False
                failed_cases.append(f"{case_name}_exception")

        passed = all_passed
        return print_test_result(
            passed,
            (
                "All edge cases handled correctly"
                if passed
                else f"Edge case failures: {failed_cases}"
            ),
        )

    except Exception as e:
        return print_test_result(False, f"Exception in edge case test: {e}")


def main():
    """Run the comprehensive BF16 Metal fix validation test suite."""
    print("BF16 Metal Fix Validation Test Suite")
    print("=" * 80)
    print(
        "This test suite validates the fix for BF16 NaN issues in Metal accumulation."
    )
    print(
        "The fix forces FP32 accumulation for BF16 inputs to prevent overflow/underflow."
    )
    print("=" * 80)

    print(f"PyTorch version: {torch.__version__}")
    print(f"BFloat16 available: {hasattr(torch, 'bfloat16')}")
    print(f"Metal extension available: {METAL_AVAILABLE}")

    if not METAL_AVAILABLE:
        print("\n⚠️  Metal extension not available - tests will be skipped")
        return

    # Run all validation tests
    test_results = []

    test_results.append(test_1_simple_pattern_baseline())
    test_results.append(test_2_complex_pattern_fix_validation())
    test_results.append(test_3_flux_pattern_validation())
    test_results.append(test_4_performance_impact_analysis())
    test_results.append(test_5_edge_case_robustness())

    # Final summary and assessment
    print(f"\n{'='*80}")
    print("BF16 METAL FIX VALIDATION SUMMARY")
    print("=" * 80)

    passed_count = sum(test_results)
    total_count = len(test_results)

    print(f"Tests passed: {passed_count}/{total_count}")

    if passed_count == total_count:
        print("🎉 BF16 METAL FIX VALIDATION SUCCESSFUL!")
        print("✓ All test cases pass - the fix resolves BF16 NaN issues")
        print("✓ Simple patterns maintain high accuracy")
        print("✓ Complex patterns no longer produce NaN/Inf")
        print("✓ FLUX-style patterns work correctly")
        print("✓ Performance impact is acceptable")
        print("✓ Edge cases are handled robustly")
    else:
        print("❌ BF16 METAL FIX VALIDATION FAILED")
        print("\nFAILED TESTS:")
        test_names = [
            "Simple Baseline",
            "Complex Fix",
            "FLUX Validation",
            "Performance",
            "Edge Cases",
        ]
        for i, (name, result) in enumerate(zip(test_names, test_results)):
            if not result:
                print(f"  ❌ {name}")

        print(f"\nRECOMMENDATIONS:")
        if not test_results[0]:
            print(
                "• Check if the precision fix was properly applied to AttentionDescriptor+Precisions.swift"
            )
        if not test_results[1]:
            print("• Verify FP32 accumulation is enforced for BF16 register precision")
        if not test_results[2]:
            print("• Test with FLUX-specific precision configurations")
        if not test_results[3]:
            print("• Optimize the fix to reduce performance overhead")
        if not test_results[4]:
            print("• Add additional robustness checks for edge cases")


if __name__ == "__main__":
    main()
