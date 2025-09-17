#!/usr/bin/env python3
"""
BF16 Metal Accumulation Diagnostic Test

This test specifically isolates the accumulation precision issue in Metal kernels
that causes NaN values in complex attention patterns with bf16 inputs.

Key investigation areas:
1. BF16 vs FP32 accumulation precision comparison
2. Overflow/underflow detection in attention computation pipeline
3. Metal kernel accumulation dtype mismatch detection
4. Complex vs simple attention pattern NaN generation isolation
"""

import os
import sys
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
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print("=" * 70)


def print_test_result(passed, message=""):
    """Print test result with clear PASS/FAIL status."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status}: {message}")
    return passed


def create_simple_attention_pattern(
    batch_size=1, num_heads=1, seq_len=32, head_dim=64, dtype=torch.float32
):
    """Create simple, well-conditioned attention tensors that should not cause overflow."""
    # Use small, normalized values to prevent overflow
    scale = 0.05  # Very small scale to avoid accumulation overflow

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype) * scale
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype) * scale
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype) * scale

    # Normalize to unit length for numerical stability
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    k = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)

    return q, k, v


def create_complex_attention_pattern(
    batch_size=2, num_heads=8, seq_len=128, head_dim=64, dtype=torch.float32
):
    """Create complex attention patterns that stress the accumulation pipeline."""
    # Create patterns that will stress accumulation precision
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)

    # Add some challenging patterns for accumulation
    # Pattern 1: Large dynamic range
    q[:, :, : seq_len // 4, :] *= 10.0  # Some elements much larger
    k[:, :, seq_len // 2 :, :] *= 0.1  # Some elements much smaller

    # Pattern 2: Create correlation patterns that will produce large attention scores
    k[:, :, : seq_len // 2, :] = q[:, :, : seq_len // 2, :] * 2.0  # High correlation

    return q, k, v


def analyze_attention_scores(q, k, scale=None):
    """Analyze attention score distribution to detect overflow conditions."""
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)

    # Compute attention scores manually
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    stats = {
        "min": scores.min().item(),
        "max": scores.max().item(),
        "mean": scores.mean().item(),
        "std": scores.std().item(),
        "has_nan": torch.isnan(scores).any().item(),
        "has_inf": torch.isinf(scores).any().item(),
    }

    # Check for potential overflow in softmax
    max_score = scores.max().item()
    stats["softmax_overflow_risk"] = max_score > 80.0  # exp(80) is near float32 limit

    return stats


def test_1_bf16_simple_accumulation():
    """
    Test 1: BF16 simple pattern accumulation vs FP32

    This isolates whether bf16 accumulation works correctly for simple patterns
    that should not stress the precision limits.
    """
    print_test_header("Test 1: BF16 Simple Pattern Accumulation")

    if not METAL_AVAILABLE:
        return print_test_result(False, "Metal extension not available")

    if not hasattr(torch, "bfloat16"):
        return print_test_result(False, "BFloat16 not available in this PyTorch build")

    try:
        # Create simple, well-conditioned test data
        q_fp32, k_fp32, v_fp32 = create_simple_attention_pattern(dtype=torch.float32)

        # Convert to bf16
        q_bf16 = q_fp32.to(torch.bfloat16)
        k_bf16 = k_fp32.to(torch.bfloat16)
        v_bf16 = v_fp32.to(torch.bfloat16)

        print(f"Input tensor shapes: {q_fp32.shape}")
        print(f"Input data range - FP32: [{q_fp32.min():.6f}, {q_fp32.max():.6f}]")
        print(f"Input data range - BF16: [{q_bf16.min():.6f}, {q_bf16.max():.6f}]")

        # Analyze attention scores for overflow risk
        fp32_scores = analyze_attention_scores(q_fp32, k_fp32)
        bf16_scores = analyze_attention_scores(q_bf16.float(), k_bf16.float())

        print(
            f"FP32 attention scores: min={fp32_scores['min']:.3f}, max={fp32_scores['max']:.3f}, "
            f"overflow_risk={fp32_scores['softmax_overflow_risk']}"
        )
        print(
            f"BF16 attention scores: min={bf16_scores['min']:.3f}, max={bf16_scores['max']:.3f}, "
            f"overflow_risk={bf16_scores['softmax_overflow_risk']}"
        )

        # Compute attention
        output_fp32 = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q_fp32, k_fp32, v_fp32
        )
        output_bf16 = metal_sdpa_extension.metal_scaled_dot_product_attention(
            q_bf16, k_bf16, v_bf16
        )

        # Check for NaN/Inf
        fp32_valid = (
            not torch.isnan(output_fp32).any() and not torch.isinf(output_fp32).any()
        )
        bf16_valid = (
            not torch.isnan(output_bf16).any() and not torch.isinf(output_bf16).any()
        )

        print(f"FP32 output valid: {fp32_valid}")
        print(f"BF16 output valid: {bf16_valid}")

        if fp32_valid and bf16_valid:
            # Compare numerical results
            output_bf16_as_fp32 = output_bf16.to(torch.float32)
            abs_diff = torch.abs(output_fp32 - output_bf16_as_fp32)
            rel_diff = abs_diff / (torch.abs(output_fp32) + 1e-8)

            max_abs_diff = abs_diff.max().item()
            max_rel_diff = rel_diff.max().item()

            print(f"Max absolute difference: {max_abs_diff:.6f}")
            print(f"Max relative difference: {max_rel_diff:.4f}")

            # For simple patterns, differences should be small
            diff_reasonable = max_abs_diff < 0.01 and max_rel_diff < 0.1

            passed = fp32_valid and bf16_valid and diff_reasonable
            return print_test_result(
                passed,
                (
                    "Simple BF16 accumulation works correctly"
                    if passed
                    else f"Simple BF16 accumulation failed (abs_diff={max_abs_diff:.6f}, rel_diff={max_rel_diff:.4f})"
                ),
            )
        else:
            return print_test_result(
                False,
                f"Invalid outputs: FP32_valid={fp32_valid}, BF16_valid={bf16_valid}",
            )

    except Exception as e:
        return print_test_result(False, f"Exception in simple accumulation test: {e}")


def test_2_bf16_complex_accumulation():
    """
    Test 2: BF16 complex pattern accumulation

    This test uses complex attention patterns that stress accumulation precision
    to isolate where bf16 accumulation breaks down compared to fp32.
    """
    print_test_header("Test 2: BF16 Complex Pattern Accumulation")

    if not METAL_AVAILABLE:
        return print_test_result(False, "Metal extension not available")

    if not hasattr(torch, "bfloat16"):
        return print_test_result(False, "BFloat16 not available in this PyTorch build")

    try:
        # Create complex, challenging test data
        q_fp32, k_fp32, v_fp32 = create_complex_attention_pattern(dtype=torch.float32)

        # Convert to bf16
        q_bf16 = q_fp32.to(torch.bfloat16)
        k_bf16 = k_fp32.to(torch.bfloat16)
        v_bf16 = v_fp32.to(torch.bfloat16)

        print(f"Input tensor shapes: {q_fp32.shape}")
        print(f"Input data range - FP32: [{q_fp32.min():.6f}, {q_fp32.max():.6f}]")
        print(f"Input data range - BF16: [{q_bf16.min():.6f}, {q_bf16.max():.6f}]")

        # Analyze attention scores for overflow risk
        fp32_scores = analyze_attention_scores(q_fp32, k_fp32)
        bf16_scores = analyze_attention_scores(q_bf16.float(), k_bf16.float())

        print(
            f"FP32 attention scores: min={fp32_scores['min']:.3f}, max={fp32_scores['max']:.3f}, "
            f"overflow_risk={fp32_scores['softmax_overflow_risk']}"
        )
        print(
            f"BF16 attention scores: min={bf16_scores['min']:.3f}, max={bf16_scores['max']:.3f}, "
            f"overflow_risk={bf16_scores['softmax_overflow_risk']}"
        )

        # This is the critical test - complex patterns with bf16
        try:
            output_fp32 = metal_sdpa_extension.metal_scaled_dot_product_attention(
                q_fp32, k_fp32, v_fp32
            )
            fp32_success = True
        except Exception as e:
            print(f"FP32 computation failed: {e}")
            output_fp32 = None
            fp32_success = False

        try:
            output_bf16 = metal_sdpa_extension.metal_scaled_dot_product_attention(
                q_bf16, k_bf16, v_bf16
            )
            bf16_success = True
        except Exception as e:
            print(f"BF16 computation failed: {e}")
            output_bf16 = None
            bf16_success = False

        if not fp32_success or not bf16_success:
            return print_test_result(
                False, f"Computation failed: FP32={fp32_success}, BF16={bf16_success}"
            )

        # Check for NaN/Inf in outputs
        fp32_valid = (
            not torch.isnan(output_fp32).any() and not torch.isinf(output_fp32).any()
        )
        bf16_valid = (
            not torch.isnan(output_bf16).any() and not torch.isinf(output_bf16).any()
        )

        print(f"FP32 output valid: {fp32_valid}")
        print(f"BF16 output valid: {bf16_valid}")

        if fp32_valid:
            print(
                f"FP32 output range: [{output_fp32.min():.6f}, {output_fp32.max():.6f}]"
            )
        if bf16_valid:
            print(
                f"BF16 output range: [{output_bf16.min():.6f}, {output_bf16.max():.6f}]"
            )

        # The key insight: does complex attention cause NaN in bf16 but not fp32?
        bf16_fails_complex = fp32_valid and not bf16_valid

        if bf16_fails_complex:
            print(
                "🚨 CRITICAL FINDING: BF16 fails on complex patterns while FP32 succeeds"
            )
            print(
                "    This indicates bf16 accumulation precision issues in Metal kernels"
            )
            return print_test_result(
                False, "BF16 accumulation fails on complex patterns (IDENTIFIED ISSUE)"
            )

        if fp32_valid and bf16_valid:
            # Both succeeded - compare quality
            output_bf16_as_fp32 = output_bf16.to(torch.float32)
            abs_diff = torch.abs(output_fp32 - output_bf16_as_fp32)
            max_abs_diff = abs_diff.max().item()

            print(f"Max absolute difference: {max_abs_diff:.6f}")

            # For complex patterns, larger differences are expected but should be finite
            diff_reasonable = max_abs_diff < 1.0  # Much more generous threshold

            passed = diff_reasonable
            return print_test_result(
                passed,
                (
                    "Complex BF16 accumulation works within tolerance"
                    if passed
                    else f"Complex BF16 accumulation shows excessive error (abs_diff={max_abs_diff:.6f})"
                ),
            )
        else:
            return print_test_result(
                False,
                f"Both computations failed: FP32_valid={fp32_valid}, BF16_valid={bf16_valid}",
            )

    except Exception as e:
        return print_test_result(False, f"Exception in complex accumulation test: {e}")


def test_3_accumulation_dtype_detection():
    """
    Test 3: Accumulation dtype mismatch detection

    This test attempts to detect if the Metal kernels are using mismatched
    accumulation types (e.g., bf16 inputs with fp32 accumulation vs bf16 accumulation).
    """
    print_test_header("Test 3: Accumulation Dtype Mismatch Detection")

    if not METAL_AVAILABLE:
        return print_test_result(False, "Metal extension not available")

    if not hasattr(torch, "bfloat16"):
        return print_test_result(False, "BFloat16 not available in this PyTorch build")

    try:
        # Create test patterns that would show different behavior with different accumulation types
        patterns = []

        # Pattern 1: Values that lose precision in bf16 accumulation
        q1 = torch.ones(1, 1, 32, 64, dtype=torch.bfloat16) * 0.001  # Very small values
        k1 = torch.ones(1, 1, 32, 64, dtype=torch.bfloat16) * 1000.0  # Large values
        v1 = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16) * 0.1
        patterns.append(("small*large", q1, k1, v1))

        # Pattern 2: Many small accumulations (would underflow in bf16 but not fp32)
        q2 = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16) * 0.01
        k2 = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16) * 0.01
        v2 = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16) * 0.01
        patterns.append(("many_small", q2, k2, v2))

        # Pattern 3: High precision accumulation test
        q3 = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
        k3 = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
        v3 = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
        # Add tiny perturbations that would be lost in bf16 accumulation
        v3 += torch.randn_like(v3) * 1e-6
        patterns.append(("high_precision", q3, k3, v3))

        all_patterns_valid = True
        accumulation_issues = []

        for pattern_name, q, k, v in patterns:
            print(f"\nTesting pattern: {pattern_name}")
            print(
                f"  Input ranges: Q[{q.min():.6f}, {q.max():.6f}], "
                f"K[{k.min():.6f}, {k.max():.6f}], V[{v.min():.6f}, {v.max():.6f}]"
            )

            try:
                output = metal_sdpa_extension.metal_scaled_dot_product_attention(
                    q, k, v
                )
                valid = not torch.isnan(output).any() and not torch.isinf(output).any()

                print(f"  Output valid: {valid}")
                if valid:
                    print(f"  Output range: [{output.min():.6f}, {output.max():.6f}]")
                else:
                    print(
                        f"  Output contains: NaN={torch.isnan(output).any()}, Inf={torch.isinf(output).any()}"
                    )
                    accumulation_issues.append(pattern_name)

                all_patterns_valid = all_patterns_valid and valid

            except Exception as e:
                print(f"  Exception: {e}")
                accumulation_issues.append(f"{pattern_name}_exception")
                all_patterns_valid = False

        if accumulation_issues:
            print(
                f"\n🚨 ACCUMULATION ISSUES DETECTED in patterns: {accumulation_issues}"
            )
            print(
                "    This suggests bf16 accumulation precision problems in Metal kernels"
            )

        passed = all_patterns_valid
        return print_test_result(
            passed,
            (
                "No accumulation dtype issues detected"
                if passed
                else f"Accumulation dtype issues found in: {accumulation_issues}"
            ),
        )

    except Exception as e:
        return print_test_result(
            False, f"Exception in accumulation dtype detection: {e}"
        )


def test_4_overflow_underflow_detection():
    """
    Test 4: Overflow/underflow detection in bf16 Metal pipeline

    This test specifically targets conditions that would cause overflow or underflow
    in bf16 accumulation but not in fp32 accumulation.
    """
    print_test_header("Test 4: Overflow/Underflow Detection in BF16 Pipeline")

    if not METAL_AVAILABLE:
        return print_test_result(False, "Metal extension not available")

    if not hasattr(torch, "bfloat16"):
        return print_test_result(False, "BFloat16 not available in this PyTorch build")

    try:
        test_cases = []

        # Case 1: Potential overflow in attention scores
        q_big = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16) * 10.0
        k_big = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16) * 10.0
        v_big = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16)
        test_cases.append(("overflow_scores", q_big, k_big, v_big))

        # Case 2: Potential underflow in small values
        q_small = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16) * 1e-4
        k_small = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16) * 1e-4
        v_small = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16) * 1e-4
        test_cases.append(("underflow_values", q_small, k_small, v_small))

        # Case 3: Mixed scale (common in real models)
        q_mixed = torch.randn(1, 8, 128, 64, dtype=torch.bfloat16)
        k_mixed = torch.randn(1, 8, 128, 64, dtype=torch.bfloat16)
        v_mixed = torch.randn(1, 8, 128, 64, dtype=torch.bfloat16)
        # Make some heads have very different scales
        q_mixed[:, :4, :, :] *= 0.001  # Very small
        k_mixed[:, 4:, :, :] *= 100.0  # Very large
        test_cases.append(("mixed_scales", q_mixed, k_mixed, v_mixed))

        overflow_cases = []
        underflow_cases = []
        successful_cases = []

        for case_name, q, k, v in test_cases:
            print(f"\nTesting case: {case_name}")

            # Analyze the attention score ranges before Metal computation
            with torch.no_grad():
                scale = 1.0 / (q.shape[-1] ** 0.5)
                scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
                score_min, score_max = scores.min().item(), scores.max().item()

                print(f"  Attention score range: [{score_min:.3f}, {score_max:.3f}]")

                # Predict overflow/underflow
                overflow_risk = score_max > 80.0  # exp(80) approaches float32 limit
                underflow_risk = score_min < -80.0 or score_max < -10.0

                print(f"  Predicted overflow risk: {overflow_risk}")
                print(f"  Predicted underflow risk: {underflow_risk}")

            try:
                output = metal_sdpa_extension.metal_scaled_dot_product_attention(
                    q, k, v
                )

                has_nan = torch.isnan(output).any().item()
                has_inf = torch.isinf(output).any().item()
                is_valid = not has_nan and not has_inf

                print(f"  Output valid: {is_valid} (NaN: {has_nan}, Inf: {has_inf})")

                if is_valid:
                    successful_cases.append(case_name)
                    print(f"  Output range: [{output.min():.6f}, {output.max():.6f}]")
                else:
                    if has_inf or score_max > 50.0:
                        overflow_cases.append(case_name)
                        print(f"  → Classified as OVERFLOW case")
                    elif score_max < -5.0 or torch.all(output == 0):
                        underflow_cases.append(case_name)
                        print(f"  → Classified as UNDERFLOW case")
                    else:
                        print(f"  → Unclassified failure")

            except Exception as e:
                print(f"  Exception: {e}")
                overflow_cases.append(f"{case_name}_exception")

        print(f"\n📊 OVERFLOW/UNDERFLOW ANALYSIS:")
        print(f"  Successful cases: {successful_cases}")
        print(f"  Overflow cases: {overflow_cases}")
        print(f"  Underflow cases: {underflow_cases}")

        # Determine the primary issue
        has_overflow_issues = len(overflow_cases) > 0
        has_underflow_issues = len(underflow_cases) > 0

        if has_overflow_issues:
            print(
                "🚨 OVERFLOW DETECTED: BF16 accumulation overflows on large attention scores"
            )
        if has_underflow_issues:
            print("🚨 UNDERFLOW DETECTED: BF16 accumulation underflows on small values")

        passed = len(successful_cases) == len(test_cases)
        return print_test_result(
            passed,
            (
                "No overflow/underflow issues detected"
                if passed
                else f"Issues found - Overflow: {overflow_cases}, Underflow: {underflow_cases}"
            ),
        )

    except Exception as e:
        return print_test_result(
            False, f"Exception in overflow/underflow detection: {e}"
        )


def main():
    """Run all bf16 Metal accumulation diagnostic tests."""
    print("BF16 Metal Accumulation Diagnostic Test Suite")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"BFloat16 available: {hasattr(torch, 'bfloat16')}")
    print(f"Metal extension available: {METAL_AVAILABLE}")

    if not METAL_AVAILABLE:
        print("\n⚠️  Metal extension not available - tests will be skipped")
        return

    print(
        "\nThis test suite isolates bf16 accumulation precision issues in Metal kernels"
    )
    print(
        "that cause NaN values in complex attention patterns while simple patterns work."
    )

    # Run all diagnostic tests
    test_results = []

    test_results.append(test_1_bf16_simple_accumulation())
    test_results.append(test_2_bf16_complex_accumulation())
    test_results.append(test_3_accumulation_dtype_detection())
    test_results.append(test_4_overflow_underflow_detection())

    # Summary and analysis
    print(f"\n{'='*70}")
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    passed_count = sum(test_results)
    total_count = len(test_results)

    print(f"Tests passed: {passed_count}/{total_count}")

    if passed_count == total_count:
        print("✓ NO BF16 ACCUMULATION ISSUES DETECTED")
        print("  The Metal pipeline appears to handle bf16 accumulation correctly.")
    else:
        print("🚨 BF16 ACCUMULATION ISSUES DETECTED")
        print("\nDIAGNOSIS:")

        if not test_results[0] and not test_results[1]:
            print(
                "• Both simple and complex patterns fail → Fundamental bf16 support issue"
            )
        elif test_results[0] and not test_results[1]:
            print(
                "• Simple patterns work, complex patterns fail → Accumulation precision issue"
            )
            print("• LIKELY CAUSE: Metal kernels use bf16 accumulation instead of fp32")

        if not test_results[2]:
            print(
                "• Accumulation dtype mismatch detected → Metal kernel configuration issue"
            )

        if not test_results[3]:
            print(
                "• Overflow/underflow detected → BF16 range limitations in accumulation"
            )


if __name__ == "__main__":
    main()
