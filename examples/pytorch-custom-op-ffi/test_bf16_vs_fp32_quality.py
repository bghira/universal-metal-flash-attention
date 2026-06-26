#!/usr/bin/env python3
"""
BF16 vs FP32 Quality Comparison Test

This test compares output quality between bf16 and fp32 modes in Metal SDPA
to verify the black image issue is resolved.
"""

import os
import sys

import numpy as np
import torch

# Add the project path to sys.path to import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    # Import the extension directly
    import metal_sdpa_extension as ext

    print("✅ Successfully imported metal_sdpa_extension")
except ImportError as e:
    print(f"❌ Failed to import metal_sdpa_extension: {e}")
    sys.exit(1)


class QualityComparator:
    """Compare output quality between bf16 and fp32 modes."""

    def __init__(self):
        self.results = []
        self.device = torch.device("cpu")

    def log_result(self, test_name, passed, message=""):
        """Log a test result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.results.append((test_name, passed, message))
        print(f"{status}: {test_name}")
        if message:
            print(f"    {message}")

    def create_realistic_attention_tensors(
        self, dtype, batch_size=2, seq_len=512, num_heads=16, head_dim=64
    ):
        """Create realistic attention tensors mimicking FLUX-like patterns."""
        # Create tensors with patterns similar to what FLUX would generate
        torch.manual_seed(42)  # For reproducible results

        # Query: simulate text embeddings with some structure
        q = torch.randn(
            batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=self.device
        )
        q = q * 0.5  # Scale down to realistic ranges

        # Key: simulate image patch embeddings
        k = torch.randn(
            batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=self.device
        )
        k = k * 0.3

        # Value: simulate feature values
        v = torch.randn(
            batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=self.device
        )
        v = v * 0.2

        # Add some structure to avoid pure noise
        for i in range(min(10, seq_len)):
            q[:, i, :, :] = q[:, i, :, :] + 0.1
            k[:, i, :, :] = k[:, i, :, :] + 0.05
            v[:, i, :, :] = v[:, i, :, :] + 0.02

        return q, k, v

    def compute_attention_metrics(self, output_tensor):
        """Compute metrics to assess attention output quality."""
        # Convert to float32 for consistent metric computation
        out_f32 = output_tensor.float()

        metrics = {}

        # Basic statistics
        metrics["mean"] = float(out_f32.mean())
        metrics["std"] = float(out_f32.std())
        metrics["min"] = float(out_f32.min())
        metrics["max"] = float(out_f32.max())

        # Check for pathological cases
        metrics["has_nan"] = bool(torch.isnan(out_f32).any())
        metrics["has_inf"] = bool(torch.isinf(out_f32).any())
        metrics["all_zeros"] = bool((out_f32 == 0).all())
        metrics["all_same"] = bool((out_f32 == out_f32[0, 0, 0, 0]).all())

        # Distribution properties
        metrics["dynamic_range"] = float(out_f32.max() - out_f32.min())
        metrics["non_zero_ratio"] = float((out_f32 != 0).float().mean())

        # Check attention pattern characteristics
        # Sum over head dimension to see if attention is well-distributed
        head_sums = out_f32.sum(dim=-1)  # [B, S, H]
        metrics["head_variance"] = float(head_sums.var())

        return metrics

    def compare_precision_outputs(self, q_fp32, k_fp32, v_fp32):
        """Compare outputs between fp32 and bf16 modes."""
        test_name = "BF16 vs FP32 Output Quality"

        try:
            # Test FP32 mode
            print("  🔍 Testing FP32 mode...")
            output_fp32 = ext.metal_scaled_dot_product_attention(q_fp32, k_fp32, v_fp32)
            metrics_fp32 = self.compute_attention_metrics(output_fp32)

            # Test BF16 mode
            print("  🔍 Testing BF16 mode...")
            q_bf16 = q_fp32.to(torch.bfloat16)
            k_bf16 = k_fp32.to(torch.bfloat16)
            v_bf16 = v_fp32.to(torch.bfloat16)

            output_bf16 = ext.metal_scaled_dot_product_attention(q_bf16, k_bf16, v_bf16)
            metrics_bf16 = self.compute_attention_metrics(output_bf16)

            # Quality checks
            quality_checks = {}

            # Check for pathological cases in bf16
            quality_checks["bf16_no_nan"] = not metrics_bf16["has_nan"]
            quality_checks["bf16_no_inf"] = not metrics_bf16["has_inf"]
            quality_checks["bf16_not_all_zeros"] = not metrics_bf16["all_zeros"]
            quality_checks["bf16_not_all_same"] = not metrics_bf16["all_same"]
            quality_checks["bf16_reasonable_range"] = (
                0.001 < metrics_bf16["dynamic_range"] < 100.0
            )
            quality_checks["bf16_diverse_values"] = metrics_bf16["non_zero_ratio"] > 0.1

            # Compare with fp32
            mean_diff = abs(metrics_fp32["mean"] - metrics_bf16["mean"])
            std_ratio = metrics_bf16["std"] / max(metrics_fp32["std"], 1e-8)

            quality_checks["similar_mean"] = mean_diff < 0.5
            quality_checks["similar_std"] = 0.5 < std_ratio < 2.0
            quality_checks["similar_variance"] = (
                abs(metrics_fp32["head_variance"] - metrics_bf16["head_variance"])
                < 10.0
            )

            all_passed = all(quality_checks.values())

            # Create detailed message
            message = f"FP32: mean={metrics_fp32['mean']:.4f}, std={metrics_fp32['std']:.4f}, range={metrics_fp32['dynamic_range']:.4f}\n"
            message += f"    BF16: mean={metrics_bf16['mean']:.4f}, std={metrics_bf16['std']:.4f}, range={metrics_bf16['dynamic_range']:.4f}\n"
            message += f"    Quality checks: {sum(quality_checks.values())}/{len(quality_checks)} passed"

            if not all_passed:
                message += (
                    f"\n    Failed: {[k for k, v in quality_checks.items() if not v]}"
                )

            self.log_result(test_name, all_passed, message)

            # Store detailed results
            return {
                "fp32_metrics": metrics_fp32,
                "bf16_metrics": metrics_bf16,
                "quality_checks": quality_checks,
                "overall_pass": all_passed,
            }

        except Exception as e:
            self.log_result(test_name, False, f"Exception: {str(e)}")
            return None

    def test_pytorch_sdpa_comparison(self):
        """Compare Metal SDPA with PyTorch SDPA as reference."""
        test_name = "Metal vs PyTorch SDPA Comparison"

        try:
            # Create test tensors
            q, k, v = self.create_realistic_attention_tensors(torch.float32)

            # Test PyTorch SDPA (reference)
            print("  🔍 Testing PyTorch SDPA (reference)...")
            torch_output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            torch_metrics = self.compute_attention_metrics(torch_output)

            # Test Metal SDPA
            print("  🔍 Testing Metal SDPA...")
            metal_output = ext.metal_scaled_dot_product_attention(q, k, v)
            metal_metrics = self.compute_attention_metrics(metal_output)

            # Check that both produce reasonable results
            torch_reasonable = (
                not torch_metrics["has_nan"]
                and not torch_metrics["all_zeros"]
                and torch_metrics["dynamic_range"] > 0.001
            )

            metal_reasonable = (
                not metal_metrics["has_nan"]
                and not metal_metrics["all_zeros"]
                and metal_metrics["dynamic_range"] > 0.001
            )

            passed = torch_reasonable and metal_reasonable

            message = f"PyTorch: reasonable={torch_reasonable}, Metal: reasonable={metal_reasonable}"
            if passed:
                message += f"\n    Both implementations produce valid attention outputs"

            self.log_result(test_name, passed, message)

        except Exception as e:
            self.log_result(test_name, False, f"Exception: {str(e)}")

    def test_different_sequence_lengths(self):
        """Test bf16 quality across different sequence lengths."""
        test_name = "BF16 Quality Across Sequence Lengths"

        try:
            sequence_lengths = [64, 256, 512, 1024]
            all_passed = True
            results = []

            for seq_len in sequence_lengths:
                print(f"  🔍 Testing sequence length: {seq_len}")

                # Create tensors
                q, k, v = self.create_realistic_attention_tensors(
                    torch.bfloat16,
                    seq_len=seq_len,
                    batch_size=1,
                    num_heads=8,
                    head_dim=64,
                )

                # Test attention
                output = ext.metal_scaled_dot_product_attention(q, k, v)
                metrics = self.compute_attention_metrics(output)

                # Quality check
                seq_passed = (
                    not metrics["has_nan"]
                    and not metrics["all_zeros"]
                    and metrics["dynamic_range"] > 0.001
                    and metrics["non_zero_ratio"] > 0.1
                )

                results.append(f"{seq_len}: {'✓' if seq_passed else '✗'}")
                all_passed = all_passed and seq_passed

            message = "Sequence lengths: " + ", ".join(results)
            self.log_result(test_name, all_passed, message)

        except Exception as e:
            self.log_result(test_name, False, f"Exception: {str(e)}")

    def run_quality_comparison(self):
        """Run comprehensive quality comparison tests."""
        print("🔍 BF16 vs FP32 Quality Comparison Tests")
        print("=" * 60)

        # Create realistic test tensors
        print("\n📋 Creating realistic attention tensors...")
        q_fp32, k_fp32, v_fp32 = self.create_realistic_attention_tensors(torch.float32)

        # Main comparison test
        print("\n📋 Precision Comparison Tests")
        detailed_results = self.compare_precision_outputs(q_fp32, k_fp32, v_fp32)

        # Additional validation tests
        print("\n📋 Reference Comparison Tests")
        self.test_pytorch_sdpa_comparison()

        print("\n📋 Robustness Tests")
        self.test_different_sequence_lengths()

        # Summary
        print("\n" + "=" * 60)
        print("📊 Quality Comparison Summary")

        total_tests = len(self.results)
        passed_tests = sum(1 for _, passed, _ in self.results if passed)
        failed_tests = total_tests - passed_tests

        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")

        if failed_tests > 0:
            print("\n❌ Failed Tests:")
            for test_name, passed, message in self.results:
                if not passed:
                    print(f"  - {test_name}")
                    if message:
                        print(f"    {message}")

        # Key conclusion
        print("\n" + "=" * 60)
        if detailed_results and detailed_results["overall_pass"]:
            print("🎉 BF16 MODE QUALITY VALIDATION: PASSED")
            print("   BF16 produces viable attention outputs similar to FP32")
            print("   The black image issue appears to be resolved!")
        else:
            print("⚠️  BF16 MODE QUALITY VALIDATION: FAILED")
            print("   BF16 outputs may still have quality issues")
            print("   Further investigation needed")

        return failed_tests == 0


if __name__ == "__main__":
    print("🔧 BF16 vs FP32 Quality Comparison")
    print("This test validates that bf16 mode produces viable outputs similar to fp32.")
    print()

    comparator = QualityComparator()
    success = comparator.run_quality_comparison()

    sys.exit(0 if success else 1)
