#!/usr/bin/env python3
"""
PyTorch Reference Validation Script for Swift MFA Tests

This script generates PyTorch reference values using the same deterministic
data generation as the Swift tests, allowing for numerical correctness validation.

Usage:
    python pytorch_reference_validation.py

Author: Claude Code (generated for MFA test validation)
"""

import torch
import torch.nn.functional as F
import numpy as np


def lcg_deterministic_data(count: int, seed: int) -> np.ndarray:
    """
    Generate the same deterministic data as Swift's generateDeterministicData()
    Uses Linear Congruential Generator with same constants.
    """
    rng = seed
    data = []

    for _ in range(count):
        # Same LCG constants as Swift: rng = rng * 1664525 + 1013904223
        rng = (rng * 1664525 + 1013904223) % (2**64)  # 64-bit arithmetic

        # Normalize to [0, 1] then convert to [-1, 1]
        normalized = (rng % 1000000) / 1000000.0
        value = (normalized - 0.5) * 2.0
        data.append(value)

    return np.array(data, dtype=np.float32)


def validate_configuration(batch_size: int, num_heads: int, seq_len: int, head_dim: int, seed: int = 12345):
    """
    Validate a specific configuration against PyTorch reference.

    Args:
        batch_size: Batch size (typically 1 for tests)
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Head dimension
        seed: Seed for deterministic data generation
    """
    print(f"\n--- Validating Configuration ---")
    print(f"B={batch_size}, H={num_heads}, S={seq_len}, D={head_dim}")
    print(f"Seed: {seed}")

    total_elements = batch_size * num_heads * seq_len * head_dim

    # Generate the same deterministic data as Swift
    query_data = lcg_deterministic_data(total_elements, seed)
    key_data = lcg_deterministic_data(total_elements, seed + 1)
    value_data = lcg_deterministic_data(total_elements, seed + 2)

    # Reshape to proper tensor dimensions: [batch, seq_len, num_heads, head_dim]
    # Note: This matches the Swift test's expected layout
    q = torch.from_numpy(query_data.reshape(batch_size, seq_len, num_heads, head_dim))
    k = torch.from_numpy(key_data.reshape(batch_size, seq_len, num_heads, head_dim))
    v = torch.from_numpy(value_data.reshape(batch_size, seq_len, num_heads, head_dim))

    print(f"Input shapes: Q={list(q.shape)}, K={list(k.shape)}, V={list(v.shape)}")

    # Print first few values for comparison with Swift output
    print(f"First 5 Q values: {q.flatten()[:5].tolist()}")
    print(f"First 5 K values: {k.flatten()[:5].tolist()}")
    print(f"First 5 V values: {v.flatten()[:5].tolist()}")

    # PyTorch SDPA expects [batch, num_heads, seq_len, head_dim]
    q_pytorch = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
    k_pytorch = k.transpose(1, 2)
    v_pytorch = v.transpose(1, 2)

    print(f"PyTorch shapes: Q={list(q_pytorch.shape)}, K={list(k_pytorch.shape)}, V={list(v_pytorch.shape)}")

    # Calculate softmax scale (same as Swift)
    scale = 1.0 / (head_dim ** 0.5)
    print(f"Softmax scale: {scale}")

    # Compute PyTorch reference
    with torch.no_grad():
        output_pytorch = F.scaled_dot_product_attention(
            q_pytorch, k_pytorch, v_pytorch,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=scale
        )

    # Transpose back to match Swift layout: [batch, seq_len, num_heads, head_dim]
    output_pytorch = output_pytorch.transpose(1, 2)
    output_flat = output_pytorch.flatten()

    print(f"Output shape: {list(output_pytorch.shape)}")
    print(f"First 5 output values: {output_flat[:5].tolist()}")
    print(f"Last 5 output values: {output_flat[-5:].tolist()}")

    # Calculate statistics
    output_np = output_flat.numpy()
    output_range = (output_np.min(), output_np.max())
    mean = output_np.mean()
    std = output_np.std()
    max_abs = np.abs(output_np).max()

    print(f"Output range: [{output_range[0]:.6f}, {output_range[1]:.6f}]")
    print(f"Mean: {mean:.6f}, Std: {std:.6f}, Max abs: {max_abs:.6f}")

    # Check for NaN/Inf
    has_nan = np.isnan(output_np).any()
    has_inf = np.isinf(output_np).any()
    print(f"Has NaN: {has_nan}, Has Inf: {has_inf}")

    non_zero_count = np.count_nonzero(np.abs(output_np) > 1e-8)
    non_zero_ratio = non_zero_count / len(output_np)
    print(f"Non-zero ratio: {non_zero_ratio:.1%}")

    return output_pytorch


def main():
    """Main validation function"""
    print("🧪 PyTorch Reference Validation for Swift MFA Tests")
    print("=" * 60)
    print("This script generates PyTorch reference values using the same")
    print("deterministic data generation as the Swift tests.")
    print()

    # Test configurations that match Swift tests
    configs = [
        # From testNumericalCorrectnessAgainstPyTorch
        {"name": "Tiny", "batch_size": 1, "num_heads": 2, "seq_len": 4, "head_dim": 8, "seed": 12345},
        {"name": "Small", "batch_size": 1, "num_heads": 4, "seq_len": 8, "head_dim": 16, "seed": 12345},

        # Additional configurations for comprehensive testing
        {"name": "FLUX Joint", "batch_size": 1, "num_heads": 24, "seq_len": 512, "head_dim": 64, "seed": 42},
        {"name": "FLUX Large", "batch_size": 1, "num_heads": 16, "seq_len": 1024, "head_dim": 88, "seed": 42},
    ]

    for config in configs:
        try:
            print(f"\n{'='*60}")
            print(f"Testing {config['name']} Configuration")
            validate_configuration(
                config["batch_size"],
                config["num_heads"],
                config["seq_len"],
                config["head_dim"],
                config["seed"]
            )
            print(f"✅ {config['name']} validation completed")

        except Exception as e:
            print(f"❌ {config['name']} validation failed: {e}")

    print(f"\n{'='*60}")
    print("📝 Usage Instructions:")
    print("1. Run Swift tests to get MFA output values")
    print("2. Compare the printed values with this PyTorch reference")
    print("3. Expected tolerance: < 1e-5 for FP32, < 1e-3 for FP16")
    print("4. Focus on first/last values and statistics for quick comparison")

    print(f"\n🔧 Debugging Tips:")
    print("• If shapes don't match, check tensor layout assumptions")
    print("• If scale differs, verify head_dim calculation")
    print("• If first values differ, check LCG implementation")
    print("• If patterns differ, check transpose operations")


if __name__ == "__main__":
    main()