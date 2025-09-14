#!/usr/bin/env python3
"""
Debug tensor layout differences between PyTorch integration and Swift test.
Author: bghira
"""

import torch
import metal_sdpa_extension
import numpy as np


def debug_tensor_layouts():
    """Compare tensor layout between PyTorch and Swift test parameters"""
    print("Tensor Layout Debug")
    print("Author: bghira")
    print("=" * 70)

    if not metal_sdpa_extension.is_metal_available():
        print("❌ Metal not available")
        return

    # EXACT same parameters as Swift test
    seq_len = 4
    head_dim = 4

    print(f"Swift test configuration:")
    print(f"  seqLen: {seq_len} (UInt32)")
    print(f"  headDim: {head_dim} (UInt16)")
    print(
        f"  Data: Array(repeating: 1.0, count: {seq_len * head_dim}) = 16 Float32 elements"
    )
    print(f"  Expected shape: [{seq_len}, {head_dim}]")

    # Test different PyTorch tensor configurations
    configurations = [
        ("2D: (seq_len, head_dim)", torch.ones(seq_len, head_dim, dtype=torch.float32)),
        (
            "2D: (head_dim, seq_len) then .T",
            torch.ones(head_dim, seq_len, dtype=torch.float32).T,
        ),
        (
            "4D: (1, seq_len, 1, head_dim)",
            torch.ones(1, seq_len, 1, head_dim, dtype=torch.float32),
        ),
        (
            "1D: (seq_len * head_dim) then reshape",
            torch.ones(seq_len * head_dim, dtype=torch.float32).reshape(
                seq_len, head_dim
            ),
        ),
    ]

    for name, tensor in configurations:
        print(f"\n--- {name} ---")
        print(f"Shape: {tensor.shape}")
        print(f"Stride: {tensor.stride()}")
        print(f"Is contiguous: {tensor.is_contiguous()}")
        print(f"Data ptr offset: {tensor.storage_offset()}")
        print(f"Element size: {tensor.element_size()}")
        print(f"Total bytes: {tensor.numel() * tensor.element_size()}")

        # Print first few data values
        print(f"Data (first 8): {tensor.flatten()[:8].tolist()}")

        try:
            output = metal_sdpa_extension.metal_scaled_dot_product_attention(
                tensor, tensor, tensor, scale=1.0 / np.sqrt(head_dim), is_causal=False
            )
            print(f"✅ Success! Output shape: {output.shape}")
            print(f"Output data (first 8): {output.flatten()[:8].tolist()}")
            print(f"Output sum: {output.sum().item():.6f}")

            # Check if output matches expected (all 1.0s)
            expected = torch.ones_like(tensor)
            diff = torch.abs(output - expected).max().item()
            print(f"Max diff from expected: {diff:.2e}")

        except Exception as e:
            print(f"❌ Error: {e}")


def debug_contiguous_vs_noncontiguous():
    """Test if contiguity affects results"""
    print(f"\n{'='*70}")
    print("Contiguous vs Non-contiguous Debug")

    seq_len, head_dim = 4, 4

    # Create contiguous tensor
    contiguous = torch.ones(seq_len, head_dim, dtype=torch.float32)

    # Create non-contiguous tensor (transpose then transpose back)
    non_contiguous = torch.ones(head_dim, seq_len, dtype=torch.float32).T

    # Verify one is contiguous, one isn't
    print(f"Contiguous tensor is_contiguous(): {contiguous.is_contiguous()}")
    print(f"Non-contiguous tensor is_contiguous(): {non_contiguous.is_contiguous()}")
    print(f"Shapes match: {contiguous.shape == non_contiguous.shape}")
    print(f"Data equal: {torch.equal(contiguous, non_contiguous)}")

    tensors = [
        ("Contiguous", contiguous),
        ("Non-contiguous", non_contiguous),
        ("Non-contiguous made contiguous", non_contiguous.contiguous()),
    ]

    for name, tensor in tensors:
        print(f"\n--- {name} ---")
        print(f"Stride: {tensor.stride()}")
        print(f"Storage offset: {tensor.storage_offset()}")

        try:
            output = metal_sdpa_extension.metal_scaled_dot_product_attention(
                tensor, tensor, tensor, scale=1.0 / np.sqrt(head_dim), is_causal=False
            )
            print(f"✅ Output sum: {output.sum().item():.6f}")
            print(
                f"All close to 1.0: {torch.allclose(output, torch.ones_like(output), atol=1e-5)}"
            )

        except Exception as e:
            print(f"❌ Error: {e}")


def debug_data_ordering():
    """Test if data ordering affects results"""
    print(f"\n{'='*70}")
    print("Data Ordering Debug")

    seq_len, head_dim = 4, 4

    # Test with different data patterns
    patterns = [
        ("All 1.0s", torch.ones(seq_len, head_dim, dtype=torch.float32)),
        (
            "Sequential",
            torch.arange(seq_len * head_dim, dtype=torch.float32).reshape(
                seq_len, head_dim
            ),
        ),
        (
            "Row-major fill",
            torch.tensor(
                [[i * head_dim + j for j in range(head_dim)] for i in range(seq_len)],
                dtype=torch.float32,
            ),
        ),
        (
            "Column-major fill",
            torch.tensor(
                [[j * seq_len + i for j in range(head_dim)] for i in range(seq_len)],
                dtype=torch.float32,
            ),
        ),
    ]

    for name, tensor in patterns:
        print(f"\n--- {name} ---")
        print(f"Data layout:")
        print(tensor)
        print(f"Flattened: {tensor.flatten().tolist()}")

        try:
            output = metal_sdpa_extension.metal_scaled_dot_product_attention(
                tensor, tensor, tensor, scale=1.0 / np.sqrt(head_dim), is_causal=False
            )
            print(f"✅ Output shape: {output.shape}")
            print(f"Output:\n{output}")

        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    debug_tensor_layouts()
    debug_contiguous_vs_noncontiguous()
    debug_data_ordering()


if __name__ == "__main__":
    main()
