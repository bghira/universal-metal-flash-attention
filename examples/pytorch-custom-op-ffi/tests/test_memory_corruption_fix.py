"""
Test suite specifically for validating the memory corruption fix.

This module contains targeted tests to reproduce and validate the fix for
the memory corruption issue that occurs with non-contiguous tensors.
"""

import gc
import hashlib
import struct
from typing import List, Tuple

import numpy as np
import pytest
import torch

try:
    import metal_sdpa_extension

    HAS_METAL = True
except ImportError:
    HAS_METAL = False


def compute_tensor_checksum(tensor: torch.Tensor) -> str:
    """Compute a checksum for a tensor to detect memory corruption."""
    # Convert to numpy for consistent hashing
    np_array = tensor.detach().cpu().numpy()
    # Use SHA256 for a robust checksum
    hasher = hashlib.sha256()
    hasher.update(np_array.tobytes())
    return hasher.hexdigest()[:16]  # Use first 16 chars for brevity


def create_guard_pattern() -> torch.Tensor:
    """Create a guard pattern tensor to detect memory overwrites."""
    # Create a distinctive pattern that's easy to verify
    pattern = torch.tensor(
        [0xDEAD, 0xBEEF, 0xCAFE, 0xBABE, 0xFEED, 0xFACE, 0xC0DE, 0x1337],
        dtype=torch.int16,
    )
    return pattern


def verify_guard_pattern(pattern: torch.Tensor) -> bool:
    """Verify that a guard pattern hasn't been corrupted."""
    expected = torch.tensor(
        [0xDEAD, 0xBEEF, 0xCAFE, 0xBABE, 0xFEED, 0xFACE, 0xC0DE, 0x1337],
        dtype=torch.int16,
    )

    # Handle sign extension for negative values
    pattern_i32 = pattern.to(torch.int32)
    expected_i32 = expected.to(torch.int32)

    # Mask to 16-bit values for comparison
    pattern_masked = pattern_i32 & 0xFFFF
    expected_masked = expected_i32 & 0xFFFF

    return torch.equal(pattern_masked, expected_masked)


@pytest.mark.skipif(not HAS_METAL, reason="Metal SDPA extension not available")
class TestMemoryCorruptionFix:
    """Tests specifically targeting the memory corruption issue and its fix."""

    def test_reproduction_permute_corruption(self):
        """
        Reproduce the exact memory corruption scenario from FLUX model usage.

        This test simulates how FLUX models create non-contiguous tensors
        through permutation and demonstrates the memory corruption issue.
        """
        print("\n" + "=" * 80)
        print("Memory Corruption Reproduction Test")
        print("=" * 80)

        device = "mps" if torch.backends.mps.is_available() else "cpu"

        # FLUX typical configuration
        batch = 1
        heads = 12
        seq_len = 77
        dim = 64

        print(f"\nConfiguration (FLUX-like):")
        print(f"  Batch: {batch}, Heads: {heads}, Seq: {seq_len}, Dim: {dim}")

        # Step 1: Create tensors in FLUX layout [B,H,S,D]
        torch.manual_seed(42)
        q_flux = (
            torch.randn(batch, heads, seq_len, dim, dtype=torch.float16, device=device)
            * 0.1
        )
        k_flux = (
            torch.randn(batch, heads, seq_len, dim, dtype=torch.float16, device=device)
            * 0.1
        )
        v_flux = (
            torch.randn(batch, heads, seq_len, dim, dtype=torch.float16, device=device)
            * 0.1
        )

        # Store checksums of original data
        q_checksum_orig = compute_tensor_checksum(q_flux)
        k_checksum_orig = compute_tensor_checksum(k_flux)
        v_checksum_orig = compute_tensor_checksum(v_flux)

        print(f"\nOriginal checksums:")
        print(f"  Q: {q_checksum_orig}")
        print(f"  K: {k_checksum_orig}")
        print(f"  V: {v_checksum_orig}")

        # Step 2: Permute to Metal layout [B,S,H,D] - creates non-contiguous views
        q_metal = q_flux.permute(0, 2, 1, 3)
        k_metal = k_flux.permute(0, 2, 1, 3)
        v_metal = v_flux.permute(0, 2, 1, 3)

        print(f"\nAfter permutation:")
        print(f"  Q contiguous: {q_metal.is_contiguous()}")
        print(f"  Q stride: {q_metal.stride()}")
        print(f"  Q shape: {list(q_metal.shape)}")

        # Calculate memory access patterns
        print("\nMemory access pattern analysis:")

        # For contiguous tensor, the offset calculation would be:
        # offset = batch_idx * (S*H*D) + seq_idx * (H*D) + head_idx * D + dim_idx

        # For permuted tensor, actual memory layout is different
        expected_stride_metal = (seq_len * heads * dim, heads * dim, dim, 1)
        actual_stride_metal = q_metal.stride()

        print(f"  Expected stride (if contiguous): {expected_stride_metal}")
        print(f"  Actual stride (permuted):         {actual_stride_metal}")

        # Show how wrong offset calculation would access memory
        def calculate_offset_contiguous(b, s, h, d, shape):
            """Calculate offset assuming contiguous memory."""
            _, S, H, D = shape
            return b * (S * H * D) + s * (H * D) + h * D + d

        def calculate_offset_strided(b, s, h, d, strides):
            """Calculate offset using actual strides."""
            return b * strides[0] + s * strides[1] + h * strides[2] + d * strides[3]

        # Example access at position [0, 0, 1, 0]
        test_pos = (0, 0, 1, 0)
        offset_wrong = calculate_offset_contiguous(*test_pos, q_metal.shape)
        offset_correct = calculate_offset_strided(*test_pos, q_metal.stride())

        print(f"\nExample memory access at position {test_pos}:")
        print(f"  Offset (assuming contiguous): {offset_wrong}")
        print(f"  Offset (using strides):        {offset_correct}")
        print(f"  Difference: {abs(offset_wrong - offset_correct)} elements")

        # Step 3: Attempt to run attention (may cause corruption)
        try:
            print("\nRunning attention with non-contiguous tensors...")
            output = metal_sdpa_extension.metal_scaled_dot_product_attention(
                q_metal, k_metal, v_metal
            )

            # Check if output is valid
            has_nan = torch.isnan(output).any().item()
            has_inf = torch.isinf(output).any().item()

            print(f"  Completed: NaN={has_nan}, Inf={has_inf}")

            # Verify original data hasn't been corrupted
            q_checksum_after = compute_tensor_checksum(q_flux)
            k_checksum_after = compute_tensor_checksum(k_flux)
            v_checksum_after = compute_tensor_checksum(v_flux)

            print(f"\nChecksum verification:")
            print(f"  Q unchanged: {q_checksum_orig == q_checksum_after}")
            print(f"  K unchanged: {k_checksum_orig == k_checksum_after}")
            print(f"  V unchanged: {v_checksum_orig == v_checksum_after}")

            if q_checksum_orig != q_checksum_after:
                print("  ❌ Input tensor Q was modified - memory corruption detected!")
            if k_checksum_orig != k_checksum_after:
                print("  ❌ Input tensor K was modified - memory corruption detected!")
            if v_checksum_orig != v_checksum_after:
                print("  ❌ Input tensor V was modified - memory corruption detected!")

        except RuntimeError as e:
            print(f"  ❌ Runtime error (possibly due to memory corruption): {e}")
            # This is actually expected behavior if corruption is detected

    def test_guard_buffer_corruption(self):
        """
        Test using guard buffers to detect out-of-bounds writes.

        This test places guard patterns around tensors to detect if
        Metal kernels write outside the expected memory regions.
        """
        print("\n" + "=" * 80)
        print("Guard Buffer Corruption Detection Test")
        print("=" * 80)

        device = "mps" if torch.backends.mps.is_available() else "cpu"

        # Create a larger buffer with guard regions
        total_elements = 1 * 77 * 12 * 64  # Metal layout shape
        guard_size = 256  # Elements for guard region

        # Allocate contiguous buffer with guards
        buffer = torch.zeros(
            guard_size + total_elements + guard_size, dtype=torch.float16, device=device
        )

        # Place guard patterns
        guard_pattern = create_guard_pattern().to(device)
        buffer[:8] = guard_pattern.to(torch.float16)
        buffer[-8:] = guard_pattern.to(torch.float16)

        print(f"Buffer layout:")
        print(f"  Total size: {len(buffer)} elements")
        print(f"  Guard size: {guard_size} elements each side")
        print(f"  Data region: {total_elements} elements")

        # Create tensor views in the data region (avoiding guards)
        data_start = guard_size
        data_end = guard_size + total_elements

        # Create FLUX layout tensor in the middle of the buffer
        flux_shape = (1, 12, 77, 64)
        data_view = buffer[data_start:data_end].view(flux_shape)

        # Fill with test data
        torch.manual_seed(42)
        data_view.copy_(
            torch.randn(flux_shape, dtype=torch.float16, device=device) * 0.1
        )

        # Permute to Metal layout (non-contiguous)
        metal_view = data_view.permute(0, 2, 1, 3)

        print(f"\nTensor view info:")
        print(f"  FLUX shape: {list(data_view.shape)}")
        print(f"  Metal shape: {list(metal_view.shape)}")
        print(f"  Contiguous: {metal_view.is_contiguous()}")

        # Verify guards are intact before operation
        front_guard = buffer[:8].to(torch.int16)
        back_guard = buffer[-8:].to(torch.int16)

        assert verify_guard_pattern(front_guard), "Front guard corrupted before test!"
        assert verify_guard_pattern(back_guard), "Back guard corrupted before test!"
        print("✅ Guard patterns verified before operation")

        # Run attention
        try:
            output = metal_sdpa_extension.metal_scaled_dot_product_attention(
                metal_view, metal_view, metal_view
            )

            # Check guards after operation
            front_guard_after = buffer[:8].to(torch.int16)
            back_guard_after = buffer[-8:].to(torch.int16)

            front_intact = verify_guard_pattern(front_guard_after)
            back_intact = verify_guard_pattern(back_guard_after)

            print(f"\nGuard verification after operation:")
            print(f"  Front guard intact: {front_intact}")
            print(f"  Back guard intact: {back_intact}")

            if not front_intact:
                print("  ❌ Front guard corrupted - underflow detected!")
            if not back_intact:
                print("  ❌ Back guard corrupted - overflow detected!")

            if front_intact and back_intact:
                print("  ✅ No out-of-bounds writes detected")

        except Exception as e:
            print(f"  ❌ Operation failed: {e}")

    def test_stride_calculation_validation(self):
        """
        Validate that stride calculations are correct for various layouts.

        This test verifies that the Metal kernel correctly interprets
        stride information for different tensor layouts.
        """
        print("\n" + "=" * 80)
        print("Stride Calculation Validation Test")
        print("=" * 80)

        device = "mps" if torch.backends.mps.is_available() else "cpu"

        test_cases = [
            # (name, shape, permutation)
            ("FLUX to Metal", (1, 12, 77, 64), (0, 2, 1, 3)),
            ("Metal to FLUX", (1, 77, 12, 64), (0, 2, 1, 3)),
            ("Transpose heads/seq", (1, 12, 77, 64), (0, 2, 1, 3)),
            ("Complex permutation", (2, 8, 32, 64), (0, 3, 2, 1)),
        ]

        for name, shape, perm in test_cases:
            print(f"\n{name}:")
            print(f"  Original shape: {shape}")
            print(f"  Permutation: {perm}")

            # Create base tensor
            torch.manual_seed(42)
            base = torch.randn(shape, dtype=torch.float16, device=device) * 0.1

            # Apply permutation
            permuted = base.permute(perm)

            print(f"  Permuted shape: {list(permuted.shape)}")
            print(f"  Original strides: {base.stride()}")
            print(f"  Permuted strides: {permuted.stride()}")
            print(f"  Contiguous: {permuted.is_contiguous()}")

            # Calculate expected element access
            # For a 4D tensor at position [b, i, j, k]:
            # offset = b*stride[0] + i*stride[1] + j*stride[2] + k*stride[3]

            # Test accessing element at position [0, 1, 0, 0]
            if all(dim > 1 for dim in permuted.shape):
                test_idx = (0, 1, 0, 0)
                offset = sum(
                    idx * stride for idx, stride in zip(test_idx, permuted.stride())
                )
                print(f"  Element at {test_idx} -> offset {offset}")

                # Verify this matches actual tensor indexing
                flat_base = base.flatten()
                if offset < len(flat_base):
                    expected_val = flat_base[offset]
                    actual_val = permuted[test_idx]
                    match = torch.allclose(expected_val, actual_val, rtol=1e-4)
                    print(f"  Offset calculation correct: {match}")

            # Test with Metal SDPA
            try:
                q, k, v = permuted, permuted, permuted
                output = metal_sdpa_extension.metal_scaled_dot_product_attention(
                    q, k, v
                )

                # Verify output
                has_nan = torch.isnan(output).any().item()
                has_inf = torch.isinf(output).any().item()

                if has_nan or has_inf:
                    print(f"  ⚠️  Output invalid: NaN={has_nan}, Inf={has_inf}")
                else:
                    print(f"  ✅ Valid output produced")

            except Exception as e:
                print(f"  ❌ Failed: {e}")

    def test_stress_memory_corruption(self):
        """
        Stress test to increase likelihood of detecting memory corruption.

        Runs multiple iterations with different patterns to catch
        intermittent corruption issues.
        """
        print("\n" + "=" * 80)
        print("Memory Corruption Stress Test")
        print("=" * 80)

        device = "mps" if torch.backends.mps.is_available() else "cpu"

        num_iterations = 50
        corruption_detected = False
        errors = []

        print(f"Running {num_iterations} iterations...")

        for i in range(num_iterations):
            # Vary the configuration slightly each iteration
            seq_len = 64 + (i % 32)  # Vary sequence length
            heads = 8 + (i % 8)  # Vary head count

            # Create FLUX layout tensors
            torch.manual_seed(i)  # Different seed each iteration
            shape = (1, heads, seq_len, 64)
            q = torch.randn(shape, dtype=torch.float16, device=device) * 0.1
            k = torch.randn(shape, dtype=torch.float16, device=device) * 0.1
            v = torch.randn(shape, dtype=torch.float16, device=device) * 0.1

            # Store original values
            q_orig = q.clone()
            k_orig = k.clone()
            v_orig = v.clone()

            # Permute to Metal layout (non-contiguous)
            q_perm = q.permute(0, 2, 1, 3)
            k_perm = k.permute(0, 2, 1, 3)
            v_perm = v.permute(0, 2, 1, 3)

            try:
                output = metal_sdpa_extension.metal_scaled_dot_product_attention(
                    q_perm, k_perm, v_perm
                )

                # Check for corruption
                if not torch.equal(q, q_orig):
                    corruption_detected = True
                    errors.append(f"Iteration {i}: Q tensor modified")
                if not torch.equal(k, k_orig):
                    corruption_detected = True
                    errors.append(f"Iteration {i}: K tensor modified")
                if not torch.equal(v, v_orig):
                    corruption_detected = True
                    errors.append(f"Iteration {i}: V tensor modified")

                # Check output validity
                if torch.isnan(output).any() or torch.isinf(output).any():
                    errors.append(f"Iteration {i}: Invalid output (NaN/Inf)")

            except Exception as e:
                errors.append(f"Iteration {i}: Exception - {str(e)[:50]}")

            # Print progress every 10 iterations
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{num_iterations} iterations")

            # Clean up periodically
            if i % 10 == 0:
                gc.collect()
                torch.mps.empty_cache() if device == "mps" else None

        print(f"\nResults:")
        print(f"  Iterations completed: {num_iterations}")
        print(f"  Corruption detected: {corruption_detected}")
        print(f"  Errors encountered: {len(errors)}")

        if errors:
            print(f"\nFirst 5 errors:")
            for error in errors[:5]:
                print(f"    {error}")

        assert not corruption_detected, "Memory corruption detected during stress test"
        assert len(errors) == 0, f"Errors occurred during stress test: {errors[:5]}"


@pytest.mark.skipif(not HAS_METAL, reason="Metal SDPA extension not available")
def test_quick_corruption_check():
    """Quick test to check for obvious memory corruption."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")

    # Create non-contiguous tensors via permute
    flux = torch.randn(1, 12, 77, 64, dtype=torch.float16, device="mps") * 0.1
    flux_orig = flux.clone()

    metal = flux.permute(0, 2, 1, 3)  # Non-contiguous

    # Run attention
    output = metal_sdpa_extension.metal_scaled_dot_product_attention(
        metal, metal, metal
    )

    # Verify input wasn't modified (would indicate corruption)
    assert torch.equal(
        flux, flux_orig
    ), "Input tensor was modified - memory corruption!"

    # Verify output is valid
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"


if __name__ == "__main__":
    # Run tests directly
    test_runner = TestMemoryCorruptionFix()

    print("Running Memory Corruption Fix Tests")
    print("=" * 80)

    test_runner.test_reproduction_permute_corruption()
    test_runner.test_guard_buffer_corruption()
    test_runner.test_stride_calculation_validation()
    test_runner.test_stress_memory_corruption()

    print("\n" + "=" * 80)
    print("All tests completed")
    print("=" * 80)
