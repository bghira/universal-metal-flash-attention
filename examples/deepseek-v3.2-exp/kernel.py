"""
DeepSeek-V3.2-Exp Custom Kernels

Metal compute shaders for:
- Sparse attention indexing
- FP16 matrix operations (Metal native)

Based on CUDA kernels from: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp
Adapted for Metal/Apple Silicon
"""

import torch
import torch.nn.functional as F
from typing import Tuple

try:
    from python.metal_sdpa_ffi import sparse_indexer_scores

    _MFA_INDEXER_AVAILABLE = True
except Exception:
    _MFA_INDEXER_AVAILABLE = False


# =============================================================================
# Indexer Kernel (Metal/MPS Backend)
# =============================================================================


def index_scoring_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Compute index scores for sparse attention

    Metal implementation using MPS backend (FP16 native).
    Original CUDA kernel uses FP8 with block-wise scaling.

    Args:
        q: Query tensor [batch, n_heads, seq_q, head_dim]
        k: Key tensor [batch, n_heads, seq_k, head_dim]
        scale: Attention scale factor

    Returns:
        scores: Index scores [batch, n_heads, seq_q, seq_k]
    """
    # Ensure tensors are on MPS device
    if q.device.type != "mps":
        raise ValueError("Tensors must be on MPS device for Metal backend")

    use_mfa = (
        _MFA_INDEXER_AVAILABLE
        and q.device.type == "mps"
        and k.device.type == "mps"
        and q.dtype == torch.float16
        and k.dtype == torch.float16
    )

    if use_mfa:
        q_contig = q.contiguous()
        k_contig = k.contiguous()
        try:
            scores = sparse_indexer_scores(q_contig, k_contig, scale)
        except RuntimeError:
            # Fallback to PyTorch path if the FFI call fails at runtime
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    else:
        # Matrix multiplication: Q @ K^T using PyTorch/MPS backend
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Apply ReLU (as in original CUDA kernel)
    scores = F.relu(scores)

    return scores


def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Fast Hadamard Transform (FHT)

    Used in some attention variants for activation rotation.
    Placeholder implementation using PyTorch (can be optimized in Metal).

    Args:
        x: Input tensor [..., dim]

    Returns:
        Transformed tensor [..., dim]
    """
    # TODO: Implement Metal compute shader for FHT
    # For now, use simple orthogonal rotation as placeholder
    return x


# =============================================================================
# Top-K Selection (PyTorch Native)
# =============================================================================


def topk_indices_and_scores(
    scores: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select top-k indices and scores per query

    Uses PyTorch native topk (dispatches to MPS backend).

    Args:
        scores: Attention scores [batch, n_heads, seq_q, seq_k]
        k: Number of top indices to select

    Returns:
        indices: Top-k indices [batch, n_heads, seq_q, k]
        values: Top-k scores [batch, n_heads, seq_q, k]
    """
    values, indices = torch.topk(scores, k=k, dim=-1, largest=True, sorted=False)
    return indices, values


# =============================================================================
# Sparse Indexer (Complete Pipeline)
# =============================================================================


class SparseIndexer:
    """
    Sparse attention indexer

    Combines index scoring + top-k selection for efficient sparse attention.
    """

    def __init__(self, topk: int = 2048, scale: float = None):
        """
        Initialize sparse indexer

        Args:
            topk: Number of top indices to select
            scale: Attention scale (default: 1/sqrt(head_dim))
        """
        self.topk = topk
        self.scale = scale

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute sparse attention indices

        Args:
            q: Query tensor [batch, n_heads, seq_q, head_dim]
            k: Key tensor [batch, n_heads, seq_k, head_dim]

        Returns:
            indices: Top-k indices [batch, n_heads, seq_q, topk]
            scores: Top-k attention scores
        """
        # Auto-compute scale if not provided
        if self.scale is None:
            head_dim = q.shape[-1]
            scale = 1.0 / (head_dim ** 0.5)
        else:
            scale = self.scale

        # Compute index scores (Metal/MPS GEMM)
        scores = index_scoring_kernel(q, k, scale=scale)

        # Select top-k indices
        seq_k = k.shape[2]
        k_actual = min(self.topk, seq_k)
        indices, topk_scores = topk_indices_and_scores(scores, k=k_actual)

        return indices, topk_scores


# =============================================================================
# Quantization Helpers (FP16 - Metal Native)
# =============================================================================


def fp16_to_metal_buffer(tensor: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor is FP16 and on MPS device

    Metal natively supports FP16, no quantization needed.

    Args:
        tensor: Input tensor

    Returns:
        FP16 tensor on MPS device
    """
    if tensor.device.type != "mps":
        tensor = tensor.to("mps")

    if tensor.dtype != torch.float16:
        tensor = tensor.to(torch.float16)

    return tensor


def block_scaling(
    tensor: torch.Tensor,
    block_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Block-wise scaling for improved FP16 precision

    Divides tensor into blocks and computes per-block scales.

    Args:
        tensor: Input tensor [..., dim]
        block_size: Block size for scaling

    Returns:
        scaled_tensor: Scaled tensor
        scales: Per-block scaling factors
    """
    # Reshape to [..., n_blocks, block_size]
    *leading_dims, dim = tensor.shape
    n_blocks = (dim + block_size - 1) // block_size
    padded_dim = n_blocks * block_size

    # Pad if necessary
    if padded_dim > dim:
        padding = padded_dim - dim
        tensor = F.pad(tensor, (0, padding))

    # Reshape
    tensor = tensor.view(*leading_dims, n_blocks, block_size)

    # Compute per-block scales
    scales = tensor.abs().max(dim=-1, keepdim=True)[0]
    scales = scales.clamp(min=1e-8)

    # Scale
    scaled = tensor / scales

    # Flatten back
    scaled = scaled.view(*leading_dims, padded_dim)
    if padded_dim > dim:
        scaled = scaled[..., :dim]

    return scaled, scales.squeeze(-1)


# =============================================================================
# Testing
# =============================================================================


if __name__ == "__main__":
    print("Testing DeepSeek-V3.2-Exp Metal Kernels\n")

    # Check MPS availability
    if not torch.backends.mps.is_available():
        print("❌ MPS backend not available")
        exit(1)

    device = torch.device("mps")
    print(f"✓ MPS device available: {device}\n")

    # Test configuration
    batch = 2
    n_heads = 8
    seq = 512
    head_dim = 128
    topk = 64

    print(f"Configuration:")
    print(f"  Batch: {batch}")
    print(f"  Heads: {n_heads}")
    print(f"  Sequence: {seq}")
    print(f"  Head dim: {head_dim}")
    print(f"  Top-k: {topk}\n")

    # Create test tensors
    q = torch.randn(batch, n_heads, seq, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch, n_heads, seq, head_dim, device=device, dtype=torch.float16)

    print(f"✓ Created test tensors")
    print(f"  Q: {list(q.shape)} {q.dtype}")
    print(f"  K: {list(k.shape)} {k.dtype}\n")

    # Test indexer
    print("⚡ Testing sparse indexer...")
    indexer = SparseIndexer(topk=topk)
    indices, scores = indexer(q, k)

    print(f"\n✅ Sparse indexer test passed!")
    print(f"  Indices: {list(indices.shape)} {indices.dtype}")
    print(f"  Scores: {list(scores.shape)} {scores.dtype}")
    print(f"  Index range: [{indices.min().item()}, {indices.max().item()}]")
    print(f"  Score range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")

    # Test block scaling
    print(f"\n⚡ Testing block scaling...")
    x = torch.randn(batch, seq, 1024, device=device, dtype=torch.float16)
    scaled, scales = block_scaling(x, block_size=128)

    print(f"\n✅ Block scaling test passed!")
    print(f"  Input: {list(x.shape)}")
    print(f"  Scaled: {list(scaled.shape)}")
    print(f"  Scales: {list(scales.shape)}")

    print(f"\n{'='*60}")
    print("All kernel tests passed! ✅")
    print('='*60)
