"""
PyTorch Custom SDPA Backend with Metal Flash Attention

This module provides a custom PyTorch backend that integrates Metal Flash Attention
via Swift FFI for high-performance scaled dot product attention on Apple Silicon.

Usage:
    import torch
    from pytorch_custom_op_ffi import register_metal_sdpa_backend, use_metal_sdpa

    # Register the backend
    register_metal_sdpa_backend()

    # Use as context manager
    with use_metal_sdpa():
        output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    # Or enable globally
    torch.backends.metal_sdpa.enabled = True
"""

from .backend import (
    register_metal_sdpa_backend,
    unregister_metal_sdpa_backend,
    use_metal_sdpa,
    is_metal_sdpa_available,
    metal_sdpa_version,
    MetalSDPAContext,
)

__version__ = "0.1.0"
__all__ = [
    "register_metal_sdpa_backend",
    "unregister_metal_sdpa_backend",
    "use_metal_sdpa",
    "is_metal_sdpa_available",
    "metal_sdpa_version",
    "MetalSDPAContext",
]
