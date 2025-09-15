"""
Universal Metal Flash Attention Python Bindings

High-performance, zero-copy Python interface to Metal Flash Attention.
Provides Flash Attention 3 compatible API with automatic memory management.
"""

from ._ffi import MFAError
from ._version import __version__
from .core import (
    MFABuffer,
    MFAContext,
    attention,
    flash_attention_forward,
    quantized_attention,
)
from .utils import create_context, get_version, is_metal_available, print_system_info

__all__ = [
    "MFAContext",
    "MFABuffer",
    "flash_attention_forward",
    "attention",
    "quantized_attention",
    "MFAError",
    "create_context",
    "is_metal_available",
    "get_version",
    "print_system_info",
    "__version__",
]
