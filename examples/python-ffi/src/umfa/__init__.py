"""
Universal Metal Flash Attention Python Bindings

High-performance, zero-copy Python interface to Metal Flash Attention.
Provides Flash Attention 3 compatible API with automatic memory management.
"""

from .core import MFAContext, MFABuffer, flash_attention_forward, attention
from .utils import create_context, is_metal_available, get_version, print_system_info
from ._ffi import MFAError
from ._version import __version__

__all__ = [
    "MFAContext",
    "MFABuffer",
    "flash_attention_forward",
    "attention",
    "MFAError",
    "create_context",
    "is_metal_available",
    "get_version",
    "print_system_info",
    "__version__",
]
