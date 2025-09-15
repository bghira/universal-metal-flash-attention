"""
Low-level FFI bindings to Universal Metal Flash Attention C library.

This module provides the raw ctypes interface to the MFA library.
Users should generally use the higher-level API in core.py instead.
"""

import ctypes
import ctypes.util
import os
import platform
from pathlib import Path
from typing import Optional

# Error codes
MFA_SUCCESS = 0
MFA_ERROR_INVALID_ARGS = 1
MFA_ERROR_MEMORY_ALLOCATION = 2
MFA_ERROR_DEVICE_NOT_SUPPORTED = 3
MFA_ERROR_KERNEL_COMPILATION = 4
MFA_ERROR_EXECUTION_FAILED = 5

# Precision types
MFA_PRECISION_FP16 = 0
MFA_PRECISION_BF16 = 1
MFA_PRECISION_FP32 = 2
MFA_PRECISION_INT8 = 3
MFA_PRECISION_INT4 = 4

# Type aliases
mfa_error_t = ctypes.c_int32
mfa_precision_t = ctypes.c_int32
mfa_context_t = ctypes.c_void_p
mfa_buffer_t = ctypes.c_void_p


class MFAError(Exception):
    """Exception raised for MFA library errors."""

    def __init__(self, code: int, message: str = ""):
        self.code = code
        self.message = message or _get_error_string(code)
        super().__init__(f"MFA Error {code}: {self.message}")


def _find_library() -> str:
    """Find the MFA library file."""
    # Look for the library in common locations
    search_paths = [
        Path(__file__).parent.parent.parent.parent.parent / ".build" / "release",
        Path(__file__).parent.parent.parent.parent.parent / ".build" / "debug",
        Path("/usr/local/lib"),
        Path("/opt/homebrew/lib"),
    ]

    lib_names = ["libMFAFFI.dylib", "MFAFFI.dylib"]

    for search_path in search_paths:
        if search_path.exists():
            for lib_name in lib_names:
                lib_path = search_path / lib_name
                if lib_path.exists():
                    return str(lib_path)

    # Fallback to system search
    for lib_name in lib_names:
        try:
            lib = ctypes.util.find_library(
                lib_name.replace("lib", "").replace(".dylib", "")
            )
            if lib:
                return lib
        except:
            pass

    raise RuntimeError(
        "Could not find MFA library. Make sure Universal Metal Flash Attention is built.\n"
        "Run: make release"
    )


def _load_library():
    """Load the MFA library and set up function signatures."""
    lib_path = _find_library()
    lib = ctypes.CDLL(lib_path)

    # Context management
    lib.mfa_create_context.argtypes = [ctypes.POINTER(mfa_context_t)]
    lib.mfa_create_context.restype = mfa_error_t

    lib.mfa_destroy_context.argtypes = [mfa_context_t]
    lib.mfa_destroy_context.restype = None

    # Buffer management
    lib.mfa_create_buffer.argtypes = [
        mfa_context_t,
        ctypes.c_size_t,
        ctypes.POINTER(mfa_buffer_t),
    ]
    lib.mfa_create_buffer.restype = mfa_error_t

    lib.mfa_buffer_from_ptr.argtypes = [
        mfa_context_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(mfa_buffer_t),
    ]
    lib.mfa_buffer_from_ptr.restype = mfa_error_t

    lib.mfa_buffer_contents.argtypes = [mfa_buffer_t]
    lib.mfa_buffer_contents.restype = ctypes.c_void_p

    lib.mfa_destroy_buffer.argtypes = [mfa_buffer_t]
    lib.mfa_destroy_buffer.restype = None

    # Attention operations
    lib.mfa_attention_forward.argtypes = [
        mfa_context_t,  # context
        mfa_buffer_t,  # q
        mfa_buffer_t,  # k
        mfa_buffer_t,  # v
        mfa_buffer_t,  # out
        ctypes.c_uint32,  # batch_size
        ctypes.c_uint32,  # seq_len_q
        ctypes.c_uint32,  # seq_len_kv
        ctypes.c_uint32,  # num_heads
        ctypes.c_uint16,  # head_dim
        ctypes.c_float,  # softmax_scale
        ctypes.c_bool,  # causal
        mfa_precision_t,  # input_precision
        mfa_precision_t,  # intermediate_precision
        mfa_precision_t,  # output_precision
        ctypes.c_bool,  # transpose_q
        ctypes.c_bool,  # transpose_k
        ctypes.c_bool,  # transpose_v
        ctypes.c_bool,  # transpose_o
    ]
    lib.mfa_attention_forward.restype = mfa_error_t

    # Quantized attention forward
    lib.mfa_attention_forward_quantized.argtypes = [
        mfa_context_t,  # context
        mfa_buffer_t,  # q
        mfa_buffer_t,  # k
        mfa_buffer_t,  # v
        mfa_buffer_t,  # out
        ctypes.c_uint32,  # batch_size
        ctypes.c_uint32,  # seq_len_q
        ctypes.c_uint32,  # seq_len_kv
        ctypes.c_uint32,  # num_heads
        ctypes.c_uint16,  # head_dim
        ctypes.c_float,  # softmax_scale
        ctypes.c_bool,  # causal
        ctypes.c_float,  # q_scale
        ctypes.c_int32,  # q_zero_point
        ctypes.c_float,  # k_scale
        ctypes.c_int32,  # k_zero_point
        ctypes.c_float,  # v_scale
        ctypes.c_int32,  # v_zero_point
        mfa_precision_t,  # q_precision
        mfa_precision_t,  # k_precision
        mfa_precision_t,  # v_precision
        mfa_precision_t,  # output_precision
        ctypes.c_bool,  # transpose_q
        ctypes.c_bool,  # transpose_k
        ctypes.c_bool,  # transpose_v
        ctypes.c_bool,  # transpose_o
    ]
    lib.mfa_attention_forward_quantized.restype = mfa_error_t

    # Utility functions
    lib.mfa_error_string.argtypes = [mfa_error_t]
    lib.mfa_error_string.restype = ctypes.c_char_p

    lib.mfa_is_device_supported.argtypes = []
    lib.mfa_is_device_supported.restype = ctypes.c_bool

    lib.mfa_get_version.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.mfa_get_version.restype = None

    return lib


def _get_error_string(code: int) -> str:
    """Get human-readable error message."""
    try:
        result = _lib.mfa_error_string(code)
        if result:
            message = result.decode("utf-8")
            # Note: MFA library allocates this string, we should free it
            # but ctypes doesn't give us direct access to free()
            return message
        return f"Unknown error code: {code}"
    except:
        return f"Unknown error code: {code}"


def _check_error(code: int) -> None:
    """Check error code and raise exception if needed."""
    if code != MFA_SUCCESS:
        raise MFAError(code)


# Load library on import
_lib = _load_library()

# Export the library instance and helper functions
__all__ = [
    "_lib",
    "MFAError",
    "_check_error",
    "_get_error_string",
    "MFA_SUCCESS",
    "MFA_ERROR_INVALID_ARGS",
    "MFA_ERROR_MEMORY_ALLOCATION",
    "MFA_ERROR_DEVICE_NOT_SUPPORTED",
    "MFA_ERROR_KERNEL_COMPILATION",
    "MFA_ERROR_EXECUTION_FAILED",
    "MFA_PRECISION_FP16",
    "MFA_PRECISION_BF16",
    "MFA_PRECISION_FP32",
    "MFA_PRECISION_INT8",
    "MFA_PRECISION_INT4",
    "mfa_error_t",
    "mfa_precision_t",
    "mfa_context_t",
    "mfa_buffer_t",
]
