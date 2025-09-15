"""
High-level Python API for Universal Metal Flash Attention.

Provides a clean, Pythonic interface with automatic resource management,
zero-copy numpy integration, and Flash Attention 3 compatible API.
"""

import ctypes
import weakref
from typing import Any, Optional, Tuple, Union

import numpy as np

from ._ffi import (
    MFA_PRECISION_BF16,
    MFA_PRECISION_FP16,
    MFA_PRECISION_FP32,
    MFAError,
    _check_error,
    _lib,
    mfa_buffer_t,
    mfa_context_t,
)

# Type aliases for clarity
FloatArray = np.ndarray
Precision = Union[int, str]


class MFAContext:
    """
    MFA context managing Metal device and command queue.

    This class provides automatic resource management for the MFA context.
    Use as a context manager for guaranteed cleanup.

    Example:
        with MFAContext() as ctx:
            # Use context
            pass
    """

    def __init__(self):
        """Create a new MFA context."""
        self._handle = mfa_context_t()
        _check_error(_lib.mfa_create_context(ctypes.byref(self._handle)))

        # Set up automatic cleanup
        self._finalizer = weakref.finalize(self, self._cleanup, self._handle)

    def __enter__(self) -> "MFAContext":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()

    def close(self):
        """Explicitly close the context and free resources."""
        if self._finalizer.detach():
            self._cleanup(self._handle)

    @staticmethod
    def _cleanup(handle: mfa_context_t):
        """Clean up MFA context resources."""
        if handle:
            _lib.mfa_destroy_context(handle)

    @property
    def handle(self) -> mfa_context_t:
        """Get the underlying context handle."""
        return self._handle

    def __bool__(self) -> bool:
        """Check if context is valid."""
        return bool(self._handle)


class MFABuffer:
    """
    MFA buffer wrapping Metal buffer for zero-copy tensor operations.

    Supports both creation of new buffers and wrapping existing numpy arrays
    for zero-copy operations.
    """

    def __init__(
        self,
        context: MFAContext,
        data: Optional[FloatArray] = None,
        size: Optional[int] = None,
    ):
        """
        Create MFA buffer.

        Args:
            context: MFA context
            data: Optional numpy array to wrap (zero-copy)
            size: Buffer size in bytes (if data is None)
        """
        self._context = context
        self._handle = mfa_buffer_t()
        self._array = data

        if data is not None:
            # Zero-copy from numpy array
            if not data.flags.c_contiguous:
                raise ValueError("Array must be C-contiguous for zero-copy")

            data_ptr = data.ctypes.data_as(ctypes.c_void_p)
            size_bytes = data.nbytes
            _check_error(
                _lib.mfa_buffer_from_ptr(
                    context.handle, data_ptr, size_bytes, ctypes.byref(self._handle)
                )
            )
        elif size is not None:
            # Create new buffer
            _check_error(
                _lib.mfa_create_buffer(context.handle, size, ctypes.byref(self._handle))
            )
        else:
            raise ValueError("Must provide either data array or buffer size")

        # Set up automatic cleanup
        self._finalizer = weakref.finalize(self, self._cleanup, self._handle)

    def close(self):
        """Explicitly close the buffer and free resources."""
        if self._finalizer.detach():
            self._cleanup(self._handle)

    @staticmethod
    def _cleanup(handle: mfa_buffer_t):
        """Clean up MFA buffer resources."""
        if handle:
            _lib.mfa_destroy_buffer(handle)

    @property
    def handle(self) -> mfa_buffer_t:
        """Get the underlying buffer handle."""
        return self._handle

    def contents_ptr(self) -> ctypes.c_void_p:
        """Get pointer to buffer contents."""
        return _lib.mfa_buffer_contents(self._handle)

    def __bool__(self) -> bool:
        """Check if buffer is valid."""
        return bool(self._handle)


def _parse_precision(precision: Precision) -> int:
    """Parse precision specification to MFA precision constant."""
    if isinstance(precision, int):
        return precision

    precision_map = {
        "fp16": MFA_PRECISION_FP16,
        "half": MFA_PRECISION_FP16,
        "bf16": MFA_PRECISION_BF16,
        "bfloat16": MFA_PRECISION_BF16,
        "fp32": MFA_PRECISION_FP32,
        "float": MFA_PRECISION_FP32,
        "float32": MFA_PRECISION_FP32,
    }

    precision_str = str(precision).lower()
    if precision_str not in precision_map:
        raise ValueError(
            f"Unknown precision: {precision}. Use one of {list(precision_map.keys())}"
        )

    return precision_map[precision_str]


def flash_attention_forward(
    context: MFAContext,
    q: FloatArray,
    k: FloatArray,
    v: FloatArray,
    *,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    input_precision: Precision = "fp16",
    intermediate_precision: Precision = "fp16",
    output_precision: Precision = "fp16",
) -> FloatArray:
    """
    Perform Flash Attention forward pass with zero-copy numpy arrays.

    Args:
        context: MFA context
        q: Query tensor [batch_size, seq_len_q, num_heads, head_dim] or [seq_len_q, head_dim] for single head
        k: Key tensor [batch_size, seq_len_kv, num_heads, head_dim] or [seq_len_kv, head_dim] for single head
        v: Value tensor [batch_size, seq_len_kv, num_heads, head_dim] or [seq_len_kv, head_dim] for single head
        causal: Whether to apply causal (lower triangular) mask
        softmax_scale: Scaling factor for attention scores (default: 1/√head_dim)
        input_precision: Precision for Q, K, V tensors
        intermediate_precision: Precision for intermediate computations
        output_precision: Precision for output tensor

    Returns:
        Output tensor with same shape as query

    Example:
        >>> with MFAContext() as ctx:
        ...     q = np.random.randn(512, 64).astype(np.float16)
        ...     k = np.random.randn(512, 64).astype(np.float16)
        ...     v = np.random.randn(512, 64).astype(np.float16)
        ...     out = flash_attention_forward(ctx, q, k, v)
    """
    # Validate inputs
    if not all(isinstance(x, np.ndarray) for x in [q, k, v]):
        raise TypeError("q, k, v must be numpy arrays")

    # Handle both single-head [seq_len, head_dim] and multi-head [batch, seq_len, heads, head_dim]
    if q.ndim == 2:
        # Single head case: [seq_len, head_dim]
        batch_size = 1
        seq_len_q, head_dim = q.shape
        seq_len_kv = k.shape[0]
        num_heads = 1

        # Validate shapes
        if k.shape != (seq_len_kv, head_dim) or v.shape != (seq_len_kv, head_dim):
            raise ValueError(f"Shape mismatch: q={q.shape}, k={k.shape}, v={v.shape}")

    elif q.ndim == 4:
        # Multi-head case: [batch_size, seq_len, num_heads, head_dim]
        batch_size, seq_len_q, num_heads, head_dim = q.shape
        seq_len_kv = k.shape[1]

        # Validate shapes
        expected_k_shape = (batch_size, seq_len_kv, num_heads, head_dim)
        expected_v_shape = (batch_size, seq_len_kv, num_heads, head_dim)
        if k.shape != expected_k_shape or v.shape != expected_v_shape:
            raise ValueError(f"Shape mismatch: q={q.shape}, k={k.shape}, v={v.shape}")

        # Current MFA implementation supports single head only
        if num_heads != 1:
            raise ValueError(
                "Multi-head attention not yet supported. Use num_heads=1 or 2D arrays."
            )
    else:
        raise ValueError(
            f"Invalid tensor dimensions. Expected 2D or 4D, got q.shape={q.shape}"
        )

    # Set default softmax scale
    if softmax_scale is None:
        softmax_scale = 1.0 / np.sqrt(head_dim)

    # Parse precisions
    input_prec = _parse_precision(input_precision)
    intermediate_prec = _parse_precision(intermediate_precision)
    output_prec = _parse_precision(output_precision)

    # Create output array
    output = np.zeros_like(q)

    # Create MFA buffers (zero-copy)
    q_buf = MFABuffer(context, q)
    k_buf = MFABuffer(context, k)
    v_buf = MFABuffer(context, v)
    out_buf = MFABuffer(context, output)

    try:
        # Call MFA attention
        _check_error(
            _lib.mfa_attention_forward(
                context.handle,
                q_buf.handle,
                k_buf.handle,
                v_buf.handle,
                out_buf.handle,
                batch_size,
                seq_len_q,
                seq_len_kv,
                num_heads,
                head_dim,
                softmax_scale,
                causal,
                input_prec,
                intermediate_prec,
                output_prec,
                False,  # transpose_q
                False,  # transpose_k
                False,  # transpose_v
                False,  # transpose_o
            )
        )
    finally:
        # Clean up buffers
        q_buf.close()
        k_buf.close()
        v_buf.close()
        out_buf.close()

    return output


def attention(
    q: FloatArray,
    k: FloatArray,
    v: FloatArray,
    context: Optional[MFAContext] = None,
    **kwargs,
) -> FloatArray:
    """
    Convenience function for flash attention with automatic context management.

    Args:
        q, k, v: Query, key, value tensors
        context: Optional MFA context (created automatically if None)
        **kwargs: Additional arguments passed to flash_attention_forward

    Returns:
        Attention output tensor

    Example:
        >>> q = np.random.randn(512, 64).astype(np.float16)
        >>> k = np.random.randn(512, 64).astype(np.float16)
        >>> v = np.random.randn(512, 64).astype(np.float16)
        >>> out = attention(q, k, v, causal=True)
    """
    if context is not None:
        return flash_attention_forward(context, q, k, v, **kwargs)
    else:
        with MFAContext() as ctx:
            return flash_attention_forward(ctx, q, k, v, **kwargs)
