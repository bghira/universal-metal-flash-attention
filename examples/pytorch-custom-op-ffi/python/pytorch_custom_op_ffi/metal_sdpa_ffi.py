"""
Metal SDPA FFI module that exposes the low-level type management functions.

This module provides direct access to the type management functions implemented
in the C++ extension for testing and advanced usage.
"""

import os
import sys

try:
    # Import the compiled extension
    import metal_sdpa_extension as _ext

    # Re-export the type management functions
    determine_output_precision = _ext.determine_output_precision
    create_typed_output_tensor = _ext.create_typed_output_tensor
    validate_output_buffer_type = _ext.validate_output_buffer_type
    convert_output_precision = _ext.convert_output_precision
    calculate_expected_buffer_size = _ext.calculate_expected_buffer_size

    # Re-export configuration classes and enums
    QuantizationConfig = _ext.QuantizationConfig
    OutputPrecision = _ext.OutputPrecision
    QuantizationGranularity = _ext.QuantizationGranularity
    QuantizationPrecision = _ext.QuantizationPrecision
    HybridStrategy = _ext.HybridStrategy
    BlockSizeConfig = _ext.BlockSizeConfig

    # Re-export main functions
    quantized_scaled_dot_product_attention_unified = (
        _ext.quantized_scaled_dot_product_attention_unified
    )
    quantized_scaled_dot_product_attention = _ext.quantized_scaled_dot_product_attention
    sparse_indexer_scores = _ext.sparse_indexer_scores

    # Re-export utility functions
    is_metal_available = _ext.is_metal_available
    get_version = _ext.get_version

    # Re-export MLA class and functions
    try:
        MlaContext = _ext.MlaContext
        mla_create_context = _ext.mla_create_context
        mla_destroy_context = _ext.mla_destroy_context
        mla_init_weights = _ext.mla_init_weights
        mla_load_weights = _ext.mla_load_weights
        mla_forward = _ext.mla_forward
    except AttributeError:
        # MLA functions not yet bound in C++ extension
        pass

    _EXTENSION_AVAILABLE = True

except ImportError as e:
    _EXTENSION_AVAILABLE = False
    _import_error = e

    # Create dummy classes/functions for testing when extension is not available
    class OutputPrecision:
        FP16 = "fp16"
        FP32 = "fp32"
        BF16 = "bf16"

    class QuantizationConfig:
        pass

    def determine_output_precision(*args, **kwargs):
        raise RuntimeError(f"Extension not available: {_import_error}")

    def create_typed_output_tensor(*args, **kwargs):
        raise RuntimeError(f"Extension not available: {_import_error}")

    def validate_output_buffer_type(*args, **kwargs):
        raise RuntimeError(f"Extension not available: {_import_error}")

    def convert_output_precision(*args, **kwargs):
        raise RuntimeError(f"Extension not available: {_import_error}")

    def calculate_expected_buffer_size(*args, **kwargs):
        raise RuntimeError(f"Extension not available: {_import_error}")

    def sparse_indexer_scores(*args, **kwargs):
        raise RuntimeError(f"Extension not available: {_import_error}")


def is_available():
    """Check if the Metal SDPA FFI extension is available."""
    return _EXTENSION_AVAILABLE


def ensure_available():
    """Ensure the Metal SDPA FFI extension is available or raise an error."""
    if not _EXTENSION_AVAILABLE:
        raise RuntimeError(
            f"Metal SDPA FFI extension not available. Import error: {_import_error}\n"
            "Make sure the extension is compiled and available in the Python path."
        )


__all__ = [
    # Type management functions
    "determine_output_precision",
    "create_typed_output_tensor",
    "validate_output_buffer_type",
    "convert_output_precision",
    "calculate_expected_buffer_size",
    # Configuration classes
    "QuantizationConfig",
    "OutputPrecision",
    "QuantizationGranularity",
    "QuantizationPrecision",
    "HybridStrategy",
    "BlockSizeConfig",
    # Main functions
    "quantized_scaled_dot_product_attention_unified",
    "quantized_scaled_dot_product_attention",
    "sparse_indexer_scores",
    # MLA class and functions
    "MlaContext",
    "mla_create_context",
    "mla_destroy_context",
    "mla_init_weights",
    "mla_load_weights",
    "mla_forward",
    # Utility functions
    "is_metal_available",
    "get_version",
    "is_available",
    "ensure_available",
]
