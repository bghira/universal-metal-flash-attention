#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include "metal_sdpa_backend.h"

// Forward declarations from the header
extern "C" {
    bool mfa_is_device_supported(void);
    void mfa_get_version(int* major, int* minor, int* patch);
}

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "PyTorch Custom SDPA Backend with Metal Flash Attention via Swift FFI";

    // Register/unregister backend functions
    m.def("register_backend", &metal_sdpa::MetalSDPABackend::register_backend,
          "Register Metal SDPA backend with PyTorch");

    m.def("unregister_backend", &metal_sdpa::MetalSDPABackend::unregister_backend,
          "Unregister Metal SDPA backend from PyTorch");

    // Direct SDPA call (for testing/debugging)
    m.def("metal_scaled_dot_product_attention",
          &metal_sdpa::MetalSDPABackend::scaled_dot_product_attention,
          "Direct call to Metal Flash Attention SDPA",
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("attn_mask") = py::none(),
          py::arg("dropout_p") = 0.0,
          py::arg("is_causal") = false,
          py::arg("scale") = py::none(),
          py::arg("enable_gqa") = false);

    // Quantized SDPA call
    m.def("quantized_scaled_dot_product_attention",
          &metal_sdpa::MetalSDPABackend::quantized_scaled_dot_product_attention,
          "Direct call to Metal Flash Attention Quantized SDPA",
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("precision") = "int8",
          py::arg("is_causal") = false,
          py::arg("scale") = py::none());

    // Configurable Quantized SDPA call with output precision control
    m.def("quantized_scaled_dot_product_attention_with_config",
          &metal_sdpa::MetalSDPABackend::quantized_scaled_dot_product_attention_with_config,
          "Direct call to Metal Flash Attention Quantized SDPA with configurable output precision",
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("config"));

    // Enhanced Quantized SDPA with hybrid granularity support
    m.def("quantized_scaled_dot_product_attention_enhanced",
          &metal_sdpa::MetalSDPABackend::quantized_scaled_dot_product_attention_enhanced,
          "Enhanced Metal Flash Attention Quantized SDPA with hybrid granularity selection",
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("config"));

    // Unified Quantized SDPA - Primary interface with all features
    m.def("quantized_scaled_dot_product_attention_unified",
          &metal_sdpa::MetalSDPABackend::quantized_scaled_dot_product_attention_unified,
          "Unified Metal Flash Attention Quantized SDPA - Primary interface with all quantization features",
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("config"));

    // QuantizationPrecision enum - order matches FFI header values
    py::enum_<metal_sdpa::QuantizationPrecision>(m, "QuantizationPrecision")
        .value("FP16", metal_sdpa::QuantizationPrecision::FP16)
        .value("BF16", metal_sdpa::QuantizationPrecision::BF16)
        .value("FP32", metal_sdpa::QuantizationPrecision::FP32)
        .value("INT8", metal_sdpa::QuantizationPrecision::INT8)
        .value("INT4", metal_sdpa::QuantizationPrecision::INT4);

    // QuantizationGranularity enum
    py::enum_<metal_sdpa::QuantizationGranularity>(m, "QuantizationGranularity")
        .value("TENSOR_WISE", metal_sdpa::QuantizationGranularity::TENSOR_WISE)
        .value("ROW_WISE", metal_sdpa::QuantizationGranularity::ROW_WISE)
        .value("BLOCK_WISE", metal_sdpa::QuantizationGranularity::BLOCK_WISE)
        .value("HYBRID", metal_sdpa::QuantizationGranularity::HYBRID);

    // HybridStrategy enum
    py::enum_<metal_sdpa::HybridStrategy>(m, "HybridStrategy")
        .value("PERFORMANCE_FIRST", metal_sdpa::HybridStrategy::PERFORMANCE_FIRST)
        .value("ACCURACY_FIRST", metal_sdpa::HybridStrategy::ACCURACY_FIRST)
        .value("BALANCED", metal_sdpa::HybridStrategy::BALANCED);

    // OutputPrecision enum
    py::enum_<metal_sdpa::OutputPrecision>(m, "OutputPrecision")
        .value("FP32", metal_sdpa::OutputPrecision::FP32)
        .value("FP16", metal_sdpa::OutputPrecision::FP16)
        .value("BF16", metal_sdpa::OutputPrecision::BF16);

    // BlockSizeConfig class
    py::class_<metal_sdpa::BlockSizeConfig>(m, "BlockSizeConfig")
        .def(py::init<>())
        .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t>(),
             py::arg("q_blk"), py::arg("k_blk"), py::arg("v_blk"), py::arg("h_blk") = 1)
        .def_readwrite("query_block_size", &metal_sdpa::BlockSizeConfig::query_block_size)
        .def_readwrite("key_block_size", &metal_sdpa::BlockSizeConfig::key_block_size)
        .def_readwrite("value_block_size", &metal_sdpa::BlockSizeConfig::value_block_size)
        .def_readwrite("head_block_size", &metal_sdpa::BlockSizeConfig::head_block_size);

    // TensorAnalysisMetrics class
    py::class_<metal_sdpa::TensorAnalysisMetrics>(m, "TensorAnalysisMetrics")
        .def(py::init<>())
        .def_readwrite("dynamic_range", &metal_sdpa::TensorAnalysisMetrics::dynamic_range)
        .def_readwrite("variance", &metal_sdpa::TensorAnalysisMetrics::variance)
        .def_readwrite("mean_abs_value", &metal_sdpa::TensorAnalysisMetrics::mean_abs_value)
        .def_readwrite("tensor_size", &metal_sdpa::TensorAnalysisMetrics::tensor_size)
        .def_readwrite("memory_footprint", &metal_sdpa::TensorAnalysisMetrics::memory_footprint)
        .def_readwrite("sparsity_ratio", &metal_sdpa::TensorAnalysisMetrics::sparsity_ratio)
        .def_readwrite("has_outliers", &metal_sdpa::TensorAnalysisMetrics::has_outliers)
        .def_readwrite("quantization_error_estimate", &metal_sdpa::TensorAnalysisMetrics::quantization_error_estimate);

    // HybridGranularityConfig class
    py::class_<metal_sdpa::HybridGranularityConfig>(m, "HybridGranularityConfig")
        .def(py::init<>())
        .def_readwrite("query_granularity", &metal_sdpa::HybridGranularityConfig::query_granularity)
        .def_readwrite("key_granularity", &metal_sdpa::HybridGranularityConfig::key_granularity)
        .def_readwrite("value_granularity", &metal_sdpa::HybridGranularityConfig::value_granularity)
        .def_readwrite("query_blocks", &metal_sdpa::HybridGranularityConfig::query_blocks)
        .def_readwrite("key_blocks", &metal_sdpa::HybridGranularityConfig::key_blocks)
        .def_readwrite("value_blocks", &metal_sdpa::HybridGranularityConfig::value_blocks)
        .def_readwrite("selection_reasoning", &metal_sdpa::HybridGranularityConfig::selection_reasoning);

    // QuantizationConfig class
    py::class_<metal_sdpa::QuantizationConfig>(m, "QuantizationConfig")
        .def(py::init<>())
        .def_readwrite("precision", &metal_sdpa::QuantizationConfig::precision)
        .def_readwrite("query_precision", &metal_sdpa::QuantizationConfig::query_precision)
        .def_readwrite("key_precision", &metal_sdpa::QuantizationConfig::key_precision)
        .def_readwrite("value_precision", &metal_sdpa::QuantizationConfig::value_precision)
        .def_readwrite("granularity", &metal_sdpa::QuantizationConfig::granularity)
        .def_readwrite("block_sizes", &metal_sdpa::QuantizationConfig::block_sizes)
        .def_readwrite("output_precision", &metal_sdpa::QuantizationConfig::output_precision)
        .def_readwrite("is_causal", &metal_sdpa::QuantizationConfig::is_causal)
        .def_readwrite("scale", &metal_sdpa::QuantizationConfig::scale)
        .def_readwrite("enable_mixed_precision", &metal_sdpa::QuantizationConfig::enable_mixed_precision)
        .def_readwrite("force_symmetric_quantization", &metal_sdpa::QuantizationConfig::force_symmetric_quantization)
        .def_readwrite("hybrid_strategy", &metal_sdpa::QuantizationConfig::hybrid_strategy)
        .def_readwrite("enable_per_tensor_granularity", &metal_sdpa::QuantizationConfig::enable_per_tensor_granularity)
        .def_readwrite("enable_adaptive_block_sizes", &metal_sdpa::QuantizationConfig::enable_adaptive_block_sizes)
        .def_static("string_to_quantization_precision", &metal_sdpa::QuantizationConfig::string_to_quantization_precision)
        .def_static("string_to_precision", &metal_sdpa::QuantizationConfig::string_to_precision)
        .def_static("precision_to_string", &metal_sdpa::QuantizationConfig::precision_to_string)
        .def_static("quantization_precision_to_string", &metal_sdpa::QuantizationConfig::quantization_precision_to_string)
        .def_static("granularity_to_string", &metal_sdpa::QuantizationConfig::granularity_to_string)
        .def_static("string_to_granularity", &metal_sdpa::QuantizationConfig::string_to_granularity)
        .def("validate_config", &metal_sdpa::QuantizationConfig::validate_config)
        .def("get_recommended_output_precision", &metal_sdpa::QuantizationConfig::get_recommended_output_precision);


    /* TEMPORARILY COMMENTED OUT - NAMESPACE ISSUES
    // Hybrid quantization utility functions
    m.def("analyze_tensor_characteristics",
          &metal_sdpa::metal_sdpa::analyze_tensor_characteristics,
          "Analyze tensor characteristics for hybrid quantization",
          py::arg("tensor"),
          py::arg("precision"));

    m.def("select_optimal_granularity",
          &metal_sdpa::metal_sdpa::select_optimal_granularity,
          "Select optimal granularity based on tensor characteristics",
          py::arg("metrics"),
          py::arg("precision"),
          py::arg("strategy") = metal_sdpa::HybridStrategy::BALANCED);

    m.def("select_hybrid_granularities",
          &metal_sdpa::metal_sdpa::select_hybrid_granularities,
          "Select hybrid granularities for Q, K, V tensors",
          py::arg("query"),
          py::arg("key"),
          py::arg("value"),
          py::arg("config"));

    m.def("select_optimal_block_sizes",
          &metal_sdpa::metal_sdpa::select_optimal_block_sizes,
          "Select optimal block sizes for block-wise quantization",
          py::arg("tensor"),
          py::arg("precision"));

    m.def("estimate_quantization_overhead",
          py::overload_cast<metal_sdpa::QuantizationGranularity, const metal_sdpa::TensorAnalysisMetrics&, metal_sdpa::QuantizationPrecision>(&metal_sdpa::metal_sdpa::estimate_quantization_overhead),
          "Estimate computational overhead for quantization granularity (metrics-based)",
          py::arg("granularity"),
          py::arg("metrics"),
          py::arg("precision"));

    m.def("estimate_quantization_overhead",
          py::overload_cast<metal_sdpa::QuantizationGranularity, const torch::Tensor&, metal_sdpa::QuantizationPrecision>(&metal_sdpa::metal_sdpa::estimate_quantization_overhead),
          "Estimate computational overhead for quantization granularity (tensor-based)",
          py::arg("granularity"),
          py::arg("tensor"),
          py::arg("precision"));

    m.def("estimate_accuracy_loss",
          &metal_sdpa::metal_sdpa::estimate_accuracy_loss,
          "Estimate accuracy loss for quantization granularity",
          py::arg("granularity"),
          py::arg("metrics"),
          py::arg("precision"));

    // Return buffer type management functions
    m.def("determine_output_precision",
          &metal_sdpa::metal_sdpa::determine_output_precision,
          "Intelligently determine optimal output precision based on quantization config and input tensors",
          py::arg("config"),
          py::arg("query"),
          py::arg("key"),
          py::arg("value"));

    m.def("create_typed_output_tensor",
          &metal_sdpa::metal_sdpa::create_typed_output_tensor,
          "Create type-safe output tensor with specified precision and validation",
          py::arg("reference_tensor"),
          py::arg("output_precision"),
          py::arg("validate_size") = true);

    m.def("validate_output_buffer_type",
          &metal_sdpa::metal_sdpa::validate_output_buffer_type,
          "Validate output buffer type matches expected precision and size",
          py::arg("output_tensor"),
          py::arg("expected_precision"),
          py::arg("expected_size"));

    m.def("convert_output_precision",
          &metal_sdpa::metal_sdpa::convert_output_precision,
          "Convert output tensor from source precision to target precision",
          py::arg("output_tensor"),
          py::arg("source_precision"),
          py::arg("target_precision"));

    m.def("calculate_expected_buffer_size",
          &metal_sdpa::metal_sdpa::calculate_expected_buffer_size,
          "Calculate expected buffer size for given tensor and precision",
          py::arg("reference_tensor"),
          py::arg("precision"));
    */

    // Utility functions
    m.def("is_metal_available", []() { return mfa_is_device_supported(); },
          "Check if Metal is available on this device");

    m.def("get_version", []() {
        int major, minor, patch;
        mfa_get_version(&major, &minor, &patch);
        return std::make_tuple(major, minor, patch);
    }, "Get Metal Flash Attention version");

    // Backend context manager helper
    py::class_<metal_sdpa::MetalSDPABackend>(m, "MetalSDPABackend")
        .def_static("register_backend", &metal_sdpa::MetalSDPABackend::register_backend)
        .def_static("unregister_backend", &metal_sdpa::MetalSDPABackend::unregister_backend);
}
